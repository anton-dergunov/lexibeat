"""The versioned HTTP surface: ``/api/v1``.

What came before this was built to *demonstrate* the music engine — the lesson flow was reachable
in-process or through Gradio and nowhere else, and its input was a two-column table capped at six
rows. This is the surface a host integrates against, and three things about it are deliberate.

**A render is an operation, not a request.** Seventy-odd speech calls plus a bed and a mix is
minutes, not seconds. ``POST /api/v1/loops`` answers an operation id and the caller follows it, so
progress is watchable, cancellation lands between utterances, and nothing holds a connection open
for four minutes. The vocabulary — ``operation_id``, ``status``, ``successful``, ``error`` — is the
one the host already follows for its other companion service, so following this one is a second
instance of a pattern rather than a second pattern.

**The speech backend is injected per render.** LexiBeat holds no provider credential, no chain and
no voice map, because the host holds all three and copying them here would mean a second copy of
every key on the same machine. ``backend_factory`` is given the validated request and whatever
render-scoped credential came with it, and returns something satisfying the ``Backend`` protocol.
Anything it returns works, because dispatch reads capabilities rather than names.

**The bundle is not a readiness gate.** ``/health`` asserts liveness only. A fresh deployment has no
sample bundle and still serves — the engine simply offers the sample-free palette — and a readiness
gate would fail the install of a working service.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from .arrange import PATTERNS, Cancelled
from .bedspec import TIMBRE_PALETTES
from .generator import ENGINE_VERSION
from .language import Language
from .loop import (
    MAX_DIRECTION_CHARACTERS,
    MAX_ITEMS,
    MAX_TEXT_CHARACTERS,
    MP3_BITRATE_KBPS,
    MP3_MIME,
    LoopError,
    LoopRequest,
    LoopResult,
    render_loop,
)
from .paths import configured_bundle_root
from .profiles import PROFILES, get_profile
from .vocab import Item
from .voice import Backend, register_secret

API_VERSION = "1.0.0"
API_PREFIX = "/api/v1"
MAX_REQUEST_BYTES = 1_000_000

QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"
FINISHED = (COMPLETED, FAILED, CANCELLED)


@dataclass(frozen=True)
class ServiceConfig:
    output_root: Path = Path("out/service")
    max_pending: int = 8
    retain_loops: int = 32

    @classmethod
    def from_environment(cls) -> "ServiceConfig":
        import os

        return cls(
            output_root=Path(os.environ.get("LEXIBEAT_SERVICE_OUT", "out/service")),
            max_pending=max(int(os.environ.get("LEXIBEAT_SERVICE_QUEUE", "8")), 1),
            retain_loops=max(int(os.environ.get("LEXIBEAT_SERVICE_RETAIN", "32")), 1),
        )

    @property
    def loops_root(self) -> Path:
        return self.output_root / "loops"


@dataclass(frozen=True)
class RenderContext:
    """What a host needs to build a backend for one render.

    ``credentials`` is whatever short-lived secret travelled with the request. It is registered with
    the provider-text redactor and is otherwise never written down: not in an operation, not in a
    log line, not in the result.
    """

    operation_id: str
    request: LoopRequest
    credentials: str = ""


BackendFactory = Callable[[RenderContext], Backend]


@dataclass
class Operation:
    id: str
    status: str = QUEUED
    fraction: float = 0.0
    message: str = "Queued"
    error: str | None = None
    result: LoopResult | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel: threading.Event = field(default_factory=threading.Event)

    @property
    def finished(self) -> bool:
        return self.status in FINISHED

    def to_dict(self) -> dict[str, Any]:
        successful: bool | None = None
        if self.status == COMPLETED:
            successful = True
        elif self.status in (FAILED, CANCELLED):
            successful = False
        body: dict[str, Any] = {
            "operation_id": self.id,
            "status": self.status,
            "successful": successful,
            "error": self.error,
            "progress": {"fraction": round(self.fraction, 4), "message": self.message},
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": None,
        }
        if self.result is not None:
            result = self.result.to_dict()
            result.pop("audio_path", None)
            # The bed replays from style, seed and engine version, and `bed_fingerprint` proves a
            # replay produced the same one. Sending the whole resolved BedSpec on every poll would
            # be kilobytes of JSON a host is told not to store.
            result.pop("bed_spec", None)
            result["audio_url"] = f"{API_PREFIX}/loops/{self.id}/audio"
            result["bitrate_kbps"] = MP3_BITRATE_KBPS
            body["result"] = result
        return body


class Operations:
    """One render at a time, a bounded queue, and the oldest finished work pruned."""

    def __init__(self, config: ServiceConfig, backend_factory: BackendFactory) -> None:
        self.config = config
        self.backend_factory = backend_factory
        self._lock = threading.Lock()
        self._operations: dict[str, Operation] = {}
        self._order: list[str] = []
        self._queue: queue.Queue[str] = queue.Queue()
        self._worker: threading.Thread | None = None
        # The request and its credential live here and nowhere else, and are popped the moment the
        # worker picks the render up: an operation a caller can read must not carry a token.
        self._pending: dict[str, tuple[LoopRequest, str]] = {}

    # -- lifecycle -------------------------------------------------------

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run, name="lexibeat-render",
                                            daemon=True)
            self._worker.start()

    def submit(self, request: LoopRequest, credentials: str = "") -> Operation:
        with self._lock:
            pending = sum(1 for row in self._operations.values() if not row.finished)
            if pending >= self.config.max_pending:
                raise BusyError("The render queue is full; try again shortly.")
            operation = Operation(id=uuid.uuid4().hex)
            self._operations[operation.id] = operation
            self._order.append(operation.id)
            self._pending[operation.id] = (request, credentials)
        self._ensure_worker()
        self._queue.put(operation.id)
        return operation

    def get(self, operation_id: str) -> Operation:
        with self._lock:
            operation = self._operations.get(operation_id)
        if operation is None:
            raise KeyError(operation_id)
        return operation

    def cancel(self, operation_id: str) -> Operation:
        operation = self.get(operation_id)
        if not operation.finished:
            operation.cancel.set()
            if operation.status == QUEUED:
                self._finish(operation, CANCELLED, "The render was cancelled.")
        return operation

    def path_for(self, operation_id: str) -> Path:
        operation = self.get(operation_id)
        if operation.status != COMPLETED or operation.result is None:
            raise FileNotFoundError(f"Operation {operation_id} has no track.")
        path = Path(operation.result.audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"The track for {operation_id} is no longer on disk.")
        return path

    # -- the worker ------------------------------------------------------

    def _note(self, operation: Operation, fraction: float, message: str) -> None:
        operation.fraction = float(fraction)
        operation.message = message
        operation.updated_at = time.time()

    def _finish(self, operation: Operation, status: str, error: str | None = None,
                result: LoopResult | None = None) -> None:
        operation.status = status
        operation.error = error
        operation.result = result
        operation.updated_at = time.time()
        if status == COMPLETED:
            operation.fraction = 1.0
            operation.message = "Loop ready"
        elif status == CANCELLED:
            operation.message = "Cancelled"
        else:
            operation.message = "Failed"

    def _run(self) -> None:
        while True:
            operation_id = self._queue.get()
            try:
                self._render(operation_id)
            finally:
                self._queue.task_done()
                self._prune()

    def _render(self, operation_id: str) -> None:
        with self._lock:
            operation = self._operations.get(operation_id)
            pending = self._pending.pop(operation_id, None)
        if operation is None or pending is None or operation.finished:
            return
        request, credentials = pending
        operation.status = RUNNING
        self._note(operation, 0.01, "Starting")
        output = self.config.loops_root / f"{operation_id}.mp3"
        try:
            backend = self.backend_factory(
                RenderContext(operation_id=operation_id, request=request,
                              credentials=credentials))
            result = render_loop(
                request, backend=backend, output=output,
                progress=lambda fraction, message: self._note(operation, fraction, message),
                cancel_check=operation.cancel.is_set)
        except Cancelled:
            self._finish(operation, CANCELLED, "The render was cancelled.")
        except Exception as exc:  # the operation carries the failure; the process carries on
            self._finish(operation, FAILED, f"{type(exc).__name__}: {exc}")
        else:
            self._finish(operation, COMPLETED, result=result)

    def _prune(self) -> None:
        """Keep the most recent finished renders and drop the rest, file and record together."""
        with self._lock:
            finished = [oid for oid in self._order
                        if oid in self._operations and self._operations[oid].finished]
            excess = finished[:max(len(finished) - self.config.retain_loops, 0)]
            for oid in excess:
                operation = self._operations.pop(oid, None)
                self._order.remove(oid)
                if operation and operation.result:
                    Path(operation.result.audio_path).unlink(missing_ok=True)


class BusyError(RuntimeError):
    """More renders are outstanding than this service will hold."""


# -- request bodies ------------------------------------------------------


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LanguageBody(StrictModel):
    code: str = Field(min_length=1, max_length=32)
    name: str = Field(default="", max_length=64)


class ItemBody(StrictModel):
    source: str = Field(min_length=1, max_length=MAX_TEXT_CHARACTERS)
    target: str = Field(min_length=1, max_length=MAX_TEXT_CHARACTERS)
    direction: str = Field(default="", max_length=MAX_DIRECTION_CHARACTERS)


class SpeechBody(StrictModel):
    """A render-scoped credential for the injected backend. Never stored, never logged."""

    token: str = Field(default="", max_length=8192)


class LoopBody(StrictModel):
    items: list[ItemBody] = Field(min_length=1, max_length=MAX_ITEMS)
    source_language: LanguageBody
    target_language: LanguageBody
    pattern: str = "retrieval"
    family: str = "auto"
    energy: str = "balanced"
    rhythm: str = "steady"
    palette: str = "hybrid"
    seed: int | None = None
    profile: str = "production-v1"
    prosody_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    voice_seed: int | None = None
    speech: SpeechBody | None = None

    def to_request(self) -> LoopRequest:
        return LoopRequest(
            items=tuple(Item(row.source, row.target, row.direction) for row in self.items),
            source_language=Language(self.source_language.code, self.source_language.name),
            target_language=Language(self.target_language.code, self.target_language.name),
            pattern=self.pattern, family=self.family, energy=self.energy,
            rhythm=self.rhythm, palette=self.palette, seed=self.seed,
            profile=self.profile, prosody_strength=self.prosody_strength,
            voice_seed=self.voice_seed,
        )


def service_schema(config: ServiceConfig | None = None) -> dict[str, Any]:
    """What this deployment can be asked for. The host reads its catalogues from here.

    Patterns and families are LexiBeat's, never copied into the host: a family added in a later
    version appears in the host's dialog with nothing changing there.
    """
    del config
    bundle = configured_bundle_root().joinpath("catalog.sqlite3").is_file()
    return {
        "api_version": API_VERSION,
        "engine_version": ENGINE_VERSION,
        "production_bundle": bundle,
        "patterns": [
            {"id": name,
             "bars_per_item": len(slots),
             "utterances_per_item": sum(1 for kind, _ in slots
                                        if kind not in ("gap", "rest")),
             "has_recall_gap": any(kind == "gap" for kind, _ in slots)}
            for name, slots in sorted(PATTERNS.items())
        ],
        "profiles": {
            name: {"version": profile.version, "families": list(profile.families)}
            for name, profile in PROFILES.items()
        },
        "families": ["auto", *get_profile("production-v1").families],
        "energy": ["calm", "balanced", "bright"],
        "rhythm": ["sparse", "steady", "groovy"],
        "palette": list(TIMBRE_PALETTES) if bundle else ["electronic"],
        "limits": {
            "max_items": MAX_ITEMS,
            "max_text_characters": MAX_TEXT_CHARACTERS,
            "max_direction_characters": MAX_DIRECTION_CHARACTERS,
            "request_bytes": MAX_REQUEST_BYTES,
        },
        "audio": {"mime": MP3_MIME, "bitrate_kbps": MP3_BITRATE_KBPS},
    }


def create_service(*, config: ServiceConfig | None = None,
                   backend_factory: BackendFactory | None = None,
                   operations: Operations | None = None):
    """Build the ASGI application. Importing web dependencies stays inside this function."""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import FileResponse, JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware

    resolved = config or ServiceConfig.from_environment()
    if operations is None:
        if backend_factory is None:
            raise ValueError(
                "create_service needs a backend_factory: LexiBeat holds no provider "
                "credential, so the host supplies the voice.")
        operations = Operations(resolved, backend_factory)

    app = FastAPI(
        title="LexiBeat",
        version=API_VERSION,
        docs_url=f"{API_PREFIX}/docs",
        openapi_url=f"{API_PREFIX}/openapi.json",
    )

    class RequestSizeMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            if request.method in {"POST", "PUT", "PATCH"}:
                raw_length = request.headers.get("content-length")
                if raw_length:
                    try:
                        if int(raw_length) > MAX_REQUEST_BYTES:
                            return JSONResponse(
                                {"error": {"code": "request_too_large",
                                           "message": "Request bodies are limited to "
                                                      f"{MAX_REQUEST_BYTES} bytes."}},
                                status_code=413)
                    except ValueError:
                        return JSONResponse(
                            {"error": {"code": "invalid_content_length",
                                       "message": "Content-Length is not a number."}},
                            status_code=400)
            return await call_next(request)

    app.add_middleware(RequestSizeMiddleware)
    app.state.service_config = resolved
    app.state.operations = operations

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, exc: RequestValidationError):
        return JSONResponse({"error": {"code": "invalid_request",
                                       "message": "The request body is not valid.",
                                       "details": exc.errors()}}, status_code=422)

    @app.exception_handler(LoopError)
    async def invalid_loop(_request: Request, exc: LoopError):
        return JSONResponse({"error": {"code": "invalid_request", "message": str(exc)}},
                            status_code=422)

    @app.exception_handler(ValueError)
    async def invalid_value(_request: Request, exc: ValueError):
        return JSONResponse({"error": {"code": "invalid_request", "message": str(exc)}},
                            status_code=422)

    @app.exception_handler(BusyError)
    async def busy(_request: Request, exc: BusyError):
        return JSONResponse({"error": {"code": "queue_full", "message": str(exc)}},
                            status_code=429)

    @app.get(f"{API_PREFIX}/health")
    def health() -> dict:
        """Liveness. Deliberately not readiness: a deployment with no bundle still serves."""
        return {
            "status": "ok",
            "api_version": API_VERSION,
            "engine_version": ENGINE_VERSION,
            "production_bundle": configured_bundle_root()
            .joinpath("catalog.sqlite3").is_file(),
        }

    @app.get(f"{API_PREFIX}/schema")
    def schema() -> dict:
        return service_schema(resolved)

    @app.post(f"{API_PREFIX}/loops", status_code=202)
    def create_loop(body: LoopBody) -> dict:
        request = body.to_request().validated()
        token = body.speech.token if body.speech else ""
        if token:
            register_secret(token)
        return operations.submit(request, token).to_dict()

    @app.get(f"{API_PREFIX}/operations/{{operation_id}}")
    def read_operation(operation_id: str) -> dict:
        try:
            return operations.get(operation_id).to_dict()
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail={"code": "unknown_operation",
                        "message": f"No operation {operation_id}."}) from exc

    @app.delete(f"{API_PREFIX}/operations/{{operation_id}}")
    def cancel_operation(operation_id: str) -> dict:
        try:
            return operations.cancel(operation_id).to_dict()
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail={"code": "unknown_operation",
                        "message": f"No operation {operation_id}."}) from exc

    @app.get(f"{API_PREFIX}/loops/{{operation_id}}/audio")
    def read_audio(operation_id: str):
        try:
            path = operations.path_for(operation_id)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(
                status_code=404,
                detail={"code": "unknown_track", "message": str(exc)}) from exc
        return FileResponse(path, media_type=MP3_MIME,
                            filename=f"lexibeat-{operation_id[:12]}.mp3")

    return app


def openapi_document(**kwargs) -> dict[str, Any]:
    """The snapshot committed as `docs/openapi-v1.json`, so a wire change shows up in a diff."""
    app = create_service(backend_factory=lambda context: None, **kwargs)  # type: ignore[arg-type,return-value]
    return app.openapi()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the LexiBeat loop service.")
    parser.add_argument("command", nargs="?", default="serve",
                        choices=("serve", "openapi"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "docs" / "openapi-v1.json")
    args = parser.parse_args()

    if args.command == "openapi":
        document = json.dumps(openapi_document(), indent=2, ensure_ascii=False) + "\n"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(document, encoding="utf-8")
        print(f"Wrote {args.out}")
        return

    raise SystemExit(
        "LexiBeat holds no provider credential, so it cannot serve on its own: a host builds "
        "the application with create_service(backend_factory=...) and runs it with uvicorn. "
        "Use 'openapi' to refresh the committed wire snapshot.")


if __name__ == "__main__":
    main()
