"""FastAPI boundary for the optional LexiBeat web explorer."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from .api import BedFingerprint, MusicRequest, resolve_music
from .explorer import (
    EXPLORER_API_VERSION,
    MAX_REQUEST_BYTES,
    ArtifactStore,
    ExplorerConfig,
    RenderBusyError,
    SampleService,
    apply_safe_repairs,
    explorer_schema,
    preview_duration,
    randomize_unlocked,
    validate_bed_spec,
)
from .generator import ENGINE_VERSION
from .library import BUNDLED_ROOT


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResolveBody(StrictModel):
    music_request: dict[str, Any] = Field(default_factory=dict)
    avoid_fingerprints: list[dict[str, Any]] = Field(default_factory=list,
                                                      max_length=3)


class SpecBody(StrictModel):
    bed_spec: dict[str, Any]
    profile: str = "production-v1"


class RandomizeBody(StrictModel):
    bed_spec: dict[str, Any]
    locked_paths: list[str] = Field(default_factory=list)
    seed: int | None = None
    music_request: dict[str, Any] | None = None


class RenderBody(StrictModel):
    bed_spec: dict[str, Any]
    duration_seconds: float | None = None


class ValidationResponse(BaseModel):
    state: str
    renderable: bool
    issues: list[dict[str, Any]]
    measurements: dict[str, float]
    fingerprint: dict[str, Any] | None = None


class GenerationResponse(BaseModel):
    request: dict[str, Any]
    bed_spec: dict[str, Any]
    fingerprint: dict[str, Any]
    quality: dict[str, Any]
    sample_manifest: list[dict[str, Any]]
    engine_version: str
    profile_version: str


class RandomizeResponse(BaseModel):
    bed_spec: dict[str, Any]
    validation: ValidationResponse
    seed: int


class SafeResponse(BaseModel):
    bed_spec: dict[str, Any] | None
    validation: ValidationResponse


class RenderResponse(BaseModel):
    artifact_id: str
    audio_url: str
    duration_seconds: float
    sample_rate: int
    sha256: str
    cache_hit: bool
    validation: ValidationResponse


class SampleListResponse(BaseModel):
    items: list[dict[str, Any]]
    total: int
    offset: int
    limit: int


def create_api(*, config: ExplorerConfig | None = None, mount_ui: bool = True,
               gpu_probe: Callable[[], dict[str, str | bool]] | None = None):
    """Create the shared ASGI application without importing web dependencies early."""
    from fastapi import FastAPI, HTTPException, Query, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import FileResponse, JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware

    resolved_config = config or ExplorerConfig.from_environment()
    artifacts = ArtifactStore(resolved_config)
    samples = SampleService(resolved_config)
    work_slot = threading.BoundedSemaphore(1)
    queue_slots = threading.BoundedSemaphore(resolved_config.max_pending_renders + 1)

    @contextmanager
    def limited_work():
        if not queue_slots.acquire(blocking=False):
            raise RenderBusyError("The explorer work queue is full; try again shortly.")
        try:
            with work_slot:
                yield
        finally:
            queue_slots.release()

    class RequestSizeMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            if request.method in {"POST", "PUT", "PATCH"}:
                raw_length = request.headers.get("content-length")
                if raw_length:
                    try:
                        if int(raw_length) > MAX_REQUEST_BYTES:
                            return JSONResponse(
                                {"error": "request_too_large",
                                 "message": f"Request bodies are limited to {MAX_REQUEST_BYTES} bytes."},
                                status_code=413)
                    except ValueError:
                        return JSONResponse({"error": "invalid_content_length"},
                                            status_code=400)
                body = await request.body()
                if len(body) > MAX_REQUEST_BYTES:
                    return JSONResponse(
                        {"error": "request_too_large",
                         "message": f"Request bodies are limited to {MAX_REQUEST_BYTES} bytes."},
                        status_code=413)
            return await call_next(request)

    app = FastAPI(
        title="LexiBeat Explorer API",
        version=EXPLORER_API_VERSION,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.add_middleware(RequestSizeMiddleware)
    app.state.explorer_config = resolved_config
    app.state.artifacts = artifacts
    app.state.samples = samples

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(_request: Request, exc: RequestValidationError):
        return JSONResponse({"error": "invalid_request", "details": exc.errors()},
                            status_code=422)

    @app.exception_handler(ValueError)
    async def value_error(_request: Request, exc: ValueError):
        return JSONResponse({"error": "invalid_request", "message": str(exc)},
                            status_code=422)

    @app.exception_handler(RenderBusyError)
    async def busy_error(_request: Request, exc: RenderBusyError):
        return JSONResponse({"error": "render_queue_full", "message": str(exc)},
                            status_code=429)

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "api_version": EXPLORER_API_VERSION,
            "engine_version": ENGINE_VERSION,
            "hosted": resolved_config.hosted,
            "production_bundle": BUNDLED_ROOT.joinpath("catalog.sqlite3").exists(),
        }

    @app.get("/api/schema")
    def schema() -> dict:
        return explorer_schema(resolved_config)

    @app.post("/api/resolve", response_model=GenerationResponse)
    def resolve(body: ResolveBody) -> dict:
        request = MusicRequest.from_dict(body.music_request)
        avoid = tuple(BedFingerprint(
            family=row["family"],
            audio_features=tuple(row.get("audio_features", ())),
            motif_features=tuple(row.get("motif_features", ())),
            instrument_families=tuple(row.get("instrument_families", ())),
        ) for row in body.avoid_fingerprints)
        with limited_work():
            return resolve_music(request, avoid_fingerprints=avoid).to_dict()

    @app.post("/api/randomize", response_model=RandomizeResponse)
    def randomize(body: RandomizeBody) -> dict:
        request = (MusicRequest.from_dict(body.music_request)
                   if body.music_request is not None else None)
        with limited_work():
            return randomize_unlocked(body.bed_spec, body.locked_paths,
                                      seed=body.seed, request=request).to_dict()

    @app.post("/api/validate", response_model=ValidationResponse)
    def validate(body: SpecBody) -> dict:
        with limited_work():
            _, report = validate_bed_spec(body.bed_spec, profile_name=body.profile)
            return report.to_dict()

    @app.post("/api/return-safe", response_model=SafeResponse)
    def return_safe(body: SpecBody) -> dict:
        with limited_work():
            spec, report = apply_safe_repairs(body.bed_spec, profile_name=body.profile)
            return {"bed_spec": asdict(spec) if spec else None,
                    "validation": report.to_dict()}

    def render_response(body: RenderBody, *, preview: bool) -> dict:
        spec, report = validate_bed_spec(body.bed_spec, analyze=False)
        if spec is None or report.state == "invalid":
            raise ValueError("BedSpec is invalid: " + "; ".join(
                issue.message for issue in report.issues if issue.severity == "error"))
        duration = body.duration_seconds
        if duration is None:
            duration = preview_duration(spec) if preview else min(
                30.0, resolved_config.max_duration_seconds)
        with limited_work():
            artifact = artifacts.render(spec, duration, preview=preview)
        return {**artifact.to_dict(), "validation": report.to_dict()}

    @app.post("/api/render-preview", response_model=RenderResponse)
    def render_preview(body: RenderBody) -> dict:
        return render_response(body, preview=True)

    @app.post("/api/render", response_model=RenderResponse)
    def render(body: RenderBody) -> dict:
        return render_response(body, preview=False)

    @app.get("/api/audio/{artifact_id}")
    def audio(artifact_id: str):
        try:
            path = artifacts.path_for(artifact_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, media_type="audio/wav",
                            filename=f"lexibeat-{artifact_id[:12]}.wav")

    @app.get("/api/samples", response_model=SampleListResponse)
    def list_samples(category: str | None = None, collection: str | None = None,
                     availability: str | None = None,
                     offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=200)) -> dict:
        return samples.list(category=category, collection=collection,
                            availability=availability, offset=offset, limit=limit)

    @app.get("/api/samples/{sample_id}")
    def sample(sample_id: str) -> dict:
        try:
            return samples.get(sample_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/samples/{sample_id}/promote")
    def promote(sample_id: str) -> dict:
        try:
            return samples.promote(sample_id)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    if mount_ui:
        from .explorer_ui import build_demo
        import gradio as gr

        demo = build_demo(
            resolved_config, artifacts=artifacts, samples=samples,
            gpu_probe=gpu_probe)
        app = gr.mount_gradio_app(app, demo, path="/", ssr_mode=False)
        app.state.explorer_config = resolved_config
        app.state.artifacts = artifacts
        app.state.samples = samples
        app.state.demo = demo
    return app


def run_local(*, host: str = "127.0.0.1", port: int = 7860,
              open_browser: bool = True) -> None:
    import threading
    import webbrowser

    import uvicorn

    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_api(), host=host, port=port)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the LexiBeat music explorer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-open", action="store_true",
                        help="do not open the default browser")
    args = parser.parse_args()
    run_local(host=args.host, port=args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
