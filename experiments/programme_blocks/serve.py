"""Serve the page to a tablet on the same network, and keep the scores it sends back.

    uv run python experiments/programme_blocks/run.py serve [--port 8765]

It serves `out/index.html`, the MP3s under `out/` (with byte ranges, since Safari will neither play
nor seek an `<audio>` without them) and `out/labels.json`, which the page reads and writes. Nothing
else is served, and the caches in particular are not. Standard library only, like
`scripts/listening/serve.py`, whose range and atomic-write handling this copies: this is a tool for
one listener on a home network, not a service.
"""

from __future__ import annotations

import json
import os
import re
import socket
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from common import OUT

LABELS = OUT / "labels.json"
MAX_LABELS_BYTES = 4 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        if not self.path.endswith(".mp3"):
            super().log_message(fmt, *args)

    def _send(self, status: int, body: bytes, kind: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].lstrip("/")
        if path in ("", "index.html"):
            self._send(200, (OUT / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "labels.json":
            self._send(200, LABELS.read_bytes() if LABELS.exists() else b"{}", "application/json")
        elif path.endswith(".mp3") and not path.startswith("cache/"):
            file = (OUT / path).resolve()
            if OUT.resolve() in file.parents and file.is_file():
                self._audio(file)
            else:
                self._send(404, b"not found", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def _audio(self, file: Path) -> None:
        data = file.read_bytes()
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if not match:
            self._send(200, data, "audio/mpeg", {"Accept-Ranges": "bytes"})
            return
        first, last = match.groups()
        start = int(first) if first else max(len(data) - int(last or 0), 0)
        end = min(int(last), len(data) - 1) if first and last else len(data) - 1
        if start > end:
            self._send(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, b"", "audio/mpeg",
                       {"Content-Range": f"bytes */{len(data)}"})
            return
        self._send(206, data[start:end + 1], "audio/mpeg",
                   {"Accept-Ranges": "bytes", "Content-Range": f"bytes {start}-{end}/{len(data)}"})

    def do_PUT(self) -> None:
        if self.path.split("?", 1)[0].lstrip("/") != "labels.json":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_LABELS_BYTES:
            self._send(413, b"too large", "text/plain")
            return
        try:
            body = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._send(400, b"not JSON", "text/plain")
            return
        if not isinstance(body, dict):
            self._send(400, b"expected an object", "text/plain")
            return
        partial = LABELS.with_suffix(".partial.json")
        partial.write_text(json.dumps(body, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(partial, LABELS)
        self._send(204, b"", "text/plain")


def addresses() -> list[str]:
    found = {"127.0.0.1"}
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))  # a documentation address; nothing is sent
            found.add(probe.getsockname()[0])
    except OSError:
        pass
    return sorted(found)


def main(port: int = 8765) -> int:
    if not (OUT / "index.html").exists():
        raise SystemExit("No page yet: run a stage first, or `run.py page`.")
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print("Open one of these on the tablet:")
    for address in addresses():
        print(f"  http://{address}:{port}/")
    print(f"Scores are saved to {LABELS}. Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
