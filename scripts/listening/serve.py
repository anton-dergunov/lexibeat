"""Serve one listening round to a tablet on the same network, and keep what is said about it.

    uv run python -m scripts.listening.serve out/listening/round-01 [--port 8765]

Then open the printed address on the tablet. The page reads `round.json` and the clips, and every
change to a rating is written straight back to `labels.json` beside them — the page also keeps a
draft in the browser, so a dropped connection loses nothing. Standard library only: this is a tool
for one listener on a home network, not a service.

It serves exactly the page, `round.json`, `labels.json` and the audio files the round lists — never
`key.json`, which is what keeps a blind round blind.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).with_name("page.html")
MAX_LABELS_BYTES = 2 * 1024 * 1024


def handler_for(root: Path):
    round_data = json.loads((root / "round.json").read_text(encoding="utf-8"))
    audio = {name for clip in round_data["clips"]
             for name in (clip.get("file"), clip.get("a"), clip.get("b")) if name}
    labels = root / "labels.json"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # quieter than the default
            if not self.path.endswith(".mp3"):
                super().log_message(fmt, *args)

        def _send(self, status: int, body: bytes, kind: str,
                  extra: dict[str, str] | None = None) -> None:
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
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "round.json":
                self._send(200, json.dumps(round_data).encode(), "application/json")
            elif path == "labels.json":
                body = labels.read_bytes() if labels.exists() else b"{}"
                self._send(200, body, "application/json")
            elif path in audio:
                self._audio(root / path)
            else:
                self._send(404, b"not found", "text/plain")

        def _audio(self, file: Path) -> None:
            """With byte ranges: Safari will neither play nor seek an `<audio>` without them."""
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
                       {"Accept-Ranges": "bytes",
                        "Content-Range": f"bytes {start}-{end}/{len(data)}"})

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
            partial = labels.with_suffix(".partial.json")
            partial.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(partial, labels)
            self._send(204, b"", "text/plain")

    return Handler


def addresses() -> list[str]:
    """The addresses a device on the network might reach this machine at."""
    found = {"127.0.0.1"}
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))  # a documentation address; nothing is sent
            found.add(probe.getsockname()[0])
    except OSError:
        pass
    try:
        found.update(info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None,
                                                               socket.AF_INET))
    except OSError:
        pass
    return sorted(found)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("round", type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if not (args.round / "round.json").is_file():
        raise SystemExit(f"No round.json in {args.round}")
    server = ThreadingHTTPServer((args.host, args.port), handler_for(args.round))
    print("Open one of these on the tablet:")
    for address in addresses():
        print(f"  http://{address}:{args.port}/")
    print(f"Labels are saved to {args.round / 'labels.json'}. Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
