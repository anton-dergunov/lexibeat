"""The listening tools: what the page is told a bed is made of, and what the server will hand out."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from lexibeat.bedspec import BedSpec
from scripts.listening.common import note_name, numeral, off_key_fifth
from scripts.listening.serve import handler_for


class FactsTests(unittest.TestCase):
    def test_numerals_carry_quality(self) -> None:
        spec = BedSpec.from_style("radiant", 1)
        spec.scale = "major"
        self.assertEqual([numeral(spec, degree) for degree in range(7)],
                         ["I", "ii", "iii", "IV", "V", "vi", "vii°"])
        self.assertTrue(off_key_fifth(spec, 6))
        self.assertFalse(off_key_fifth(spec, 4))

    def test_note_names(self) -> None:
        self.assertEqual(note_name(60), "C4")
        self.assertEqual(note_name(88), "E6")


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "round.json").write_text(json.dumps({"kind": "pairs", "clips": [
            {"id": "p01", "listen_for": "the chords", "a": "p01-a.mp3", "b": "p01-b.mp3"}]}))
        (root / "key.json").write_text(json.dumps({"p01": {"fixed": "a"}}))
        (root / "p01-a.mp3").write_bytes(bytes(range(200)))
        (root / "p01-b.mp3").write_bytes(b"b" * 50)
        self.root = root
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(root))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def status(self, path: str, **kwargs) -> int:
        try:
            with urllib.request.urlopen(urllib.request.Request(self.base + path, **kwargs)) as r:
                return r.status
        except urllib.error.HTTPError as error:
            return error.code

    def test_a_blind_round_stays_blind(self) -> None:
        self.assertEqual(self.status("/round.json"), 200)
        self.assertEqual(self.status("/key.json"), 404)
        self.assertEqual(self.status("/../key.json"), 404)

    def test_audio_answers_a_byte_range(self) -> None:
        request = urllib.request.Request(self.base + "/p01-a.mp3",
                                         headers={"Range": "bytes=10-19"})
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), bytes(range(10, 20)))
            self.assertEqual(response.headers["Content-Range"], "bytes 10-19/200")

    def test_labels_are_kept_and_nothing_else_can_be_written(self) -> None:
        body = json.dumps({"p01": {"choice": "a"}}).encode()
        self.assertEqual(self.status("/labels.json", data=body, method="PUT"), 204)
        self.assertEqual(json.loads((self.root / "labels.json").read_text()),
                         {"p01": {"choice": "a"}})
        self.assertEqual(self.status("/key.json", data=body, method="PUT"), 404)
        self.assertEqual(self.status("/labels.json", data=b"[1]", method="PUT"), 400)


if __name__ == "__main__":
    unittest.main()
