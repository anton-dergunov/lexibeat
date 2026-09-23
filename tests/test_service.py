"""Tests for the versioned `/api/v1` surface a host integrates against."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from lexibeat.service import (
    API_PREFIX,
    API_VERSION,
    Operations,
    RenderContext,
    ServiceConfig,
    openapi_document,
    service_schema,
)
from lexibeat.voice import BackendCapabilities, SpeechRequest, SynthesisResult

try:
    from fastapi.testclient import TestClient

    from lexibeat.service import create_service
except ImportError:  # The service surface is an optional install.
    TestClient = None
    create_service = None

SNAPSHOT = Path(__file__).resolve().parents[1] / "docs" / "openapi-v1.json"


class StubBackend:
    name = "stub"
    sample_rate = 24000
    model_id = "stub/v1"
    load_seconds = 0.0
    capabilities = BackendCapabilities("instruction", "instruction", "preset",
                                       languages=())

    def synth(self, request: SpeechRequest) -> SynthesisResult:
        count = int(self.sample_rate * 0.5)
        ramp = np.linspace(0, 0.5, count, endpoint=False)
        audio = (0.4 * np.sin(2 * np.pi * 200 * ramp) * np.hanning(count)).astype(np.float32)
        return SynthesisResult(audio, self.sample_rate, 0.0, {})


def body(**overrides) -> dict:
    payload = {
        "items": [{"source": "asco", "target": "disgust",
                   "direction": "repulsed, recoiling slightly"}],
        "source_language": {"code": "es", "name": "Spanish"},
        "target_language": {"code": "en", "name": "English"},
        "seed": 5,
    }
    payload.update(overrides)
    return payload


class SchemaTests(unittest.TestCase):
    def test_the_catalogues_are_this_service_s_and_the_host_copies_none_of_them(self) -> None:
        schema = service_schema()
        self.assertEqual(schema["api_version"], API_VERSION)
        names = {row["id"] for row in schema["patterns"]}
        self.assertEqual(names, {"retrieval", "alternating"})
        retrieval = next(row for row in schema["patterns"] if row["id"] == "retrieval")
        self.assertTrue(retrieval["has_recall_gap"])
        self.assertEqual(retrieval["utterances_per_item"], 6)
        self.assertIn("auto", schema["families"])
        self.assertEqual(schema["audio"]["mime"], "audio/mpeg")
        self.assertEqual(schema["audio"]["bitrate_kbps"], 128)


@unittest.skipUnless(TestClient is not None, "install the service extra")
class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.seen: list[RenderContext] = []
        self.config = ServiceConfig(output_root=Path(self.temp.name))
        self.client = TestClient(create_service(
            config=self.config, backend_factory=self.factory))
        self.addCleanup(self.client.close)

    def factory(self, context: RenderContext):
        self.seen.append(context)
        return StubBackend()

    def follow(self, operation_id: str, limit: int = 900) -> dict:
        for _ in range(limit):
            operation = self.client.get(
                f"{API_PREFIX}/operations/{operation_id}").json()
            if operation["status"] not in ("queued", "running"):
                return operation
            time.sleep(0.2)
        raise AssertionError("The operation never finished.")

    def test_health_is_liveness_and_not_readiness(self) -> None:
        """A deployment with no sample bundle still serves; gating on it would fail the install."""
        payload = self.client.get(f"{API_PREFIX}/health").json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("production_bundle", payload)
        self.assertEqual(payload["api_version"], API_VERSION)

    def test_the_schema_says_which_bundle_is_mounted_and_whether_all_of_it_is(self) -> None:
        schema = self.client.get(f"{API_PREFIX}/schema").json()
        self.assertEqual(set(schema["bundle"]), {"present", "complete", "bundle", "version",
                                                 "assets", "missing", "expanded"})
        self.assertEqual(schema["production_bundle"], schema["bundle"]["complete"])

    def test_a_render_is_an_operation_and_the_track_is_fetched_from_it(self) -> None:
        created = self.client.post(f"{API_PREFIX}/loops", json=body())
        self.assertEqual(created.status_code, 202)
        self.assertEqual(created.json()["status"], "queued")
        operation = self.follow(created.json()["operation_id"])
        self.assertEqual(operation["status"], "completed")
        self.assertTrue(operation["successful"])
        self.assertIsNone(operation["error"])
        self.assertEqual(operation["progress"]["fraction"], 1.0)
        result = operation["result"]
        self.assertEqual(result["bitrate_kbps"], 128)
        self.assertEqual(len(result["timeline"]), 1)
        self.assertEqual(result["timeline"][0]["source"], "asco")
        audio = self.client.get(result["audio_url"])
        self.assertEqual(audio.status_code, 200)
        self.assertEqual(audio.headers["content-type"], "audio/mpeg")
        self.assertGreater(len(audio.content), 1000)

    def test_a_poll_does_not_carry_the_resolved_bed(self) -> None:
        """Style, seed and engine version replay it; the fingerprint proves the replay."""
        operation = self.follow(
            self.client.post(f"{API_PREFIX}/loops", json=body()).json()["operation_id"])
        result = operation["result"]
        self.assertNotIn("bed_spec", result)
        self.assertNotIn("audio_path", result)
        for field in ("style_id", "seed", "engine_version", "bed_fingerprint"):
            self.assertTrue(result[field] != "" and result[field] is not None, field)
        self.assertEqual(result["seed"], body()["seed"])

    def test_the_backend_is_built_per_render_from_the_credential_that_came_with_it(self) -> None:
        secret = "render-scoped-token-value"
        self.follow(self.client.post(
            f"{API_PREFIX}/loops",
            json=body(speech={"token": secret})).json()["operation_id"])
        self.assertEqual(len(self.seen), 1)
        self.assertEqual(self.seen[0].credentials, secret)
        self.assertEqual(self.seen[0].request.source_language.name, "Spanish")

    def test_the_hosts_delivery_reaches_the_backend_factory_untouched(self) -> None:
        """LexiBeat has no opinion about the value, and that is the point of passing it.

        A host voice that takes a director note and one that does not are told apart by the
        backend the host builds, which can only be built once this has arrived. Without it the
        factory has to guess, and a deployment that chose the plain voice was paying for a
        directed take on every repetition of every line.
        """
        self.follow(self.client.post(
            f"{API_PREFIX}/loops",
            json=body(speech={"token": "t", "delivery": "plain"})).json()["operation_id"])
        self.assertEqual(self.seen[0].delivery, "plain")

    def test_a_render_that_says_nothing_about_delivery_still_renders(self) -> None:
        self.follow(self.client.post(f"{API_PREFIX}/loops", json=body()).json()["operation_id"])
        self.assertEqual(self.seen[0].delivery, "")

    def test_the_credential_never_appears_in_anything_a_caller_can_read(self) -> None:
        secret = "render-scoped-token-value"
        operation = self.follow(self.client.post(
            f"{API_PREFIX}/loops",
            json=body(speech={"token": secret})).json()["operation_id"])
        self.assertNotIn(secret, json.dumps(operation))

    def test_a_render_can_be_cancelled_and_reports_itself_unsuccessful(self) -> None:
        many = [{"source": f"palabra {index}", "target": f"word {index}"}
                for index in range(40)]
        created = self.client.post(f"{API_PREFIX}/loops", json=body(items=many))
        operation_id = created.json()["operation_id"]
        cancelled = self.client.delete(f"{API_PREFIX}/operations/{operation_id}")
        self.assertEqual(cancelled.status_code, 200)
        operation = self.follow(operation_id)
        self.assertEqual(operation["status"], "cancelled")
        self.assertFalse(operation["successful"])

    def test_an_unknown_operation_and_an_unfinished_track_are_both_404(self) -> None:
        self.assertEqual(
            self.client.get(f"{API_PREFIX}/operations/missing").status_code, 404)
        self.assertEqual(
            self.client.get(f"{API_PREFIX}/loops/missing/audio").status_code, 404)

    def test_a_request_that_cannot_be_rendered_is_refused_before_the_queue(self) -> None:
        self.assertEqual(
            self.client.post(f"{API_PREFIX}/loops", json=body(items=[])).status_code, 422)
        self.assertEqual(
            self.client.post(f"{API_PREFIX}/loops",
                             json=body(pattern="waltz")).status_code, 422)
        self.assertEqual(
            self.client.post(f"{API_PREFIX}/loops",
                             json=body(unexpected="field")).status_code, 422)
        self.assertEqual(self.seen, [])

    def test_a_backend_that_fails_fails_the_operation_and_not_the_process(self) -> None:
        class Broken(StubBackend):
            def synth(self, request):
                raise RuntimeError("synthetic failure")

        self.seen.clear()
        client = TestClient(create_service(
            config=self.config, backend_factory=lambda context: Broken()))
        self.addCleanup(client.close)
        created = client.post(f"{API_PREFIX}/loops", json=body())
        operation_id = created.json()["operation_id"]
        for _ in range(900):
            operation = client.get(f"{API_PREFIX}/operations/{operation_id}").json()
            if operation["status"] not in ("queued", "running"):
                break
            time.sleep(0.2)
        self.assertEqual(operation["status"], "failed")
        self.assertFalse(operation["successful"])
        self.assertIn("synthetic failure", operation["error"])
        self.assertEqual(client.get(f"{API_PREFIX}/health").json()["status"], "ok")

    def test_the_committed_openapi_snapshot_matches_the_running_application(self) -> None:
        """A wire change that nobody meant to make shows up as a diff in this file."""
        self.assertEqual(json.loads(SNAPSHOT.read_text(encoding="utf-8")),
                         openapi_document())


class QueueTests(unittest.TestCase):
    def test_more_outstanding_renders_than_the_queue_holds_is_refused(self) -> None:
        from lexibeat.service import BusyError
        from lexibeat.loop import LoopRequest
        from lexibeat.language import ENGLISH, SPANISH
        from lexibeat.vocab import Item

        with tempfile.TemporaryDirectory() as tmp:
            operations = Operations(
                ServiceConfig(output_root=Path(tmp), max_pending=1),
                lambda context: StubBackend())
            operations._ensure_worker = lambda: None  # never drain the queue
            request = LoopRequest(items=(Item("a", "b"),), source_language=SPANISH,
                                  target_language=ENGLISH)
            operations.submit(request)
            with self.assertRaises(BusyError):
                operations.submit(request)


if __name__ == "__main__":
    unittest.main()
