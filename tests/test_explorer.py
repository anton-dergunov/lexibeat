"""Tests for the optional music explorer service and HTTP boundary."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from lexibeat import MusicRequest, resolve_music
from lexibeat.explorer import (
    MAX_REQUEST_BYTES,
    ArtifactStore,
    ExplorerConfig,
    RenderCancelledError,
    SampleService,
    apply_safe_repairs,
    explorer_schema,
    randomize_unlocked,
    validate_bed_spec,
)
from lexibeat.library import SampleLibrary, configured_bundle_root, local_root
from lexibeat.explorer_ui import build_demo
from lexibeat.lesson import (
    DEFAULT_LESSON_ROWS,
    LESSON_MODEL,
    finalize_lesson,
    lesson_gpu_duration,
    normalize_lesson_rows,
    render_lesson_speech,
    resolve_lesson_spec,
)
from lexibeat.music import SR, Grid
from lexibeat.voice import CAPABILITIES, SynthesisResult

try:
    from fastapi.testclient import TestClient
    from lexibeat.explorer_web import create_api
except ImportError:  # The explorer remains an optional install.
    TestClient = None
    create_api = None


class ExplorerCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = resolve_music(MusicRequest(
            family="warm-motion", palette="electronic", seed=1234))
        cls.data = asdict(cls.result.bed_spec)

    def test_schema_reports_versions_controls_and_hosted_limits(self) -> None:
        schema = explorer_schema(ExplorerConfig(hosted=True))
        self.assertEqual(schema["api_version"], "explorer-v1")
        self.assertEqual(schema["engine_version"], "1.0.0")
        self.assertEqual(schema["limits"]["render_seconds"], 30.0)
        self.assertFalse(schema["capabilities"]["sample_promotion"])
        self.assertIn("/bpm", schema["lockable_paths"])

    def test_schema_offers_only_electronic_without_production_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
                "lexibeat.explorer.BUNDLED_ROOT", Path(tmp)):
            schema = explorer_schema(ExplorerConfig(hosted=True))
        self.assertEqual(schema["simple"]["palette"], ["electronic"])
        self.assertFalse(schema["capabilities"]["production_bundle"])

    def test_strict_unknown_field_and_invalid_pattern_are_rejected(self) -> None:
        unknown = copy.deepcopy(self.data)
        unknown["client_path"] = "/tmp/anything.wav"
        _, report = validate_bed_spec(unknown, analyze=False)
        self.assertEqual(report.state, "invalid")
        self.assertIn("/client_path", {issue.path for issue in report.issues})

        invalid = copy.deepcopy(self.data)
        invalid["phrase"]["percussion"][0]["pattern"] = "x"
        _, report = validate_bed_spec(invalid, analyze=False)
        self.assertEqual(report.state, "invalid")
        self.assertTrue(any(issue.code == "pattern_length" for issue in report.issues))

    def test_experimental_value_has_explicit_repair(self) -> None:
        experimental = copy.deepcopy(self.data)
        experimental["bpm"] = 120
        _, report = validate_bed_spec(experimental, analyze=False)
        self.assertEqual(report.state, "experimental")
        issue = next(issue for issue in report.issues if issue.path == "/bpm")
        self.assertTrue(issue.has_safe_value)
        self.assertEqual(issue.safe_value, 104)

        repaired, repaired_report = apply_safe_repairs(experimental)
        self.assertIsNotNone(repaired)
        self.assertEqual(repaired.bpm, 104)
        self.assertNotEqual(repaired_report.state, "invalid")

    def test_lock_aware_randomization_is_deterministic_and_preserves_lock(self) -> None:
        request = MusicRequest(family="warm-motion", palette="electronic")
        with mock.patch("urllib.request.urlopen") as network:
            first = randomize_unlocked(self.data, ["/bpm"], seed=55, request=request)
            second = randomize_unlocked(self.data, ["/bpm"], seed=55, request=request)
        network.assert_not_called()
        self.assertEqual(first.bed_spec.to_json(), second.bed_spec.to_json())
        self.assertEqual(first.bed_spec.bpm, self.data["bpm"])
        with self.assertRaisesRegex(ValueError, "cannot be locked"):
            randomize_unlocked(self.data, ["/seed"], seed=55, request=request)

    def test_cache_key_deduplication_invalidation_and_managed_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(ExplorerConfig(output_root=Path(tmp)))
            first = store.render(self.result.bed_spec, 3.0, preview=True)
            second = store.render(self.result.bed_spec, 3.0, preview=True)
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(first.artifact_id, second.artifact_id)
            audio, rate = sf.read(store.path_for(first.artifact_id), always_2d=True)
            self.assertEqual(rate, 44_100)
            self.assertEqual(audio.shape[1], 2)
            self.assertTrue(np.isfinite(audio).all())
            with self.assertRaises(FileNotFoundError):
                store.path_for("../../etc/passwd")
            with mock.patch("lexibeat.explorer.ENGINE_VERSION", "test-next"):
                changed = store.render(self.result.bed_spec, 3.0, preview=True)
            self.assertNotEqual(first.artifact_id, changed.artifact_id)

    def test_cancellation_leaves_no_partial_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(ExplorerConfig(output_root=Path(tmp)))
            with self.assertRaises(RenderCancelledError):
                store.render(self.result.bed_spec, 3.0, preview=True,
                             cancelled=lambda: True)
            self.assertEqual(list(Path(tmp).glob("*")), [])

    def test_hosted_sample_promotion_is_disabled_before_any_copy(self) -> None:
        service = SampleService(ExplorerConfig(hosted=True))
        with self.assertRaisesRegex(PermissionError, "disabled"):
            service.promote("vcsl:00000000000000000000")

    def test_default_sample_cache_is_portable(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True), \
                mock.patch("pathlib.Path.home", return_value=Path("/home/runner")):
            self.assertEqual(local_root(), Path("/home/runner/.cache/lexibeat"))
            library = SampleLibrary()
            self.assertEqual(library.local, Path("/home/runner/.cache/lexibeat"))

        with mock.patch.dict("os.environ", {"XDG_CACHE_HOME": "/tmp/cache"}, clear=True):
            self.assertEqual(local_root(), Path("/tmp/cache/lexibeat"))

    def test_bundle_root_can_select_an_attached_volume(self) -> None:
        with mock.patch.dict(
                "os.environ", {"LEXIBEAT_BUNDLE_ROOT": "/data/custom/v1"},
                clear=True):
            self.assertEqual(configured_bundle_root(), Path("/data/custom/v1"))

    def test_space_entrypoint_builds_a_local_gradio_demo(self) -> None:
        import app as space_entrypoint

        self.assertFalse(space_entrypoint.config.hosted)
        self.assertIsNotNone(space_entrypoint.demo)
        self.assertTrue(callable(space_entrypoint.generate_hosted_lesson))
        source = Path(space_entrypoint.__file__).read_text(encoding="utf-8")
        self.assertIn("@spaces.GPU(duration=lesson_gpu_duration)", source)
        self.assertIn(
            "lesson_generate=generate_hosted_lesson if config.hosted else None",
            source)


class LessonTests(unittest.TestCase):
    class FakeBackend:
        name = "chatterbox"
        model_id = "fake-chatterbox"
        sample_rate = SR
        load_seconds = 0.0
        capabilities = CAPABILITIES["chatterbox"]

        def synth(self, text, lang, prosody, emotion, target_seconds=None,
                  seed=None):
            del text, lang, prosody, emotion, target_seconds, seed
            audio = np.sin(np.linspace(0, 50, SR // 10)).astype(np.float32) * 0.1
            return SynthesisResult(audio, SR, 0.001)

    def test_vocabulary_table_validation_and_duration(self) -> None:
        rows = [[" hola ", " hello "], ["", ""], ["hola", "hello"]]
        items = normalize_lesson_rows(rows)
        self.assertEqual([(item.source, item.target) for item in items],
                         [("hola", "hello"), ("hola", "hello")])
        self.assertEqual(lesson_gpu_duration(DEFAULT_LESSON_ROWS), 90)
        self.assertEqual(lesson_gpu_duration([["hola", "hello"]] * 6), 120)
        with self.assertRaisesRegex(ValueError, "both Spanish and English"):
            normalize_lesson_rows([["hola", ""]])
        with self.assertRaisesRegex(ValueError, "at least one"):
            normalize_lesson_rows([["", ""]])
        with self.assertRaisesRegex(ValueError, "120-character"):
            normalize_lesson_rows([["a" * 121, "word"]])
        with self.assertRaisesRegex(ValueError, "limited to 6"):
            normalize_lesson_rows([[str(i), str(i)] for i in range(7)])

    def test_current_bed_is_reused_and_default_is_safe(self) -> None:
        current = resolve_music(MusicRequest(
            family="warm-motion", palette="electronic", seed=42)).bed_spec
        reused = resolve_lesson_spec({"bed_spec": asdict(current)})
        self.assertEqual(reused.to_json(), current.to_json())
        generated = resolve_lesson_spec({}, palette="electronic")
        _, report = validate_bed_spec(generated, analyze=False)
        self.assertEqual(report.state, "production-safe")

    def test_speech_cache_downbeats_subtitles_and_final_mix(self) -> None:
        result = resolve_music(MusicRequest(
            family="warm-motion", palette="electronic", seed=73))
        state = {"bed_spec": asdict(result.bed_spec)}
        with tempfile.TemporaryDirectory() as tmp:
            config = ExplorerConfig(output_root=Path(tmp))
            first = render_lesson_speech(
                [["hola", "hello"]], LESSON_MODEL, state,
                backend=self.FakeBackend(), config=config)
            second = render_lesson_speech(
                [["hola", "hello"]], LESSON_MODEL, state,
                backend=self.FakeBackend(), config=config)
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(len(first["subtitles"]), 6)
            self.assertEqual(
                [row["text"] for row in first["subtitles"]],
                ["hola", "hello", "hola", "hello", "hola", "hello"])
            grid = Grid.from_spec(result.bed_spec)
            self.assertEqual(
                [round(row["timestamp"][0] / grid.bar)
                 for row in first["subtitles"]],
                [2, 4, 5, 6, 7, 8])
            self.assertEqual(
                [round(row["timestamp"][1] / grid.bar)
                 for row in first["subtitles"]],
                [4, 5, 6, 7, 8, 10])
            for index, row in enumerate(first["subtitles"]):
                self.assertAlmostEqual(row["timestamp"][0] / grid.bar,
                                       round(row["timestamp"][0] / grid.bar))
                self.assertGreater(row["timestamp"][1], row["timestamp"][0])
                if index + 1 < len(first["subtitles"]):
                    self.assertEqual(
                        row["timestamp"][1],
                        first["subtitles"][index + 1]["timestamp"][0])

            mixed = finalize_lesson(first, config=config)
            audio, rate = sf.read(mixed["audio_path"], always_2d=True)
            self.assertEqual(rate, SR)
            self.assertEqual(audio.shape[1], 2)
            self.assertTrue(np.isfinite(audio).all())
            self.assertLessEqual(float(np.abs(audio).max()), 0.97 + 1e-4)
            self.assertEqual(len(mixed["subtitles"]), 6)

            changed = render_lesson_speech(
                [["adiós", "goodbye"]], LESSON_MODEL, state,
                backend=self.FakeBackend(), config=config)
            self.assertNotEqual(first["speech_path"], changed["speech_path"])

            class OtherModel(self.FakeBackend):
                model_id = "fake-chatterbox-next"

            model_changed = render_lesson_speech(
                [["hola", "hello"]], LESSON_MODEL, state,
                backend=OtherModel(), config=config)
            self.assertNotEqual(first["speech_path"], model_changed["speech_path"])

            seed_changed = render_lesson_speech(
                [["hola", "hello"]], LESSON_MODEL, state,
                backend=self.FakeBackend(), config=config, voice_seed=999)
            self.assertNotEqual(first["speech_path"], seed_changed["speech_path"])

            other_bed = resolve_music(MusicRequest(
                family="warm-motion", palette="electronic", seed=74)).bed_spec
            bed_changed = render_lesson_speech(
                [["hola", "hello"]], LESSON_MODEL,
                {"bed_spec": asdict(other_bed)}, backend=self.FakeBackend(),
                config=config)
            self.assertNotEqual(first["speech_path"], bed_changed["speech_path"])

            progress_events: list[tuple[float, str]] = []
            render_lesson_speech(
                [["uno", "one"]], LESSON_MODEL, state,
                backend=self.FakeBackend(), config=config,
                progress=lambda value, message:
                    progress_events.append((value, message)))
            messages = [message for _, message in progress_events]
            self.assertIn("Validating vocabulary", messages)
            self.assertTrue(any("Synthesizing 1 of 6" in message
                                for message in messages))
            self.assertEqual(messages[-1], "Speech synthesis complete")

    def test_failed_speech_render_removes_partial_files(self) -> None:
        class BrokenBackend(self.FakeBackend):
            def synth(self, *args, **kwargs):
                raise RuntimeError("synthetic failure")

        result = resolve_music(MusicRequest(palette="electronic", seed=19))
        with tempfile.TemporaryDirectory() as tmp:
            config = ExplorerConfig(output_root=Path(tmp))
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                render_lesson_speech(
                    [["hola", "hello"]], LESSON_MODEL,
                    {"bed_spec": asdict(result.bed_spec)},
                    backend=BrokenBackend(), config=config)
            self.assertEqual(list(Path(tmp).rglob("*.partial.*")), [])


@unittest.skipUnless(TestClient is not None, "install the explorer extra")
class ExplorerHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        config = ExplorerConfig(output_root=Path(cls.temp.name))
        cls.client = TestClient(create_api(config=config, mount_ui=False))
        response = cls.client.post("/api/resolve", json={
            "music_request": {
                "family": "warm-motion", "palette": "electronic", "seed": 1234,
            }
        })
        if response.status_code != 200:
            raise AssertionError(response.text)
        cls.spec = response.json()["bed_spec"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
        cls.temp.cleanup()

    def test_health_schema_and_strict_request_models(self) -> None:
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["api_version"], "explorer-v1")
        self.assertEqual(self.client.get("/api/schema").status_code, 200)
        bad = self.client.post("/api/resolve", json={"music_request": {}, "extra": 1})
        self.assertEqual(bad.status_code, 422)

    def test_oversized_request_is_rejected(self) -> None:
        response = self.client.post(
            "/api/resolve", content=b"{}",
            headers={"content-type": "application/json",
                     "content-length": str(MAX_REQUEST_BYTES + 1)})
        self.assertEqual(response.status_code, 413)

    def test_bedspec_round_trip_validation_render_and_audio_access(self) -> None:
        validation = self.client.post("/api/validate", json={"bed_spec": self.spec})
        self.assertEqual(validation.status_code, 200)
        self.assertEqual(validation.json()["state"], "production-safe")
        rendered = self.client.post("/api/render-preview", json={
            "bed_spec": self.spec, "duration_seconds": 3,
        })
        self.assertEqual(rendered.status_code, 200, rendered.text)
        artifact = rendered.json()
        audio = self.client.get(artifact["audio_url"])
        self.assertEqual(audio.status_code, 200)
        self.assertEqual(audio.headers["content-type"], "audio/wav")
        self.assertEqual(self.client.get("/api/audio/not-a-hash").status_code, 404)

    def test_randomize_and_return_safe_endpoints(self) -> None:
        randomized = self.client.post("/api/randomize", json={
            "bed_spec": self.spec,
            "locked_paths": ["/bpm"],
            "seed": 55,
            "music_request": {"family": "warm-motion", "palette": "electronic"},
        })
        self.assertEqual(randomized.status_code, 200, randomized.text)
        self.assertEqual(randomized.json()["bed_spec"]["bpm"], self.spec["bpm"])

        experimental = copy.deepcopy(self.spec)
        experimental["bpm"] = 120
        repaired = self.client.post("/api/return-safe", json={"bed_spec": experimental})
        self.assertEqual(repaired.status_code, 200, repaired.text)
        self.assertEqual(repaired.json()["bed_spec"]["bpm"], 104)

    def test_sample_listing_is_paginated_and_contains_no_physical_path(self) -> None:
        response = self.client.get("/api/samples?limit=1")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["limit"], 1)
        self.assertLessEqual(len(payload["items"]), 1)
        if payload["items"]:
            self.assertNotIn("path", payload["items"][0])
            self.assertIn("logical_id", payload["items"][0])

    def test_hosted_promotion_returns_forbidden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(create_api(
                config=ExplorerConfig(hosted=True, output_root=Path(tmp)),
                mount_ui=False))
            response = client.post(
                "/api/samples/vcsl:00000000000000000000/promote")
            self.assertEqual(response.status_code, 403)

    def test_gradio_application_mounts_without_rendering_on_startup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_api(config=ExplorerConfig(output_root=Path(tmp)), mount_ui=True)
            self.assertFalse(app.state.demo.ssr_mode)
            with TestClient(app) as client:
                self.assertEqual(client.get("/").status_code, 200)
                config = client.get("/config")
                self.assertEqual(config.status_code, 200)
                labels = {component.get("props", {}).get("label")
                          for component in config.json().get("components", [])}
                self.assertIn("Vocabulary", labels)
                self.assertIn("Voice model", labels)
                self.assertIn("Style", labels)
                self.assertIn("Current BedSpec JSON", labels)

    def test_lesson_handler_is_directly_registered_and_tabs_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = ExplorerConfig(output_root=Path(tmp))

            def lesson_generate(rows, model, state):
                return {"rows": rows, "model": model, "state": state}

            demo = build_demo(
                config, artifacts=ArtifactStore(config),
                samples=SampleService(config), lesson_generate=lesson_generate)
            handlers = [block.fn for block in demo.fns.values()]
            self.assertIn(lesson_generate, handlers)
            labels = {getattr(block, "label", None) for block in demo.blocks.values()}
            self.assertIn("Vocabulary", labels)
            self.assertIn("Voice model", labels)
            vocabulary = next(block for block in demo.blocks.values()
                              if getattr(block, "label", None) == "Vocabulary")
            self.assertEqual(vocabulary.value["data"], DEFAULT_LESSON_ROWS)
            voice_model = next(block for block in demo.blocks.values()
                               if getattr(block, "label", None) == "Voice model")
            self.assertEqual(voice_model.value, LESSON_MODEL)
            self.assertEqual(voice_model.choices,
                             [("Chatterbox Multilingual", LESSON_MODEL)])
            start_lesson = next(
                block for block in demo.fns.values()
                if block.fn is not None and block.fn.__name__ == "start_lesson")
            self.assertIn("Waiting for local Chatterbox",
                          start_lesson.fn(DEFAULT_LESSON_ROWS))
            tabs = next(block for block in demo.blocks.values()
                        if block.__class__.__name__ == "Tabs")
            self.assertEqual(tabs.selected, "lesson")
            tab_labels = {getattr(block, "label", None)
                          for block in demo.blocks.values()
                          if block.__class__.__name__ == "Tab"}
            self.assertTrue({"Lesson", "Music", "Lab"}.issubset(tab_labels))

    def test_simple_generate_renders_a_fresh_reset_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = ExplorerConfig(output_root=Path(tmp))
            demo = build_demo(
                config, artifacts=ArtifactStore(config),
                samples=SampleService(config))
            generate = next(
                block for block in demo.fns.values()
                if block.fn is not None and block.fn.__name__ == "generate")
            progress_events: list[tuple[float, str]] = []

            def record_progress(value: float, *, desc: str = "") -> None:
                progress_events.append((value, desc))

            with mock.patch("lexibeat.explorer_ui.secrets.randbits", return_value=1234):
                values = generate.fn(
                    "warm-motion", "balanced", "steady", "electronic",
                    record_progress)
            by_label = {
                output.label: values[index]
                for index, output in enumerate(generate.outputs)
                if getattr(output, "label", None)
            }
            preview = by_label["Current music preview"]
            self.assertTrue(preview.autoplay)
            self.assertEqual(preview.playback_position, 0)
            self.assertEqual(preview.buttons, ["download"])
            self.assertFalse(preview.editable)
            self.assertTrue(Path(by_label["Download WAV"]).is_file())
            button_values = {
                component.value for component in demo.blocks.values()
                if component.__class__.__name__ == "Button"
            }
            self.assertIn("Generate", button_values)
            self.assertNotIn("Generate another", button_values)
            self.assertNotIn("Play", button_values)
            self.assertNotIn("Stop", button_values)
            self.assertTrue(any("Testing candidate" in message
                                and "of 24" in message
                                for _, message in progress_events))
            self.assertTrue(any("of 6 accepted" in message
                                for _, message in progress_events))
            fractions = [fraction for fraction, _ in progress_events]
            self.assertEqual(fractions, sorted(fractions))


if __name__ == "__main__":
    unittest.main()
