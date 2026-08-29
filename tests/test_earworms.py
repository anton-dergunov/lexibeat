from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from earworms.bedspec import BedSpec
from earworms.arrange import arrange
from earworms.emotion import EMOTIONS, NEUTRAL, VECTOR_ORDER
from earworms.instruments import SampledInstrument
from earworms.mix import duck_envelope, mix_stems
from earworms.music import Grid, SR, filter_curve, render_bed, render_stems
from earworms.samples import PACKS, Sample, SamplePack, midi, missing
from earworms.voice import (
    CAPABILITIES,
    DEFAULT_MODELS,
    FISH_TAGS,
    IndexTTS25Backend,
    MlxAudioBackend,
    Prosody,
    Qwen3Backend,
    Speaker,
    SynthesisResult,
    delivery_instruction,
    reference_path,
)
from earworms.vocab import Item
from benchmark_voices import pressure_snapshot, write_comparison


class VoiceTests(unittest.TestCase):
    def test_experimental_capabilities_and_defaults_are_explicit(self) -> None:
        expected = {"indextts25", "voxcpm2", "qwen3", "tada", "fish-s2"}
        self.assertTrue(expected.issubset(DEFAULT_MODELS))
        self.assertTrue(all(CAPABILITIES[name].experimental for name in expected))
        self.assertEqual(CAPABILITIES["tada"].emotion, "unsupported")
        self.assertEqual(CAPABILITIES["indextts25"].emotion, "8-float vector")

    def test_index_emotion_vector_keeps_official_order(self) -> None:
        vector = EMOTIONS["surprised"].vector()
        self.assertEqual(len(vector), 8)
        self.assertEqual(VECTOR_ORDER[6], "surprised")
        self.assertEqual(vector[6], EMOTIONS["surprised"].exaggeration)
        self.assertEqual(sum(value != 0 for value in vector), 1)

    def test_instruction_and_fish_tag_map_emotion_and_prosody(self) -> None:
        text = delivery_instruction(EMOTIONS["sad"],
                                    Prosody(speed=0.96, semitones=-0.4))
        self.assertIn("sad", text)
        self.assertIn("slowly", text)
        self.assertIn("lower", text)
        self.assertEqual(FISH_TAGS["emphatic"], "emphasis")

    def test_qwen_uses_language_voice_instruction_and_seed(self) -> None:
        calls = []

        class Result:
            audio = np.ones(32, dtype=np.float32)
            sample_rate = 24000

        class Model:
            def generate(self, **kwargs):
                calls.append(kwargs)
                yield Result()

        backend = Qwen3Backend.__new__(Qwen3Backend)
        backend._model = Model()
        backend.voices = {"es": "Serena", "en": "Ryan"}
        backend.sample_rate = 24000
        result = backend.synth("hola", "es", Prosody(), EMOTIONS["happy"], seed=11)
        self.assertEqual(result.sample_rate, 24000)
        self.assertEqual(calls[0]["voice"], "Serena")
        self.assertEqual(calls[0]["lang_code"], "spanish")
        self.assertIn("happy", calls[0]["instruct"])

    def test_speaker_records_target_fit_and_deterministic_seed(self) -> None:
        class FakeBackend:
            name = "voxcpm2"
            model_id = "fake"
            sample_rate = SR
            load_seconds = 0.0
            calls = []

            def synth(self, text, lang, prosody, emotion, target_seconds=None,
                      seed=None):
                self.calls.append((target_seconds, seed))
                return SynthesisResult(np.ones(SR * 3, dtype=np.float32), SR,
                                       0.1, {"native": True})

        backend = FakeBackend()
        with mock.patch("earworms.voice.make_backend", return_value=backend), \
                mock.patch("earworms.voice.warnings.warn"):
            speaker = Speaker(backend="voxcpm2", voice_seed=90)
            audio = speaker.say("hola", "es", target_seconds=1.0)
        self.assertLessEqual(len(audio), SR)
        self.assertEqual(backend.calls, [(1.0, 90)])
        self.assertTrue(speaker.stats[0]["post_fit_applied"])

    def test_index_worker_request_carries_vector_and_duration_target(self) -> None:
        class Input:
            def __init__(self):
                self.value = ""

            def write(self, value):
                self.value += value

            def flush(self):
                pass

        class Process:
            stdin = Input()

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "worker.wav"
            sf.write(output, np.ones(100, dtype=np.float32), 22050)
            backend = IndexTTS25Backend.__new__(IndexTTS25Backend)
            backend._tmp = types.SimpleNamespace(name=tmp)
            backend._process = Process()
            backend.ref_audios = {"es": "/ref.wav"}
            backend.sample_rate = 22050
            backend.close = lambda: None
            backend._read_response = lambda: {
                "generation_seconds": 0.2, "passes": 2,
                "peak_memory_bytes": 123,
                "controls": {"duration_factor": 0.8},
            }
            with mock.patch("earworms.voice.hashlib.sha1") as sha:
                sha.return_value.hexdigest.return_value = output.stem
                result = backend.synth("hola", "es", Prosody(speed=1.25),
                                       EMOTIONS["happy"], target_seconds=1.1,
                                       seed=3)
            request = json.loads(backend._process.stdin.value)
            self.assertAlmostEqual(request["duration_factor"], 0.8)
            self.assertEqual(request["target_seconds"], 1.1)
            self.assertEqual(request["emotion"], EMOTIONS["happy"].vector())
            self.assertEqual(result.passes, 2)
    def test_reference_cache_is_language_specific(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
                os.environ, {"EARWORMS_CACHE": tmp}):
            self.assertEqual(reference_path("say:Paulina", "es").name,
                             "say-Paulina-es.wav")
            self.assertEqual(reference_path("say:Daniel", "en").name,
                             "say-Daniel-en.wav")

    def test_chatterbox_routes_native_reference_by_language(self) -> None:
        calls: list[dict] = []

        class Result:
            audio = np.ones(16, dtype=np.float32)
            sample_rate = 24000

        class Model:
            sample_rate = 24000

            def generate(self, **kwargs):
                calls.append(kwargs)
                yield Result()

        utils = types.ModuleType("mlx_audio.tts.utils")
        utils.load_model = lambda _: Model()
        modules = {
            "mlx_audio": types.ModuleType("mlx_audio"),
            "mlx_audio.tts": types.ModuleType("mlx_audio.tts"),
            "mlx_audio.tts.utils": utils,
        }
        resolved = lambda source, lang: Path(f"/{lang}-{source.split(':')[-1]}.wav")
        with mock.patch.dict(sys.modules, modules), mock.patch(
                "earworms.voice.ensure_reference", side_effect=resolved):
            backend = MlxAudioBackend()
            backend.synth("hola", "es", Prosody(), NEUTRAL)
            backend.synth("hello", "en", Prosody(), NEUTRAL)

        self.assertEqual(calls[0]["ref_audio"], "/es-Paulina.wav")
        self.assertEqual(calls[1]["ref_audio"], "/en-Daniel.wav")
        self.assertEqual(calls[0]["lang_code"], "es")
        self.assertEqual(calls[1]["lang_code"], "en")


class BedSpecTests(unittest.TestCase):
    def test_legacy_dict_gets_compatible_defaults(self) -> None:
        spec = BedSpec.from_dict({"bpm": 80, "pad": {"level": 0.4}})
        self.assertEqual((spec.beats_per_bar, spec.beat_unit), (4, 4))
        self.assertEqual(spec.chord_extension, "none")
        self.assertEqual(spec.pad.instrument, "synth")
        self.assertEqual(spec.pad.duck_db, 5.0)

    def test_json_round_trip_preserves_new_fields(self) -> None:
        spec = BedSpec.from_style("nocturne", 9)
        rebuilt = BedSpec.from_dict(json.loads(spec.to_json()))
        self.assertEqual(rebuilt.to_json(), spec.to_json())

    def test_extensions_add_notes(self) -> None:
        base = BedSpec(chord_extension="none").chord(0)
        seventh = BedSpec(chord_extension="seventh").chord(0)
        add9 = BedSpec(chord_extension="add9").chord(0)
        ninth = BedSpec(chord_extension="ninth").chord(0)
        self.assertGreater(len(seventh), len(base))
        self.assertGreater(len(add9), len(base))
        self.assertEqual(set(ninth), set(seventh) | set(add9))

    def test_supported_meters_have_dynamic_grids(self) -> None:
        for beats, expected in ((3, 12), (4, 16), (5, 20)):
            spec = BedSpec(bpm=60, beats_per_bar=beats, beat_unit=4)
            grid = Grid.from_spec(spec)
            self.assertEqual(grid.steps_per_bar, expected)
            self.assertAlmostEqual(grid.bar, float(beats))
            self.assertAlmostEqual(grid.step_time(1, 0), grid.bar)

    def test_swing_uses_meter_specific_step_length(self) -> None:
        grid = Grid(bpm=60, beats_per_bar=3, beat_unit=4, swing=0.25)
        self.assertAlmostEqual(grid.step_time(0, 2), 0.5 + 0.25 * 0.25)

    def test_styles_are_seeded_and_keep_speech_sized_bars(self) -> None:
        meters = set()
        for seed in range(30):
            first = BedSpec.from_style("yoga", seed)
            second = BedSpec.from_style("yoga", seed)
            self.assertEqual(first.to_json(), second.to_json())
            self.assertGreaterEqual(Grid.from_spec(first).bar, 2.2)
            meters.add(first.beats_per_bar)
        self.assertEqual(meters, {3, 4, 5})

    def test_filter_curves_are_bounded_and_deterministic(self) -> None:
        for name in ("sine", "triangle", "random_walk"):
            spec = BedSpec()
            spec.pad.cutoff_curve = name
            a = filter_curve(spec, 8, 100, np.random.default_rng(4))
            b = filter_curve(spec, 8, 100, np.random.default_rng(4))
            np.testing.assert_allclose(a, b)
            self.assertGreaterEqual(float(a.min()), 120.0)
            self.assertLessEqual(float(a.max()), SR * 0.45)


class SampleTests(unittest.TestCase):
    def test_pack_manifests_cover_requested_instruments(self) -> None:
        self.assertEqual(midi("F#3"), 54)
        self.assertTrue({"salamander", "vsco-marimba", "vsco-glockenspiel",
                         "vsco-strings"}.issubset(PACKS))
        self.assertTrue(all(entry.remote_path for pack in PACKS.values()
                            for entry in pack.entries()))

    def test_sampled_instrument_uses_manifest_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
                os.environ, {"EARWORMS_CACHE": tmp}):
            entry = Sample("tone.wav", "remote/tone.wav", 60, 1, "test")
            pack = SamplePack("test", "CC0", "test", "https://example.test",
                              "https://example.test/", (entry,))
            directory = Path(tmp) / "samples" / "test"
            directory.mkdir(parents=True)
            sf.write(directory / entry.filename,
                     np.sin(2 * np.pi * 261.6 * np.arange(SR) / SR), SR)
            self.assertFalse(missing(pack))
            audio = SampledInstrument(pack).render(60, 0.5, 0.2)
            self.assertEqual(len(audio), int(0.2 * SR))
            self.assertAlmostEqual(float(np.abs(audio).max()), 0.5, places=3)


class RenderAndMixTests(unittest.TestCase):
    def _spec(self, beats: int = 4) -> BedSpec:
        spec = BedSpec(bpm=72, beats_per_bar=beats, beat_unit=4, seed=3)
        spec.pad.instrument = "synth"
        spec.lead.instrument = "synth"
        spec.space.reverb_seconds = 0.15
        spec.space.reverb_mix = 0.1
        return spec

    def test_render_stems_sum_to_bed_shape_for_each_meter(self) -> None:
        for beats in (3, 4, 5):
            spec = self._spec(beats)
            stems = render_stems(spec, 1)
            bed = render_bed(spec, 1)
            self.assertEqual(set(stems), {"pad", "bass", "drums", "lead"})
            self.assertEqual(bed.shape, next(iter(stems.values())).shape)
            self.assertTrue(np.isfinite(bed).all())

    def test_sampled_pad_handles_non_integral_tempo_rounding(self) -> None:
        spec = self._spec()
        spec.bpm = 66
        spec.pad.instrument = "strings"
        stems = render_stems(spec, 1)
        self.assertTrue(np.isfinite(stems["pad"]).all())

    def test_per_layer_ducking_and_limiter(self) -> None:
        seconds = 3
        n = seconds * SR
        t = np.arange(n) / SR
        speech = np.zeros(n, dtype=np.float32)
        speech[SR:2 * SR] = 0.2 * np.sin(2 * np.pi * 220 * t[:SR])
        shallow = duck_envelope(speech, 2.0)
        deep = duck_envelope(speech, 8.0)
        self.assertLess(float(deep[int(1.5 * SR)]),
                        float(shallow[int(1.5 * SR)]))
        tone = (0.05 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
        stems = {"pad": np.stack([tone, tone], axis=1),
                 "bass": np.stack([tone, tone], axis=1)}
        track = mix_stems(stems, speech, {"pad": 8.0, "bass": 2.0})
        self.assertTrue(np.isfinite(track).all())
        self.assertLessEqual(float(np.abs(track).max()), 0.97001)

    def test_arrangement_passes_slot_duration_and_uses_downbeats(self) -> None:
        class FakeSpeaker:
            prosody_strength = 1.0
            targets: list[float] = []

            def say(self, text, lang, prosody, emotion, target_seconds=None):
                self.targets.append(target_seconds)
                return np.ones(100, dtype=np.float32)

        grid = Grid(bpm=60, beats_per_bar=4, beat_unit=4)
        speaker = FakeSpeaker()
        events, _ = arrange([Item("hola", "hello")], speaker, grid,
                            progress=False)
        self.assertEqual(len(events), 6)
        self.assertTrue(all(event.start % grid.bar == 0 for event in events))
        self.assertTrue(all(value == grid.bar * 0.92
                            for value in speaker.targets))


class BenchmarkTests(unittest.TestCase):
    def test_pressure_snapshot_parses_swap_for_numeric_delta(self) -> None:
        outputs = iter([
            "System-wide memory free percentage: 27%",
            "total = 4096.00M  used = 1536.50M  free = 2559.50M",
            "Pages free: 42.",
        ])
        with mock.patch("benchmark_voices.command_output",
                        side_effect=lambda _: next(outputs)):
            snapshot = pressure_snapshot()
        self.assertEqual(snapshot["free_percent"], 27)
        self.assertEqual(snapshot["swap_used_bytes"], int(1536.5 * 2**20))

    def test_comparison_schema_handles_success_and_failure_without_models(self) -> None:
        rows = [
            {
                "backend": "indextts25", "success": True,
                "timing": {"model_load_seconds": 1.0,
                           "model_generation_seconds": 2.0},
                "benchmark_wall_seconds": 3.0,
                "audio": {"duration_seconds": 4.0},
                "memory": {"mlx_peak_bytes": 1024},
                "system_memory": {"peak_process_tree_rss_bytes": 2048,
                                  "lowest_free_percent": 25},
            },
            {"backend": "tada", "success": False},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_comparison(directory, rows)
            payload = json.loads((directory / "comparison.json").read_text())
            markdown = (directory / "comparison.md").read_text()
        self.assertEqual([row["backend"] for row in payload],
                         ["indextts25", "tada"])
        self.assertIn("| tada | failed |", markdown)
        self.assertIn("does not certify pronunciation", markdown)


if __name__ == "__main__":
    unittest.main()
