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
from earworms.emotion import NEUTRAL
from earworms.instruments import SampledInstrument
from earworms.mix import duck_envelope, mix_stems
from earworms.music import Grid, SR, filter_curve, render_bed, render_stems
from earworms.samples import PACKS, Sample, SamplePack, midi, missing
from earworms.voice import MlxAudioBackend, Prosody, reference_path


class VoiceTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
