from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from lexibeat.demo import (
    DEMO_HEIGHT,
    DEMO_WIDTH,
    PersistentSpeaker,
    arrange_demo,
    cache_key,
    encode_visual_track,
    load_demo_config,
    mux_audio,
    resolve_demo_specs,
)
from lexibeat.arrange import TARGET
from lexibeat.language import ENGLISH, SPANISH
from lexibeat.loop import build_timeline
from lexibeat.music import SR, Grid
from lexibeat.voice import CAPABILITIES, Delivery, Speaker, SynthesisResult

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "examples" / "readme_demo" / "full.json"
HEADLINE_MANIFEST = ROOT / "examples" / "readme_demo" / "headline.json"


class DemoConfigTests(unittest.TestCase):
    def test_headline_manifest_uses_selected_story_and_acoustic_bed(self) -> None:
        config = load_demo_config(HEADLINE_MANIFEST)
        self.assertEqual([item.source for item in config.items], [
            "¿arrancamos?", "espectacular", "preciosa", "asombroso",
            "asco", "dar bronca", "¡Qué susto!", "la pesadilla",
            "qué lástima", "estar envuelto en sus pensamientos",
            "vencer tus miedos", "descansar",
        ])
        self.assertEqual(
            [(variant.name, variant.style, variant.seed)
             for variant in config.variants],
            [("acoustic", "acoustic", 9)],
        )
        specs = resolve_demo_specs(config)
        self.assertEqual(
            {(spec.bpm, spec.beats_per_bar, spec.beat_unit)
             for spec in specs.values()},
            {(78, 4, 4)},
        )

    def test_manifest_has_experiment_items_and_matching_beds(self) -> None:
        config = load_demo_config(MANIFEST)
        self.assertEqual(len(config.items), 20)
        self.assertEqual(config.items[0].source, "¿arrancamos?")
        self.assertEqual(config.items[1].source, "¡Que cante!")
        self.assertEqual(config.items[-2].source, "descansar")
        self.assertEqual(config.items[-1].source, "tranquilizarse")
        spans = dict(zip((item.source for item in config.items),
                         config.bars_per_utterance))
        self.assertEqual(spans["estar envuelto en sus pensamientos"], 2)
        directions = dict(zip((item.source for item in config.items),
                              (item.direction for item in config.items)))
        self.assertEqual(directions["descansar"], "drowsy and unhurried")
        self.assertEqual(directions["tranquilizarse"], "calm, settling")
        specs = resolve_demo_specs(config)
        self.assertEqual(list(specs), [
            "gentle-movement", "playful-plucked", "sunlit", "lofi-wide",
            "nocturnal", "meditative", "acoustic", "radiant",
            "warm-motion", "bright-pastoral"])
        grids = {(spec.bpm, spec.beats_per_bar, spec.beat_unit)
                 for spec in specs.values()}
        self.assertEqual(grids, {(78, 4, 4)})

    def test_full_timeline_has_downbeats_and_progressive_reveals(self) -> None:
        class DemoFakeSpeaker:
            prosody_strength = 1.0
            capabilities = CAPABILITIES["gemini"]

            @staticmethod
            def say(text, language, delivery, target_seconds=None, slot_seconds=None):
                del text, language, delivery, target_seconds
                return np.ones(SR // 20, dtype=np.float32)

        config = load_demo_config(MANIFEST)
        spec = next(iter(resolve_demo_specs(config).values()))
        grid = Grid.from_spec(spec)
        events, total_bars = arrange_demo(config, DemoFakeSpeaker(), grid)
        timeline = build_timeline(config.items, events, grid, total_bars,
                                  config.pattern)
        self.assertEqual(len(events), 120)
        self.assertEqual(total_bars, 170)
        self.assertTrue(all(abs(event.start / grid.bar -
                                round(event.start / grid.bar)) < 1e-8
                            for event in events))
        for row in timeline:
            self.assertLess(row["source_reveal"], row["target_reveal"])
            self.assertTrue(all(
                utterance["start"] >= row["target_reveal"]
                for utterance in row["utterances"]
                if utterance["role"] == TARGET))
            self.assertGreater(row["end"], row["target_reveal"])


class PersistentSpeakerTests(unittest.TestCase):
    class FakeBackend:
        name = "fake"
        model_id = "fake-v1"
        voices = {"es": "one", "en": "two"}
        sample_rate = SR
        calls = 0

        capabilities = CAPABILITIES["gemini"]
        load_seconds = 0.0

        def synth(self, request):
            del request
            type(self).calls += 1
            return SynthesisResult(
                np.full(SR // 10, 0.1, dtype=np.float32), SR, 0.01,
                {"provider": "fake"})

    def test_completed_take_is_reused_from_disk(self) -> None:
        self.FakeBackend.calls = 0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = Speaker(backend_instance=self.FakeBackend())
            cached = PersistentSpeaker(first, root)
            audio = cached.say("hola", SPANISH, Delivery(direction="warmly"), 1.0)
            cached.close()
            self.assertEqual(self.FakeBackend.calls, 1)
            self.assertTrue(np.isfinite(audio).all())

            second = Speaker(backend_instance=self.FakeBackend())
            resumed = PersistentSpeaker(second, root)
            again = resumed.say("hola", SPANISH, Delivery(direction="warmly"), 1.0)
            resumed.close()
            self.assertEqual(self.FakeBackend.calls, 1)
            self.assertTrue(second.stats[-1]["cache_hit"])
            np.testing.assert_allclose(audio, again, atol=2e-5)

    def test_vertex_and_direct_gemini_use_distinct_cache_keys(self) -> None:
        class GeminiLike:
            name = "gemini"
            model_id = "gemini-test"
            voices = {"es": "one", "en": "two"}

        direct = Speaker.__new__(Speaker)
        direct.backend = GeminiLike()
        direct.voice_seed = 7
        vertex = Speaker.__new__(Speaker)
        vertex.backend = GeminiLike()
        vertex.backend.vertex = True
        vertex.voice_seed = 7
        args = ("hola", SPANISH, Delivery(direction="warmly"), 1.0)
        self.assertNotEqual(cache_key(direct, *args), cache_key(vertex, *args))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe") and
                     importlib.util.find_spec("PIL"),
                     "FFmpeg, FFprobe and Pillow are required")
class DemoVideoIntegrationTests(unittest.TestCase):
    def test_short_video_has_expected_codecs_and_dimensions(self) -> None:
        config = load_demo_config(MANIFEST)
        grid = Grid.from_spec(next(iter(resolve_demo_specs(config).values())))
        timeline = [{
            "index": 0, "source": "hola", "target": "hello",
            "direction": "warmly, greeting a friend", "start": 0.0,
            "source_reveal": 0.0, "target_reveal": 0.12, "end": 0.5,
            "utterances": [
                {"role": "source", "repetition": 0, "start": 0.0, "end": 0.1},
                {"role": "target", "repetition": 0, "start": 0.12, "end": 0.22},
            ],
        }]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            visual = root / "visual.mp4"
            audio = root / "audio.wav"
            output = root / "demo.mp4"
            encode_visual_track("LexiBeat", timeline, 0.5, grid, visual)
            sf.write(audio, np.zeros((SR // 2, 2), dtype=np.float32), SR)
            mux_audio(visual, audio, output)
            probe = subprocess.run([
                "ffprobe", "-v", "error", "-show_streams", "-of", "json",
                str(output)], capture_output=True, text=True, check=True)
            streams = json.loads(probe.stdout)["streams"]
            video = next(row for row in streams if row["codec_type"] == "video")
            audio_stream = next(row for row in streams if row["codec_type"] == "audio")
            self.assertEqual(video["codec_name"], "h264")
            self.assertEqual((video["width"], video["height"]),
                             (DEMO_WIDTH, DEMO_HEIGHT))
            self.assertEqual(video["pix_fmt"], "yuv420p")
            self.assertEqual(audio_stream["codec_name"], "aac")


if __name__ == "__main__":
    unittest.main()
