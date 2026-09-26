"""Tests for the one-phase loop render and the timeline it produces."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from lexibeat.arrange import SOURCE, TARGET, Cancelled
from lexibeat.formats import renderable
from lexibeat.language import ENGLISH, SPANISH, Language
from lexibeat.loop import (
    MAX_ITEMS,
    MP3_BITRATE_KBPS,
    LoopError,
    LoopRequest,
    bed_fingerprint,
    build_timeline,
    estimated_seconds,
    render_loop,
)
from lexibeat.music import SR
from lexibeat.vocab import Item
from lexibeat.voice import BackendCapabilities, SpeechRequest, SynthesisResult

WORDS = (
    Item("asco", "disgust", "repulsed, recoiling slightly"),
    Item("la rabia", "rage", "angry, with heat behind it"),
)


class RecordingBackend:
    """Satisfies the protocol structurally, and declares no languages, so it speaks anything."""

    name = "recording"
    sample_rate = 24000
    model_id = "recording/v1"
    load_seconds = 0.0
    capabilities = BackendCapabilities("instruction", "instruction", "preset",
                                       languages=())

    def __init__(self, seconds: float = 0.6) -> None:
        self.seen: list[SpeechRequest] = []
        self.seconds = seconds

    def synth(self, request: SpeechRequest) -> SynthesisResult:
        self.seen.append(request)
        count = int(self.sample_rate * self.seconds)
        ramp = np.linspace(0, self.seconds, count, endpoint=False)
        audio = (0.4 * np.sin(2 * np.pi * 190 * ramp) * np.hanning(count)).astype(np.float32)
        return SynthesisResult(audio, self.sample_rate, 0.0, {})


def request(**overrides) -> LoopRequest:
    values = dict(items=WORDS, source_language=SPANISH, target_language=ENGLISH,
                  seed=7)
    values.update(overrides)
    return LoopRequest(**values)


class LoopRequestTests(unittest.TestCase):
    def test_a_bad_request_is_refused_by_the_part_that_is_wrong(self) -> None:
        with self.assertRaisesRegex(LoopError, "unknown format 'waltz'"):
            request(format="waltz").validated()
        with self.assertRaisesRegex(LoopError, "no switch 'fast'"):
            request(switches={"fast": True}).validated()
        with self.assertRaisesRegex(LoopError, "at least one word"):
            request(items=()).validated()
        with self.assertRaisesRegex(LoopError, "direction"):
            request(items=(Item("a", "b", "x" * 400),)).validated()
        with self.assertRaisesRegex(LoopError, "limited to"):
            request(items=tuple(Item(f"a{n}", f"b{n}")
                                for n in range(MAX_ITEMS + 1))).validated()

    def test_the_six_pair_cap_is_gone(self) -> None:
        many = tuple(Item(f"palabra {n}", f"word {n}") for n in range(40))
        self.assertEqual(len(request(items=many).validated().items), 40)

    def test_a_loop_estimates_its_own_length_before_a_note_is_synthesised(self) -> None:
        self.assertAlmostEqual(estimated_seconds(12, renderable("classic"), bpm=80),
                               (2 + 12 * 8 + 2) * 3.0, places=5)


class TimelineTests(unittest.TestCase):
    def test_reveals_and_utterances_line_up_with_the_words(self) -> None:
        backend = RecordingBackend()
        with tempfile.TemporaryDirectory() as tmp:
            result = render_loop(request(), backend=backend,
                                 output=Path(tmp) / "loop.mp3")
        self.assertEqual(len(result.timeline), len(WORDS))
        row = result.timeline[0]
        self.assertEqual(row["source"], "asco")
        self.assertEqual(row["direction"], "repulsed, recoiling slightly")
        # The answer arrives after the word and after the recall gap, never before it.
        self.assertLess(row["source_reveal"], row["target_reveal"])
        self.assertLessEqual(row["target_reveal"], row["end"])
        roles = [utterance["role"] for utterance in row["utterances"]]
        self.assertEqual(roles, [SOURCE, TARGET] * 3)
        self.assertEqual([utterance["repetition"] for utterance in row["utterances"]],
                         [0, 0, 1, 1, 2, 2])

    def test_a_timeline_refuses_events_that_are_not_the_words_it_was_given(self) -> None:
        with self.assertRaises(LoopError):
            build_timeline(list(WORDS), [], None, 10, renderable("classic"))


class RenderTests(unittest.TestCase):
    def test_a_render_writes_one_mp3_at_the_bitrate_this_repository_measured(self) -> None:
        backend = RecordingBackend()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "loop.mp3"
            result = render_loop(request(), backend=backend, output=output)
            self.assertTrue(output.is_file())
            self.assertEqual(list(Path(tmp).glob("*.partial.*")), [])
            info = sf.info(output)
            self.assertEqual(info.format, "MP3")
            measured = output.stat().st_size * 8 / result.duration_seconds / 1000
            self.assertAlmostEqual(measured, MP3_BITRATE_KBPS, delta=8)
        self.assertEqual(result.audio_mime, "audio/mpeg")
        self.assertAlmostEqual(info.duration, result.duration_seconds, delta=0.2)

    def test_every_line_is_spoken_in_its_own_language_three_times(self) -> None:
        backend = RecordingBackend()
        with tempfile.TemporaryDirectory() as tmp:
            render_loop(request(), backend=backend, output=Path(tmp) / "loop.mp3")
        self.assertEqual(len(backend.seen), len(WORDS) * 6)
        spanish = [row for row in backend.seen if row.language is SPANISH]
        self.assertEqual({row.text for row in spanish}, {"asco", "la rabia"})
        self.assertEqual({row.language.name for row in backend.seen},
                         {"Spanish", "English"})
        takes = sorted(row.delivery.take for row in backend.seen if row.text == "asco")
        self.assertEqual(takes, [0, 1, 2])

    def test_the_caller_direction_reaches_every_take_of_that_line(self) -> None:
        backend = RecordingBackend()
        with tempfile.TemporaryDirectory() as tmp:
            render_loop(request(), backend=backend, output=Path(tmp) / "loop.mp3")
        directions = {row.delivery.direction for row in backend.seen if row.text == "asco"}
        self.assertEqual(directions, {"repulsed, recoiling slightly"})

    def test_the_bed_replays_from_style_seed_and_engine_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = render_loop(request(), backend=RecordingBackend(),
                                output=Path(tmp) / "a.mp3")
            second = render_loop(request(), backend=RecordingBackend(),
                                 output=Path(tmp) / "b.mp3")
        self.assertEqual(first.style_id, second.style_id)
        self.assertEqual(first.seed, second.seed)
        # The request's seed, which is what a host can send again; not the winning candidate's.
        self.assertEqual(first.seed, request().seed)
        self.assertEqual(first.bed_fingerprint, second.bed_fingerprint)
        self.assertEqual(len(first.bed_fingerprint), 16)

    def test_a_render_stops_between_utterances_when_asked(self) -> None:
        backend = RecordingBackend()
        stop = {"after": 3}

        def cancel() -> bool:
            stop["after"] -= 1
            return stop["after"] <= 0

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "loop.mp3"
            with self.assertRaises(Cancelled):
                render_loop(request(), backend=backend, output=output,
                            cancel_check=cancel)
            self.assertFalse(output.exists())

    def test_a_language_the_backend_refuses_names_the_language(self) -> None:
        class SpanishOnly(RecordingBackend):
            capabilities = BackendCapabilities("instruction", "instruction", "preset",
                                               languages=("es",))

        with tempfile.TemporaryDirectory() as tmp, \
                self.assertRaisesRegex(RuntimeError, "English"):
            render_loop(request(), backend=SpanishOnly(),
                        output=Path(tmp) / "loop.mp3")

    def test_a_format_teaches_any_pair_of_languages(self) -> None:
        backend = RecordingBackend()
        mandarin = Language("zh-Hans", "Mandarin Chinese")
        portuguese = Language("pt-BR", "Brazilian Portuguese")
        with tempfile.TemporaryDirectory() as tmp:
            result = render_loop(
                request(items=(Item("苹果", "maçã"),), source_language=mandarin,
                        target_language=portuguese),
                backend=backend, output=Path(tmp) / "loop.mp3")
        self.assertEqual({row.language.name for row in backend.seen},
                         {"Mandarin Chinese", "Brazilian Portuguese"})
        self.assertEqual(result.timeline[0]["source"], "苹果")


class FingerprintTests(unittest.TestCase):
    def test_a_fingerprint_is_short_stable_and_sensitive(self) -> None:
        from lexibeat.api import BedFingerprint

        one = BedFingerprint("sunlit", (1.0, 2.0), (3.0,), ("piano",))
        again = BedFingerprint("sunlit", (1.0, 2.0), (3.0,), ("piano",))
        other = BedFingerprint("sunlit", (1.0, 2.5), (3.0,), ("piano",))
        self.assertEqual(bed_fingerprint(one), bed_fingerprint(again))
        self.assertNotEqual(bed_fingerprint(one), bed_fingerprint(other))


if __name__ == "__main__":
    unittest.main()


class SlimRuntimeTests(unittest.TestCase):
    def test_the_render_path_pulls_in_neither_librosa_nor_numba(self) -> None:
        """The reason the service image is small, asserted rather than remembered.

        `librosa.resample` was the music path's only use of librosa, and through it numba and
        llvmlite rode into an image whose job is to render a bed and mix speech it did not
        synthesise. A fresh interpreter, because this one may have loaded them for a voice test.
        """
        import subprocess
        import sys

        probe = (
            "import lexibeat.loop, lexibeat.service, sys;"
            "print(sorted(name for name in ('librosa', 'numba', 'llvmlite', 'torch', 'gradio')"
            " if name in sys.modules))"
        )
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                                text=True, check=True,
                                cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.stdout.strip(), "[]")
