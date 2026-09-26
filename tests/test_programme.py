"""Tests for planning a loop from its format, and rendering the template formats."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from lexibeat import formats
from lexibeat.formats import FormatError
from lexibeat.language import ENGLISH, SPANISH, Language
from lexibeat.loop import LoopError, render_loop
from lexibeat.programme import count_words, phrases, plan
from lexibeat.vocab import Item
from lexibeat.voice import BackendCapabilities, Delivery, Prosody, delivery_instruction

from test_loop import RecordingBackend, request

WORDS = [Item("el atasco", "traffic jam"), Item("la cebolla", "onion"),
         Item("el banco", "bank"), Item("la multa", "fine"), Item("el bigote", "moustache")]


def planned(data_or_id, words=WORDS, seed=0, target=ENGLISH):
    fmt = data_or_id if isinstance(data_or_id, formats.Format) else \
        formats.renderable(data_or_id)
    return plan(fmt, words, source_language=SPANISH, target_language=target, seed=seed)


def fmt(*sections, **extra) -> dict:
    return {"id": "test", "label": "Test", "description": "A test format.",
            "sections": list(sections), **extra}


WORDS_SECTION = {"kind": "words", "block": [{"say": "word"}, {"say": "translation"}]}


class PlanTests(unittest.TestCase):
    def test_sections_run_in_order_and_a_quiz_at_the_middle_covers_the_first_half(self) -> None:
        programme = planned(fmt(
            {"kind": "intro"}, WORDS_SECTION,
            {"kind": "quiz", "at": "middle", "block": [{"say": "translation"}, {"gap": 1},
                                                       {"say": "word"}]},
            {"kind": "review", "block": [{"say": "word"}]},
            {"kind": "outro"}))
        sections = [s.section for s in programme.lines]
        self.assertEqual(sections[0], "intro")
        self.assertEqual(sections[-1], "outro")
        # A quiz and a review each open with a line saying what is coming.
        self.assertEqual([s.section for s in programme.lines if s.kind == "announce"],
                         ["quiz", "review"])
        # Three words (the first half of five, rounded up), then the quiz over those three.
        quiz_at = sections.index("quiz")
        self.assertEqual([s.item for s in programme.lines[:quiz_at] if s.section == "words"],
                         [0, 0, 1, 1, 2, 2])
        self.assertEqual(sorted({s.item for s in programme.lines
                                 if s.section == "quiz" and s.item is not None}), [0, 1, 2])
        self.assertEqual(sorted({s.item for s in programme.lines
                                 if s.section == "review" and s.item is not None}),
                         [0, 1, 2, 3, 4])

    def test_a_quiz_at_the_end_follows_every_word_and_covers_them_all(self) -> None:
        programme = planned(fmt(WORDS_SECTION, {"kind": "quiz", "block": [{"say": "word"}]}))
        sections = [s.section for s in programme.lines]
        self.assertEqual(sections, ["words"] * 10 + ["quiz"] * 6)
        quiet = planned(fmt(WORDS_SECTION, {"kind": "quiz", "announce": False,
                                            "block": [{"say": "word"}]}))
        self.assertFalse(any(s.kind == "announce" for s in quiet.lines))

    def test_take_indices_continue_across_sections(self) -> None:
        programme = planned("review")
        first = [(s.section, s.side, s.take) for s in programme.lines if s.item == 0]
        self.assertEqual(first, [("words", "target", 0), ("words", "source", 0),
                                 ("review", "source", 1), ("review", "target", 1)])

    def test_roles_and_languages_follow_the_side(self) -> None:
        for segment in planned("echo").lines:
            if segment.side == "source":
                self.assertEqual((segment.role, segment.language), ("native", SPANISH))
            else:
                self.assertEqual((segment.role, segment.language), ("guide", ENGLISH))

    def test_cues_vary_between_words_and_repeat_on_a_re_render(self) -> None:
        cues = [s.text for s in planned("echo", seed=4).lines if s.kind == "cue"]
        self.assertEqual(len(set(cues)), len(cues))
        self.assertEqual(cues, [s.text for s in planned("echo", seed=4).lines if s.kind == "cue"])
        self.assertTrue(set(cues) <= set(phrases("en")["your_turn"]))

    def test_an_intro_counts_the_words_in_the_learner_s_own_grammar(self) -> None:
        self.assertEqual(count_words("ru", 1), "1 слово")
        self.assertEqual(count_words("ru", 3), "3 слова")
        self.assertEqual(count_words("ru", 5), "5 слов")
        self.assertEqual(count_words("ru", 22), "22 слова")
        self.assertEqual(count_words("en-US", 1), "1 word")
        programme = planned(fmt({"kind": "intro"}, WORDS_SECTION),
                            target=Language("ru", "Russian"))
        self.assertIn("5 слов", programme.lines[0].text)

    def test_a_learner_language_without_phrases_is_refused_by_name(self) -> None:
        with self.assertRaisesRegex(FormatError, r"no[ne]* .*Portuguese \(pt\)"):
            planned("echo", target=Language("pt", "Portuguese"))
        # A format that needs no phrases does not care.
        planned("classic", target=Language("pt", "Portuguese"))

    def test_a_request_is_refused_before_the_queue_when_its_phrases_are_missing(self) -> None:
        with self.assertRaisesRegex(LoopError, r"Portuguese"):
            request(format="echo", target_language=Language("pt", "Portuguese")).validated()


class DeliveryTests(unittest.TestCase):
    def test_a_pace_the_format_asks_for_reaches_the_director_note(self) -> None:
        slow = delivery_instruction(Delivery(pace="slow"))
        self.assertIn("slowly", slow)
        fast = delivery_instruction(Delivery(pace="fast"))
        self.assertIn("fast", fast)
        # Without a pace, the prosody's band still speaks.
        self.assertIn("slowly and deliberately",
                      delivery_instruction(Delivery(prosody=Prosody(speed=0.9))))


class RenderTests(unittest.TestCase):
    def render(self, **overrides):
        backend = RecordingBackend()
        with tempfile.TemporaryDirectory() as tmp:
            result = render_loop(request(**overrides), backend=backend,
                                 output=Path(tmp) / "loop.mp3")
        return result, backend

    def test_a_review_announces_itself_and_says_every_word_again(self) -> None:
        result, backend = self.render(format="review")
        review = [cue for cue in result.cues if cue["section"] == "review"]
        self.assertEqual(review[0]["kind"], "announce")
        self.assertIn(review[0]["text"], phrases("en")["review"])
        pairs = [cue for cue in review if cue["kind"] == "say"]
        self.assertEqual(len(pairs), 2 * len(result.items))
        for cue in pairs:
            self.assertEqual(sum(1 for c in result.cues if c["text"] == cue["text"]), 2)
        # The meaning comes first in this format, so it is revealed first.
        row = result.items[0]
        self.assertLess(row["target_reveal"], row["source_reveal"])

    def test_a_stretched_line_is_shorter(self) -> None:
        stretched = fmt(WORDS_SECTION, {"kind": "review", "announce": False, "stretch": 1.2,
                                        "block": [{"say": "word"}]})
        result, _ = self.render(format=stretched)
        drill = max(c["end"] - c["start"] for c in result.cues if c["section"] == "words")
        brisk = max(c["end"] - c["start"] for c in result.cues if c["section"] == "review")
        self.assertAlmostEqual(brisk, drill / 1.2, delta=0.02)

    def test_the_listener_chooses_how_many_times_each_word_is_said(self) -> None:
        chosen = fmt(
            {"kind": "words", "switch": "times", "repetitions": 2,
             "choice": {"2": {"repetitions": 1}, "4": {"repetitions": 3}},
             "block": [{"say": "word"}, {"gap": 1}, {"say": "translation"},
                       {"say": "word", "repeat": True}, {"say": "translation", "repeat": True},
                       {"rest": 1}]},
            switches={"times": {"label": "Times", "default": "3", "choices": ["2", "3", "4"]}})
        for times, pairs in (("2", 2), ("3", 3), ("4", 4)):
            programme = planned(formats.resolve(formats.parse(chosen), {"times": times}))
            said = [s for s in programme.lines if s.item == 0 and s.side == "source"]
            self.assertEqual(len(said), pairs, times)
            self.assertEqual([s.take for s in said], list(range(pairs)))

    def test_a_classic_word_s_drill_is_one_group(self) -> None:
        result, _ = self.render(format="classic")
        self.assertEqual([cue["group"] for cue in result.cues], [0] * 6 + [1] * 6)

    def test_every_line_tells_the_voice_who_says_it(self) -> None:
        result, backend = self.render(format="echo")
        roles = {(seen.language.code, seen.role) for seen in backend.seen}
        self.assertEqual(roles, {("es", "native"), ("en", "guide")})
        cues = [cue for cue in result.cues if cue["kind"] == "cue"]
        self.assertEqual(len(cues), len(result.items))
        self.assertTrue(all(cue["role"] == "guide" and cue["side"] is None for cue in cues))

    def test_a_missing_requirement_falls_back_and_says_so(self) -> None:
        needs = fmt(WORDS_SECTION, requires=["multilingual_voice"], fallback="classic")
        result, _ = self.render(format=needs)
        self.assertEqual((result.format, result.fallback_from), ("classic", "test"))

    def test_a_missing_requirement_without_a_fallback_is_refused_naming_it(self) -> None:
        needs = fmt(WORDS_SECTION, requires=["multilingual_voice"])
        with self.assertRaisesRegex(LoopError, r"requires multilingual_voice"):
            self.render(format=needs)

    def test_a_voice_that_mixes_languages_meets_the_requirement(self) -> None:
        needs = fmt(WORDS_SECTION, requires=["multilingual_voice"], fallback="classic")
        backend = RecordingBackend()
        backend.capabilities = replace(backend.capabilities, mixes_languages=True)
        with tempfile.TemporaryDirectory() as tmp:
            result = render_loop(request(format=needs), backend=backend,
                                 output=Path(tmp) / "loop.mp3")
        self.assertEqual((result.format, result.fallback_from), ("test", None))

    def test_intro_and_outro_are_lines_of_no_word(self) -> None:
        result, _ = self.render(format=fmt({"kind": "intro"}, WORDS_SECTION, {"kind": "outro"}))
        self.assertEqual(result.cues[0]["kind"], "intro")
        self.assertIsNone(result.cues[0]["item"])
        self.assertEqual(result.cues[-1]["kind"], "outro")
        self.assertLessEqual(result.items[-1]["end"], result.cues[-1]["start"])


if __name__ == "__main__":
    unittest.main()
