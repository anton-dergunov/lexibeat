"""Tests for programme formats: the grammar, switches, what renders, and the built-in files."""

from __future__ import annotations

import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

from lexibeat import formats
from lexibeat.formats import FormatError
from lexibeat.loop import LoopError, render_loop

from test_loop import RecordingBackend, request

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "docs" / "programme-format.md"

# The bars of the two drills this package has always made, written out rather than derived, so a
# change to either built-in format has to change this too.
CLASSIC = [("source", 0), ("gap", 0), ("target", 0), ("source", 1), ("target", 1),
           ("source", 2), ("target", 2), ("rest", 0)]
ALTERNATING = [("source", 0), ("target", 0), ("source", 1), ("target", 1),
               ("source", 2), ("target", 2), ("rest", 0), ("rest", 0)]


def minimal(**overrides) -> dict:
    data = {"id": "test", "label": "Test", "description": "A test format.",
            "sections": [{"kind": "words", "block": [{"say": "word"},
                                                     {"say": "translation"}]}]}
    data.update(overrides)
    return data


def switched() -> dict:
    return minimal(
        switches={"extra": {"label": "Say it again", "default": True},
                  "review": {"label": "Final review", "default": "normal",
                             "choices": ["off", "normal", "fast"]}},
        sections=[
            {"kind": "words", "block": [{"say": "word"}, {"say": "translation"},
                                        {"say": "word", "switch": "extra"}]},
            {"kind": "review", "switch": "review", "block": [{"say": "word"}],
             "choice": {"fast": {"stretch": 1.2, "bed": "quicker"}}},
        ])


def spec_examples() -> list[dict]:
    blocks = re.findall(r"```json\n(.*?)```", SPEC.read_text(encoding="utf-8"), re.S)
    return [data for data in (json.loads(block) for block in blocks)
            if isinstance(data, dict) and "sections" in data]


class BuiltinTests(unittest.TestCase):
    def test_the_built_in_formats_are_today_s_two_drills_bar_for_bar(self) -> None:
        self.assertEqual(sorted(formats.builtin()), ["alternating", "classic"])
        self.assertEqual(formats.slots(formats.renderable("classic")), CLASSIC)
        self.assertEqual(formats.slots(formats.renderable("alternating")), ALTERNATING)

    def test_every_built_in_format_file_is_named_after_its_id_and_renders(self) -> None:
        for path in formats.HERE.glob("*.json"):
            fmt = formats.renderable(path.stem)
            self.assertEqual(fmt.id, path.stem)

    def test_a_summary_is_what_a_host_lists_a_format_by(self) -> None:
        summary = formats.builtin()["classic"].summary()
        self.assertEqual(summary["bars_per_item"], 8)
        self.assertEqual(summary["utterances_per_item"], 6)
        self.assertTrue(summary["has_recall_gap"])
        self.assertEqual(summary["label"], "Classic drill")


class GrammarTests(unittest.TestCase):
    def test_every_example_in_the_spec_parses(self) -> None:
        examples = spec_examples()
        self.assertGreaterEqual(len(examples), 6)
        for data in examples:
            with self.subTest(data["id"]):
                self.assertEqual(formats.parse(data).id, data["id"])

    def test_the_spec_s_classic_example_is_the_built_in_file(self) -> None:
        example = next(data for data in spec_examples() if data["id"] == "classic")
        shipped = json.loads((formats.HERE / "classic.json").read_text(encoding="utf-8"))
        self.assertEqual(example, shipped)

    def test_a_refusal_names_where_the_format_is_wrong(self) -> None:
        cases = [
            (minimal(sections=[{"kind": "words", "block": [{"shout": "word"}]}]),
             r"sections\[0\]\.block\[0\]: a step names exactly one of"),
            (minimal(sections=[{"kind": "words", "block": [{"say": "word", "loud": True}]}]),
             r"sections\[0\]\.block\[0\]: unknown key 'loud'"),
            (minimal(sections=[{"kind": "words", "block": [{"say": "meaning"}]}]),
             r"'say' is one of word, translation"),
            (minimal(sections=[{"kind": "words", "block": [
                {"say": "word", "when": "sometimes"}]}]), r"'when' is one of"),
            (minimal(sections=[{"kind": "words", "block": [
                {"say": "word", "then": [{"say": "word"}]}]}]), r"needs a 'when'"),
            (minimal(sections=[{"kind": "words", "block": [
                {"remark": {"kinds": ["grammar"]}, "when": "writer_decides"}]}]),
             r"a kind is one of contrast"),
            (minimal(sections=[{"kind": "words", "block": [{"say": "word", "stretch": 1.5}]}]),
             r"'stretch' is from 1\.0 to 1\.2"),
            (minimal(sections=[{"kind": "words", "block": [{"pronounce": "syllables"}]}]),
             r"'pronounce' is one of slow_whole"),
            (minimal(sections=[{"kind": "review", "block": [{"example": "writer"}]}]),
             r"'example' is not a step this section takes"),
            (minimal(sections=[{"kind": "outro"}]), r"needs a 'words' section"),
            (minimal(sections=[{"kind": "words", "block": []}]), r"non-empty list of steps"),
            (minimal(id="Not An Id"), r"'id' is lower-case"),
            (minimal(colour="blue"), r"format: unknown key 'colour'"),
            (minimal(sections=[{"kind": "words", "switch": "missing",
                                "block": [{"say": "word"}]}]),
             r"switch 'missing' is not declared"),
        ]
        for data, message in cases:
            with self.subTest(message):
                with self.assertRaisesRegex(FormatError, message):
                    formats.parse(data)

    def test_a_then_step_cannot_nest_or_carry_a_condition(self) -> None:
        with self.assertRaisesRegex(FormatError, r"then\[0\]: a 'then' step cannot have its own"):
            formats.parse(minimal(sections=[{"kind": "words", "block": [
                {"say": "word", "when": "hard_to_say",
                 "then": [{"say": "word", "when": "hard_to_say",
                           "then": [{"say": "word"}]}]}]}]))


class SwitchTests(unittest.TestCase):
    def test_defaults_apply_when_a_switch_is_not_sent(self) -> None:
        fmt = formats.resolve(formats.parse(switched()))
        self.assertEqual([step.value for step in fmt.sections[0].block],
                         ["word", "translation", "word"])
        self.assertEqual(fmt.sections[1].params, {})
        self.assertEqual(fmt.switches, {})

    def test_an_off_switch_removes_what_it_names(self) -> None:
        fmt = formats.resolve(formats.parse(switched()), {"extra": False, "review": "off"})
        self.assertEqual([step.value for step in fmt.sections[0].block], ["word", "translation"])
        self.assertEqual([section.kind for section in fmt.sections], ["words"])

    def test_a_choice_applies_its_overrides(self) -> None:
        fmt = formats.resolve(formats.parse(switched()), {"review": "fast"})
        self.assertEqual(fmt.sections[1].params, {"stretch": 1.2, "bed": "quicker"})

    def test_a_switch_the_format_does_not_offer_is_refused(self) -> None:
        with self.assertRaisesRegex(FormatError, r"no switch 'speed'"):
            formats.resolve(formats.parse(switched()), {"speed": True})
        with self.assertRaisesRegex(FormatError, r"switch 'review' is one of off, normal, fast"):
            formats.resolve(formats.parse(switched()), {"review": "slow"})
        with self.assertRaisesRegex(FormatError, r"switch 'extra' is true or false"):
            formats.resolve(formats.parse(switched()), {"extra": "yes"})


class RenderableTests(unittest.TestCase):
    def test_what_this_version_cannot_render_is_named_not_dropped(self) -> None:
        story = next(data for data in spec_examples() if data["id"] == "story")
        with self.assertRaisesRegex(FormatError, r"format 'story' uses .*story_beat.*cannot "
                                                 r"render yet"):
            formats.renderable(story)
        missing = formats.unsupported(formats.resolve(formats.parse(story)))
        self.assertIn("order 'writer'", missing)
        self.assertIn("the 'review' section", missing)

    def test_a_words_block_must_say_both_sides_for_now(self) -> None:
        one_sided = minimal(sections=[{"kind": "words", "block": [{"say": "word"}, {"gap": 1}]}])
        with self.assertRaisesRegex(FormatError, r"does not say both"):
            formats.renderable(one_sided)

    def test_a_request_with_an_unrenderable_format_is_refused_before_any_speech(self) -> None:
        story = next(data for data in spec_examples() if data["id"] == "story")
        backend = RecordingBackend()
        with tempfile.TemporaryDirectory() as tmp, \
                self.assertRaisesRegex(LoopError, r"cannot render yet"):
            render_loop(request(format=story), backend=backend, output=Path(tmp) / "x.mp3")
        self.assertEqual(backend.seen, [])

    def test_an_inline_copy_of_classic_renders_exactly_as_classic_does(self) -> None:
        inline = copy.deepcopy(json.loads((formats.HERE / "classic.json").read_text("utf-8")))
        inline["id"] = "my-classic"
        with tempfile.TemporaryDirectory() as tmp:
            by_name = render_loop(request(format="classic"), backend=RecordingBackend(),
                                  output=Path(tmp) / "a.mp3")
            by_value = render_loop(request(format=inline), backend=RecordingBackend(),
                                   output=Path(tmp) / "b.mp3")
        self.assertEqual(by_name.format, "classic")
        self.assertEqual(by_value.format, "my-classic")
        self.assertEqual(by_name.timeline, by_value.timeline)
        self.assertEqual(by_name.total_bars, by_value.total_bars)

    def test_a_switch_changes_what_is_rendered(self) -> None:
        data = minimal(switches={"again": {"label": "Say it again", "default": False}},
                       sections=[{"kind": "words", "block": [
                           {"say": "word"}, {"say": "translation"},
                           {"say": "word", "switch": "again"}]}])
        with tempfile.TemporaryDirectory() as tmp:
            plain = render_loop(request(format=data), backend=RecordingBackend(),
                                output=Path(tmp) / "a.mp3")
            again = render_loop(request(format=data, switches={"again": True}),
                                backend=RecordingBackend(), output=Path(tmp) / "b.mp3")
        self.assertEqual(len(plain.timeline[0]["utterances"]), 2)
        self.assertEqual(len(again.timeline[0]["utterances"]), 3)
        self.assertEqual(again.total_bars - plain.total_bars, len(again.timeline))


if __name__ == "__main__":
    unittest.main()
