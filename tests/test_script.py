"""Tests for the writer formats: what a writer is asked, how its reply is read, and what it renders."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from lexibeat import formats, script
from lexibeat.language import ENGLISH, SPANISH
from lexibeat.loop import LoopError, render_loop
from lexibeat.programme import plan
from lexibeat.script import Needs, ScriptError
from lexibeat.vocab import Item

from test_loop import RecordingBackend, request

WORDS = (Item("el atasco", "traffic jam"), Item("la cebolla", "onion"), Item("el banco", "bank"),
         Item("la multa", "fine"))


def reply(**overrides) -> dict:
    """A reply that answers everything the built-in writer formats ask for."""
    data = {
        "order": [2, 0, 1, 3],
        "groups": [{"title": "Around town", "items": [2, 0]},
                   {"title": "Food and fines", "items": [1, 3]}],
        "intro": {"text": "Four words today.", "direction": "warm"},
        "outro": {"text": "See you tomorrow.", "direction": "warm"},
        "title": "A day out",
        "beats": [{"lines": [{"text": "Hay un atasco en el banco.",
                              "translation": "There's a jam at the bank.", "direction": ""}]},
                  {"lines": [{"text": "Una multa por una cebolla.",
                              "translation": "A fine for an onion.", "direction": ""}]}],
        "words": [{"item": i,
                   "example": {"text": f"Ejemplo {i}.", "translation": f"Example {i}.",
                               "direction": "deadpan"},
                   "remark": ({"kind": "mnemonic", "text": "Picture a crying onion.",
                               "direction": ""} if i == 1 else None),
                   "hard_to_say": i == 0}
                  for i in range(len(WORDS))],
    }
    data.update(overrides)
    return data


class FakeWriter:
    def __init__(self, answer: dict | str | None = None) -> None:
        self.answer = answer if answer is not None else reply()
        self.prompts: list[str] = []

    def write(self, request) -> str:
        self.prompts.append(request.prompt)
        text = self.answer if isinstance(self.answer, str) else json.dumps(self.answer)
        return f"Here it is:\n```json\n{text}\n```"


def mixing() -> RecordingBackend:
    backend = RecordingBackend()
    backend.capabilities = replace(backend.capabilities, mixes_languages=True)
    return backend


def render(format_id: str, writer=None, backend=None, **overrides):
    backend = backend or mixing()
    with tempfile.TemporaryDirectory() as tmp:
        result = render_loop(request(items=WORDS, format=format_id, **overrides),
                             backend=backend, writer=writer, output=Path(tmp) / "loop.mp3")
    return result, backend


class NeedsTests(unittest.TestCase):
    def test_a_format_asks_the_writer_for_exactly_what_it_uses(self) -> None:
        radio = script.needs(formats.resolve(formats.load("radio-lesson")))
        self.assertTrue(radio.example and radio.example_required and radio.hard_to_say)
        self.assertEqual(set(radio.remark_kinds),
                         {"contrast", "register", "false_friend", "mnemonic", "culture"})
        self.assertTrue(radio.groups and radio.order and radio.intro and radio.outro)
        self.assertFalse(radio.chunk or radio.callback_kinds)
        story = script.needs(formats.resolve(formats.load("story")))
        self.assertEqual(story.chunk, 2)
        self.assertFalse(story.example or story.remark_kinds or story.groups)

    def test_a_switched_off_remark_is_not_asked_for(self) -> None:
        radio = script.needs(formats.resolve(formats.load("radio-lesson"), {"remarks": False}))
        self.assertEqual(radio.remark_kinds, ())

    def test_the_prompt_holds_only_the_parts_asked_for(self) -> None:
        story = script.prompt(script.needs(formats.resolve(formats.load("story"))), WORDS,
                              source_language=SPANISH, target_language=ENGLISH)
        self.assertIn("A story", story)
        self.assertNotIn("Remarks", story)
        self.assertIn("0. el atasco — traffic jam", story)
        self.assertIn('"beats"', story)
        radio = script.prompt(script.needs(formats.resolve(formats.load("radio-lesson"))), WORDS,
                              source_language=SPANISH, target_language=ENGLISH)
        self.assertIn("Hard to say", radio)
        self.assertIn("contrast, register", radio)
        self.assertNotIn("{{", radio)


class ParseTests(unittest.TestCase):
    need = Needs(example=True, example_required=True, remark_kinds=("mnemonic",),
                 hard_to_say=True, order=True)

    def test_a_reply_is_read_from_a_fence_or_bare(self) -> None:
        data = reply()
        fenced = script.parse(f"```json\n{json.dumps(data)}\n```", self.need, len(WORDS))
        bare = script.parse(f"Sure! {json.dumps(data)} Enjoy.", self.need, len(WORDS))
        self.assertEqual(fenced, bare)
        self.assertEqual(fenced.order, (2, 0, 1, 3))
        self.assertTrue(fenced.words[0].hard_to_say)
        self.assertEqual(fenced.words[1].remark[0], "mnemonic")
        self.assertIsNone(fenced.words[0].remark)

    def test_a_refusal_names_what_is_wrong(self) -> None:
        words = reply()["words"]
        cases = [
            ("no JSON here", "holds no JSON object"),
            ('{"order": [0, 1,', "not valid JSON"),
            (json.dumps(reply(order=[0, 1, 2])), r"order: must name every one of the 4 words"),
            (json.dumps(reply(order=[0, 0, 1, 2])), r"order: names a word twice"),
            (json.dumps(reply(order=[0, 1, 2, 9])), r"order: names a word outside"),
            (json.dumps(reply(words=words[:3])), r"words: no example for word 3"),
            (json.dumps(reply(words=[{**words[0], "remark": {"kind": "grammar", "text": "x"}},
                                     *words[1:]])),
             r"words\[0\]\.remark: kind 'grammar' is not one of mnemonic"),
            (json.dumps(reply(words=[{**words[0], "example": {"text": "", "translation": "x"}},
                                     *words[1:]])),
             r"words\[0\]\.example: 'text' is empty"),
            (json.dumps(reply(words=[{**words[0], "example": {"text": "x" * 600,
                                                               "translation": "y"}},
                                     *words[1:]])), r"longer than 500 characters"),
            (json.dumps(reply(words=[{**words[0], "hard_to_say": "yes"}, *words[1:]])),
             r"'hard_to_say' is true or false"),
            (json.dumps(reply(words=[words[0], words[0], *words[2:]])), r"word 0 has two entries"),
        ]
        for text, message in cases:
            with self.subTest(message):
                with self.assertRaisesRegex(ScriptError, message):
                    script.parse(text, self.need, len(WORDS))

    def test_a_story_needs_one_beat_per_group_of_words(self) -> None:
        need = Needs(chunk=2, order=True)
        with self.assertRaisesRegex(ScriptError, r"beats: needs one beat per group of 2 words: 2"):
            script.parse(json.dumps(reply(beats=reply()["beats"][:1])), need, len(WORDS))


class PlanTests(unittest.TestCase):
    def plan(self, format_id: str, answer: dict | None = None, **switches):
        fmt = formats.resolve(formats.load(format_id), switches)
        written = script.parse(json.dumps(answer or reply()), script.needs(fmt), len(WORDS))
        return plan(fmt, WORDS, source_language=SPANISH, target_language=ENGLISH, script=written)

    def test_then_follows_only_a_step_that_happened(self) -> None:
        lines = self.plan("radio-lesson").lines
        # Word 1 has a remark, which comes before its example and is not followed by the word;
        # word 0 is hard to say, so it is said slowly and then once more; word 2 has neither.
        def after(item, kind):
            mine = [s for s in lines if s.item == item and s.section == "words"]
            at = next(i for i, s in enumerate(mine) if s.kind == kind)
            return [(s.kind, s.side) for s in mine[at + 1:]]
        self.assertEqual(after(1, "remark")[:2], [("example", None), ("translation", None)])
        self.assertEqual(after(0, "pronounce"), [("say", "source")])
        # Three pairs by default, before anything the writer wrote.
        self.assertEqual(sum(1 for s in lines if s.item == 2 and s.section == "words"
                             and s.side == "source"), 3)
        self.assertFalse(any(s.kind in ("remark", "pronounce") for s in lines if s.item == 2))
        slow = next(s for s in lines if s.kind == "pronounce")
        self.assertEqual((slow.pace, slow.bars, slow.role), ("slow", None, "native"))

    def test_the_writer_s_order_and_groups_shape_the_words_section(self) -> None:
        lines = self.plan("radio-lesson").lines
        taught = [s.item for s in lines if s.section == "words" and s.kind == "say"]
        self.assertEqual(list(dict.fromkeys(taught)), [2, 0, 1, 3])
        headers = [(s.text, lines[i + 1].item) for i, s in enumerate(lines) if s.kind == "header"]
        self.assertEqual(headers, [("Around town", 2), ("Food and fines", 1)])
        # The midpoint quiz covers the first half as taught: words 2 and 0, each pair back to
        # back after the quiz is announced.
        quiz = [s for s in lines if s.section == "quiz"]
        self.assertEqual(quiz[0].kind, "announce")
        self.assertEqual([(s.item, s.side) for s in quiz[1:]],
                         [(2, "target"), (2, "source"), (0, "target"), (0, "source")])

    def test_a_story_beat_follows_each_group_of_words(self) -> None:
        lines = self.plan("story").lines
        kinds = [(s.kind, s.item) for s in lines if s.section == "words" and s.kind != "say"]
        self.assertEqual(kinds, [("story", None), ("translation", None)] * 2)
        first = next(i for i, s in enumerate(lines) if s.kind == "story")
        self.assertEqual({s.item for s in lines[:first] if s.kind == "say"}, {2, 0})
        story = [s for s in lines if s.kind == "story"]
        self.assertTrue(all(s.role == "native" and s.language == SPANISH for s in story))

    def test_an_example_is_native_and_its_translation_the_guide_s(self) -> None:
        lines = self.plan("radio-lesson").lines
        example = next(i for i, s in enumerate(lines) if s.kind == "example")
        self.assertEqual((lines[example].role, lines[example].language), ("native", SPANISH))
        self.assertEqual((lines[example + 1].kind, lines[example + 1].role), ("translation",
                                                                              "guide"))


class ScriptRoundTripTests(unittest.TestCase):
    def test_a_render_returns_its_script_and_it_renders_the_same_lines_again(self) -> None:
        first, _ = render("radio-lesson", FakeWriter())
        self.assertIsNotNone(first.script)
        backend = mixing()
        with tempfile.TemporaryDirectory() as tmp:
            again = render_loop(request(items=WORDS, format="radio-lesson", script=first.script),
                                backend=backend, writer=None, output=Path(tmp) / "loop.mp3")
        self.assertEqual((again.format, again.fallback_from), ("radio-lesson", None))
        self.assertEqual([c["text"] for c in again.cues], [c["text"] for c in first.cues])
        self.assertEqual(again.script, first.script)

    def test_a_given_script_calls_no_writer(self) -> None:
        first, _ = render("story", FakeWriter())
        writer = FakeWriter()
        render("story", writer, script=first.script)
        self.assertEqual(writer.prompts, [])

    def test_a_script_that_does_not_fit_the_format_is_refused_naming_why(self) -> None:
        # A story's script, as a real writer would answer it: no examples, no groups.
        story_only = {key: value for key, value in reply().items()
                      if key in ("order", "title", "beats", "intro", "outro")}
        story, _ = render("story", FakeWriter(story_only))
        with self.assertRaisesRegex(LoopError, r"The script sent with this render cannot be "
                                               r"used: groups: is a non-empty list"):
            render("radio-lesson", script=story.script)

    def test_a_format_with_no_writer_returns_no_script(self) -> None:
        result, _ = render("classic")
        self.assertIsNone(result.script)


class GroupTests(unittest.TestCase):
    def groups(self, format_id: str):
        result, _ = render(format_id, FakeWriter())
        by_group: dict[int, list[dict]] = {}
        for cue in result.cues:
            by_group.setdefault(cue["group"], []).append(cue)
        return result, by_group

    def test_groups_rise_in_the_order_heard(self) -> None:
        result, _ = self.groups("radio-lesson")
        groups = [cue["group"] for cue in result.cues]
        self.assertEqual(groups, sorted(groups))
        self.assertEqual(groups[0], 0)

    def test_a_drill_a_remark_and_an_example_are_three_groups(self) -> None:
        _, by_group = self.groups("radio-lesson")
        word1 = [[c["kind"] for c in cues] for cues in by_group.values()
                 if cues[0]["item"] == 1 and cues[0]["section"] == "words"]
        self.assertEqual(word1, [["say"] * 6, ["remark"], ["example", "translation"]])

    def test_a_quiz_pair_is_one_group_and_its_announcement_another(self) -> None:
        _, by_group = self.groups("radio-lesson")
        quiz = [[(c["kind"], c["side"]) for c in cues] for cues in by_group.values()
                if cues[0]["section"] == "quiz"]
        self.assertEqual(quiz[0], [("announce", None)])
        self.assertTrue(all(pair == [("say", "target"), ("say", "source")] for pair in quiz[1:]))
        review = [cues for cues in by_group.values() if cues[0]["section"] == "review"]
        self.assertEqual(len(review), 1 + len(WORDS))

    def test_a_story_line_and_its_translation_are_one_group(self) -> None:
        _, by_group = self.groups("story")
        story = [[c["kind"] for c in cues] for cues in by_group.values()
                 if cues[0]["kind"] == "story"]
        self.assertTrue(story and all(kinds == ["story", "translation"] for kinds in story))


class RenderTests(unittest.TestCase):
    def test_a_writer_format_renders_its_written_lines(self) -> None:
        writer = FakeWriter()
        result, backend = render("radio-lesson", writer)
        self.assertEqual(len(writer.prompts), 1)
        self.assertEqual((result.format, result.fallback_from), ("radio-lesson", None))
        kinds = {cue["kind"] for cue in result.cues}
        self.assertTrue({"intro", "outro", "header", "example", "remark", "pronounce"} <= kinds)
        self.assertEqual(result.cues[0]["text"], "Four words today.")
        # Written lines take the bars they need, and nothing overlaps the next line's downbeat.
        starts = [cue["start"] for cue in result.cues]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual([row["index"] for row in result.items], [0, 1, 2, 3])

    def test_without_a_writer_a_writer_format_falls_back(self) -> None:
        result, _ = render("radio-lesson")
        self.assertEqual((result.format, result.fallback_from), ("classic", "radio-lesson"))

    def test_without_a_voice_that_mixes_languages_the_radio_lesson_falls_back(self) -> None:
        result, _ = render("radio-lesson", FakeWriter(), backend=RecordingBackend())
        self.assertEqual((result.format, result.fallback_from), ("classic", "radio-lesson"))

    def test_a_writer_format_without_a_fallback_is_refused_naming_the_writer(self) -> None:
        with self.assertRaisesRegex(LoopError, r"requires writer"):
            render("story")

    def test_an_unusable_reply_fails_the_render_before_any_speech(self) -> None:
        backend = mixing()
        with self.assertRaisesRegex(LoopError, r"The writer's reply could not be used: "
                                               r"beats: needs one beat"):
            render("story", FakeWriter(reply(beats=[])), backend=backend)
        self.assertEqual(backend.seen, [])

    def test_a_learner_language_without_phrases_takes_them_from_the_writer(self) -> None:
        from lexibeat.language import Language

        echo_writer = {"id": "echo-written", "label": "E", "description": "Echo with a story.",
                       "requires": ["writer"],
                       "sections": [{"kind": "words", "block": [
                           {"say": "word"}, {"cue": "your_turn"}, {"say": "translation"},
                           {"example": "writer"}]}]}
        answer = reply(phrases={"your_turn": ["Sua vez.", "Agora você.", "Diga."]})
        result, _ = render(echo_writer, FakeWriter(answer),
                           target_language=Language("pt", "Portuguese"))
        cues = [cue["text"] for cue in result.cues if cue["kind"] == "cue"]
        self.assertTrue(cues and set(cues) <= {"Sua vez.", "Agora você.", "Diga."})


class QuoteTests(unittest.TestCase):
    """A guide line that quotes Spanish names what it quotes, so the voice says it natively."""

    REMARK = {"kind": "mnemonic", "direction": "",
              "text": "Say cebolla, not Onion: la Cebolla makes you cry, tener ganas de does not.",
              "quoted": ["la cebolla", "tener ganas de", "nothing like this", "cebolla"]}

    def written(self, answer: dict | None = None):
        answer = answer or reply(words=[{"item": i, "example": {"text": "E.", "translation": "E."},
                                         "remark": self.REMARK if i == 1 else None}
                                        for i in range(len(WORDS))])
        fmt = formats.resolve(formats.load("radio-lesson"))
        return script.parse(json.dumps(answer), script.needs(fmt), len(WORDS)), fmt

    def test_the_parser_keeps_what_the_line_says_as_the_line_says_it(self) -> None:
        written, _ = self.written()
        _, remark = written.words[1].remark
        # A span the line does not contain is dropped; casing is the line's own.
        self.assertEqual(remark.quoted, ("la Cebolla", "tener ganas de", "cebolla"))

    def test_quoted_that_is_not_a_list_is_refused_by_name(self) -> None:
        answer = reply(intro={"text": "Four words.", "quoted": "atasco"})
        with self.assertRaisesRegex(ScriptError, r"intro: 'quoted' is a list"):
            self.written(answer)

    def test_the_guide_is_told_which_words_to_say_in_spanish(self) -> None:
        written, fmt = self.written()
        lines = plan(fmt, WORDS, source_language=SPANISH, target_language=ENGLISH,
                     script=written).lines
        remark = next(s for s in lines if s.kind == "remark")
        # In the order heard; "cebolla" inside "la Cebolla" is the longer span's, and the loop's own
        # word is found whether or not the writer listed it.
        self.assertEqual(remark.quotes.spans, ("cebolla", "la Cebolla", "tener ganas de"))
        self.assertEqual(remark.quotes.sentence(),
                         "Say “cebolla”, “la Cebolla” and “tener ganas de” in Spanish, with a "
                         "native Spanish pronunciation, and everything else in English.")
        # A header names a word the writer never listed; a translation or a cue quotes nothing.
        header = plan(fmt, WORDS, source_language=SPANISH, target_language=ENGLISH,
                      script=self.written(reply(groups=[
                          {"title": "El banco and el atasco", "items": [2, 0]},
                          {"title": "Food", "items": [1, 3]}]))[0]).lines
        self.assertEqual(next(s for s in header if s.kind == "header").quotes.spans,
                         ("El banco", "el atasco"))
        self.assertTrue(all(s.quotes is None for s in lines
                            if s.kind in ("translation", "cue", "announce", "say")))

    def test_the_note_reaches_a_voice_that_mixes_languages_and_no_other(self) -> None:
        answer = reply(words=[{"item": i, "example": {"text": "E.", "translation": "E."},
                               "remark": self.REMARK if i == 1 else None}
                              for i in range(len(WORDS))])
        _, backend = render("radio-lesson", FakeWriter(answer))
        from lexibeat.voice import delivery_instruction

        told = [delivery_instruction(seen.delivery) for seen in backend.seen
                if seen.text == self.REMARK["text"]]
        self.assertTrue(told)
        self.assertTrue(all(note.endswith("and everything else in English.") for note in told))
        # A plain voice is not told: the radio lesson falls back, and a classic drill quotes nothing.
        _, plain = render("radio-lesson", FakeWriter(answer), backend=RecordingBackend())
        self.assertTrue(all(seen.delivery.quotes is None for seen in plain.seen))

    def test_the_prompt_asks_for_quoted_on_guide_lines(self) -> None:
        fmt = formats.resolve(formats.load("radio-lesson"))
        text = script.prompt(script.needs(fmt), WORDS, source_language=SPANISH,
                             target_language=ENGLISH)
        self.assertIn('"quoted": ["..."]', text)
        self.assertIn("with a native Spanish pronunciation", " ".join(text.split()))


class GeminiWriterTests(unittest.TestCase):
    def test_a_busy_model_is_waited_out_and_one_out_of_quota_passed_over(self) -> None:
        try:
            from google.genai import types as _  # noqa: F401
        except ImportError:
            self.skipTest("the hosted-tts extra is not installed")
        from unittest import mock

        from lexibeat.writer import GeminiWriter, WriteRequest, WriterError

        calls = []

        def answer(model, contents, config):
            calls.append(model)
            if model == "a":
                raise RuntimeError("429 RESOURCE_EXHAUSTED quota")
            if model == "b" and calls.count("b") < 3:
                raise RuntimeError("503 UNAVAILABLE high demand")
            return mock.Mock(text="{}")

        writer = GeminiWriter.__new__(GeminiWriter)
        writer._client = mock.Mock()
        writer._client.models.generate_content.side_effect = answer
        writer.models, writer.temperature = ("a", "b", "c"), None
        with mock.patch.object(GeminiWriter, "_sleep") as sleep:
            self.assertEqual(writer.write(WriteRequest("p")), "{}")
        self.assertEqual(calls, ["a", "b", "b", "b"])
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(writer.last_model, "b")
        writer._client.models.generate_content.side_effect = RuntimeError("400 INVALID_ARGUMENT")
        with self.assertRaisesRegex(WriterError, "a refused: 400"):
            writer.write(WriteRequest("p"))


class ServiceTests(unittest.TestCase):
    def test_the_service_builds_a_writer_per_render(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("the service extra is not installed")
        import time

        from lexibeat.service import API_PREFIX, ServiceConfig, create_service

        built = []

        def writer_for(context):
            built.append(context.operation_id)
            return FakeWriter()

        with tempfile.TemporaryDirectory() as tmp:
            app = create_service(config=ServiceConfig(output_root=Path(tmp)),
                                 backend_factory=lambda context: mixing(),
                                 writer_factory=writer_for)
            client = TestClient(app)
            body = {"items": [{"source": w.source, "target": w.target} for w in WORDS],
                    "source_language": {"code": "es", "name": "Spanish"},
                    "target_language": {"code": "en", "name": "English"},
                    "format": "radio-lesson", "seed": 5}
            operation = client.post(f"{API_PREFIX}/loops", json=body).json()
            for _ in range(3000):  # five minutes: a loaded machine renders slowly
                state = client.get(f"{API_PREFIX}/operations/{operation['operation_id']}").json()
                if state["status"] in ("completed", "failed"):
                    break
                time.sleep(0.1)
        self.assertEqual(state["status"], "completed", state.get("error"))
        self.assertEqual(state["result"]["format"], "radio-lesson")
        self.assertEqual(built, [operation["operation_id"]])


if __name__ == "__main__":
    unittest.main()
