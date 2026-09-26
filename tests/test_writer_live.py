"""The writer formats against a real model. Gated: it spends calls on GEMINI_API_KEY.

    RUN_LIVE_WRITER_TESTS=true uv run --extra hosted-tts --env-file .env \\
        python -m unittest discover -s tests -p test_writer_live.py -v

Four calls a run: `radio-lesson` and `story`, each over Spanish taught from English and English
taught from Russian. Each asserts what the parser cannot — that remarks stay off grammar — and saves
the prompt, the model that answered and the script to `out/writer-live/`, for a person to read,
since whether a line is any good is not something a test can say.
"""

from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path

from lexibeat import formats, script
from lexibeat.language import ENGLISH, SPANISH, Language
from lexibeat.loop import write_script
from lexibeat.vocab import Item

RUSSIAN = Language("ru", "Russian")
OUT = Path(__file__).resolve().parent.parent / "out" / "writer-live"

# From experiments/programme_blocks/words.json, the tracked sample of a real vocabulary.
SPANISH_WORDS = (Item("el atasco", "traffic jam"), Item("el bigote", "mustache"),
                 Item("desaprovechar", "waste"), Item("el ayuntamiento", "town hall"),
                 Item("broncearse", "to get a tan"), Item("el chamuyo", "smooth talk"),
                 Item("el moretón", "bruise"), Item("anhelar", "long"))
ENGLISH_WORDS = (Item("concussion", "сотрясение мозга"), Item("chuckle", "посмеиваться"),
                 Item("dignity", "достоинство"), Item("coax", "уговаривать"),
                 Item("drizzle", "моросить"), Item("belligerent", "воинственный"),
                 Item("doppelganger", "двойник"), Item("chivalrous", "галантный"))

# Words a remark must not lean on. The prompt forbids grammar and linguistic terms; the parser cannot
# see meaning, so this is the check that the prompt held.
GRAMMAR_TERMS = ("conjugat", "infinitive", "syllable", "preterite", "subjunctive", "palatal",
                 "vowel", "consonant", "spelling", "spelled", "masculine", "feminine",
                 "спряжен", "инфинитив", "слог", "гласн", "согласн", "орфограф", "ударени")

LIVE = os.environ.get("RUN_LIVE_WRITER_TESTS", "").lower() == "true"


@unittest.skipUnless(LIVE, "set RUN_LIVE_WRITER_TESTS=true to spend GEMINI_API_KEY calls")
class LiveWriterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from lexibeat.writer import GeminiWriter

        cls.writer = GeminiWriter()
        OUT.mkdir(parents=True, exist_ok=True)
        cls.stamp = time.strftime("%Y%m%d-%H%M%S")

    def written(self, format_id: str, words, source: Language, target: Language):
        fmt = formats.resolve(formats.load(format_id))
        need = script.needs(fmt)
        prompt = script.prompt(need, words, source_language=source, target_language=target)
        result = write_script(fmt, words, source_language=source, target_language=target,
                              writer=self.writer)
        path = OUT / f"{self.stamp}-{format_id}-{source.code}-{target.code}.json"
        path.write_text(json.dumps({"format": format_id, "model": self.writer.last_model,
                                    "seconds": self.writer.last_seconds, "prompt": prompt,
                                    "script": result.raw}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        return result

    def check_radio(self, words, source, target) -> None:
        result = self.written("radio-lesson", words, source, target)
        self.assertEqual(sorted(result.order), list(range(len(words))))
        self.assertTrue(all(w.example for w in result.words))
        remarks = [w.remark[1].text.lower() for w in result.words if w.remark]
        self.assertLess(len(remarks), len(words), "a remark for every word is not 'sometimes'")
        for text in remarks:
            self.assertFalse([t for t in GRAMMAR_TERMS if t in text], text)
        self.assertTrue(result.intro and result.outro and result.groups)

    def check_story(self, words, source, target) -> None:
        result = self.written("story", words, source, target)
        self.assertEqual(len(result.beats), -(-len(words) // 2))
        self.assertEqual(sorted(result.order), list(range(len(words))))

    def test_a_radio_lesson_of_spanish_for_an_english_speaker(self) -> None:
        self.check_radio(SPANISH_WORDS, SPANISH, ENGLISH)

    def test_a_radio_lesson_of_english_for_a_russian_speaker(self) -> None:
        self.check_radio(ENGLISH_WORDS, ENGLISH, RUSSIAN)

    def test_a_story_in_spanish_for_an_english_speaker(self) -> None:
        self.check_story(SPANISH_WORDS, SPANISH, ENGLISH)

    def test_a_story_in_english_for_a_russian_speaker(self) -> None:
        self.check_story(ENGLISH_WORDS, ENGLISH, RUSSIAN)


if __name__ == "__main__":
    unittest.main()
