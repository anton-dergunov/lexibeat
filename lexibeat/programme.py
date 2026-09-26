"""Plan a loop: a resolved format and its words, as the ordered segments the arranger places.

A format is a recipe and carries no text; a programme is that recipe applied to these words, in
these languages, with every line's text, voice role, take index and pace decided. The arranger then
only has to put each segment on the grid, which is why a planning mistake is testable here without
synthesising a note.

**Sections run in the format's order**, with one exception: a `quiz` placed `at: "middle"` goes
after the first half of the words and covers that half. A quiz at the end, and a review, cover every
word.

**Take indices continue across sections.** A word's third line on one side is take 2 whether it is
in the drill or the review, so a voice keeps varying its delivery instead of repeating take 0.

**Learner-language lines that are not the words themselves** — a cue, an intro, an outro — come from
the phrase files beside the built-in formats. A line is picked from the seed and the word's index,
so neighbouring words hear different cues and a re-render hears the same ones.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

from .formats import HERE, SAY_ROLES, SOURCE, TARGET, Format, FormatError, Step, needs_writer
from .language import Language
from .script import Script, needs as script_needs, placeholder
from .vocab import Item

PHRASES = HERE / "phrases"

# The director note for a cue: an invitation, not an announcement.
CUE_DIRECTION = "warmly, as an invitation to speak"
TEMPLATE_DIRECTION = "warmly, like a presenter"


@dataclass(frozen=True)
class Segment:
    """One bar-aligned thing in a loop: a line, a silence, or music alone.

    `bars` is a fixed count, or None for a line that takes as many bars as it needs — anything a
    writer wrote, and a word said slowly. `side` is which half of the pair a word's own line says
    (`source` or `target`); `role` is who says it; `block` numbers a word's own block in the words
    section, which is what the long-take retry compares repetitions within.
    """

    kind: str
    section: str
    bars: int | None
    item: int | None = None
    side: str | None = None
    role: str = "native"
    language: Language | None = None
    text: str = ""
    take: int = 0
    pace: str = ""
    stretch: float = 1.0
    direction: str = ""
    block: int | None = None

    @property
    def spoken(self) -> bool:
        return bool(self.text)


@dataclass(frozen=True)
class Programme:
    format_id: str
    segments: tuple[Segment, ...] = field(default_factory=tuple)

    @property
    def lines(self) -> list[Segment]:
        return [segment for segment in self.segments if segment.spoken]

    @property
    def bars(self) -> int:
        """The bars of a programme whose every segment has a fixed length."""
        return sum(segment.bars or 0 for segment in self.segments)


# -- phrases -----------------------------------------------------------------------------------


def _base(code: str) -> str:
    return code.split("-")[0].split("_")[0].lower()


@lru_cache(maxsize=None)
def phrases(code: str) -> dict | None:
    """The phrase file for a language code (`en`, `en-US` alike), or None when there is none."""
    path = PHRASES / f"{_base(code)}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _plural(code: str, count: int) -> str:
    """The plural category a phrase file's `word_forms` is keyed by (CLDR's, for these languages)."""
    if _base(code) == "ru":
        if count % 10 == 1 and count % 100 != 11:
            return "one"
        if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
            return "few"
        return "many"
    return "one" if count == 1 else "other"


def count_words(code: str, count: int) -> str:
    table = phrases(code)
    forms = table["word_forms"] if table else {}
    return f"{count} {forms.get(_plural(code, count), forms.get('other', ''))}".strip()


def needs_phrases(fmt: Format) -> bool:
    return any(step.kind == "cue" for section in fmt.sections for step in section.block) or \
        any(section.kind in ("intro", "outro") and section.params.get("text", "template")
            == "template" for section in fmt.sections)


def _pick(options: Sequence[str], seed: int, index: int) -> str:
    return options[(seed + index) % len(options)]


# -- planning ----------------------------------------------------------------------------------


class _Takes:
    """Each word's take counter per side, shared by every section."""

    def __init__(self) -> None:
        self._next: dict[tuple[int, str], int] = {}

    def __call__(self, item: int, side: str) -> int:
        key = (item, side)
        take = self._next.get(key, 0)
        self._next[key] = take + 1
        return take


def plan(fmt: Format, items: Sequence[Item], *, source_language: Language,
         target_language: Language, seed: int = 0, script: Script | None = None) -> Programme:
    """The programme for a resolved, renderable format.

    `script` is what the writer wrote for this render; a format that takes text from a writer
    cannot be planned without one. Raises `FormatError` when a learner-language phrase is needed
    and neither a phrase file nor the script has it.
    """
    if needs_writer(fmt) and script is None:
        raise FormatError(f"format '{fmt.id}' takes text from a writer, and this render has none")
    table = dict(phrases(target_language.code) or {})
    if script is not None and script.your_turn and "your_turn" not in table:
        table["your_turn"] = list(script.your_turn)
    if needs_phrases(fmt) and not _has_phrases(fmt, table):
        raise FormatError(f"format '{fmt.id}' needs learner-language phrases, and there are none "
                          f"for {target_language.name} ({target_language.code}) yet")
    languages = {SOURCE: source_language, TARGET: target_language}
    roles = {SOURCE: "native", TARGET: "guide"}
    takes = _Takes()
    words = next(section for section in fmt.sections if section.kind == "words")
    count = len(items)
    order = list(script.order) if script is not None and script.order else list(range(count))
    middle = math.ceil(count / 2)
    segments: list[Segment] = []
    block = 0

    def native(text: str, section: str, index: int | None, kind: str, direction: str = "",
               block_id: int | None = None) -> None:
        segments.append(Segment(kind=kind, section=section, bars=None, item=index,
                                role="native", language=source_language, text=text,
                                direction=direction, block=block_id))

    def guide(text: str, section: str, index: int | None, kind: str, direction: str = "",
              block_id: int | None = None) -> None:
        segments.append(Segment(kind=kind, section=section, bars=None, item=index,
                                role="guide", language=target_language, text=text,
                                direction=direction, block=block_id))

    def happened(step: Step, index: int) -> bool:
        written = script.words[index] if script is not None else None
        if step.when == "always":
            return True
        if step.when == "hard_to_say":
            return bool(written and written.hard_to_say)
        # writer_decides and natural_link: the step happens when the writer wrote it.
        return bool(written and {"example": written.example, "remark": written.remark,
                                 "callback": written.callback}.get(step.kind))

    def steps(section_kind: str, block_steps: Sequence[Step], index: int, item: Item, *,
              stretch: float = 1.0, block_id: int | None = None) -> None:
        for step in block_steps:
            if not happened(step, index):
                continue
            one(section_kind, step, index, item, stretch=stretch, block_id=block_id)
            if step.then:
                steps(section_kind, step.then, index, item, stretch=stretch, block_id=block_id)

    def one(section_kind: str, step: Step, index: int, item: Item, *, stretch: float,
            block_id: int | None) -> None:
        written = script.words[index] if script is not None else None
        if step.kind in ("say", "pronounce"):
            side = SAY_ROLES[step.value] if step.kind == "say" else SOURCE
            slow = step.kind == "pronounce"
            segments.append(Segment(
                kind=step.kind, section=section_kind,
                # Said slowly and whole, a word can outrun its bar; it is never squeezed back.
                bars=None if slow else 1, item=index, side=side, role=roles[side],
                language=languages[side], text=item.source if side == SOURCE else item.target,
                take=takes(index, side), pace="slow" if slow else step.params.get("pace", ""),
                stretch=1.0 if slow else step.params.get("stretch", stretch),
                direction=item.direction, block=block_id))
        elif step.kind in ("gap", "rest"):
            segments.append(Segment(kind=step.kind, section=section_kind, bars=step.value,
                                    item=index, block=block_id))
        elif step.kind == "cue":
            segments.append(Segment(
                kind="cue", section=section_kind, bars=1, item=index, role="guide",
                language=target_language, text=_pick(table[step.value], seed, index),
                direction=CUE_DIRECTION, block=block_id))
        elif step.kind == "example":
            line = written.example
            native(line.text, section_kind, index, "example", line.direction, block_id)
            if step.params.get("translate") and line.translation:
                guide(line.translation, section_kind, index, "translation", "", block_id)
        elif step.kind == "remark":
            _, line = written.remark
            guide(line.text, section_kind, index, "remark", line.direction, block_id)
        elif step.kind == "callback":
            _, lines = written.callback
            for line in lines:
                if line.speaker == "native":
                    native(line.text, section_kind, index, "callback", line.direction, block_id)
                    if line.translation:
                        guide(line.translation, section_kind, index, "translation", "",
                              block_id)
                else:
                    guide(line.text, section_kind, index, "callback", line.direction, block_id)
        else:  # story_beat runs after a chunk; unsupported() refuses anything else
            raise FormatError(f"'{step.kind}' cannot be planned in a block")

    def framing(section) -> None:
        kind = section.kind
        if section.params.get("text") == "writer":
            line = script.intro if kind == "intro" else script.outro
            guide(line.text, kind, None, kind, line.direction)
            return
        line = _pick(table[kind], seed, len(segments)).format(
            count_words=count_words(target_language.code, count))
        segments.append(Segment(kind=kind, section=kind, bars=1, role="guide",
                                language=target_language, text=line,
                                direction=TEMPLATE_DIRECTION))

    def over(section, indices: Sequence[int]) -> None:
        stretch = section.params.get("stretch", 1.0)
        for index in indices:
            steps(section.kind, section.block, index, items[index], stretch=stretch)

    def beat(section, number: int) -> None:
        for step in section.after_chunk:
            if step.kind == "story_beat":
                for line in script.beats[number]:
                    native(line.text, "words", None, "story", line.direction)
                    if line.translation:
                        guide(line.translation, "words", None, "translation")
            elif step.kind == "rest":
                segments.append(Segment(kind="rest", section="words", bars=step.value))

    headers = {}
    if words.params.get("group_headers") and script is not None:
        headers = {members[0]: title for title, members in script.groups}
    chunk = words.params.get("chunk")
    middle_quizzes = [s for s in fmt.sections if s.kind == "quiz" and s.params.get("at") == "middle"]
    for section in fmt.sections:
        if section.kind in ("intro", "outro"):
            framing(section)
        elif section.kind == "words":
            for position, index in enumerate(order):
                if index in headers:
                    guide(headers[index], "words", None, "header", TEMPLATE_DIRECTION)
                steps("words", words.block, index, items[index], block_id=block)
                block += 1
                taught = position + 1
                if chunk and (taught % chunk == 0 or taught == count):
                    beat(words, (taught - 1) // chunk)
                if taught == middle and taught < count:
                    for quiz in middle_quizzes:
                        over(quiz, order[:middle])
            if count == 1:
                for quiz in middle_quizzes:
                    over(quiz, order)
        elif section.kind == "quiz":
            if section.params.get("at") != "middle":
                over(section, order)
        elif section.kind == "review":
            over(section, order)
    return Programme(fmt.id, tuple(segments))


def _has_phrases(fmt: Format, table: dict) -> bool:
    for section in fmt.sections:
        if section.kind in ("intro", "outro") and section.params.get("text", "template") \
                == "template" and section.kind not in table:
            return False
        if any(step.kind == "cue" and step.value not in table for step in section.block):
            return False
    return True


# A line whose length depends on its text is estimated at this many bars before it is written.
ESTIMATED_BARS_PER_LINE = 2


def estimated_bars(fmt: Format, count: int) -> int:
    """How many bars a format takes for so many words, before any of them is known.

    Exact for a format of fixed lines; a written line — a remark, a story beat — is counted at
    `ESTIMATED_BARS_PER_LINE`, and every optional one as though it happens.
    """
    words = [Item(f"w{i}", f"t{i}") for i in range(max(count, 0))]
    need = script_needs(fmt, phrases_missing=False)
    written = placeholder(need, len(words)) if need.any else None
    programme = plan(fmt, words, source_language=Language("xx", "x"),
                     target_language=Language("en", "English"), script=written)
    return sum(segment.bars if segment.bars is not None else ESTIMATED_BARS_PER_LINE
               for segment in programme.segments)
