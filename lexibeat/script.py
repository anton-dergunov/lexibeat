"""What a writer model is asked for, and how its answer is read: the programme script.

A format that takes text from a writer — examples, remarks, a story, an intro — gets it from **one**
call per render. `needs()` reads the resolved format and says which parts to ask for; `prompt()`
assembles the prompt from `lexibeat/prompts/`, one part per need, so a writer is never asked for
what the format will not use; `parse()` reads the reply.

**The reply is read, never constrained.** No JSON mode and no schema: a schema guarantees a shape,
not a good line, and moves a model's drafting into the fields a listener hears. The prompt states
the shape, and `parse()` checks every part it relies on and refuses the reply naming the first thing
wrong with it. A missing optional part — no remark for a word — means that part did not happen.
Retrying belongs to the host's model chain, not here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from .formats import CALLBACK_KINDS, Format
from .language import Language
from .vocab import Item

PROMPTS = Path(__file__).resolve().parent / "prompts"
# What each phrase is, for a writer asked to supply the ones a learner language has no file for.
PHRASE_ASKS = {
    "your_turn": "very short invitations to say the word aloud (\"Your turn.\")",
    "quiz": "one line announcing a quick check of the words just heard (\"Quick check: do you "
            "remember these?\")",
    "review": "one line announcing that every word comes once more (\"Now, all the words "
              "once more.\")",
    "intro": "one opening line with `{count_words}` where the number of words goes",
    "outro": "one closing line (\"That's all for today.\")",
}
MAX_LINE = 500
MAX_DIRECTION = 200


class ScriptError(ValueError):
    """The writer's reply cannot be used. The message names what is wrong with it."""


@dataclass(frozen=True)
class Needs:
    """Which parts of a script a resolved format asks the writer for."""

    example: bool = False
    remark_kinds: tuple[str, ...] = ()
    hard_to_say: bool = False
    callback_kinds: tuple[str, ...] = ()
    chunk: int | None = None
    order: bool = False
    groups: bool = False
    intro: bool = False
    outro: bool = False
    # Learner-language phrases the format speaks and no phrase file has, by phrase-file key.
    phrases: tuple[str, ...] = ()
    # An example the format always plays must be written for every word.
    example_required: bool = False

    @property
    def any(self) -> bool:
        return any((self.example, self.remark_kinds, self.hard_to_say, self.callback_kinds,
                    self.chunk, self.order, self.groups, self.intro, self.outro, self.phrases))


@dataclass(frozen=True)
class Line:
    text: str
    translation: str = ""
    direction: str = ""
    speaker: str = "native"


@dataclass(frozen=True)
class WordScript:
    example: Line | None = None
    remark: tuple[str, Line] | None = None
    hard_to_say: bool = False
    callback: tuple[str, tuple[Line, ...]] | None = None


@dataclass(frozen=True)
class Script:
    words: tuple[WordScript, ...]
    order: tuple[int, ...] | None = None
    groups: tuple[tuple[str, tuple[int, ...]], ...] = ()
    intro: Line | None = None
    outro: Line | None = None
    title: str = ""
    beats: tuple[tuple[Line, ...], ...] = ()
    phrases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, compare=False)


# -- what to ask for -----------------------------------------------------------------------------


def needs(fmt: Format, *, missing_phrases: Sequence[str] = ()) -> Needs:
    """What to ask the writer for. `missing_phrases` are the phrase-file keys the format speaks
    that the learner language has no phrase file for."""
    example = example_required = hard = False
    remarks: list[str] = []
    callbacks: list[str] = []
    chunk = None
    intro = outro = groups = False
    for section in fmt.sections:
        if section.kind == "intro" and section.params.get("text") == "writer":
            intro = True
        if section.kind == "outro" and section.params.get("text") == "writer":
            outro = True
        if section.params.get("group_headers"):
            groups = True
        if section.params.get("chunk") and any(s.kind == "story_beat"
                                               for s in section.after_chunk):
            chunk = section.params["chunk"]
        for step in (*section.block, *section.after_chunk):
            for one in (step, *step.then):
                if one.kind == "example":
                    example = True
                    example_required |= one.when == "always"
                elif one.kind == "remark":
                    remarks += [k for k in one.value if k not in remarks]
                elif one.kind == "callback":
                    callbacks += [k for k in one.value if k not in callbacks]
                if one.when == "hard_to_say":
                    hard = True
    return Needs(example=example, remark_kinds=tuple(remarks), hard_to_say=hard,
                 callback_kinds=tuple(callbacks), chunk=chunk,
                 order=fmt.order in ("writer", "group_by_topic") or chunk is not None,
                 groups=groups or fmt.order == "group_by_topic", intro=intro, outro=outro,
                 phrases=tuple(missing_phrases),
                 example_required=example_required)


@lru_cache(maxsize=None)
def _part(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8").strip()


def _fill(text: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def prompt(need: Needs, items: Sequence[Item], *, source_language: Language,
           target_language: Language) -> str:
    """The one prompt for this render, asking for exactly what the format will use."""
    values = {"language": source_language.name, "learner_language": target_language.name,
              "count": str(len(items)), "remark_kinds": ", ".join(need.remark_kinds),
              "callback_kinds": ", ".join(need.callback_kinds), "chunk": str(need.chunk or 0),
              "order_note": ("Choose the teaching order (`order`) so the story can use the words "
                             "as they arrive." if need.chunk else "")}
    parts: list[str] = []
    word_shape: dict[str, str] = {}
    shape: dict[str, str] = {}
    if need.order:
        parts.append(_part("part_order"))
        shape["order"] = "[2, 0, 1]"
    if need.groups:
        parts.append(_part("part_groups"))
        shape["groups"] = '[{"title": "...", "items": [2, 0]}]'
    if need.intro:
        parts.append(_part("part_intro"))
        shape["intro"] = '{"text": "...", "direction": "..."}'
    if need.outro:
        parts.append(_part("part_outro"))
        shape["outro"] = '{"text": "...", "direction": "..."}'
    if need.example:
        parts.append(_part("part_example"))
        word_shape["example"] = '{"text": "...", "translation": "...", "direction": "..."}'
    if need.remark_kinds:
        parts.append(_part("part_remark"))
        word_shape["remark"] = '{"kind": "...", "text": "...", "direction": "..."} or null'
    if need.hard_to_say:
        parts.append(_part("part_hard_to_say"))
        word_shape["hard_to_say"] = "false"
    if need.callback_kinds:
        parts.append(_part("part_callback"))
        word_shape["callback"] = ('{"kind": "...", "refers_to": 0, "lines": [{"speaker": '
                                  '"native", "text": "...", "translation": "..."}]} or null')
    if need.chunk:
        parts.append(_part("part_story"))
        shape["title"] = '"..."'
        shape["beats"] = '[{"lines": [{"text": "...", "translation": "...", "direction": "..."}]}]'
    if need.phrases:
        parts.append(_part("part_phrases") + "\n" + "\n".join(
            f"- `{key}`: {PHRASE_ASKS[key]}" for key in need.phrases))
        shape["phrases"] = "{" + ", ".join(f'"{key}": ["...", "..."]'
                                           for key in need.phrases) + "}"
    if word_shape:
        inner = ", ".join(f'"{k}": {v}' for k, v in {"item": "0", **word_shape}.items())
        shape = {"words": f"[{{{inner}}}]", **shape}
    words = "\n".join(f"{index}. {item.source} — {item.target}"
                      for index, item in enumerate(items))
    body = "{\n" + ",\n".join(f'  "{k}": {v}' for k, v in shape.items()) + "\n}"
    return _fill(_part("programme"), {**values, "words": words,
                                      "parts": _fill("\n\n".join(parts), values), "shape": body})


# -- reading the reply ---------------------------------------------------------------------------


def _object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    elif not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ScriptError("the reply holds no JSON object")
        text = text[start:end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ScriptError(f"the reply is not valid JSON ({exc.msg} at line {exc.lineno})") from exc
    if not isinstance(data, dict):
        raise ScriptError("the reply is not a JSON object")
    return data


def _need(condition: bool, where: str, message: str) -> None:
    if not condition:
        raise ScriptError(f"{where}: {message}")


def _text(value: Any, where: str, what: str, *, limit: int = MAX_LINE,
          empty: bool = False) -> str:
    if value is None and empty:
        return ""
    _need(isinstance(value, str), where, f"'{what}' is not text")
    value = value.strip()
    _need(empty or value != "", where, f"'{what}' is empty")
    _need(len(value) <= limit, where, f"'{what}' is longer than {limit} characters")
    return value


def _line(data: Any, where: str, *, translated: bool, speaker: str = "native") -> Line:
    _need(isinstance(data, dict), where, "a line is an object")
    return Line(text=_text(data.get("text"), where, "text"),
                translation=_text(data.get("translation"), where, "translation",
                                  empty=not translated),
                direction=_text(data.get("direction"), where, "direction",
                                limit=MAX_DIRECTION, empty=True),
                speaker=speaker)


def _indices(value: Any, where: str, count: int, *, every: bool) -> tuple[int, ...]:
    _need(isinstance(value, list) and all(isinstance(i, int) and not isinstance(i, bool)
                                          for i in value), where, "is a list of word numbers")
    _need(all(0 <= i < count for i in value), where, f"names a word outside 0–{count - 1}")
    _need(len(set(value)) == len(value), where, "names a word twice")
    if every:
        _need(len(value) == count, where, f"must name every one of the {count} words")
    return tuple(value)


def parse(text: str, need: Needs, count: int) -> Script:
    """The script in a writer's reply, or `ScriptError` naming the first thing wrong with it."""
    data = _object(text)
    order = None
    if need.order:
        order = _indices(data.get("order"), "order", count, every=True)
    groups: list[tuple[str, tuple[int, ...]]] = []
    if need.groups:
        raw = data.get("groups")
        _need(isinstance(raw, list) and raw, "groups", "is a non-empty list")
        seen: list[int] = []
        for g, group in enumerate(raw):
            where = f"groups[{g}]"
            _need(isinstance(group, dict), where, "a group is an object")
            members = _indices(group.get("items"), f"{where}.items", count, every=False)
            _need(bool(members), where, "has no words")
            seen += members
            groups.append((_text(group.get("title"), where, "title", limit=120), members))
        _indices(seen, "groups", count, every=True)
        if order is None:
            order = tuple(seen)
    intro = _line(data.get("intro"), "intro", translated=False, speaker="guide") \
        if need.intro else None
    outro = _line(data.get("outro"), "outro", translated=False, speaker="guide") \
        if need.outro else None

    words = [WordScript() for _ in range(count)]
    per_word = need.example or need.remark_kinds or need.hard_to_say or need.callback_kinds
    if per_word:
        raw = data.get("words")
        _need(isinstance(raw, list), "words", "is a list, one entry per word")
        found: set[int] = set()
        for w, entry in enumerate(raw):
            where = f"words[{w}]"
            _need(isinstance(entry, dict), where, "an entry is an object")
            item = entry.get("item")
            _need(isinstance(item, int) and not isinstance(item, bool) and 0 <= item < count,
                  where, f"'item' is a word number from 0 to {count - 1}")
            _need(item not in found, where, f"word {item} has two entries")
            found.add(item)
            example = remark = callback = None
            if need.example and entry.get("example") is not None:
                example = _line(entry["example"], f"{where}.example", translated=True)
            if need.remark_kinds and entry.get("remark") is not None:
                at = f"{where}.remark"
                raw_remark = entry["remark"]
                _need(isinstance(raw_remark, dict), at, "a remark is an object or null")
                kind = raw_remark.get("kind")
                _need(kind in need.remark_kinds, at,
                      f"kind {kind!r} is not one of {', '.join(need.remark_kinds)}")
                remark = (kind, _line(raw_remark, at, translated=False, speaker="guide"))
            if need.callback_kinds and entry.get("callback") is not None:
                at = f"{where}.callback"
                raw_back = entry["callback"]
                _need(isinstance(raw_back, dict), at, "a callback is an object or null")
                kind = raw_back.get("kind")
                _need(kind in need.callback_kinds and kind in CALLBACK_KINDS, at,
                      f"kind {kind!r} is not one of {', '.join(need.callback_kinds)}")
                lines_raw = raw_back.get("lines")
                _need(isinstance(lines_raw, list) and 1 <= len(lines_raw) <= 3, at,
                      "'lines' holds one to three lines")
                lines = []
                for n, line in enumerate(lines_raw):
                    speaker = line.get("speaker") if isinstance(line, dict) else None
                    _need(speaker in ("native", "guide"), f"{at}.lines[{n}]",
                          "'speaker' is native or guide")
                    lines.append(_line(line, f"{at}.lines[{n}]", translated=False,
                                       speaker=speaker))
                callback = (kind, tuple(lines))
            hard = False
            if need.hard_to_say:
                hard = entry.get("hard_to_say", False)
                _need(isinstance(hard, bool), where, "'hard_to_say' is true or false")
            words[item] = WordScript(example, remark, hard, callback)
        if need.example_required:
            missing = [i for i in range(count) if words[i].example is None]
            if missing:
                raise ScriptError(f"words: no example for word {missing[0]}")

    title, beats = "", []
    if need.chunk:
        title = _text(data.get("title", ""), "title", "title", limit=120, empty=True)
        raw = data.get("beats")
        wanted = -(-count // need.chunk)
        _need(isinstance(raw, list) and len(raw) == wanted, "beats",
              f"needs one beat per group of {need.chunk} words: {wanted}")
        for b, beat in enumerate(raw):
            where = f"beats[{b}]"
            lines_raw = beat.get("lines") if isinstance(beat, dict) else None
            _need(isinstance(lines_raw, list) and 1 <= len(lines_raw) <= 3, where,
                  "'lines' holds one to three lines")
            beats.append(tuple(_line(line, f"{where}.lines[{n}]", translated=True)
                               for n, line in enumerate(lines_raw)))

    phrases: dict[str, tuple[str, ...]] = {}
    if need.phrases:
        raw = data.get("phrases")
        _need(isinstance(raw, dict), "phrases", "is an object of phrase lists")
        for key in need.phrases:
            lines = raw.get(key)
            _need(isinstance(lines, list) and len(lines) >= 3, f"phrases.{key}",
                  "holds at least three lines")
            phrases[key] = tuple(_text(line, f"phrases.{key}", "line", limit=120)
                                 for line in lines)
    return Script(tuple(words), order, tuple(groups), intro, outro, title, tuple(beats),
                  phrases, data)


def placeholder(need: Needs, count: int) -> Script:
    """A script with every part present, for estimating length before anything is written."""
    line = Line("…", "…")
    return Script(
        words=tuple(WordScript(example=line if need.example else None,
                               remark=(need.remark_kinds[0], line) if need.remark_kinds else None,
                               hard_to_say=need.hard_to_say,
                               callback=((need.callback_kinds[0], (line,))
                                         if need.callback_kinds else None))
                    for _ in range(count)),
        order=tuple(range(count)) if need.order else None,
        groups=((("…", tuple(range(count))),) if need.groups and count else ()),
        intro=line if need.intro else None, outro=line if need.outro else None,
        beats=tuple((line,) for _ in range(-(-count // need.chunk))) if need.chunk else (),
        phrases={key: ("…",) for key in need.phrases})
