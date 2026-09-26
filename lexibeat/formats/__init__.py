"""Programme formats: what a loop is made of, as a small JSON document per format.

A format is a recipe and never carries text. It names sections (`intro`, `words`, `quiz`, `review`,
`outro`), and a `words`, `quiz` or `review` section runs its `block` of steps once per word. The
grammar is closed on purpose: a step's only condition is `when`, taken from a fixed list of named
checks, and a format cannot combine them, count, or refer from one segment to another. Anything
that would need that becomes a new named check, a new section kind, or a decision the writer makes.
`docs/programme-format.md` is the contract; this module parses the whole of it.

Not everything the grammar describes is rendered yet. `unsupported()` names what this version cannot
render, and a render refuses such a format by name instead of dropping part of it.

The built-in formats are the JSON files beside this module. The host names one, and may set the
switches it declares; it never builds a format. An inline format goes through the same parser.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent

# The two sides of a pair, as the arranger and the timeline name them. `say: "word"` is the source.
SOURCE = "source"
TARGET = "target"
SAY_ROLES = {"word": SOURCE, "translation": TARGET}

SECTION_KINDS = ("intro", "words", "quiz", "review", "outro")
STEP_KINDS = ("say", "gap", "rest", "cue", "example", "remark", "pronounce", "story_beat",
              "callback")
WHEN = ("always", "writer_decides", "hard_to_say", "natural_link")
ORDERS = ("as_given", "group_by_topic", "writer")
REQUIRES = ("writer", "guide_voice", "multilingual_voice")
PACES = ("slow", "natural", "fast")
BEDS = ("same", "quicker")
CUES = ("your_turn",)
# No grammar kinds: a remark is about hearing, meaning and use, never about how a word is written.
REMARK_KINDS = ("contrast", "register", "false_friend", "mnemonic", "culture", "joke")
CALLBACK_KINDS = ("reuse", "contrast", "chain", "quiz")
# Stretching past this sounds processed; slowness is asked of the voice, never stretched.
MAX_STRETCH = 1.2

_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")
_SWITCH = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

# What this version renders. Everything else in the grammar parses, and is refused at render.
RENDERED_SECTIONS = {"words"}
RENDERED_STEPS = {"say", "gap", "rest"}


class FormatError(ValueError):
    """A format, or the switches sent with it, is not valid. The message names where."""


@dataclass(frozen=True)
class Switch:
    label: str
    default: bool | str
    choices: tuple[str, ...] = ()

    def check(self, name: str, value: Any) -> bool | str:
        if self.choices:
            if not isinstance(value, str) or value not in self.choices:
                raise FormatError(f"switch '{name}' is one of {', '.join(self.choices)}, "
                                  f"not {value!r}")
        elif not isinstance(value, bool):
            raise FormatError(f"switch '{name}' is true or false, not {value!r}")
        return value


@dataclass(frozen=True)
class Step:
    kind: str
    value: Any
    params: Mapping[str, Any] = field(default_factory=dict)
    when: str = "always"
    then: tuple["Step", ...] = ()
    switch: str | None = None


@dataclass(frozen=True)
class Section:
    kind: str
    block: tuple[Step, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)
    after_chunk: tuple[Step, ...] = ()
    choice: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    switch: str | None = None


@dataclass(frozen=True)
class Format:
    id: str
    label: str
    description: str
    sections: tuple[Section, ...]
    order: str = "as_given"
    requires: tuple[str, ...] = ()
    fallback: str | None = None
    switches: Mapping[str, Switch] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """What `/schema` says about this format, for a host to list it by."""
        words = [s for s in self.sections if s.kind == "words"]
        slots_ = [slot for section in words for slot in _section_slots(section)]
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "switches": {name: {"label": s.label, "default": s.default,
                                **({"choices": list(s.choices)} if s.choices else {})}
                         for name, s in self.switches.items()},
            "requires": list(self.requires),
            "bars_per_item": len(slots_),
            "utterances_per_item": sum(1 for kind, _ in slots_ if kind in (SOURCE, TARGET)),
            "has_recall_gap": any(kind == "gap" for kind, _ in slots_),
        }


# -- parsing ---------------------------------------------------------------------------------


def _need(condition: bool, where: str, message: str) -> None:
    if not condition:
        raise FormatError(f"{where}: {message}")


def _keys(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    extra = sorted(set(data) - allowed)
    _need(not extra, where, f"unknown {'key' if len(extra) == 1 else 'keys'} "
                            f"{', '.join(repr(k) for k in extra)}")


def _text(data: Mapping[str, Any], key: str, where: str, *, limit: int = 300) -> str:
    value = data.get(key)
    _need(isinstance(value, str) and value.strip() != "", where, f"'{key}' must be some text")
    _need(len(value) <= limit, where, f"'{key}' is longer than {limit} characters")
    return value


def _one_of(value: Any, allowed: tuple[str, ...], where: str, what: str) -> str:
    _need(isinstance(value, str) and value in allowed, where,
          f"{what} is one of {', '.join(allowed)}, not {value!r}")
    return value


def _count(value: Any, where: str, what: str) -> int:
    _need(isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 8, where,
          f"{what} is a whole number of bars from 1 to 8, not {value!r}")
    return value


def _stretch(value: Any, where: str) -> float:
    _need(isinstance(value, (int, float)) and not isinstance(value, bool)
          and 1.0 <= value <= MAX_STRETCH, where,
          f"'stretch' is from 1.0 to {MAX_STRETCH}, not {value!r}")
    return float(value)


def _kinds(value: Any, allowed: tuple[str, ...], where: str) -> tuple[str, ...]:
    _need(isinstance(value, dict) and set(value) == {"kinds"}, where,
          "takes {\"kinds\": [...]}")
    kinds = value["kinds"]
    _need(isinstance(kinds, list) and kinds, where, "'kinds' is a non-empty list")
    for kind in kinds:
        _one_of(kind, allowed, where, "a kind")
    return tuple(kinds)


_STEP_PARAMS = {
    "say": {"pace", "stretch"},
    "example": {"translate"},
}
_COMMON_STEP_KEYS = {"when", "then", "switch"}


def _step(data: Any, where: str, *, nested: bool = False) -> Step:
    _need(isinstance(data, dict), where, "a step is an object")
    kinds = [k for k in data if k in STEP_KINDS]
    _need(len(kinds) == 1, where, "a step names exactly one of " + ", ".join(STEP_KINDS)
          + (f"; it names {', '.join(kinds)}" if kinds else ""))
    kind = kinds[0]
    _keys(data, {kind, *_STEP_PARAMS.get(kind, set()), *_COMMON_STEP_KEYS}, where)
    raw = data[kind]
    if kind == "say":
        value = _one_of(raw, tuple(SAY_ROLES), where, "'say'")
    elif kind in ("gap", "rest"):
        value = _count(raw, where, f"'{kind}'")
    elif kind == "cue":
        value = _one_of(raw, CUES, where, "'cue'")
    elif kind in ("example", "story_beat"):
        value = _one_of(raw, ("writer",), where, f"'{kind}'")
    elif kind == "pronounce":
        # The whole word, slowly: splitting into syllables was wrong on half the words it was
        # tried on, on every voice, with or without a written pronunciation.
        value = _one_of(raw, ("slow_whole",), where, "'pronounce'")
    elif kind == "remark":
        value = _kinds(raw, REMARK_KINDS, where)
    else:  # callback
        value = _kinds(raw, CALLBACK_KINDS, where)

    params: dict[str, Any] = {}
    if "pace" in data:
        params["pace"] = _one_of(data["pace"], PACES, where, "'pace'")
    if "stretch" in data:
        params["stretch"] = _stretch(data["stretch"], where)
    if "translate" in data:
        _need(isinstance(data["translate"], bool), where, "'translate' is true or false")
        params["translate"] = data["translate"]

    when = _one_of(data.get("when", "always"), WHEN, where, "'when'")
    then: tuple[Step, ...] = ()
    if "then" in data:
        _need(not nested, where, "a 'then' step cannot have its own 'then'")
        _need(when != "always", where, "'then' follows a step that may not happen, so it "
                                       "needs a 'when' other than 'always'")
        steps = data["then"]
        _need(isinstance(steps, list) and steps, where, "'then' is a non-empty list of steps")
        then = tuple(_step(s, f"{where}.then[{i}]", nested=True) for i, s in enumerate(steps))
    if nested:
        _need("when" not in data and "switch" not in data, where,
              "a 'then' step takes neither 'when' nor 'switch'")
    switch = data.get("switch")
    if switch is not None:
        _need(isinstance(switch, str), where, "'switch' names a switch")
    return Step(kind, value, params, when, then, switch)


def _block(data: Any, where: str, allowed: tuple[str, ...]) -> tuple[Step, ...]:
    _need(isinstance(data, list) and data, where, "a non-empty list of steps")
    steps = tuple(_step(s, f"{where}[{i}]") for i, s in enumerate(data))
    for i, step in enumerate(steps):
        _need(step.kind in allowed, f"{where}[{i}]",
              f"'{step.kind}' is not a step this section takes ({', '.join(allowed)})")
    return steps


_SECTION_KEYS = {
    "intro": {"text"},
    "outro": {"text"},
    "words": {"block", "group_headers", "chunk", "after_chunk"},
    "quiz": {"block", "at"},
    "review": {"block", "stretch", "bed", "choice"},
}
_SECTION_STEPS = {
    "words": STEP_KINDS,
    "quiz": ("say", "gap", "rest", "cue"),
    "review": ("say", "gap", "rest"),
}
_AFTER_CHUNK_STEPS = ("story_beat", "callback", "remark", "rest")


def _section_params(kind: str, data: Mapping[str, Any], where: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if "text" in data:
        params["text"] = _one_of(data["text"], ("template", "writer"), where, "'text'")
    if "group_headers" in data:
        _need(isinstance(data["group_headers"], bool), where, "'group_headers' is true or false")
        params["group_headers"] = data["group_headers"]
    if "chunk" in data:
        params["chunk"] = _count(data["chunk"], where, "'chunk'")
    if "at" in data:
        params["at"] = _one_of(data["at"], ("middle", "end"), where, "'at'")
    if "stretch" in data:
        params["stretch"] = _stretch(data["stretch"], where)
    if "bed" in data:
        params["bed"] = _one_of(data["bed"], BEDS, where, "'bed'")
    return params


def _section(data: Any, where: str) -> Section:
    _need(isinstance(data, dict), where, "a section is an object")
    kind = _one_of(data.get("kind"), SECTION_KINDS, where, "'kind'")
    _keys(data, {"kind", "switch", *_SECTION_KEYS[kind]}, where)
    block: tuple[Step, ...] = ()
    if kind in _SECTION_STEPS:
        _need("block" in data, where, f"a '{kind}' section needs a 'block'")
        block = _block(data["block"], f"{where}.block", _SECTION_STEPS[kind])
    after: tuple[Step, ...] = ()
    if "after_chunk" in data:
        _need("chunk" in data, where, "'after_chunk' runs after each 'chunk', so it needs one")
        after = _block(data["after_chunk"], f"{where}.after_chunk", _AFTER_CHUNK_STEPS)
    choice: dict[str, dict[str, Any]] = {}
    if "choice" in data:
        raw = data["choice"]
        _need(isinstance(raw, dict) and raw, where, "'choice' maps a switch's choices to "
                                                    "overrides")
        for name, overrides in raw.items():
            at = f"{where}.choice.{name}"
            _need(isinstance(overrides, dict) and overrides, at, "an object of overrides")
            _keys(overrides, _SECTION_KEYS[kind] - {"block", "choice"}, at)
            choice[name] = _section_params(kind, overrides, at)
    switch = data.get("switch")
    if switch is not None:
        _need(isinstance(switch, str), where, "'switch' names a switch")
    return Section(kind, block, _section_params(kind, data, where), after, choice, switch)


def _switch(name: str, data: Any, where: str) -> Switch:
    _need(bool(_SWITCH.match(name)), where, f"a switch name is lower-case letters, digits and "
                                           f"underscores, not {name!r}")
    _need(isinstance(data, dict), where, "a switch is an object")
    _keys(data, {"label", "default", "choices"}, where)
    label = _text(data, "label", where, limit=60)
    choices: tuple[str, ...] = ()
    if "choices" in data:
        raw = data["choices"]
        _need(isinstance(raw, list) and len(raw) >= 2 and all(isinstance(c, str) for c in raw)
              and len(set(raw)) == len(raw), where, "'choices' is two or more distinct names")
        choices = tuple(raw)
    _need("default" in data, where, "a switch needs a 'default'")
    switch = Switch(label, data["default"], choices)
    switch.check(name, data["default"])
    return switch


def parse(data: Any) -> Format:
    """A format from its JSON, with every part of the grammar checked. Raises `FormatError`."""
    _need(isinstance(data, dict), "format", "a format is an object")
    _keys(data, {"id", "label", "description", "order", "requires", "fallback", "switches",
                 "sections"}, "format")
    format_id = data.get("id")
    _need(isinstance(format_id, str) and bool(_ID.match(format_id)), "format",
          f"'id' is lower-case letters, digits and hyphens, not {format_id!r}")
    label = _text(data, "label", "format", limit=60)
    description = _text(data, "description", "format")
    order = _one_of(data.get("order", "as_given"), ORDERS, "format", "'order'")
    requires = data.get("requires", [])
    _need(isinstance(requires, list), "format", "'requires' is a list")
    for need in requires:
        _one_of(need, REQUIRES, "format.requires", "a requirement")
    fallback = data.get("fallback")
    if fallback is not None:
        _need(isinstance(fallback, str) and bool(_ID.match(fallback)) and fallback != format_id,
              "format", "'fallback' names another format")
    raw_switches = data.get("switches", {})
    _need(isinstance(raw_switches, dict), "format", "'switches' is an object")
    switches = {name: _switch(name, value, f"switches.{name}")
                for name, value in raw_switches.items()}
    sections_raw = data.get("sections")
    _need(isinstance(sections_raw, list) and sections_raw, "format",
          "'sections' is a non-empty list")
    sections = tuple(_section(s, f"sections[{i}]") for i, s in enumerate(sections_raw))
    _need(any(s.kind == "words" for s in sections), "format", "a format needs a 'words' section")

    def named(where: str, name: str | None, *, choosing: bool = False) -> None:
        if name is None:
            return
        _need(name in switches, where, f"switch '{name}' is not declared in 'switches'")
        _need(not choosing or bool(switches[name].choices), where,
              f"'choice' needs switch '{name}' to have 'choices'")

    for i, section in enumerate(sections):
        where = f"sections[{i}]"
        named(where, section.switch)
        if section.choice:
            _need(section.switch is not None, where, "'choice' is keyed by the section's switch")
            named(where, section.switch, choosing=True)
            unknown = sorted(set(section.choice) - set(switches[section.switch].choices))
            _need(not unknown, where, f"'choice' names {', '.join(unknown)}, which switch "
                                      f"'{section.switch}' does not offer")
        for j, step in enumerate((*section.block, *section.after_chunk)):
            named(f"{where}.block[{j}]", step.switch)
    return Format(format_id, label, description, sections, order, tuple(requires), fallback,
                  switches)


# -- the built-in formats ----------------------------------------------------------------------


def builtin() -> dict[str, Format]:
    """Every format this version ships, by id. A file whose name is not its id is an error."""
    found: dict[str, Format] = {}
    for path in sorted(HERE.glob("*.json")):
        try:
            fmt = parse(json.loads(path.read_text(encoding="utf-8")))
        except FormatError as exc:
            raise FormatError(f"{path.name}: {exc}") from exc
        if fmt.id != path.stem:
            raise FormatError(f"{path.name}: its id is '{fmt.id}', not '{path.stem}'")
        found[fmt.id] = fmt
    return found


def load(format: str | Mapping[str, Any]) -> Format:
    """A built-in format by id, or an inline one."""
    if isinstance(format, str):
        formats = builtin()
        if format not in formats:
            raise FormatError(f"unknown format '{format}'. Try: {', '.join(sorted(formats))}")
        return formats[format]
    return parse(dict(format))


# -- switches ----------------------------------------------------------------------------------


def _on(value: bool | str) -> bool:
    return value is not False and value != "off"


def resolve(fmt: Format, switches: Mapping[str, Any] | None = None) -> Format:
    """The format as this render will run it: switches checked, what they turn off removed, and
    the chosen overrides applied. The result carries no switches."""
    switches = dict(switches or {})
    unknown = sorted(set(switches) - set(fmt.switches))
    if unknown:
        offered = ", ".join(sorted(fmt.switches)) or "none"
        raise FormatError(f"format '{fmt.id}' has no switch {', '.join(repr(u) for u in unknown)}"
                          f" (it offers: {offered})")
    values = {name: s.check(name, switches.get(name, s.default))
              for name, s in fmt.switches.items()}

    def keep(name: str | None) -> bool:
        return name is None or _on(values[name])

    def steps(block: tuple[Step, ...]) -> tuple[Step, ...]:
        return tuple(replace(step, switch=None) for step in block if keep(step.switch))

    sections = []
    for section in fmt.sections:
        if not keep(section.switch):
            continue
        params = dict(section.params)
        if section.switch is not None:
            params.update(section.choice.get(str(values[section.switch]), {}))
        sections.append(replace(section, block=steps(section.block),
                                after_chunk=steps(section.after_chunk), params=params,
                                choice={}, switch=None))
    for section in sections:
        if section.kind in _SECTION_STEPS and not section.block:
            raise FormatError(f"with these switches, format '{fmt.id}' has a '{section.kind}' "
                              "section with nothing left in it")
    if not any(s.kind == "words" for s in sections):
        raise FormatError(f"with these switches, format '{fmt.id}' has no 'words' section left")
    return replace(fmt, sections=tuple(sections), switches={})


# -- what this version renders -------------------------------------------------------------------


def unsupported(fmt: Format) -> list[str]:
    """What in a resolved format this version cannot render, each named as a reader would say it.

    Empty means it renders. The limits are this version's, not the grammar's: a format that
    parses is a valid format, and a later version renders more of it.
    """
    missing: list[str] = []

    def add(what: str) -> None:
        if what not in missing:
            missing.append(what)

    if fmt.order != "as_given":
        add(f"order '{fmt.order}'")
    for need in fmt.requires:
        add(f"requires '{need}'")
    words = [s for s in fmt.sections if s.kind == "words"]
    for section in fmt.sections:
        if section.kind not in RENDERED_SECTIONS:
            # Named once: what is inside a section this version cannot render does not matter yet.
            add(f"the '{section.kind}' section")
            continue
        for key in section.params:
            add(f"'{key}' on the '{section.kind}' section")
        for step in section.after_chunk:
            add(step.kind)
        for step in section.block:
            if step.kind not in RENDERED_STEPS:
                add(step.kind)
            for key in step.params:
                add(f"'{key}' on '{step.kind}'")
            if step.when != "always":
                add(f"when '{step.when}'")
    # Each timeline row is one word with both of its reveals, so this version renders one words
    # section, and it must say both sides. The cue timeline lifts both limits.
    if len(words) > 1:
        add("more than one 'words' section")
    for section in words:
        said = {step.value for step in section.block if step.kind == "say"}
        if said != set(SAY_ROLES):
            add("a 'words' block that does not say both the word and its translation")
    return missing


def renderable(format: str | Mapping[str, Any],
               switches: Mapping[str, Any] | None = None) -> Format:
    """Load, resolve and check a format for this version, or raise naming what it cannot do."""
    fmt = resolve(load(format), switches)
    missing = unsupported(fmt)
    if missing:
        raise FormatError(f"format '{fmt.id}' uses {', '.join(missing)}, which this version "
                          "cannot render yet")
    return fmt


# -- expansion ---------------------------------------------------------------------------------


def _section_slots(section: Section) -> list[tuple[str, int]]:
    slots: list[tuple[str, int]] = []
    said = {SOURCE: 0, TARGET: 0}
    for step in section.block:
        if step.kind == "say":
            role = SAY_ROLES[step.value]
            slots.append((role, said[role]))
            said[role] += 1
        elif step.kind in ("gap", "rest"):
            slots.extend((step.kind, 0) for _ in range(step.value))
    return slots


def slots(fmt: Format) -> list[tuple[str, int]]:
    """One word's bars in a renderable format: `(source | target | gap | rest, repetition)`.

    The repetition counts that side's earlier lines in the block, which is the take index a voice
    varies its delivery by.
    """
    return [slot for section in fmt.sections if section.kind == "words"
            for slot in _section_slots(section)]


def spoken_slots(fmt: Format) -> list[tuple[str, int]]:
    return [(kind, rep) for kind, rep in slots(fmt) if kind in (SOURCE, TARGET)]
