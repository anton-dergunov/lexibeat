"""Parse vocabulary items out of Obsidian-style markdown notes.

Expected shape of an entry:

    ##### **antojar**
    *to crave*
    > ¿Qué sabor se te **antoja** más? - What flavor are you most **craving**?

The example line is optional. Its two halves are separated by " - ".

This reader is the local command's convenience, not the interface LexiBeat is integrated through.
A host supplies :class:`Item` values directly, with a direction it has chosen by reading the word —
which is why the emoji column that used to colour delivery here is gone. One mechanism decides how
a line is said, and it is the caller's.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# Anything trailing the headword - an emoji a note happens to carry - is ignored rather than
# read: nothing downstream has an opinion about it any more.
HEADWORD = re.compile(r"^#{3,6}\s+\*\*(?P<word>.+?)\*\*.*$")
TRANSLATION = re.compile(r"^\*(?P<gloss>[^*].*?)\*\s*$")
EXAMPLE = re.compile(r"^>\s*(?P<body>.+)$")


def _strip_markup(text: str) -> str:
    text = text.replace("**", "").replace("*", "")
    # Emoji and other symbol characters read badly when sent to a TTS engine.
    text = "".join(c for c in text if unicodedata.category(c) not in {"So", "Sk"})
    return " ".join(text.split())


@dataclass(frozen=True)
class Item:
    """One thing to teach: two strings, and how the caller wants them said.

    ``direction`` is free text — *"repulsed, recoiling slightly"* — and empty is a legitimate
    answer meaning "read it plainly".
    """

    source: str
    target: str
    direction: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _strip_markup(self.source))
        object.__setattr__(self, "target", _strip_markup(self.target))
        object.__setattr__(self, "direction", " ".join(str(self.direction or "").split()))

    def __bool__(self) -> bool:
        return bool(self.source and self.target)

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "target": self.target,
                "direction": self.direction}


@dataclass
class Entry:
    word: Item
    example: Item | None = None


def parse_file(path: Path) -> list[Entry]:
    entries: list[Entry] = []
    word: str | None = None
    gloss: str | None = None
    example: Item | None = None

    def flush() -> None:
        nonlocal word, gloss, example
        if word and gloss:
            item = Item(word, gloss.split(";")[0])
            if item:
                entries.append(Entry(item, example))
        word, gloss, example = None, None, None

    for line in path.read_text(encoding="utf-8").splitlines():
        if m := HEADWORD.match(line):
            flush()
            word = m.group("word")
        elif word and gloss is None and (m := TRANSLATION.match(line)):
            gloss = m.group("gloss")
        elif word and example is None and (m := EXAMPLE.match(line)):
            body = m.group("body")
            # The two halves are joined by a spaced hyphen; be tolerant of dashes.
            parts = re.split(r"\s+[-–—]\s+", body, maxsplit=1)
            if len(parts) == 2:
                candidate = Item(parts[0], parts[1])
                if candidate:
                    example = candidate
    flush()
    return entries


def load(
    paths: list[Path],
    *,
    mode: str = "mixed",
    limit: int | None = None,
    seed: int | None = None,
) -> list[Item]:
    """Collect items from the given files or directories.

    mode: "words" uses headwords only, "phrases" prefers the example sentence,
    "mixed" teaches the headword and follows it with its example where one exists.
    """
    files: list[Path] = []
    for p in paths:
        files.extend(sorted(p.glob("*.md")) if p.is_dir() else [p])

    entries = [e for f in files for e in parse_file(f)]

    items: list[Item] = []
    for entry in entries:
        if mode == "words":
            items.append(entry.word)
        elif mode == "phrases":
            items.append(entry.example or entry.word)
        else:
            items.append(entry.word)
            if entry.example:
                items.append(entry.example)

    if seed is not None:
        import random

        random.Random(seed).shuffle(items)
    return items[:limit] if limit else items
