"""Parse vocabulary items out of the Obsidian markdown notes.

Expected shape of an entry:

    ##### **antojar** 🤤
    *to crave*
    > ¿Qué sabor se te **antoja** más? - What flavor are you most **craving**?

The example line is optional. Its two halves are separated by " - ".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

HEADWORD = re.compile(r"^#{3,6}\s+\*\*(?P<word>.+?)\*\*\s*(?P<emoji>.*)$")
TRANSLATION = re.compile(r"^\*(?P<gloss>[^*].*?)\*\s*$")
EXAMPLE = re.compile(r"^>\s*(?P<body>.+)$")


def _emoji_only(text: str) -> str:
    """Keep just the symbol characters trailing a headword."""
    return "".join(c for c in (text or "")
                   if unicodedata.category(c) in {"So", "Sk"})


def _strip_markup(text: str) -> str:
    text = text.replace("**", "").replace("*", "")
    # Emoji and other symbol characters read badly when sent to a TTS engine.
    text = "".join(c for c in text if unicodedata.category(c) not in {"So", "Sk"})
    return " ".join(text.split())


@dataclass
class Item:
    """One thing to teach: a source-language string and its translation."""

    source: str  # Spanish
    target: str  # English
    emoji: str = ""  # the note's own emoji, used to colour the delivery

    def __post_init__(self) -> None:
        self.source = _strip_markup(self.source)
        self.target = _strip_markup(self.target)

    def __bool__(self) -> bool:
        return bool(self.source and self.target)


@dataclass
class Entry:
    word: Item
    example: Item | None = None


def parse_file(path: Path) -> list[Entry]:
    entries: list[Entry] = []
    word: str | None = None
    emoji: str = ""
    gloss: str | None = None
    example: Item | None = None

    def flush() -> None:
        nonlocal word, emoji, gloss, example
        if word and gloss:
            item = Item(word, gloss.split(";")[0], emoji)
            if item:
                entries.append(Entry(item, example))
        word, emoji, gloss, example = None, "", None, None

    for line in path.read_text(encoding="utf-8").splitlines():
        if m := HEADWORD.match(line):
            flush()
            word = m.group("word")
            emoji = _emoji_only(m.group("emoji"))
        elif word and gloss is None and (m := TRANSLATION.match(line)):
            gloss = m.group("gloss")
        elif word and example is None and (m := EXAMPLE.match(line)):
            body = m.group("body")
            # The two halves are joined by a spaced hyphen; be tolerant of dashes.
            parts = re.split(r"\s+[-–—]\s+", body, maxsplit=1)
            if len(parts) == 2:
                candidate = Item(parts[0], parts[1], emoji)
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
