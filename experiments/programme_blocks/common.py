"""Paths, the example vocabulary, prompt templates and reports: what every stage shares."""

from __future__ import annotations

import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"
PROMPTS = HERE / "prompts"

T = TypeVar("T")
R = TypeVar("R")

# Language names as a director note or a prompt spells them.
NAMES = {"es": "Spanish", "en": "English", "ru": "Russian", "fr": "French", "de": "German",
         "it": "Italian", "zh": "Mandarin Chinese"}


def words() -> list[dict]:
    """The example vocabulary: a sample of the owner's own words, tracked on purpose."""
    return json.loads((HERE / "words.json").read_text(encoding="utf-8"))


def word(headword: str, language: str | None = None) -> dict:
    for row in words():
        if row["headword"] == headword and (language is None or row["language"] == language):
            return row
    raise KeyError(headword)


def sample(language: str, count: int, seed: int) -> list[dict]:
    pool = [row for row in words() if row["language"] == language]
    return random.Random(seed).sample(pool, count)


# Three twelve-word lessons, shared by the context, story, callback and framing stages so that
# their outputs can be read against each other.
LESSONS = {
    "lesson-es-1": ("es", 101),
    "lesson-es-2": ("es", 202),
    "lesson-en-1": ("en", 303),
}


def lesson(name: str) -> list[dict]:
    language, seed = LESSONS[name]
    return sample(language, 12, seed)


def learner_language(language: str) -> str:
    """Spanish is glossed in English and English in Russian, as in the owner's vocabulary."""
    return {"es": "en", "en": "ru"}[language]


def brief(row: dict, *, notes: bool = False) -> dict:
    """What a prompt is told about one word."""
    keep = ["headword", "primaryGloss", "shortGloss", "pos", "register", "topics", "emotion"]
    if notes:
        keep += ["definition", "notes"]
    return {key: row[key] for key in keep if row.get(key)}


def prompt(name: str, **values: Any) -> str:
    """A tracked prompt template with `{{name}}` blanks. Values that are not strings go in as JSON."""
    text = (PROMPTS / f"{name}.md").read_text(encoding="utf-8")
    for key, value in values.items():
        rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False,
                                                                   indent=1)
        text = text.replace("{{" + key + "}}", rendered)
    missing = re.findall(r"\{\{(\w+)\}\}", text)
    if missing:
        raise KeyError(f"Prompt {name} has unfilled blanks: {missing}")
    return text


def write_report(section: str, report: dict) -> Path:
    path = OUT / section / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def read_report(section: str) -> dict | None:
    path = OUT / section / "report.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def parallel(function: Callable[[T], R], items: Iterable[T], workers: int = 6) -> list[R]:
    """Map over items with a few threads; the providers answer in seconds, not milliseconds."""
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        return list(pool.map(function, items))


def relative(path: Path) -> str:
    """A path the page can link to: relative to `out/`."""
    return os.path.relpath(path, OUT)
