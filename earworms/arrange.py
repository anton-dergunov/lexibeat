"""Lay utterances onto the beat grid.

Each vocabulary item gets a fixed block of bars. Within the block, every
utterance starts exactly on a downbeat; the rest of its bar is silence. This
mirrors what the reference Earworms track does (see DESIGN.md §1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .music import Grid
from .vocab import Item
from .voice import Prosody, Speaker, fit

# Each entry is one bar: (language | "gap" | "rest", repetition index).
#   "gap"  - deliberate silence for the learner to recall the translation
#   "rest" - breathing room before the next word
PATTERNS: dict[str, list[tuple[str, int]]] = {
    # Spanish, silence to recall in, then the answer. Retrieval practice.
    "retrieval": [
        ("es", 0), ("gap", 0), ("en", 0),
        ("es", 1), ("en", 1),
        ("es", 2), ("en", 2),
        ("rest", 0),
    ],
    # Straight alternation, as in the original recordings.
    "earworms": [
        ("es", 0), ("en", 0),
        ("es", 1), ("en", 1),
        ("es", 2), ("en", 2),
        ("rest", 0), ("rest", 0),
    ],
}


@dataclass
class Event:
    start: float  # seconds
    audio: np.ndarray
    label: str


def arrange(
    items: list[Item],
    speaker: Speaker,
    grid: Grid,
    *,
    pattern: str = "retrieval",
    intro_bars: int = 2,
    outro_bars: int = 2,
    progress: bool = True,
) -> tuple[list[Event], int]:
    """Return the scheduled utterances and the total number of bars needed."""
    slots = PATTERNS[pattern]
    events: list[Event] = []
    bar = intro_bars

    for n, item in enumerate(items, 1):
        if progress:
            print(f"  [{n}/{len(items)}] {item.source} — {item.target}", flush=True)
        for kind, rep in slots:
            if kind in ("gap", "rest"):
                bar += 1
                continue
            text = item.source if kind == "es" else item.target
            audio = speaker.say(text, kind, Prosody.for_repeat(rep))
            # Leave a little of the bar clear so the next downbeat stays audible.
            audio = fit(audio, grid.bar * 0.92)
            events.append(Event(grid.bar_start(bar), audio, f"{kind}:{text}"))
            bar += 1

    return events, bar + outro_bars


def render_speech(events: list[Event], total_bars: int, grid: Grid) -> np.ndarray:
    """Flatten scheduled events into one continuous mono track."""
    track = np.zeros(grid.samples(total_bars * grid.bar) + grid.sr, dtype=np.float32)
    for ev in events:
        at = grid.samples(ev.start)
        end = min(len(track), at + len(ev.audio))
        track[at:end] += ev.audio[: end - at]
    return track
