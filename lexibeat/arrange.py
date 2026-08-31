"""Lay utterances onto the beat grid.

Each vocabulary item gets a fixed block of bars. Within the block, every
utterance starts exactly on a downbeat; the rest of its bar is silence. This
preserves the downbeat-aligned structure established by the prototype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .emotion import for_item
from .music import Grid
from .vocab import Item
from .voice import Prosody, Speaker

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
    # Straight alternation without a retrieval gap.
    "alternating": [
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
    emotions: bool = True,
    progress: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[Event], int]:
    """Return the scheduled utterances and the total number of bars needed."""
    slots = PATTERNS[pattern]
    events: list[Event] = []
    bar = intro_bars
    completed = 0
    spoken_slots = sum(kind in ("es", "en") for kind, _ in slots)
    total_utterances = len(items) * spoken_slots

    for n, item in enumerate(items, 1):
        emotion = for_item(item.source, item.emoji, enabled=emotions)
        if progress:
            print(f"  [{n}/{len(items)}] {item.emoji or ' '} {item.source} — "
                  f"{item.target}  ({emotion.name})", flush=True)
        item_events: dict[str, list[tuple[Event, str, Prosody]]] = {
            "es": [], "en": [],
        }
        for kind, rep in slots:
            if kind in ("gap", "rest"):
                bar += 1
                continue
            text = item.source if kind == "es" else item.target
            if getattr(getattr(speaker, "backend", None), "name", None) == "chatterbox":
                prosody = Prosody.for_chatterbox_repeat(
                    rep, speaker.prosody_strength)
            else:
                prosody = Prosody.for_repeat(rep, speaker.prosody_strength)
            prosody = prosody.with_emotion(emotion, speaker.prosody_strength)
            if progress_callback:
                language = "Spanish" if kind == "es" else "English"
                progress_callback(
                    completed, total_utterances,
                    f"Synthesizing {completed + 1} of {total_utterances}: "
                    f"{language} — {text}")
            # Leave a little of the bar clear so the next downbeat stays audible.
            audio = speaker.say(text, kind, prosody, emotion,
                                target_seconds=grid.bar * 0.92)
            event = Event(grid.bar_start(bar), audio, f"{kind}:{text}")
            events.append(event)
            item_events[kind].append((event, text, prosody))
            completed += 1
            bar += 1

        if getattr(getattr(speaker, "backend", None), "name", None) == "chatterbox":
            for kind, repetitions in item_events.items():
                longest = _long_duration_outlier(
                    [len(event.audio) for event, _, _ in repetitions], grid.sr)
                if longest is None:
                    continue
                event, text, prosody = repetitions[longest]
                peers = [len(row[0].audio) for index, row in enumerate(repetitions)
                         if index != longest]
                peer_median = float(np.median(peers))
                if progress_callback:
                    language = "Spanish" if kind == "es" else "English"
                    progress_callback(
                        completed, total_utterances,
                        f"Retrying an unusually long {language} repetition — {text}")
                replacement = speaker.say(
                    text, kind, prosody, emotion,
                    target_seconds=grid.bar * 0.92, retry=True)
                if abs(len(replacement) - peer_median) < \
                        abs(len(event.audio) - peer_median):
                    event.audio = replacement
                else:
                    remember = getattr(speaker, "remember_take", None)
                    if remember:
                        remember(text, kind, prosody, emotion,
                                 grid.bar * 0.92, event.audio)

    return events, bar + outro_bars


def _long_duration_outlier(lengths: list[int], sample_rate: int) -> int | None:
    """Return a clearly long take among three repetitions, if one exists."""
    if len(lengths) != 3:
        return None
    longest = int(np.argmax(lengths))
    peers = [length for index, length in enumerate(lengths) if index != longest]
    peer_median = float(np.median(peers))
    threshold = max(peer_median * 1.6, peer_median + 0.45 * sample_rate)
    return longest if lengths[longest] > threshold else None


def render_speech(events: list[Event], total_bars: int, grid: Grid) -> np.ndarray:
    """Flatten scheduled events into one continuous mono track."""
    track = np.zeros(grid.samples(total_bars * grid.bar) + grid.sr, dtype=np.float32)
    for ev in events:
        at = grid.samples(ev.start)
        end = min(len(track), at + len(ev.audio))
        track[at:end] += ev.audio[: end - at]
    return track
