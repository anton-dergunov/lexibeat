"""Lay utterances onto the beat grid.

Each vocabulary item gets the block of bars its format's `words` section describes. Within the
block, every utterance starts exactly on a downbeat; the rest of its bar is silence.

A format's slots are `source` and `target`, never languages: which language each one is comes from
the request, so the same format teaches Mandarin from Portuguese without a line changing here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .formats import SOURCE, TARGET, Format, slots as format_slots, spoken_slots
from .language import Language
from .music import Grid
from .vocab import Item
from .voice import Delivery, Speaker


class Cancelled(RuntimeError):
    """The caller asked for the render to stop, between utterances."""


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
    source_language: Language,
    target_language: Language,
    format: Format,
    intro_bars: int = 2,
    outro_bars: int = 2,
    progress: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[list[Event], int]:
    """Return the scheduled utterances and the total number of bars needed.

    `format` is resolved and renderable (`formats.renderable`): each bar of its `words` block is a
    line on its downbeat, a recall gap, or a rest.
    """
    slots = format_slots(format)
    languages = {SOURCE: source_language, TARGET: target_language}
    events: list[Event] = []
    bar = intro_bars
    completed = 0
    total_utterances = len(items) * len(spoken_slots(format))

    for n, item in enumerate(items, 1):
        if progress:
            print(f"  [{n}/{len(items)}] {item.source} — {item.target}"
                  f"{'  (' + item.direction + ')' if item.direction else ''}", flush=True)
        item_events: dict[str, list[tuple[Event, str, Delivery]]] = {
            SOURCE: [], TARGET: [],
        }
        for kind, rep in slots:
            if kind in ("gap", "rest"):
                bar += 1
                continue
            if cancel_check and cancel_check():
                raise Cancelled("The render was cancelled.")
            text = item.source if kind == SOURCE else item.target
            language = languages[kind]
            delivery = speaker.take(rep, item.direction)
            if progress_callback:
                progress_callback(
                    completed, total_utterances,
                    f"Synthesizing {completed + 1} of {total_utterances}: "
                    f"{language.name} — {text}")
            # Aim to leave a little of the bar clear so the next downbeat stays audible; a take that
            # still runs past the bar fades under the next voice rather than being cut off.
            audio = speaker.say(text, language, delivery,
                                target_seconds=grid.bar * 0.92, slot_seconds=grid.bar)
            event = Event(grid.bar_start(bar), audio, f"{kind}:{text}")
            events.append(event)
            item_events[kind].append((event, text, delivery))
            completed += 1
            bar += 1

        # A backend that cannot be told to go faster, and cannot be stretched locally, is the one
        # that returns a take far longer than its peers. That is a capability, not a model name.
        if not speaker.capabilities.schedulable:
            for kind, repetitions in item_events.items():
                longest = _long_duration_outlier(
                    [len(event.audio) for event, _, _ in repetitions], grid.sr)
                if longest is None:
                    continue
                event, text, delivery = repetitions[longest]
                peers = [len(row[0].audio) for index, row in enumerate(repetitions)
                         if index != longest]
                peer_median = float(np.median(peers))
                if progress_callback:
                    progress_callback(
                        completed, total_utterances,
                        f"Retrying an unusually long {languages[kind].name} repetition — {text}")
                replacement = speaker.say(
                    text, languages[kind], delivery,
                    target_seconds=grid.bar * 0.92, slot_seconds=grid.bar, retry=True)
                if abs(len(replacement) - peer_median) < \
                        abs(len(event.audio) - peer_median):
                    event.audio = replacement
                else:
                    remember = getattr(speaker, "remember_take", None)
                    if remember:
                        remember(text, languages[kind], delivery,
                                 grid.bar * 0.92, event.audio)

    return events, bar + outro_bars


def _long_duration_outlier(lengths: list[int], sample_rate: int) -> int | None:
    """Return a clearly long take among the repetitions, if one exists.

    This required exactly three lengths, which is the whole of what stood between three repetitions
    and four. Any count from three up now works; two cannot have an outlier, because with one peer
    there is nothing to be an outlier from.
    """
    if len(lengths) < 3:
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
