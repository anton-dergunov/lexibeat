"""Lay a loop's segments onto the beat grid.

The planner (`programme.py`) turns a format and its words into segments; this places them. Every
line starts exactly on a downbeat, a fixed line is fitted to its bar, and the rest of the bar is
silence. A take that still overflows fades under the next voice rather than being cut.

A segment names its side (`source`, `target`) and its language, never a particular language pair,
so the same format teaches Mandarin from Portuguese without a line changing here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .dsp import time_stretch
from .formats import SOURCE, TARGET, Format
from .language import Language
from .music import Grid
from .programme import Programme, Segment, plan
from .script import Script
from .vocab import Item
from .voice import Speaker

__all__ = ["SOURCE", "TARGET", "Cancelled", "Event", "arrange", "arrange_programme",
           "render_speech"]


class Cancelled(RuntimeError):
    """The caller asked for the render to stop, between utterances."""


@dataclass
class Event:
    start: float  # seconds
    audio: np.ndarray
    label: str
    segment: Segment | None = None


def arrange(
    items: list[Item],
    speaker: Speaker,
    grid: Grid,
    *,
    source_language: Language,
    target_language: Language,
    format: Format,
    seed: int = 0,
    script: Script | None = None,
    intro_bars: int = 2,
    outro_bars: int = 2,
    progress: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[list[Event], int]:
    """Plan a resolved, renderable format over these words, then place it. Returns the scheduled
    utterances and the total number of bars. `script` is what a writer wrote for this render, for a
    format that takes text from one."""
    programme = plan(format, items, source_language=source_language,
                     target_language=target_language, seed=seed, script=script)
    return arrange_programme(programme, speaker, grid, intro_bars=intro_bars,
                             outro_bars=outro_bars, progress=progress,
                             progress_callback=progress_callback, cancel_check=cancel_check)


def arrange_programme(
    programme: Programme,
    speaker: Speaker,
    grid: Grid,
    *,
    intro_bars: int = 2,
    outro_bars: int = 2,
    progress: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[list[Event], int]:
    """Place every segment of a programme, in order, each on its own downbeat."""
    events: list[Event] = []
    bar = intro_bars
    completed = 0
    total = len(programme.lines)
    open_block: int | None = None
    block_events: list[Event] = []

    def settle() -> None:
        nonlocal completed
        if block_events and not speaker.capabilities.schedulable:
            completed = _retry_long_takes(block_events, speaker, grid, completed, total,
                                          progress_callback)
        block_events.clear()

    for segment in programme.segments:
        if segment.block != open_block:
            settle()
            open_block = segment.block
            if progress and segment.block is not None and segment.item is not None:
                print(f"  [{segment.item + 1}] {segment.text or segment.kind}", flush=True)
        if not segment.spoken:
            bar += segment.bars
            continue
        if cancel_check and cancel_check():
            raise Cancelled("The render was cancelled.")
        if progress_callback:
            progress_callback(completed, total,
                              f"Synthesizing {completed + 1} of {total}: "
                              f"{segment.language.name} — {segment.text}")
        audio = _say(speaker, segment, grid)
        label = f"{segment.side}:{segment.text}" if segment.side else \
            f"{segment.kind}:{segment.text}"
        event = Event(grid.bar_start(bar), audio, label, segment)
        events.append(event)
        if segment.kind == "say":
            block_events.append(event)
        completed += 1
        bar += segment.bars if segment.bars is not None else _bars_for(audio, grid)
    settle()
    return events, bar + outro_bars


# A line of its own length ends at least this long before the next downbeat, so the next line
# does not arrive on the heels of the last word.
LINE_TAIL_SECONDS = 0.35


def _bars_for(audio: np.ndarray, grid: Grid) -> int:
    return max(1, math.ceil((len(audio) / grid.sr + LINE_TAIL_SECONDS) / grid.bar))


def _say(speaker: Speaker, segment: Segment, grid: Grid, *, retry: bool = False) -> np.ndarray:
    delivery = speaker.take(segment.take, segment.direction, pace=segment.pace,
                            quotes=segment.quotes)
    fixed = segment.bars is not None
    # A fixed line aims to leave a little of its bar clear so the next downbeat stays audible. A
    # line of its own length is spoken as it comes and given the bars it needs.
    audio = speaker.say(segment.text, segment.language, delivery,
                        target_seconds=grid.bar * 0.92 * segment.bars if fixed else None,
                        slot_seconds=grid.bar * segment.bars if fixed else None,
                        retry=retry, role=segment.role)
    if segment.stretch > 1.0 + 1e-6 and len(audio) > grid.sr // 10:
        # Faster by stretching, never slower: a slowed recording sounds metallic, and slowness
        # is asked of the voice instead.
        audio = time_stretch(audio, grid.sr, segment.stretch)
    return audio


def _retry_long_takes(events: list[Event], speaker: Speaker, grid: Grid, completed: int,
                      total: int, progress_callback) -> int:
    """Record once more the one repetition of a side that came back far longer than its peers.

    A backend that can be neither told to go faster nor stretched locally is the one that returns
    such a take. That is a capability, not a model name.
    """
    for side in (SOURCE, TARGET):
        repetitions = [event for event in events if event.segment.side == side]
        longest = _long_duration_outlier([len(event.audio) for event in repetitions], grid.sr)
        if longest is None:
            continue
        event = repetitions[longest]
        segment = event.segment
        peer_median = float(np.median([len(row.audio) for index, row in enumerate(repetitions)
                                       if index != longest]))
        if progress_callback:
            progress_callback(completed, total,
                              f"Retrying an unusually long {segment.language.name} repetition — "
                              f"{segment.text}")
        replacement = _say(speaker, segment, grid, retry=True)
        if abs(len(replacement) - peer_median) < abs(len(event.audio) - peer_median):
            event.audio = replacement
        else:
            remember = getattr(speaker, "remember_take", None)
            if remember:
                delivery = speaker.take(segment.take, segment.direction, pace=segment.pace,
                            quotes=segment.quotes)
                remember(segment.text, segment.language, delivery,
                         grid.bar * 0.92 * segment.bars, event.audio,
                         slot_seconds=grid.bar * segment.bars, role=segment.role)
    return completed


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
