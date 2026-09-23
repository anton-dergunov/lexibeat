"""Render one loop: words in, a finished track and its timeline out.

A *loop* is what this library makes — a handful of words, each spoken over a bar grid with a
silence in the middle to recall the answer in, over a bed that replays byte-identically from a
style and a seed. It is deliberately not called a lesson: "lesson" will be overloaded the day a
second kind of lesson exists, and a loop is a thing that repeats.

This replaces the two-phase `lesson.py`, whose split existed only to fit a GPU reservation on a
hosted Space and whose cache was keyed on a hash of the whole vocabulary — so a loop sharing eleven
of twelve words with another one paid for all twelve again. There is one phase here and no cache:
what is worth caching is a *take*, and a take is the host's to cache because the host is the one
paying a provider for it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import soundfile as sf

from .api import MusicRequest, resolve_music
from .arrange import (
    PATTERNS,
    SOURCE,
    TARGET,
    Cancelled,
    Event,
    arrange,
    render_speech,
    spoken_slots,
)
from .language import Language
from .mix import mix_stems
from .music import SR, Grid, render_stems
from .vocab import Item
from .voice import Backend, Speaker

# libsndfile writes MP3 through LAME, and exposes quality as a 0-1 `compression_level` rather than
# a bitrate. Measured against libsndfile 1.2.2: in CONSTANT mode the level maps onto LAME's own
# bitrate ladder, and 0.65 is the rung that is 128 kbps. Anything else is guesswork dressed as a
# constant, so the pair is written down together.
MP3_BITRATE_MODE = "CONSTANT"
MP3_COMPRESSION_LEVEL = 0.65
MP3_BITRATE_KBPS = 128
MP3_MIME = "audio/mpeg"

# A bound, not a product decision: a request is a list of words and a malformed one should not
# render an hour of audio. The six-row cap this replaces *was* a product decision, and a wrong one.
MAX_ITEMS = 200
MAX_TEXT_CHARACTERS = 200
MAX_DIRECTION_CHARACTERS = 200

SPEECH_LUFS = -16.0
MUSIC_LUFS = -26.0


class LoopError(ValueError):
    """The request cannot be rendered, and the message says which part of it."""


@dataclass(frozen=True)
class LoopRequest:
    items: tuple[Item, ...]
    source_language: Language
    target_language: Language
    pattern: str = "retrieval"
    family: str = "auto"
    energy: str = "balanced"
    rhythm: str = "steady"
    palette: str = "hybrid"
    seed: int | None = None
    profile: str = "production-v1"
    prosody_strength: float = 1.0
    voice_seed: int | None = None

    def validated(self) -> "LoopRequest":
        if self.pattern not in PATTERNS:
            raise LoopError(f"Unknown pattern '{self.pattern}'. "
                            f"Try: {', '.join(sorted(PATTERNS))}")
        if not self.items:
            raise LoopError("A loop needs at least one word.")
        if len(self.items) > MAX_ITEMS:
            raise LoopError(f"A loop is limited to {MAX_ITEMS} words.")
        for index, item in enumerate(self.items, 1):
            if not item:
                raise LoopError(f"Word {index} needs both a source and a target.")
            for label, text in (("source", item.source), ("target", item.target)):
                if len(text) > MAX_TEXT_CHARACTERS:
                    raise LoopError(f"Word {index}'s {label} exceeds the "
                                    f"{MAX_TEXT_CHARACTERS}-character limit.")
            if len(item.direction) > MAX_DIRECTION_CHARACTERS:
                raise LoopError(f"Word {index}'s direction exceeds the "
                                f"{MAX_DIRECTION_CHARACTERS}-character limit.")
        # Raises with the music layer's own wording for a bad family, energy, rhythm or palette.
        self.music_request().validated()
        return self

    def music_request(self) -> MusicRequest:
        return MusicRequest(family=self.family, energy=self.energy, rhythm=self.rhythm,
                            palette=self.palette, seed=self.seed, profile=self.profile)


@dataclass(frozen=True)
class LoopResult:
    """Everything the host stores about a rendered loop.

    These are the fields and no others because a host records *what was said* — the item text is
    denormalised into the timeline on purpose, so editing a word afterwards cannot make a player
    caption a recording that no longer matches it.
    """

    audio_path: str
    audio_mime: str
    duration_seconds: float
    pattern: str
    style_id: str
    seed: int
    engine_version: str
    profile_version: str
    bed_fingerprint: str
    total_bars: int
    bpm: float
    timeline: list[dict[str, Any]] = field(default_factory=list)
    bed_spec: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bed_fingerprint(fingerprint: Any) -> str:
    """A short, stable digest of a resolved bed's identity.

    The host stores the style, the seed and the engine version, which replay the bed exactly; this
    is what *proves* a replay produced the same bed, and is why the whole BedSpec does not have to
    be stored as opaque JSON beside them.
    """
    payload = json.dumps(asdict(fingerprint), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def build_timeline(items: Sequence[Item], events: Sequence[Event], grid: Grid,
                   total_bars: int, pattern: str) -> list[dict[str, Any]]:
    """Describe progressive reveals and active utterances from arranged events.

    This lived in `demo.py`, which is a script for making a README video — so the one exposure rich
    enough to drive an interface was in the one place an interface could not reach. The subtitle
    rows the old lesson path produced held a caption until the next utterance and could not say
    when the answer arrives, which is the single thing a retrieval loop's display turns on.
    """
    slots = spoken_slots(pattern)
    expected = len(items) * len(slots)
    if len(events) != expected:
        raise LoopError(f"Expected {expected} speech events, received {len(events)}.")
    timeline: list[dict[str, Any]] = []
    cursor = 0
    for item_index, item in enumerate(items):
        utterances = []
        for kind, repetition in slots:
            event = events[cursor]
            expected_text = item.source if kind == SOURCE else item.target
            if event.label != f"{kind}:{expected_text}":
                raise LoopError("Speech events do not match the requested words.")
            utterances.append({
                "role": kind,
                "repetition": repetition,
                "start": float(event.start),
                "end": float(event.start + len(event.audio) / grid.sr),
            })
            cursor += 1
        source_reveal = next(row["start"] for row in utterances if row["role"] == SOURCE)
        target_reveal = next(row["start"] for row in utterances if row["role"] == TARGET)
        next_start = (float(events[cursor].start) if cursor < len(events)
                      else float(total_bars * grid.bar))
        timeline.append({
            "index": item_index,
            "source": item.source,
            "target": item.target,
            "direction": item.direction,
            "start": source_reveal,
            "source_reveal": source_reveal,
            "target_reveal": target_reveal,
            "end": next_start,
            "utterances": utterances,
        })
    return timeline


def write_mp3(path: Path, track: np.ndarray, sample_rate: int = SR) -> None:
    """Write the finished track once, atomically, at a bitrate this repository measured."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.partial.mp3")
    try:
        sf.write(temporary, track, sample_rate, format="MP3",
                 bitrate_mode=MP3_BITRATE_MODE,
                 compression_level=MP3_COMPRESSION_LEVEL)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render_loop(
    request: LoopRequest,
    *,
    backend: Backend,
    output: Path | str,
    progress: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    speech_lufs: float = SPEECH_LUFS,
    music_lufs: float = MUSIC_LUFS,
) -> LoopResult:
    """Speak every line, render the bed, duck it under the speech, and write the MP3."""
    request = request.validated()
    output = Path(output)

    def report(fraction: float, message: str) -> None:
        if progress:
            progress(fraction, message)

    def gate() -> None:
        if cancel_check and cancel_check():
            raise Cancelled("The render was cancelled.")

    report(0.02, "Resolving the music bed")
    gate()
    resolved = resolve_music(request.music_request())
    spec = resolved.bed_spec
    grid = Grid.from_spec(spec)
    voice_seed = (int(spec.seed % (2 ** 31 - 1)) if request.voice_seed is None
                  else int(request.voice_seed))

    speaker = Speaker(backend_instance=backend,
                      prosody_strength=request.prosody_strength,
                      voice_seed=voice_seed)
    try:
        report(0.05, "Synthesizing speech")
        events, total_bars = arrange(
            list(request.items), speaker, grid,
            source_language=request.source_language,
            target_language=request.target_language,
            pattern=request.pattern, progress=False,
            cancel_check=cancel_check,
            progress_callback=(lambda completed, total, message:
                               report(0.05 + 0.70 * completed / max(total, 1), message)))
        speech = render_speech(events, total_bars, grid)
        if not len(speech) or not np.isfinite(speech).all():
            raise LoopError("The speech renderer produced invalid audio.")
        timeline = build_timeline(list(request.items), events, grid, total_bars,
                                  request.pattern)
    finally:
        speaker.close()

    gate()
    report(0.78, "Rendering the music bed")
    stems = render_stems(
        spec, total_bars,
        progress_callback=lambda value, message: report(0.78 + 0.14 * value, message))

    gate()
    report(0.93, "Mixing")
    depths = {name: getattr(spec, name).duck_db for name in stems}
    track = mix_stems(stems, speech, depths, speech_lufs=speech_lufs,
                      music_lufs=music_lufs)
    if track.ndim != 2 or track.shape[1] != 2 or not len(track) or \
            not np.isfinite(track).all():
        raise LoopError("The loop renderer produced invalid stereo audio.")
    peak = float(np.abs(track).max())
    if peak > 0.97 + 1e-7:
        raise LoopError(f"The loop renderer exceeded the 0.97 peak limit ({peak:.3f}).")

    report(0.97, "Writing the track")
    write_mp3(output, track)
    report(1.0, "Loop ready")
    return LoopResult(
        audio_path=str(output),
        audio_mime=MP3_MIME,
        duration_seconds=len(track) / SR,
        pattern=request.pattern,
        style_id=resolved.fingerprint.family,
        # The seed the request carried (or the one minted for it), never the winning candidate's:
        # that is what the same request replays from, and a candidate's seed is not a request's.
        seed=int(resolved.request.seed),
        engine_version=resolved.engine_version,
        profile_version=resolved.profile_version,
        bed_fingerprint=bed_fingerprint(resolved.fingerprint),
        total_bars=total_bars,
        bpm=float(spec.bpm),
        timeline=timeline,
        bed_spec=asdict(spec),
    )


def estimated_seconds(item_count: int, pattern: str, bpm: float = 80.0,
                      beats_per_bar: int = 4, beat_unit: int = 4,
                      intro_bars: int = 2, outro_bars: int = 2) -> float:
    """How long a loop of this shape will run, before a note of it is synthesised."""
    grid = Grid(bpm=bpm, beats_per_bar=beats_per_bar, beat_unit=beat_unit)
    bars = intro_bars + item_count * len(PATTERNS[pattern]) + outro_bars
    return bars * grid.bar
