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
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import soundfile as sf

from .api import MusicRequest, resolve_music
from .arrange import SOURCE, TARGET, Cancelled, Event, arrange, render_speech
from .formats import Format, FormatError, missing_requirements, needs_writer, renderable
from .programme import estimated_bars, group_lines, phrase_keys, phrases
from .script import Script, ScriptError, needs as script_needs, parse as parse_script
from .script import prompt as script_prompt
from .writer import WriteRequest, Writer
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
    # A built-in format's id, or an inline format (`docs/programme-format.md`), and the values of
    # the switches that format declares.
    format: str | Mapping[str, Any] = "classic"
    switches: Mapping[str, Any] = field(default_factory=dict)
    # A script a previous render of these words returned (`LoopResult.script`). Given one, the
    # render reads it exactly as it would a writer's reply and calls no writer: new music for a
    # radio lesson keeps its lines, and its takes come from the host's cache.
    script: Mapping[str, Any] | None = None
    family: str = "auto"
    energy: str = "balanced"
    rhythm: str = "steady"
    palette: str = "hybrid"
    seed: int | None = None
    profile: str = "production-v1"
    prosody_strength: float = 1.0
    voice_seed: int | None = None

    def resolved_format(self) -> Format:
        """The format as this render runs it: loaded, switches applied, and checked renderable.

        What it *requires* is not checked here, because that depends on the backend and writer a
        render is given (`render_loop`); a request can be refused before it is queued without them.
        """
        try:
            fmt = renderable(self.format, self.switches)
        except FormatError as exc:
            raise LoopError(f"Format: {exc}") from exc
        # A writer can supply missing phrases at render time, so only a format with no writer text
        # can be refused for them here.
        if missing_phrases(fmt, self.target_language) and not needs_writer(fmt):
            raise LoopError(f"Format: format '{fmt.id}' needs learner-language phrases, and there "
                            f"are none for {self.target_language.name} "
                            f"({self.target_language.code}) yet")
        return fmt

    def validated(self) -> "LoopRequest":
        self.resolved_format()
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

    These are the fields and no others because a host records *what was said* — the text is
    denormalised into `items` and `cues` on purpose, so editing a word afterwards cannot make a
    player caption a recording that no longer matches it.

    `format` is the format that was rendered. When the one asked for needed something this render
    lacked and named a fallback, `format` is the fallback and `fallback_from` the one asked for.
    """

    audio_path: str
    audio_mime: str
    duration_seconds: float
    format: str
    style_id: str
    seed: int
    engine_version: str
    profile_version: str
    bed_fingerprint: str
    total_bars: int
    bpm: float
    items: list[dict[str, Any]] = field(default_factory=list)
    cues: list[dict[str, Any]] = field(default_factory=list)
    fallback_from: str | None = None
    # What the writer wrote for this render, as read, or None for a format with no writer. Sent
    # back as `LoopRequest.script`, it renders the same lines again.
    script: dict[str, Any] | None = None
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
                   total_bars: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The two views a player needs of an arranged loop: one row per word, and every line.

    **`items`** is one row per word, from its block in the words section: when the block starts and
    ends, and when each side is first heard — the reveals a retrieval display turns on, since the
    answer must not be on screen before the recall gap has passed. A side a format never says in
    the words section has no reveal. Rows are in the words' own order, which need not be the order
    they are taught in. An item ends where whatever follows its own block begins — the next word, a
    quiz, a piece of the story — or at the end of the loop.

    **`cues`** is every line in the order it is heard: a word may appear in it more than once, and
    a line may belong to no word at all (an intro, a cue). Lines sharing a `group` are shown
    together — a word and its translation, an example and its translation (`group_lines`).
    """
    cues: list[dict[str, Any]] = []
    if any(event.segment is None for event in events):
        raise LoopError("An arranged line has lost the segment it was planned from.")
    groups = group_lines([event.segment for event in events])
    for event, group in zip(events, groups):
        segment = event.segment
        if segment.kind == "say":
            item = items[segment.item]
            if segment.text != (item.source if segment.side == SOURCE else item.target):
                raise LoopError("Speech events do not match the requested words.")
        cues.append({
            "kind": segment.kind,
            "section": segment.section,
            "group": group,
            "item": segment.item,
            "side": segment.side,
            "role": segment.role,
            "language": segment.language.code,
            "text": segment.text,
            "take": segment.take,
            "start": float(event.start),
            "end": float(event.start + len(event.audio) / grid.sr),
        })

    in_words = [cue for cue in cues if cue["section"] == "words" and cue["item"] is not None]
    starts = {}
    for cue in in_words:
        starts.setdefault(cue["item"], cue["start"])
    if len(starts) != len(items):
        raise LoopError(f"Expected lines for {len(items)} words, found {len(starts)}.")
    total = float(total_bars * grid.bar)
    # Words are taught in the order the format (or its writer) chose, not necessarily by index.
    taught = sorted(starts, key=starts.get)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        mine = [cue for cue in in_words if cue["item"] == index]
        start = starts[index]
        position = taught.index(index)
        following = [cue["start"] for cue in cues if cue["start"] > mine[-1]["start"]
                     and not (cue["section"] == "words" and cue["item"] == index)]
        if position + 1 < len(taught):
            following.append(starts[taught[position + 1]])
        end = min(following, default=total)

        def reveal(side: str) -> float | None:
            return next((cue["start"] for cue in mine if cue["side"] == side), None)

        rows.append({
            "index": index,
            "source": item.source,
            "target": item.target,
            "direction": item.direction,
            "start": start,
            "end": end,
            "source_reveal": reveal(SOURCE),
            "target_reveal": reveal(TARGET),
        })
    return rows, cues


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
    writer: Writer | None = None,
    progress: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    speech_lufs: float = SPEECH_LUFS,
    music_lufs: float = MUSIC_LUFS,
) -> LoopResult:
    """Write the lines if the format asks for it, speak them, render the bed, duck it under the
    speech, and write the MP3."""
    request = request.validated()
    fmt, fallback_from = _format_for(request, backend,
                                     has_writer=writer is not None or request.script is not None)
    output = Path(output)

    def report(fraction: float, message: str) -> None:
        if progress:
            progress(fraction, message)

    def gate() -> None:
        if cancel_check and cancel_check():
            raise Cancelled("The render was cancelled.")

    script = None
    if needs_writer(fmt) and request.script is not None:
        script = read_script(fmt, request.items, request.script,
                             target_language=request.target_language)
    elif needs_writer(fmt):
        report(0.01, "Writing the programme")
        script = write_script(fmt, request.items, source_language=request.source_language,
                              target_language=request.target_language, writer=writer)
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
            format=fmt, seed=int(resolved.request.seed), script=script, progress=False,
            cancel_check=cancel_check,
            progress_callback=(lambda completed, total, message:
                               report(0.05 + 0.70 * completed / max(total, 1), message)))
        speech = render_speech(events, total_bars, grid)
        if not len(speech) or not np.isfinite(speech).all():
            raise LoopError("The speech renderer produced invalid audio.")
        items, cues = build_timeline(list(request.items), events, grid, total_bars)
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
        format=fmt.id,
        style_id=resolved.fingerprint.family,
        # The seed the request carried (or the one minted for it), never the winning candidate's:
        # that is what the same request replays from, and a candidate's seed is not a request's.
        seed=int(resolved.request.seed),
        engine_version=resolved.engine_version,
        profile_version=resolved.profile_version,
        bed_fingerprint=bed_fingerprint(resolved.fingerprint),
        total_bars=total_bars,
        bpm=float(spec.bpm),
        items=items,
        cues=cues,
        fallback_from=fallback_from,
        script=dict(script.raw) if script is not None else None,
        bed_spec=asdict(spec),
    )


def read_script(fmt: Format, items: Sequence[Item], script: Mapping[str, Any], *,
                target_language: Language) -> Script:
    """A script a previous render returned, read as a fresh reply would be.

    It is checked against what *this* format needs, so a script from another format, or for other
    words, is refused naming the first fault rather than half used.
    """
    need = script_needs(fmt, missing_phrases=missing_phrases(fmt, target_language))
    try:
        return parse_script(json.dumps(dict(script), ensure_ascii=False), need, len(items))
    except ScriptError as exc:
        raise LoopError(f"The script sent with this render cannot be used: {exc}") from exc


def missing_phrases(fmt: Format, language: Language) -> list[str]:
    """The learner-language phrases a format speaks that no phrase file has for this language."""
    table = phrases(language.code) or {}
    return [key for key in phrase_keys(fmt) if not table.get(key)]


def write_script(fmt: Format, items: Sequence[Item], *, source_language: Language,
                 target_language: Language, writer: Writer) -> Script:
    """One writer call for the whole programme, read by the script parser.

    A reply that cannot be used raises `LoopError` with the parser's own sentence, which says what
    was wrong; retrying is the host's model chain's business, not this one's.
    """
    need = script_needs(fmt, missing_phrases=missing_phrases(fmt, target_language))
    text = writer.write(WriteRequest(script_prompt(
        need, items, source_language=source_language, target_language=target_language)))
    try:
        return parse_script(text, need, len(items))
    except ScriptError as exc:
        raise LoopError(f"The writer's reply could not be used: {exc}") from exc


def _format_for(request: LoopRequest, backend: Backend, *,
                has_writer: bool) -> tuple[Format, str | None]:
    """The format this render runs, and the one asked for when that had to fall back.

    A requirement this render lacks falls back to the format's `fallback`, with its default switches,
    or is refused naming what is missing. Never a partial render.
    """
    fmt = request.resolved_format()
    capabilities = getattr(backend, "capabilities", None)
    missing = missing_requirements(
        fmt, mixes_languages=bool(getattr(capabilities, "mixes_languages", False)),
        has_writer=has_writer)
    if not missing:
        return fmt, None
    if fmt.fallback is None:
        raise LoopError(f"Format: format '{fmt.id}' requires {', '.join(missing)}, which this "
                        "render does not have")
    fallback = replace(request, format=fmt.fallback, switches={})
    return _format_for(fallback, backend, has_writer=has_writer)[0], fmt.id


def estimated_seconds(item_count: int, format: Format, bpm: float = 80.0,
                      beats_per_bar: int = 4, beat_unit: int = 4,
                      intro_bars: int = 2, outro_bars: int = 2) -> float:
    """How long a loop of this shape will run, before a note of it is synthesised."""
    grid = Grid(bpm=bpm, beats_per_bar=beats_per_bar, beat_unit=beat_unit)
    return (intro_bars + estimated_bars(format, item_count) + outro_bars) * grid.bar
