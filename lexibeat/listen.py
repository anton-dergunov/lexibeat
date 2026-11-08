"""Render a bed to be heard on its own: no voice, made for choosing music by ear.

A host plays this when a listener wants to hear a style before choosing it, or to hear again the
music they kept from a loop. It is the same bed a loop with this family and seed is made over, and
`bed_fingerprint` proves it, but two things differ because nothing is spoken over it:

- **It starts at full level.** A bed under a voice fades in over two bars, which is right when the
  first word lands on the downbeat after it and wrong when the music *is* the thing being heard: at
  slow tempi the first seven seconds of a twenty-second sample would be a fade. So the bed is
  rendered with two bars more than asked, and those two bars are dropped. Its fade out is kept,
  because a sample that ends by fading away sounds finished.
- **It is loudness-normalised alone** to one level, so styles are compared by ear rather than by
  how loud each happens to come out, then limited under the same 0.97 peak a loop is.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .api import MusicRequest, resolve_music
from .arrange import Cancelled
from .generator import GenerationCancelledError
from .loop import MP3_MIME, LoopError, bed_fingerprint, write_mp3
from .mix import limit, loudness_normalize
from .music import SR, Grid, render_stems

# A bound, not a product decision: what a host asks for is a sample or a minute of kept music, and
# a malformed request should not render an hour.
MAX_MUSIC_SECONDS = 300.0
LEAD_IN_BARS = 2
# Below a loop's -16 LUFS on purpose: a bed alone has more peaks per unit of loudness than a voice,
# and this leaves the limiter little to do.
MUSIC_ALONE_LUFS = -18.0
# Long enough that starting at full level does not click; short enough that it is not a fade.
ONSET_SECONDS = 0.04


@dataclass(frozen=True)
class MusicOnlyRequest:
    """What the bed of a loop is chosen by, and how long to play it for."""

    family: str
    seed: int
    seconds: float
    energy: str = "balanced"
    rhythm: str = "steady"
    palette: str = "hybrid"
    profile: str = "production-v1"

    def music_request(self) -> MusicRequest:
        return MusicRequest(family=self.family, energy=self.energy, rhythm=self.rhythm,
                            palette=self.palette, seed=self.seed, profile=self.profile)

    def validated(self) -> "MusicOnlyRequest":
        if self.family == "auto":
            raise LoopError("Music on its own needs a family: 'auto' is the absence of one.")
        if not np.isfinite(self.seconds) or not 1.0 <= self.seconds <= MAX_MUSIC_SECONDS:
            raise LoopError(f"seconds must be between 1 and {MAX_MUSIC_SECONDS:g}.")
        # Raises with the music layer's own wording for a bad family, energy, rhythm or palette.
        self.music_request().validated()
        return self


@dataclass(frozen=True)
class MusicResult:
    audio_path: str
    audio_mime: str
    duration_seconds: float
    style_id: str
    seed: int
    engine_version: str
    profile_version: str
    bed_fingerprint: str
    total_bars: int
    bpm: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def render_music_alone(
    request: MusicOnlyRequest,
    *,
    output: Path | str,
    progress: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> MusicResult:
    request = request.validated()
    output = Path(output)

    def report(fraction: float, message: str) -> None:
        if progress:
            progress(fraction, message)

    try:
        report(0.02, "Resolving the music bed")
        resolved = resolve_music(request.music_request(), cancel_check=cancel_check)
        spec = resolved.bed_spec
        grid = Grid.from_spec(spec)
        bars = max(1, round(request.seconds / grid.bar))
        report(0.3, "Rendering the music bed")
        stems = render_stems(
            spec, bars + LEAD_IN_BARS, cancel_check=cancel_check,
            progress_callback=lambda value, message: report(0.3 + 0.6 * value, message))
    except (InterruptedError, GenerationCancelledError) as exc:
        raise Cancelled("The render was cancelled.") from exc
    bed = sum(stems.values(), np.zeros_like(next(iter(stems.values()))))
    bed = bed[grid.samples(LEAD_IN_BARS * grid.bar):]
    onset = min(grid.samples(ONSET_SECONDS), len(bed))
    bed[:onset] *= np.linspace(0.0, 1.0, onset, dtype=np.float32)[:, None]

    report(0.93, "Levelling")
    track = limit(loudness_normalize(bed, MUSIC_ALONE_LUFS)).astype(np.float32)
    if track.ndim != 2 or track.shape[1] != 2 or not len(track) or \
            not np.isfinite(track).all():
        raise LoopError("The music renderer produced invalid stereo audio.")
    peak = float(np.abs(track).max())
    if peak > 0.97 + 1e-7:
        raise LoopError(f"The music renderer exceeded the 0.97 peak limit ({peak:.3f}).")

    report(0.97, "Writing the track")
    write_mp3(output, track)
    report(1.0, "Music ready")
    return MusicResult(
        audio_path=str(output),
        audio_mime=MP3_MIME,
        duration_seconds=len(track) / SR,
        style_id=resolved.fingerprint.family,
        # The seed the request carried, as a loop reports it: what the same request replays from.
        seed=int(resolved.request.seed),
        engine_version=resolved.engine_version,
        profile_version=resolved.profile_version,
        bed_fingerprint=bed_fingerprint(resolved.fingerprint),
        total_bars=bars,
        bpm=float(spec.bpm),
    )
