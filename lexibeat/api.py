"""Stable public interface for resolving and rendering LexiBeat music beds.

The small :class:`MusicRequest` is caller intent. The returned ``BedSpec`` is
the complete replayable composition. Resolution and rendering are offline and
never download samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np

from .bedspec import BedSpec

Energy = Literal["calm", "balanced", "bright"]
Rhythm = Literal["sparse", "steady", "groovy"]
Palette = Literal["acoustic", "hybrid", "electronic"]


@dataclass(frozen=True)
class MusicRequest:
    """A deliberately small set of safe, product-level controls."""

    family: str = "auto"
    energy: Energy = "balanced"
    rhythm: Rhythm = "steady"
    palette: Palette = "hybrid"
    seed: int | None = None
    profile: str = "production-v1"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "MusicRequest":
        return cls(**values).validated()

    def validated(self) -> "MusicRequest":
        from .profiles import get_profile

        profile = get_profile(self.profile)
        if self.family != "auto" and self.family not in profile.families:
            choices = ", ".join(("auto", *profile.families))
            raise ValueError(f"Unknown family '{self.family}'. Try: {choices}")
        if self.energy not in ("calm", "balanced", "bright"):
            raise ValueError("energy must be calm, balanced, or bright")
        if self.rhythm not in ("sparse", "steady", "groovy"):
            raise ValueError("rhythm must be sparse, steady, or groovy")
        if self.palette not in ("acoustic", "hybrid", "electronic"):
            raise ValueError("palette must be acoustic, hybrid, or electronic")
        if self.seed is not None and not 0 <= self.seed < 2 ** 64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        return self


@dataclass(frozen=True)
class BedFingerprint:
    family: str
    audio_features: tuple[float, ...]
    motif_features: tuple[float, ...]
    instrument_families: tuple[str, ...]


@dataclass(frozen=True)
class QualityReport:
    accepted: bool
    score: float
    rejection_reasons: tuple[str, ...]
    measurements: dict[str, float]


@dataclass(frozen=True)
class SampleUsage:
    collection: str
    asset_id: str
    sha256: str
    license: str
    attribution: str
    relative_path: str


@dataclass(frozen=True)
class MusicGenerationResult:
    request: MusicRequest
    bed_spec: BedSpec
    fingerprint: BedFingerprint
    quality: QualityReport
    sample_manifest: tuple[SampleUsage, ...]
    engine_version: str
    profile_version: str

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_music(
    request: MusicRequest,
    *,
    avoid_fingerprints: Sequence[BedFingerprint] = (),
) -> MusicGenerationResult:
    """Resolve caller intent into one validated, fully replayable bed."""
    from .generator import resolve_request

    return resolve_request(request, avoid_fingerprints=avoid_fingerprints)


def render_music(
    result_or_spec: MusicGenerationResult | BedSpec,
    *,
    duration_seconds: float,
) -> np.ndarray:
    """Render a resolved bed close to the requested duration, on whole bars."""
    from .generator import render_resolved

    spec = (result_or_spec.bed_spec if isinstance(result_or_spec, MusicGenerationResult)
            else result_or_spec)
    return render_resolved(spec, duration_seconds=duration_seconds)


def generate_music(
    request: MusicRequest,
    *,
    duration_seconds: float,
    avoid_fingerprints: Sequence[BedFingerprint] = (),
) -> tuple[np.ndarray, MusicGenerationResult]:
    """Resolve and render in one call while still returning the replay contract."""
    result = resolve_music(request, avoid_fingerprints=avoid_fingerprints)
    return render_music(result, duration_seconds=duration_seconds), result
