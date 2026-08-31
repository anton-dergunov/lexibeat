"""Immutable generation profiles for the public LexiBeat API."""

from __future__ import annotations

from dataclasses import dataclass


POSITIVE_FAMILIES = (
    "meditative",
    "organic",
    "acoustic",
    "sunlit",
    "radiant",
    "acoustic-flow",
    "playful-minimal",
    "warm-motion",
    "bright-organic",
    "gentle-game",
    "sunlit-acoustic",
    "gentle-movement",
    "playful-plucked",
    "bright-pastoral",
)

BROAD_FAMILIES = (
    "meditative",
    "organic",
    "acoustic",
    "nocturnal",
    "sunlit",
    "lofi-wide",
)


@dataclass(frozen=True)
class GenerationProfile:
    """Versioned policy applied above the fully resolved ``BedSpec``."""

    name: str
    version: str
    families: tuple[str, ...]
    candidate_count: int = 6
    candidate_attempt_multiplier: int = 4
    max_swing: float = 0.025
    max_percussion_share: float = 0.60
    top_tier_fraction: float = 0.34
    min_quality_score: float = 0.62


PRODUCTION_V1 = GenerationProfile(
    name="production-v1",
    version="1.3.0",
    families=POSITIVE_FAMILIES,
)

EXPLORATION_V1 = GenerationProfile(
    name="exploration-v1",
    version="1.3.0",
    families=BROAD_FAMILIES,
    max_swing=0.08,
    min_quality_score=0.55,
)

PROFILES = {
    PRODUCTION_V1.name: PRODUCTION_V1,
    EXPLORATION_V1.name: EXPLORATION_V1,
}


def get_profile(name: str) -> GenerationProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown generation profile '{name}'. Try: {choices}") from exc
