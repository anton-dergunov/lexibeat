"""Immutable generation profiles for the public LexiBeat API."""

from __future__ import annotations

from dataclasses import dataclass, field


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

# What each production family sounds like, in words a listener choosing one can use. Written from
# what the family's code does — its tempo range, scales, textures and the leads it prefers
# (`bedspec._WIDE_FAMILIES`, `generator.enrich_with_catalog_samples`) — and meant to be edited by
# ear. A host shows these beside the family in its picker, so a new family needs a line here.
FAMILY_DESCRIPTIONS = {
    "meditative": "Slow and spacious: long held chords, little or no percussion.",
    "organic": "Earthy mid-tempo pulse: marimba or piano over soft hand percussion.",
    "acoustic": "Gentle and melodic: piano, marimba or glockenspiel over warm strings.",
    "sunlit": "Bright and relaxed, lightly rhythmic.",
    "radiant": "The brightest and quickest: open, major-key, uplifting.",
    "acoustic-flow": "Calm and flowing: sustained piano and strings, a light pulse.",
    "playful-minimal": "Light and sparse: a few bouncy notes over a soft pad.",
    "warm-motion": "Warm, flowing, gently propulsive.",
    "bright-organic": "Upbeat and airy: marimba or piano with a moving bass line.",
    "gentle-game": "Quick, cheerful arpeggios, like a calm game menu.",
    "sunlit-acoustic": "Guitar, harp or plucked strings; bright and unhurried.",
    "gentle-movement": "A soft pulse carried by piano, vibraphone, ocarina or organ.",
    "playful-plucked": "Kalimba, mbira, strumstick and guitar: plucked and bouncy.",
    "bright-pastoral": "Ocarina and pizzicato strings; open, countryside feel.",
}


def family_label(family: str) -> str:
    """`acoustic-flow` → "Acoustic flow"."""
    return family.replace("-", " ").capitalize()


BROAD_FAMILIES = (
    "meditative",
    "organic",
    "acoustic",
    "nocturnal",
    "sunlit",
    "lofi-wide",
)


@dataclass(frozen=True)
class ListenerPolicy:
    """What the listener has ruled out, applied on top of a resolved bed.

    Every switch is written so it changes *only* a bed that shows the thing it rules out, and draws
    nothing from the bed's own random stream: a bed without the defect comes out byte-identical
    with the switch on or off. That is what lets a switch be heard in isolation — the same seed,
    rendered both ways, differs in exactly one part — before it is adopted as a profile default.

    - ``approved_catalog_only``: a catalogue lead is chosen only from the banks the bundle's
      expansion policy accepted after listening; anything else falls back to an accepted bank that
      fits, or to the named pack. Before this, banks nobody had auditioned under speech — an FM
      electric piano among them — reached the lead with no register limit at all.
    - ``lead_register_cap``: no lead note above this MIDI note; higher ones fold down by octaves.
      The README demo, which the owner likes, never went above 88.
    - ``diatonic_fifths``: a chord's fifth is the scale's own. `BedSpec.chord` stacks a perfect
      fifth on every degree, which is off-key on vii in major, #iv in lydian and vi in dorian.
    """

    approved_catalog_only: bool = False
    lead_register_cap: int | None = None
    diatonic_fifths: bool = False


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
    listener: ListenerPolicy = field(default_factory=ListenerPolicy)


PRODUCTION_V1 = GenerationProfile(
    name="production-v1",
    version="1.5.0",
    families=POSITIVE_FAMILIES,
    # Adopted after round 0 of `docs/plans/music-listening.md`, heard blind on the same seeds: the
    # approved-only lead won 4–2 and the cap at 88 won 5–0 (one "same"). Diatonic fifths lost 1–5 —
    # the perfect fifth on every chord is preferred by ear — and stays off.
    listener=ListenerPolicy(approved_catalog_only=True, lead_register_cap=88),
)

EXPLORATION_V1 = GenerationProfile(
    name="exploration-v1",
    version="1.4.0",
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
