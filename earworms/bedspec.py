"""Every parameter of the music bed, in one place.

The bed used to be hard-coded in music.py. It is now described by a BedSpec,
which can be sampled from a named style, saved to JSON, hand-edited and
replayed. Styles define *ranges*; a seeded rng picks within them, so two runs of
the same style sound related but not identical.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

SCALES: dict[str, list[int]] = {
    "natural_minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
}

# Which scale degrees make up the pentatonic subset used for the lead part.
PENTATONIC_DEGREES = {
    "natural_minor": [0, 2, 3, 4, 6],
    "dorian": [0, 2, 3, 4, 6],
    "harmonic_minor": [0, 2, 3, 4, 6],
    "major": [0, 1, 2, 4, 5],
    "lydian": [0, 1, 2, 4, 5],
}


@dataclass
class Pad:
    instrument: str = "synth"  # "synth" or "strings"
    level: float = 0.55
    detune: float = 0.06  # semitones either side of centre
    cutoff_base: float = 900.0
    cutoff_motion: float = 400.0
    cutoff_curve: str = "sine"  # sine, triangle, random_walk
    cutoff_period_bars: float = 9.0
    overlap: float = 1.6  # chord length as a multiple of a bar
    duck_db: float = 5.0
    enabled: bool = True


@dataclass
class Bass:
    level: float = 0.5
    octave: int = -1
    attack: float = 0.15
    decay_bars: float = 0.6
    duck_db: float = 5.0
    enabled: bool = True


@dataclass
class Drums:
    """Patterns are 16-step strings: 'x' strikes, '.' rests."""

    kick: str = "x.......x......."
    rim: str = "....x.......x..."
    kick_level: float = 0.55
    rim_level: float = 0.16
    shaker_level: float = 0.06
    shaker_density: float = 1.0  # 1.0 = every eighth note
    duck_db: float = 5.0
    enabled: bool = True


@dataclass
class Lead:
    instrument: str = "synth"  # synth, piano, marimba, glockenspiel
    level: float = 1.0
    bar_probability: float = 0.45  # chance a given bar gets any notes at all
    max_notes: int = 2
    register: tuple[int, int] = (24, 39)  # semitones above the root
    velocity: tuple[float, float] = (0.45, 0.75)
    humanize: float = 0.0  # seconds of timing jitter
    duck_db: float = 5.0
    enabled: bool = True


@dataclass
class Space:
    reverb_seconds: float = 3.0
    reverb_mix: float = 0.45


@dataclass
class BedSpec:
    bpm: float = 80.0
    beats_per_bar: int = 4
    beat_unit: int = 4
    swing: float = 0.0  # 0 = straight, 0.3 ≈ a gentle shuffle
    root: int = 45  # MIDI note; 45 = A2
    scale: str = "natural_minor"
    progression: list[int] = field(default_factory=lambda: [0, 5, 2, 6])
    chord_extension: str = "none"  # none, seventh, add9, ninth
    seed: int = 0
    pad: Pad = field(default_factory=Pad)
    bass: Bass = field(default_factory=Bass)
    drums: Drums = field(default_factory=Drums)
    lead: Lead = field(default_factory=Lead)
    space: Space = field(default_factory=Space)

    # -- harmony helpers -------------------------------------------------

    def scale_steps(self) -> list[int]:
        return SCALES[self.scale]

    def chord_root(self, degree: int) -> int:
        """Root of the chord on `degree`, folded to stay near the tonic."""
        steps = self.scale_steps()
        note = self.root + steps[degree % len(steps)] + 12 * (degree // len(steps))
        while note - self.root > 6:
            note -= 12
        while note - self.root < -6:
            note += 12
        return note

    def chord(self, degree: int) -> list[int]:
        """A spread voicing with an optional diatonic seventh or ninth."""
        steps = self.scale_steps()
        n = len(steps)
        base_degree = degree % n
        third = (steps[(base_degree + 2) % n] - steps[base_degree]) % 12
        r = self.chord_root(degree)
        notes = [r, r + 7, r + 12, r + 12 + third, r + 19]

        seventh = (steps[(base_degree + 6) % n] - steps[base_degree]) % 12
        second = (steps[(base_degree + 1) % n] - steps[base_degree]) % 12
        if self.chord_extension in ("seventh", "ninth"):
            notes.append(r + 12 + seventh)
        if self.chord_extension in ("add9", "ninth"):
            notes.append(r + 12 + second)
        if self.chord_extension not in ("none", "seventh", "add9", "ninth"):
            raise ValueError(f"Unknown chord extension '{self.chord_extension}'.")
        return sorted(set(notes))

    @property
    def steps_per_bar(self) -> int:
        """Number of sixteenth-note subdivisions in the configured meter."""
        steps = self.beats_per_bar * 16 / self.beat_unit
        if steps != int(steps):
            raise ValueError("Meter must divide cleanly into sixteenth notes.")
        return int(steps)

    def pentatonic(self) -> list[int]:
        """Lead pitches, as absolute MIDI notes across the configured register."""
        steps = self.scale_steps()
        offsets = [steps[d] for d in PENTATONIC_DEGREES[self.scale]]
        lo, hi = self.register_semitones()
        notes = []
        for octave in range(0, 5):
            for off in offsets:
                semis = off + 12 * octave
                if lo <= semis <= hi:
                    notes.append(self.root + semis)
        return notes or [self.root + lo]

    def register_semitones(self) -> tuple[int, int]:
        return tuple(self.lead.register)  # type: ignore[return-value]

    # -- serialisation ---------------------------------------------------

    def to_json(self, path: Path | None = None) -> str:
        text = json.dumps(asdict(self), indent=2)
        if path:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, data: dict) -> "BedSpec":
        return _build(cls, data)

    @classmethod
    def from_json(cls, path: Path) -> "BedSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- styles ----------------------------------------------------------

    @classmethod
    def from_style(cls, style: str, seed: int = 0) -> "BedSpec":
        if style not in STYLES:
            raise ValueError(f"Unknown style '{style}'. Try: {', '.join(STYLES)}")
        rng = random.Random(seed)
        spec = STYLES[style](rng)
        spec.seed = seed
        return spec


def _build(kind, data):
    """Rebuild a nested dataclass tree from plain JSON data."""
    if not is_dataclass(kind):
        return data
    kwargs = {}
    for f in fields(kind):
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(f.type) or (isinstance(value, dict) and f.name in _NESTED):
            kwargs[f.name] = _build(_NESTED[f.name], value)
        elif f.name == "register":
            kwargs[f.name] = tuple(value)
        elif f.name == "velocity":
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    return kind(**kwargs)


_NESTED = {"pad": Pad, "bass": Bass, "drums": Drums, "lead": Lead, "space": Space}


# --------------------------------------------------------------------------
# Styles. Each returns a spec with values sampled from that style's ranges.
# --------------------------------------------------------------------------

def _meter(rng: random.Random, bpms: list[int]) -> tuple[int, int, int]:
    """Choose 3/4, 4/4 or 5/4 while keeping a speech bar at least 2.2 s."""
    beats = rng.choice([4, 4, 4, 4, 3, 5])
    compatible = [bpm for bpm in bpms if 60 / bpm * beats >= 2.2]
    if not compatible:
        beats = 4
        compatible = bpms
    return rng.choice(compatible), beats, 4


def _beat_pattern(beats: int, positions: list[int]) -> str:
    steps = ["."] * (beats * 4)
    for position in positions:
        steps[position % len(steps)] = "x"
    return "".join(steps)


def _drum_patterns(beats: int, flavour: str) -> tuple[str, str]:
    steps = beats * 4
    if flavour == "lofi":
        kick = _beat_pattern(beats, [0, max(4, steps // 2 - 2), steps // 2 + 2])
        rim = _beat_pattern(beats, list(range(4, steps, 8)))
    elif flavour == "warm":
        kick = _beat_pattern(beats, list(range(0, steps, 4)))
        rim = _beat_pattern(beats, list(range(4, steps, 8)))
    elif flavour == "nocturne":
        kick = _beat_pattern(beats, [0, steps // 2])
        rim = _beat_pattern(beats, [steps // 2])
    else:
        kick = _beat_pattern(beats, list(range(0, steps, 8)))
        rim = _beat_pattern(beats, list(range(4, steps, 8)))
    return kick, rim


def _harmony(rng: random.Random) -> str:
    return rng.choice(["none", "seventh", "add9", "add9", "ninth"])


def _curve(rng: random.Random) -> str:
    return rng.choice(["sine", "sine", "triangle", "random_walk"])

def _yoga(rng: random.Random) -> BedSpec:
    """The original bed: warm, unresolved, nothing demanding attention."""
    bpm, beats, unit = _meter(rng, [72, 76, 78, 80, 82, 84])
    kick, rim = _drum_patterns(beats, "yoga")
    return BedSpec(
        bpm=bpm, beats_per_bar=beats, beat_unit=unit,
        root=rng.choice([43, 45, 47, 48]),
        scale=rng.choice(["natural_minor", "dorian"]),
        progression=rng.choice([[0, 5, 2, 6], [0, 3, 5, 4], [0, 5, 3, 6]]),
        chord_extension=_harmony(rng),
        pad=Pad(instrument=rng.choice(["synth", "synth", "strings"]),
                cutoff_base=rng.uniform(800, 1100),
                cutoff_motion=rng.uniform(300, 500), cutoff_curve=_curve(rng),
                cutoff_period_bars=rng.uniform(6, 12), duck_db=rng.uniform(5.5, 6.5)),
        bass=Bass(duck_db=rng.uniform(1.5, 2.5)),
        drums=Drums(kick=kick, rim=rim, duck_db=rng.uniform(2.5, 3.5)),
        lead=Lead(instrument=rng.choice(["synth", "piano", "marimba"]),
                  bar_probability=rng.uniform(0.35, 0.55), max_notes=2,
                  duck_db=rng.uniform(6.5, 7.5)),
        space=Space(reverb_seconds=rng.uniform(2.6, 3.4), reverb_mix=rng.uniform(0.4, 0.5)),
    )


def _nocturne(rng: random.Random) -> BedSpec:
    """Slower and darker, with the melodic instrument further forward."""
    bpm, beats, unit = _meter(rng, [58, 60, 64, 66, 68])
    kick, rim = _drum_patterns(beats, "nocturne")
    return BedSpec(
        bpm=bpm, beats_per_bar=beats, beat_unit=unit,
        root=rng.choice([40, 41, 43, 45]),
        scale=rng.choice(["natural_minor", "harmonic_minor"]),
        progression=rng.choice([[0, 4, 5, 0], [0, 6, 5, 4], [0, 2, 5, 4]]),
        chord_extension=_harmony(rng),
        pad=Pad(instrument=rng.choice(["strings", "strings", "synth"]),
                level=0.6, cutoff_base=rng.uniform(600, 850),
                cutoff_curve=_curve(rng), cutoff_period_bars=rng.uniform(8, 16),
                overlap=rng.uniform(1.6, 2.1), duck_db=rng.uniform(5.5, 6.5)),
        bass=Bass(level=0.42, attack=0.25, duck_db=rng.uniform(1.5, 2.5)),
        drums=Drums(kick=kick, rim=rim, shaker_level=0.03,
                    shaker_density=0.5, duck_db=rng.uniform(2.5, 3.5)),
        lead=Lead(instrument=rng.choice(["piano", "piano", "glockenspiel"]), level=1.1,
                  bar_probability=rng.uniform(0.55, 0.8), max_notes=3,
                  register=(19, 34), humanize=0.02,
                  duck_db=rng.uniform(6.5, 7.5)),
        space=Space(reverb_seconds=rng.uniform(3.2, 4.2), reverb_mix=rng.uniform(0.45, 0.58)),
    )


def _lofi(rng: random.Random) -> BedSpec:
    """Shuffled, muted and closer, with a busier kit."""
    bpm, beats, unit = _meter(rng, [68, 72, 74, 76, 80])
    kick, rim = _drum_patterns(beats, "lofi")
    return BedSpec(
        bpm=bpm, beats_per_bar=beats, beat_unit=unit,
        swing=rng.uniform(0.18, 0.3),
        root=rng.choice([44, 45, 46, 48]),
        scale=rng.choice(["dorian", "natural_minor"]),
        progression=rng.choice([[0, 3, 5, 4], [0, 5, 1, 4], [0, 4, 5, 3]]),
        chord_extension=_harmony(rng),
        pad=Pad(instrument=rng.choice(["synth", "synth", "strings"]), level=0.45,
                cutoff_base=rng.uniform(500, 750), cutoff_motion=250,
                cutoff_curve=_curve(rng), cutoff_period_bars=rng.uniform(4, 9),
                duck_db=rng.uniform(5.5, 6.5)),
        bass=Bass(level=0.58, attack=0.08, decay_bars=0.45,
                  duck_db=rng.uniform(1.5, 2.5)),
        drums=Drums(kick=kick, rim=rim,
                    kick_level=0.6, rim_level=0.2,
                    shaker_level=rng.uniform(0.05, 0.09),
                    duck_db=rng.uniform(2.5, 3.5)),
        lead=Lead(instrument=rng.choice(["piano", "marimba"]),
                  bar_probability=rng.uniform(0.4, 0.65),
                  max_notes=3, register=(24, 36), humanize=0.03,
                  velocity=(0.35, 0.6), duck_db=rng.uniform(6.5, 7.5)),
        space=Space(reverb_seconds=rng.uniform(1.6, 2.4), reverb_mix=rng.uniform(0.3, 0.4)),
    )


def _warm(rng: random.Random) -> BedSpec:
    """Major and open — brighter than the rest without becoming cheerful."""
    bpm, beats, unit = _meter(rng, [78, 82, 84, 88, 90, 92])
    kick, rim = _drum_patterns(beats, "warm")
    return BedSpec(
        bpm=bpm, beats_per_bar=beats, beat_unit=unit,
        root=rng.choice([48, 50, 53, 55]),
        scale=rng.choice(["major", "lydian"]),
        progression=rng.choice([[0, 4, 5, 3], [0, 3, 4, 0], [0, 5, 3, 4]]),
        chord_extension=_harmony(rng),
        pad=Pad(instrument=rng.choice(["synth", "strings"]),
                cutoff_base=rng.uniform(1000, 1400),
                cutoff_motion=rng.uniform(350, 550), cutoff_curve=_curve(rng),
                cutoff_period_bars=rng.uniform(6, 12), duck_db=rng.uniform(5.5, 6.5)),
        bass=Bass(level=0.45, duck_db=rng.uniform(1.5, 2.5)),
        drums=Drums(kick=kick, rim=rim,
                    shaker_level=rng.uniform(0.05, 0.08),
                    duck_db=rng.uniform(2.5, 3.5)),
        lead=Lead(instrument=rng.choice(["piano", "glockenspiel"]),
                  bar_probability=rng.uniform(0.45, 0.7), max_notes=2,
                  register=(24, 38), duck_db=rng.uniform(6.5, 7.5)),
        space=Space(reverb_seconds=rng.uniform(2.2, 3.0), reverb_mix=rng.uniform(0.35, 0.45)),
    )


STYLES = {"yoga": _yoga, "nocturne": _nocturne, "lofi": _lofi, "warm": _warm}
