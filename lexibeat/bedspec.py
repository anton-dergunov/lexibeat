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

from .library import InstrumentRef, InstrumentZoneRef, SampleRef

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

BASS_GRAMMARS = ("drone", "sustain", "root_fifth", "chord_tone",
                 "passing", "syncopated")
MOTIF_GRAMMARS = ("random_walk", "rising", "falling", "arch",
                  "return_home", "call_response")
RESOLVED_BASS_GRAMMARS = ("legacy", *BASS_GRAMMARS)
RESOLVED_MOTIF_GRAMMARS = ("legacy", *MOTIF_GRAMMARS)
TIMBRE_PALETTES = ("acoustic", "hybrid", "electronic", "airy", "wooden",
                   "warm", "shimmering", "plucked", "soft-electronic")


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
    level: float = 1.0
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
class NoteEvent:
    """A fully resolved note positioned on the loop's sixteenth-note grid."""

    step: int
    duration_steps: float
    midi_note: int
    velocity: float
    articulation: str = "natural"
    sample_variation: int = 0


@dataclass
class ChordEvent:
    step: int
    duration_steps: float
    midi_notes: list[int]
    velocity: float = 0.55
    articulation: str = "sustain"
    sample_variation: int = 0


@dataclass
class PercussionLane:
    """A resolved repeating lane; pattern spans the complete phrase loop."""

    sound: str
    pattern: str
    level: float
    probability: float = 1.0
    humanize: float = 0.0
    pan: float = 0.0
    sample: SampleRef | None = None
    role: str = ""  # low, mid or high; empty keeps legacy JSON compatible
    articulation: str = "natural"
    round_robin_samples: tuple[SampleRef, ...] = ()


@dataclass
class ResolvedPhrase:
    """All musical decisions needed to replay one coherent phrase."""

    family: str
    loop_bars: int
    harmony_texture: str
    pad_timbre: str
    bass_timbre: str
    round_robin_strategy: str = "cyclic"
    bass_grammar: str = "legacy"
    motif_grammar: str = "legacy"
    palette: str = "hybrid"
    chords: list[ChordEvent] = field(default_factory=list)
    bass: list[NoteEvent] = field(default_factory=list)
    lead: list[NoteEvent] = field(default_factory=list)
    percussion: list[PercussionLane] = field(default_factory=list)
    lead_sample: SampleRef | None = None
    pad_sample: SampleRef | None = None
    lead_instrument: InstrumentRef | None = None
    pad_instrument: InstrumentRef | None = None
    bass_instrument: InstrumentRef | None = None


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
    phrase: ResolvedPhrase | None = None
    schema_version: int = 2
    engine_version: str = "1.2.0"
    profile_version: str = "legacy"

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
        spec = _build(cls, data)
        if data.get("phrase") is not None:
            spec.phrase = _phrase_from_dict(data["phrase"])
        return spec

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
        _apply_variety_grammars(spec, seed)
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
        if f.name == "phrase":
            continue
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


def _sample_ref(data: dict | None) -> SampleRef | None:
    return SampleRef(**data) if data else None


def _instrument_ref(data: dict | None) -> InstrumentRef | None:
    if not data:
        return None
    return InstrumentRef(data["name"], tuple(
        InstrumentZoneRef(
            sample=_sample_ref(zone["sample"]), root_note=zone["root_note"],
            lo_note=zone.get("lo_note", 0), hi_note=zone.get("hi_note", 127),
            lo_velocity=zone.get("lo_velocity", 0),
            hi_velocity=zone.get("hi_velocity", 127),
            gain_db=zone.get("gain_db", 0.0),
            round_robin=zone.get("round_robin", 0),
            articulation=zone.get("articulation", "natural"))
        for zone in data.get("zones", [])))


def _phrase_from_dict(data: dict) -> ResolvedPhrase:
    return ResolvedPhrase(
        family=data["family"], loop_bars=int(data["loop_bars"]),
        harmony_texture=data["harmony_texture"], pad_timbre=data["pad_timbre"],
        bass_timbre=data["bass_timbre"],
        round_robin_strategy=data.get("round_robin_strategy", "cyclic"),
        bass_grammar=data.get("bass_grammar", "legacy"),
        motif_grammar=data.get("motif_grammar", "legacy"),
        palette=data.get("palette", "hybrid"),
        chords=[ChordEvent(**event) for event in data.get("chords", [])],
        bass=[NoteEvent(**event) for event in data.get("bass", [])],
        lead=[NoteEvent(**event) for event in data.get("lead", [])],
        percussion=[PercussionLane(
            sound=lane["sound"], pattern=lane["pattern"], level=lane["level"],
            probability=lane.get("probability", 1.0),
            humanize=lane.get("humanize", 0.0), pan=lane.get("pan", 0.0),
            sample=_sample_ref(lane.get("sample")), role=lane.get("role", ""),
            articulation=lane.get("articulation", "natural"),
            round_robin_samples=tuple(
                _sample_ref(ref) for ref in lane.get("round_robin_samples", [])
                if ref))
            for lane in data.get("percussion", [])],
        lead_sample=_sample_ref(data.get("lead_sample")),
        pad_sample=_sample_ref(data.get("pad_sample")),
        lead_instrument=_instrument_ref(data.get("lead_instrument")),
        pad_instrument=_instrument_ref(data.get("pad_instrument")),
        bass_instrument=_instrument_ref(data.get("bass_instrument")),
    )


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


def _euclidean(pulses: int, steps: int, rotation: int = 0) -> str:
    """Evenly distribute ``pulses`` across ``steps`` without a pattern table."""
    pulses = max(0, min(pulses, steps))
    bucket = 0
    values: list[str] = []
    for _ in range(steps):
        bucket += pulses
        if bucket >= steps:
            bucket -= steps
            values.append("x")
        else:
            values.append(".")
    if values:
        rotation %= len(values)
        values = values[-rotation:] + values[:-rotation]
    return "".join(values)


def _with_bar_downbeats(pattern: str, steps_per_bar: int) -> str:
    """Guarantee the primary rhythmic anchor beneath every spoken downbeat."""
    values = list(pattern)
    for index in range(0, len(values), steps_per_bar):
        values[index] = "x"
    return "".join(values)


def _smooth_voicing(notes: list[int], previous: list[int] | None) -> list[int]:
    """Choose an inversion with compact motion from the previous chord."""
    base = sorted(set(notes))[:5]
    candidates: list[list[int]] = []
    for inversion in range(min(4, len(base))):
        inverted = sorted(base[inversion:] + [note + 12 for note in base[:inversion]])
        for shift in (-12, 0, 12):
            candidate = [note + shift for note in inverted]
            if 35 <= min(candidate) and max(candidate) <= 84:
                candidates.append(candidate)
    if not candidates:
        return base
    if previous is None:
        return min(candidates, key=lambda chord: abs(sum(chord) / len(chord) - 57))
    return min(candidates, key=lambda chord: sum(
        abs(note - previous[min(index, len(previous) - 1)])
        for index, note in enumerate(chord)))


_WIDE_FAMILIES = {
    "meditative": {
        "bpms": [58, 62, 66, 70, 72, 76],
        "scales": ["natural_minor", "dorian", "major"],
        "roots": [40, 43, 45, 48], "textures": ["sustain", "drone", "open"],
        "bass": ["sustain", "drone", "root_fifth"],
        "pads": ["sine", "triangle", "strings"],
        "leads": ["synth", "piano", "glockenspiel"], "swing": (0.0, 0.03),
        "density": (0.12, 0.28), "brightness": (550, 1050),
    },
    "organic": {
        "bpms": [68, 72, 76, 80, 84, 88],
        "scales": ["dorian", "natural_minor", "major"],
        "roots": [43, 45, 48, 50], "textures": ["pulse", "open", "arpeggio"],
        "bass": ["root_fifth", "syncopated", "sustain"],
        "pads": ["triangle", "strings", "soft_saw"],
        "leads": ["marimba", "piano", "synth"], "swing": (0.0, 0.05),
        "density": (0.28, 0.52), "brightness": (700, 1350),
    },
    "acoustic": {
        "bpms": [64, 68, 72, 76, 80, 84],
        "scales": ["major", "dorian", "natural_minor"],
        "roots": [45, 48, 50, 53], "textures": ["arpeggio", "sustain", "pulse"],
        "bass": ["sustain", "root_fifth", "passing"],
        "pads": ["strings", "triangle"],
        "leads": ["piano", "marimba", "glockenspiel"], "swing": (0.0, 0.04),
        "density": (0.22, 0.46), "brightness": (850, 1500),
    },
    "nocturnal": {
        "bpms": [56, 60, 62, 66, 68, 72],
        "scales": ["natural_minor", "harmonic_minor", "dorian"],
        "roots": [38, 40, 41, 43, 45], "textures": ["drone", "sustain", "open"],
        "bass": ["drone", "sustain", "passing"],
        "pads": ["sine", "strings", "triangle"],
        "leads": ["piano", "glockenspiel", "synth"], "swing": (0.0, 0.04),
        "density": (0.1, 0.32), "brightness": (420, 850),
    },
    "sunlit": {
        "bpms": [78, 82, 86, 90, 94, 98],
        "scales": ["major", "lydian", "dorian"],
        "roots": [48, 50, 53, 55], "textures": ["pulse", "arpeggio", "open"],
        "bass": ["root_fifth", "syncopated", "passing"],
        "pads": ["triangle", "soft_saw", "strings"],
        "leads": ["piano", "marimba", "glockenspiel"], "swing": (0.0, 0.04),
        "density": (0.3, 0.55), "brightness": (1050, 1800),
    },
    "lofi-wide": {
        "bpms": [66, 70, 74, 78, 82, 86],
        "scales": ["dorian", "natural_minor", "major"],
        "roots": [43, 45, 46, 48, 50], "textures": ["pulse", "arpeggio", "sustain"],
        "bass": ["syncopated", "root_fifth", "passing"],
        "pads": ["soft_saw", "triangle", "sine"],
        "leads": ["piano", "marimba", "synth"], "swing": (0.02, 0.08),
        "density": (0.28, 0.5), "brightness": (500, 1100),
    },
    "radiant": {
        "bpms": [86, 90, 94, 98, 102], "scales": ["major", "lydian"],
        "roots": [48, 50, 53, 55], "textures": ["open", "pulse", "arpeggio"],
        "bass": ["root_fifth", "sustain"],
        "pads": ["strings", "triangle", "soft_saw"],
        "leads": ["piano", "marimba", "glockenspiel"], "swing": (0.0, 0.025),
        "density": (0.3, 0.5), "brightness": (1150, 1850),
    },
    "acoustic-flow": {
        "bpms": [68, 72, 76, 80, 84, 88], "scales": ["major", "dorian"],
        "roots": [45, 48, 50, 53], "textures": ["sustain", "open", "arpeggio"],
        "bass": ["sustain", "root_fifth"], "pads": ["strings", "triangle"],
        "leads": ["piano", "marimba"], "swing": (0.0, 0.025),
        "density": (0.2, 0.4), "brightness": (800, 1450),
    },
    "playful-minimal": {
        "bpms": [76, 82, 86, 90, 94], "scales": ["major", "lydian", "dorian"],
        "roots": [48, 50, 53, 55], "textures": ["pulse", "open"],
        "bass": ["root_fifth", "sustain"], "pads": ["sine", "triangle"],
        "leads": ["marimba", "piano", "glockenspiel"], "swing": (0.0, 0.02),
        "density": (0.16, 0.34), "brightness": (900, 1650),
    },
    "warm-motion": {
        "bpms": [72, 76, 80, 84, 88], "scales": ["major", "dorian"],
        "roots": [45, 48, 50, 53], "textures": ["pulse", "open"],
        "bass": ["root_fifth", "sustain"],
        "pads": ["triangle", "strings", "soft_saw"],
        "leads": ["piano", "marimba", "synth"], "swing": (0.0, 0.03),
        "density": (0.26, 0.44), "brightness": (750, 1350),
    },
    "bright-organic": {
        "bpms": [76, 80, 84, 88, 92], "scales": ["major", "dorian", "lydian"],
        "roots": [48, 50, 53, 55], "textures": ["open", "arpeggio", "pulse"],
        "bass": ["root_fifth", "passing"], "pads": ["strings", "triangle"],
        "leads": ["marimba", "piano", "synth"], "swing": (0.0, 0.035),
        "density": (0.24, 0.46), "brightness": (950, 1600),
    },
    "gentle-game": {
        "bpms": [82, 86, 90, 94, 98], "scales": ["major", "lydian"],
        "roots": [48, 50, 53, 55], "textures": ["arpeggio", "pulse"],
        "bass": ["root_fifth", "passing"], "pads": ["triangle", "sine"],
        "leads": ["marimba", "piano", "glockenspiel"], "swing": (0.0, 0.025),
        "density": (0.26, 0.44), "brightness": (1000, 1700),
    },
    "sunlit-acoustic": {
        "bpms": [76, 80, 84, 88, 92], "scales": ["major", "lydian"],
        "roots": [48, 50, 53, 55], "textures": ["open", "sustain", "arpeggio"],
        "bass": ["root_fifth", "sustain"], "pads": ["strings", "triangle"],
        "leads": ["piano", "marimba", "synth"], "swing": (0.0, 0.015),
        "density": (0.2, 0.38), "brightness": (1050, 1750),
    },
    "gentle-movement": {
        "bpms": [70, 74, 78, 82, 86], "scales": ["major", "dorian"],
        "roots": [45, 48, 50, 53], "textures": ["pulse", "open", "sustain"],
        "bass": ["root_fifth", "passing"], "pads": ["triangle", "strings"],
        "leads": ["piano", "marimba", "synth"], "swing": (0.0, 0.012),
        "density": (0.22, 0.4), "brightness": (800, 1400),
    },
    "playful-plucked": {
        "bpms": [78, 82, 86, 90, 94], "scales": ["major", "lydian"],
        "roots": [48, 50, 53, 55], "textures": ["open", "pulse"],
        "bass": ["root_fifth", "sustain"], "pads": ["sine", "triangle"],
        "leads": ["marimba", "glockenspiel", "synth"], "swing": (0.0, 0.01),
        "density": (0.18, 0.36), "brightness": (1000, 1700),
    },
    "bright-pastoral": {
        "bpms": [72, 76, 80, 84, 88], "scales": ["major", "lydian", "dorian"],
        "roots": [48, 50, 53, 55], "textures": ["sustain", "open", "arpeggio"],
        "bass": ["sustain", "root_fifth"], "pads": ["strings", "triangle"],
        "leads": ["piano", "marimba", "synth"], "swing": (0.0, 0.012),
        "density": (0.18, 0.36), "brightness": (950, 1550),
    },
}


def _progression(rng: random.Random, scale: str, bars: int) -> list[int]:
    choices = {
        0: [2, 3, 4, 5, 5, 6], 1: [4, 6, 0], 2: [4, 5, 6],
        3: [0, 4, 5], 4: [0, 5, 6], 5: [0, 2, 3, 4], 6: [0, 3, 4],
    }
    if scale == "harmonic_minor":
        choices[4] = [0, 0, 5, 6]
    degrees = [0]
    while len(degrees) < bars:
        degrees.append(rng.choice(choices[degrees[-1]]))
    return degrees


def _note_events(spec: BedSpec, rng: random.Random, bars: int,
                 mode: str) -> list[NoteEvent]:
    steps = spec.steps_per_bar
    events: list[NoteEvent] = []
    for bar in range(bars):
        degree = spec.progression[bar % len(spec.progression)]
        root = spec.chord_root(degree) - 12
        if mode in ("sustain", "drone"):
            note = spec.root - 12 if mode == "drone" else root
            events.append(NoteEvent(bar * steps, steps * 0.82, note,
                                    rng.uniform(0.42, 0.62)))
        elif mode == "root_fifth":
            events.extend([
                NoteEvent(bar * steps, steps * 0.42, root, rng.uniform(0.45, 0.65)),
                NoteEvent(bar * steps + steps // 2, steps * 0.35, root + 7,
                          rng.uniform(0.34, 0.54)),
            ])
        elif mode == "syncopated":
            for at, interval, velocity in ((0, 0, 0.62),
                                           (max(3, steps * 3 // 8), 7, 0.42),
                                           (max(6, steps * 3 // 4), 0, 0.5)):
                events.append(NoteEvent(bar * steps + min(at, steps - 1),
                                        max(1.5, steps / 5), root + interval,
                                        velocity * rng.uniform(0.86, 1.08)))
        elif mode == "chord_tone":
            scale = spec.scale_steps()
            third = (scale[(degree + 2) % len(scale)] -
                     scale[degree % len(scale)]) % 12
            colour = third if bar % 2 == 0 else 7
            for at, interval, velocity in (
                    (0, 0, 0.58), (steps // 2, colour, 0.44)):
                events.append(NoteEvent(
                    bar * steps + min(at, steps - 1), max(1.5, steps / 5),
                    root + interval, velocity * rng.uniform(0.88, 1.06)))
        else:  # passing
            next_degree = spec.progression[(bar + 1) % len(spec.progression)]
            target = spec.chord_root(next_degree) - 12
            passing = root + max(-2, min(2, target - root))
            events.extend([
                NoteEvent(bar * steps, steps * 0.55, root, rng.uniform(0.45, 0.62)),
                NoteEvent(bar * steps + steps * 3 // 4, steps * 0.2, passing,
                          rng.uniform(0.3, 0.46)),
            ])
    return events


def _lead_events(spec: BedSpec, rng: random.Random, bars: int,
                 density: float) -> list[NoteEvent]:
    notes = spec.pentatonic()
    if not notes:
        return []
    steps = spec.steps_per_bar
    index = rng.randrange(len(notes))
    motif: list[tuple[int, int, float]] = []
    motif_bars = min(4, bars)
    effective_density = max(density, 0.30 if spec.lead.instrument == "piano" else density)
    for step in range(2, steps * motif_bars, 2):
        if rng.random() > effective_density:
            continue
        index = max(0, min(len(notes) - 1, index + rng.choice([-2, -1, 0, 0, 1, 2])))
        motif.append((step, notes[index], rng.uniform(*spec.lead.velocity)))

    if spec.lead.instrument != "piano":
        available_steps = list(range(2, steps * motif_bars, 2))
        used_steps = {event[0] for event in motif}
        while len(motif) < min(2, len(available_steps)):
            choices = [step for step in available_steps if step not in used_steps]
            if not choices:
                break
            step = rng.choice(choices)
            used_steps.add(step)
            motif.append((step, rng.choice(notes), rng.uniform(*spec.lead.velocity)))
        motif.sort()
        if len(motif) >= 2 and max(note for _, note, _ in motif) - min(
                note for _, note, _ in motif) < 3:
            pairs = [(low, high) for low in notes for high in notes
                     if 3 <= high - low <= 12]
            if pairs:
                low, high = rng.choice(pairs)
                motif[0] = (motif[0][0], low, motif[0][2])
                motif[-1] = (motif[-1][0], high, motif[-1][2])

    # A piano random walk can become trapped at one edge of its register. Keep
    # it sparse, but guarantee enough pitch movement to sound intentionally
    # written rather than like one repeatedly triggered sample.
    if spec.lead.instrument == "piano":
        available_steps = list(range(2, steps * motif_bars, 2))
        used_steps = {event[0] for event in motif}
        while len(motif) < min(5, len(available_steps)):
            choices = [step for step in available_steps if step not in used_steps]
            if not choices:
                break
            step = rng.choice(choices)
            used_steps.add(step)
            motif.append((step, rng.choice(notes), rng.uniform(*spec.lead.velocity)))
        motif.sort()
        if len(motif) >= 2 and max(note for _, note, _ in motif) - min(
                note for _, note, _ in motif) < 7:
            pairs = [(low, high) for low in notes for high in notes
                     if 7 <= high - low <= 19]
            if pairs:
                low, high = rng.choice(pairs)
                first, last = motif[0], motif[-1]
                motif[0] = (first[0], low, first[2])
                motif[-1] = (last[0], high, last[2])
    events: list[NoteEvent] = []
    phrase_steps = bars * steps
    motif_span = steps * motif_bars
    for offset in range(0, phrase_steps, motif_span):
        repetition = offset // max(motif_span, 1)
        for event_index, (step, note, velocity) in enumerate(motif):
            keep = (spec.lead.instrument == "piano" and
                    event_index in (0, len(motif) - 1)) or rng.random() < 0.88
            if offset + step < phrase_steps and keep:
                if (spec.lead.instrument == "piano" and repetition % 2 and
                        event_index >= max(len(motif) - 2, 0)):
                    shifted = note + rng.choice([-12, 12])
                    if shifted in notes:
                        note = shifted
                events.append(NoteEvent(offset + step, rng.choice([1.5, 2.5, 4.0]),
                                        note, velocity * rng.uniform(0.88, 1.08)))
    return events


def _contour_lead_events(spec: BedSpec, rng: random.Random, bars: int,
                         grammar: str) -> list[NoteEvent]:
    """Write a sparse, clearly recognizable contour without renderer randomness."""
    notes = spec.pentatonic()
    if not notes:
        return []
    patterns = {
        "rising": (0, 1, 2, 3, 4),
        "falling": (4, 3, 2, 1, 0),
        "arch": (0, 2, 4, 2, 0),
        "return_home": (0, 3, 1, 4, 0),
        "call_response": (0, 2, 3, 1, 4),
    }
    pattern = patterns[grammar]
    width = min(max(pattern) + 1, len(notes))
    start = rng.randrange(max(len(notes) - width + 1, 1))
    pitches = [notes[min(start + index, len(notes) - 1)] for index in pattern]
    motif_bars = min(4, bars)
    steps = spec.steps_per_bar
    span = motif_bars * steps
    if grammar == "call_response" and motif_bars >= 2:
        positions = [
            min(2, steps - 1), max(3, steps // 2),
            steps * 2 + min(2, steps - 1),
            steps * 2 + max(3, steps // 2),
            steps * 3 + max(2, steps * 3 // 4),
        ] if motif_bars >= 4 else [
            min(2, steps - 1), max(3, steps // 2),
            steps + min(2, steps - 1), steps + max(3, steps // 2),
            steps + max(4, steps * 3 // 4),
        ]
    else:
        positions = [max(1, round((index + 1) * span / 6))
                     for index in range(5)]
    positions = [min(position, span - 1) for position in positions]
    events: list[NoteEvent] = []
    for offset in range(0, bars * steps, span):
        for index, (position, note) in enumerate(zip(positions, pitches, strict=True)):
            if offset + position >= bars * steps:
                continue
            response = grammar == "call_response" and index >= 2
            velocity = rng.uniform(*spec.lead.velocity) * (0.9 if response else 1.0)
            events.append(NoteEvent(
                offset + position, rng.choice((1.5, 2.5, 3.5)), note, velocity,
                articulation="natural", sample_variation=len(events)))
    return events


def _apply_variety_grammars(spec: BedSpec, seed: int) -> None:
    """Resolve macro-grammars from independent seeds after legacy style sampling."""
    phrase = spec.phrase
    if phrase is None:
        return
    calm = phrase.family in ("meditative", "nocturnal", "acoustic-flow",
                             "bright-pastoral")
    bass_choices = (
        ("sustain", "sustain", "drone", "root_fifth", "chord_tone")
        if calm else
        ("sustain", "root_fifth", "chord_tone", "passing", "syncopated")
    )
    bass_rng = random.Random(seed ^ 0x424153534752414D)
    phrase.bass_grammar = bass_rng.choice(bass_choices)
    phrase.bass = _note_events(
        spec, bass_rng, phrase.loop_bars, phrase.bass_grammar)
    for index, event in enumerate(phrase.bass):
        event.sample_variation = index

    motif_choices = (
        ("random_walk", "arch", "return_home", "falling") if calm else
        ("random_walk", "rising", "falling", "arch", "return_home",
         "call_response")
    )
    motif_rng = random.Random(seed ^ 0x4D4F544946475241)
    phrase.motif_grammar = motif_rng.choice(motif_choices)
    if phrase.motif_grammar != "random_walk":
        phrase.lead = _contour_lead_events(
            spec, motif_rng, phrase.loop_bars, phrase.motif_grammar)
    for index, event in enumerate(phrase.lead):
        event.sample_variation = index


def _bar_pattern(bars: int, steps: int, positions: list[int],
                 rng: random.Random, optional: set[int] | None = None) -> str:
    """Repeat a meter-aware pattern with deterministic light omissions."""
    values = ["."] * (bars * steps)
    optional = optional or set()
    for bar in range(bars):
        for position in positions:
            if position in optional and rng.random() < 0.28:
                continue
            values[bar * steps + min(max(position, 0), steps - 1)] = "x"
    return "".join(values)


def _percussion_lanes(spec: BedSpec, rng: random.Random, bars: int,
                      density: float) -> list[PercussionLane]:
    """Create a clear pulse first, then restrained meter-aligned accents."""
    steps = spec.steps_per_bar
    half = max(4, (steps // 2 // 4) * 4)
    grammar = rng.choice(["straight", "straight", "half-time",
                          "gentle-syncopation", "sparse"])
    low_positions = [0]
    low_optional: set[int] = set()
    if grammar != "sparse":
        low_positions.append(half)
        if grammar in ("half-time", "gentle-syncopation"):
            low_optional.add(half)
    low = _bar_pattern(bars, steps, low_positions, rng, low_optional)

    beat_positions = list(range(4, steps, 8)) or [min(4, steps - 1)]
    if grammar == "straight":
        mid_positions = beat_positions
    elif grammar == "gentle-syncopation":
        mid_positions = sorted(set(beat_positions + [max(2, half - 2)]))
    elif grammar == "half-time":
        mid_positions = [half]
    else:
        mid_positions = [beat_positions[-1]]
    mid = _bar_pattern(bars, steps, mid_positions, rng,
                       set(mid_positions) if grammar == "sparse" else set())

    high_positions = list(range(2, steps, 4))
    if grammar in ("sparse", "half-time"):
        high_positions = high_positions[::2]
    high = _bar_pattern(bars, steps, high_positions, rng,
                        set(high_positions) if density < 0.32 else set())
    return [
        PercussionLane("synth:kick", low, 0.25 + density * 0.14, 0.98,
                       role="low"),
        PercussionLane(rng.choice(["synth:rim", "synth:wood", "synth:brush"]),
                       mid, 0.06 + density * 0.11, 0.86, 0.002,
                       rng.uniform(-0.2, 0.2), role="mid"),
        PercussionLane(rng.choice(["synth:shaker", "synth:soft_hat"]), high,
                       0.012 + density * 0.035, 0.84, 0.002,
                       rng.choice([-0.3, 0.3]), role="high"),
    ]


def _resolve_phrase(spec: BedSpec, family: str, rng: random.Random,
                    config: dict) -> ResolvedPhrase:
    bars = rng.choice([4, 4, 8])
    spec.progression = _progression(rng, spec.scale, bars)
    steps = spec.steps_per_bar
    texture = rng.choice(config["textures"])
    chords: list[ChordEvent] = []
    previous: list[int] | None = None
    for bar, degree in enumerate(spec.progression):
        notes = _smooth_voicing(spec.chord(degree), previous)
        previous = notes
        if texture in ("sustain", "drone", "open"):
            if texture == "drone":
                notes = sorted(set([spec.root, spec.root + 7, spec.root + 12,
                                    *notes[-2:]]))
            elif texture == "open":
                notes = sorted(set([notes[0], notes[1], notes[-2], notes[-1]]))
            chords.append(ChordEvent(bar * steps, steps * spec.pad.overlap, notes,
                                     rng.uniform(0.42, 0.62)))
        elif texture == "pulse":
            for at, accent in ((0, 1.0), (steps // 2, 0.72)):
                chords.append(ChordEvent(bar * steps + at, steps * 0.34, notes,
                                         rng.uniform(0.4, 0.58) * accent))
        else:  # arpeggio
            order = notes + list(reversed(notes[1:-1]))
            for index, at in enumerate(range(0, steps, 2)):
                chords.append(ChordEvent(bar * steps + at, 1.7,
                                         [order[index % len(order)]],
                                         rng.uniform(0.34, 0.5)))

    density_lo, density_hi = config["density"]
    density = rng.uniform(density_lo, density_hi)
    lanes = _percussion_lanes(spec, rng, bars, density)
    if family in ("meditative", "nocturnal") and rng.random() < 0.55:
        lanes = lanes[:rng.choice([1, 2])]
    # Preserve the historical draw order: timbres were resolved before bass and
    # lead events. Macro-grammar RNGs are independent and applied by from_style.
    pad_timbre = rng.choice(config["pads"])
    bass_timbre = rng.choice(["sine", "round", "triangle", "pluck"])
    bass_mode = rng.choice(config["bass"])
    bass = _note_events(spec, rng, bars, bass_mode)
    lead = _lead_events(spec, rng, bars, density)
    for index, event in enumerate(chords):
        event.sample_variation = index
        event.articulation = "sustain" if texture != "arpeggio" else "plucked"
    for index, event in enumerate(bass):
        event.sample_variation = index
    for index, event in enumerate(lead):
        event.sample_variation = index
    return ResolvedPhrase(
        family=family, loop_bars=bars, harmony_texture=texture,
        pad_timbre=pad_timbre, bass_timbre=bass_timbre,
        bass_grammar=bass_mode, chords=chords, bass=bass, lead=lead,
        percussion=lanes,
    )


def _wide_style(family: str, rng: random.Random) -> BedSpec:
    config = _WIDE_FAMILIES[family]
    bpm, beats, unit = _meter(rng, config["bpms"])
    low, high = config["brightness"]
    spec = BedSpec(
        bpm=bpm, beats_per_bar=beats, beat_unit=unit,
        swing=rng.uniform(*config["swing"]), root=rng.choice(config["roots"]),
        scale=rng.choice(config["scales"]), chord_extension=_harmony(rng),
        pad=Pad(instrument="strings" if "strings" in config["pads"] and
                rng.random() < 0.35 else "synth", level=rng.uniform(0.38, 0.58),
                cutoff_base=rng.uniform(low, high), cutoff_motion=rng.uniform(120, 520),
                cutoff_curve=_curve(rng), cutoff_period_bars=rng.uniform(4, 14),
                overlap=rng.uniform(1.1, 1.9), duck_db=rng.uniform(6.0, 7.5)),
        bass=Bass(level=rng.uniform(0.34, 0.56), attack=rng.uniform(0.04, 0.22),
                  decay_bars=rng.uniform(0.35, 0.9), duck_db=rng.uniform(2.0, 3.5)),
        drums=Drums(level=rng.uniform(0.48, 0.64), duck_db=rng.uniform(4.5, 6.0)),
        lead=Lead(instrument=rng.choice(config["leads"]), level=rng.uniform(0.72, 1.08),
                  register=(rng.choice([19, 24]), rng.choice([34, 36, 39])),
                  velocity=(0.3, 0.66), humanize=rng.uniform(0.0, 0.006),
                  duck_db=rng.uniform(7.0, 9.0)),
        space=Space(reverb_seconds=rng.uniform(1.5, 4.4),
                    reverb_mix=rng.uniform(0.28, 0.58)),
    )
    if family not in ("lofi-wide", "nocturnal"):
        spec.swing = min(spec.swing, 0.025)
    if spec.lead.instrument == "piano":
        spec.lead.level = rng.uniform(0.92, 1.14)
        spec.lead.register = (rng.choice([17, 19]), rng.choice([43, 46]))
        spec.lead.velocity = (0.34, 0.7)
        spec.lead.duck_db = rng.uniform(5.5, 7.0)
    spec.phrase = _resolve_phrase(spec, family, rng, config)
    return spec


def _family_style(name: str):
    return lambda rng: _wide_style(name, rng)


STYLES = {
    "yoga": _yoga, "nocturne": _nocturne, "lofi": _lofi, "warm": _warm,
    **{name: _family_style(name) for name in _WIDE_FAMILIES},
}
