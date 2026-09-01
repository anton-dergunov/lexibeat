"""Resolved, replayable role treatments for the Wave 3 listening experiment."""

from __future__ import annotations

from dataclasses import replace

from .bedspec import BedSpec, ChordEvent, NoteEvent
from .library import InstrumentRef


SUSTAINED_LEADS = {
    "bassoon", "clarinet", "flute", "harmonica", "oboe", "ocarina", "recorder",
}
PLUCKED_LEADS = {"harp", "harpsichord", "lamellophone", "marimba", "plucked-string"}
ROLE_GAIN_DB = {
    "accordion": -5.0,
    "bassoon": -6.0,
    "clarinet": -6.0,
    "flute": -6.0,
    "harmonica": -6.0,
    "harp": 0.0,
    "harpsichord": -4.0,
    "lamellophone": 0.0,
    "marimba": -4.0,
    "oboe": -6.0,
    "ocarina": -7.0,
    "organ": -6.0,
    "plucked-string": 0.0,
    "recorder": -7.0,
}
ROLE_LEAD_LEVEL = {
    "accordion": 4.0,
    "bassoon": 4.0,
    "clarinet": 4.0,
    "flute": 4.0,
    "harmonica": 4.0,
    "harp": 2.4,
    "harpsichord": 4.0,
    "lamellophone": 2.8,
    "marimba": 3.2,
    "oboe": 4.0,
    "ocarina": 4.0,
    "plucked-string": 2.5,
    "recorder": 4.0,
}


def _raise_to_gain(instrument: InstrumentRef, target_max_db: float) -> InstrumentRef:
    """Raise a conservatively audited bank while retaining relative zone balance."""
    current_max = max(zone.gain_db for zone in instrument.zones)
    boost = max(target_max_db - current_max, 0.0)
    return InstrumentRef(instrument.name, tuple(
        replace(zone, gain_db=zone.gain_db + boost) for zone in instrument.zones
    ))


def _fit(note: int, instrument: InstrumentRef) -> int:
    lo = min(zone.root_note for zone in instrument.zones)
    hi = max(zone.root_note for zone in instrument.zones)
    while note < lo and note + 12 <= hi:
        note += 12
    while note > hi and note - 12 >= lo:
        note -= 12
    return min(max(note, lo), hi)


def _bar_note(spec: BedSpec, instrument: InstrumentRef, bar: int,
              octave: int) -> int:
    degree = spec.progression[bar % len(spec.progression)]
    return _fit(spec.chord_root(degree) + octave, instrument)


def _sustained_events(spec: BedSpec, instrument: InstrumentRef,
                      family: str) -> list[NoteEvent]:
    assert spec.phrase is not None
    steps = spec.steps_per_bar
    octave = 12 if family == "bassoon" else 24
    articulation = instrument.zones[0].articulation
    return [
        NoteEvent(
            bar * steps + max(1, round(steps * 0.22)),
            steps * 0.58,
            _bar_note(spec, instrument, bar, octave),
            0.50 if bar % 2 == 0 else 0.46,
            articulation,
            bar,
        )
        for bar in range(spec.phrase.loop_bars)
    ]


def _accordion_events(spec: BedSpec, instrument: InstrumentRef) -> list[NoteEvent]:
    assert spec.phrase is not None
    steps = spec.steps_per_bar
    articulation = instrument.zones[0].articulation
    events = []
    for bar in range(spec.phrase.loop_bars):
        note = _bar_note(spec, instrument, bar, 12)
        events.append(NoteEvent(
            bar * steps + max(1, round(steps * 0.16)), steps * 0.48,
            note, 0.50, articulation, bar))
        if bar % 2:
            events.append(NoteEvent(
                bar * steps + round(steps * 0.72), steps * 0.18,
                _fit(note + 2, instrument), 0.40, articulation, bar + 1))
    return events


def _plucked_events(spec: BedSpec, instrument: InstrumentRef,
                    family: str) -> list[NoteEvent]:
    assert spec.phrase is not None
    steps = spec.steps_per_bar
    articulation = instrument.zones[0].articulation
    events = []
    positions = (0.20, 0.58, 0.78) if family == "harp" else (0.24, 0.66)
    for bar in range(spec.phrase.loop_bars):
        base = _bar_note(spec, instrument, bar, 24 if family != "plucked-string" else 12)
        for index, position in enumerate(positions):
            interval = (0, 7, 12)[index]
            events.append(NoteEvent(
                bar * steps + round(steps * position),
                steps * (0.28 if family == "harp" else 0.24),
                _fit(base + interval, instrument),
                0.48 - index * 0.045,
                articulation,
                bar + index,
            ))
    return events


def _organ_chords(spec: BedSpec, instrument: InstrumentRef) -> list[ChordEvent]:
    assert spec.phrase is not None
    steps = spec.steps_per_bar
    articulation = instrument.zones[0].articulation
    chords = []
    scale = spec.scale_steps()
    for bar in range(spec.phrase.loop_bars):
        degree = spec.progression[bar % len(spec.progression)]
        root = spec.chord_root(degree) + 12
        third = (scale[(degree + 2) % len(scale)] - scale[degree % len(scale)]) % 12
        notes = list(dict.fromkeys(
            _fit(note, instrument) for note in (root, root + third, root + 7)
        ))
        chords.append(ChordEvent(
            bar * steps, steps * 0.90, notes, 0.68, articulation, bar))
    return chords


def apply_wave3_role_profile(
    spec: BedSpec,
    family: str,
    instrument: InstrumentRef,
) -> dict:
    """Apply one experimental family role entirely through serialized BedSpec data."""
    if spec.phrase is None:
        raise ValueError("Wave 3 role profiles require a resolved phrase.")
    if family not in ROLE_GAIN_DB:
        raise ValueError(f"Unknown Wave 3 role family '{family}'.")
    instrument = _raise_to_gain(instrument, ROLE_GAIN_DB[family])
    spec.space.reverb_seconds = min(spec.space.reverb_seconds, 2.0)
    spec.space.reverb_mix = min(spec.space.reverb_mix, 0.28)

    if family == "organ":
        spec.phrase.pad_instrument = instrument
        spec.phrase.pad_sample = None
        spec.phrase.chords = _organ_chords(spec, instrument)
        spec.phrase.motif_grammar = "organ-held-pad-v1"
        spec.pad.level = max(spec.pad.level, 0.62)
        spec.pad.duck_db = max(spec.pad.duck_db, 6.5)
        spec.lead.enabled = False
        role = "held-pad"
    else:
        spec.phrase.lead_instrument = instrument
        spec.phrase.lead_sample = None
        spec.lead.enabled = True
        # The renderer deliberately keeps generic sampled leads conservative.
        # These explicit BedSpec levels compensate for that attenuation in the
        # listening experiment and remain visible/replayable in the saved JSON.
        spec.lead.level = ROLE_LEAD_LEVEL[family]
        spec.lead.duck_db = 6.5
        if family in SUSTAINED_LEADS:
            spec.phrase.lead = _sustained_events(spec, instrument, family)
            spec.phrase.motif_grammar = f"{family}-sustained-v1"
            role = "sustained-lead"
        elif family == "accordion":
            spec.phrase.lead = _accordion_events(spec, instrument)
            spec.phrase.motif_grammar = "accordion-cantabile-v1"
            role = "cantabile-lead"
        elif family in PLUCKED_LEADS:
            spec.phrase.lead = _plucked_events(spec, instrument, family)
            spec.phrase.motif_grammar = f"{family}-plucked-v1"
            role = "plucked-lead"
        else:  # pragma: no cover - guarded by ROLE_GAIN_DB and sets above
            raise ValueError(f"No Wave 3 role profile for '{family}'.")
    return {
        "family": family,
        "role": role,
        "instrument": instrument.name,
        "target_max_gain_db": ROLE_GAIN_DB[family],
        "lead_level": None if family == "organ" else ROLE_LEAD_LEVEL[family],
        "reverb_seconds": spec.space.reverb_seconds,
        "reverb_mix": spec.space.reverb_mix,
    }
