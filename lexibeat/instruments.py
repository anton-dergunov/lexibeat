"""Instruments for the lead layer.

Both implementations answer the same question — "give me this note, at this
velocity, lasting this long" — so the bed can swap an oscillator for a recorded
piano by changing one field in the spec.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Protocol

import librosa
import numpy as np
import soundfile as sf

from . import samples as sample_packs
from .library import InstrumentRef, SampleLibrary, SampleRef
from .samples import SamplePack

SR = 44100


class Instrument(Protocol):
    def render(self, midi_note: float, velocity: float,
               seconds: float) -> np.ndarray: ...


def _midi_hz(note: float) -> float:
    return 440.0 * 2 ** ((note - 69) / 12.0)


class SynthInstrument:
    """The original additive bell: a sine plus two quiet harmonics."""

    name = "synth"

    def render(self, midi_note: float, velocity: float,
               seconds: float) -> np.ndarray:
        n = int(seconds * SR)
        t = np.arange(n) / SR
        f = _midi_hz(midi_note)
        sig = (np.sin(2 * np.pi * f * t)
               + 0.3 * np.sin(2 * np.pi * 2 * f * t)
               + 0.12 * np.sin(2 * np.pi * 3 * f * t))
        attack = np.clip(t / 0.006, 0, 1)
        return sig * attack * np.exp(-t / 0.5) * velocity


@lru_cache(maxsize=128)
def _load_sample(path: str, max_seconds: float) -> tuple[np.ndarray, int]:
    """Read a sample as mono, truncated — the full files are 15 s each."""
    with sf.SoundFile(path) as f:
        data = f.read(frames=int(max_seconds * f.samplerate), dtype="float32")
        rate = f.samplerate
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, rate


class SampledInstrument:
    """Plays back recorded samples, pitched to the requested note.

    The library is sampled in minor thirds, so the resampling ratio is never
    more than 1.5 semitones and the timbre stays convincing.
    """

    def __init__(self, pack: SamplePack, velocities: tuple[int, ...] | None = None):
        self.name = pack.name
        self.pack = pack
        directory = sample_packs.pack_dir(pack)
        self.layers: dict[int, list[tuple[int, str]]] = {}
        for entry in pack.entries(velocities):
            path = directory / entry.filename
            if path.exists():
                self.layers.setdefault(entry.velocity, []).append(
                    (entry.midi_note, str(path)))
        if not self.layers:
            raise FileNotFoundError(
                f"No samples cached for '{pack.name}'. "
                f"Run: uv run generate.py --download-samples {pack.name}")
        for entries in self.layers.values():
            entries.sort()
        self._sorted_layers = sorted(self.layers)

    def _pick(self, midi_note: float, velocity: float) -> tuple[str, float]:
        """Choose a sample, and how many semitones it must be shifted."""
        # Velocity layers run 1..layer_count across the dynamic range.
        wanted = 1 + velocity * (self.pack.layer_count - 1)
        layer = min(self._sorted_layers, key=lambda k: abs(k - wanted))
        entries = self.layers[layer]
        note, path = min(entries, key=lambda e: abs(e[0] - midi_note))
        return path, midi_note - note

    def render(self, midi_note: float, velocity: float,
               seconds: float) -> np.ndarray:
        path, semitones = self._pick(midi_note, velocity)
        # Read a little extra, since shifting up shortens the sample.
        audio, rate = _load_sample(path, seconds * 1.4 + 0.5)

        # Resampling to a lower target rate and then calling the result 44.1 kHz
        # raises the pitch — one operation covers both shift and rate change.
        ratio = 2 ** (semitones / 12.0)
        target = SR / ratio
        if abs(target - rate) > 1.0:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=target,
                                     res_type="soxr_hq")

        n = int(seconds * SR)
        if len(audio) < n:
            audio = np.pad(audio, (0, n - len(audio)))
        audio = audio[:n].copy()

        # Fade the tail so a truncated note does not click.
        release = min(int(0.25 * SR), n)
        if release:
            audio[-release:] *= np.linspace(1.0, 0.0, release) ** 1.5

        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * velocity
        return audio


class CatalogSampleInstrument:
    """Play a stable logical sample reference from the tiered library."""

    def __init__(self, ref: SampleRef, library: SampleLibrary | None = None):
        self.ref = ref
        self.library = library or SampleLibrary()
        self.asset = self.library.asset(ref)
        self.path = self.library.resolve(ref)
        self.name = f"{ref.collection}:{ref.asset_id}"

    def render(self, midi_note: float, velocity: float,
               seconds: float) -> np.ndarray:
        root = self.asset.midi_note if self.asset.midi_note is not None else midi_note
        audio, rate = _load_sample(str(self.path), seconds * 1.5 + 0.5)
        ratio = 2 ** ((midi_note - root) / 12.0)
        target = SR / ratio
        if abs(target - rate) > 1.0:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=target,
                                     res_type="soxr_hq")
        n = max(int(seconds * SR), 1)
        audio = np.pad(audio, (0, max(n - len(audio), 0)))[:n].copy()
        release = min(int(0.18 * SR), n)
        if release:
            audio[-release:] *= np.linspace(1.0, 0.0, release)
        peak = np.abs(audio).max()
        return audio / peak * velocity if peak else audio


class CatalogMultiSampleInstrument:
    """Render a complete resolved instrument rather than stretching one sample."""

    def __init__(self, ref: InstrumentRef, library: SampleLibrary | None = None):
        if not ref.zones:
            raise ValueError(f"Instrument '{ref.name}' has no sample zones.")
        self.ref = ref
        self.library = library or SampleLibrary()
        self.name = ref.name

    def _pick(self, midi_note: float, velocity: float):
        midi_velocity = int(np.clip(round(velocity * 127), 0, 127))
        matching = [zone for zone in self.ref.zones
                    if zone.lo_note <= midi_note <= zone.hi_note and
                    zone.lo_velocity <= midi_velocity <= zone.hi_velocity]
        choices = matching or list(self.ref.zones)
        return min(choices, key=lambda zone: (
            abs(zone.root_note - midi_note),
            abs((zone.lo_velocity + zone.hi_velocity) / 2 - midi_velocity),
            zone.sample.asset_id))

    def render(self, midi_note: float, velocity: float,
               seconds: float) -> np.ndarray:
        zone = self._pick(midi_note, velocity)
        path = self.library.resolve(zone.sample)
        audio, rate = _load_sample(str(path), seconds * 1.5 + 0.5)
        ratio = 2 ** ((midi_note - zone.root_note) / 12.0)
        target = SR / ratio
        if abs(target - rate) > 1.0:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=target,
                                     res_type="soxr_hq")
        n = max(int(seconds * SR), 1)
        audio = np.pad(audio, (0, max(n - len(audio), 0)))[:n].copy()
        release = min(int(0.2 * SR), n)
        if release:
            audio[-release:] *= np.linspace(1.0, 0.0, release) ** 1.3
        peak = float(np.abs(audio).max())
        gain = 10 ** (zone.gain_db / 20)
        return audio / peak * velocity * gain if peak else audio


def load_one_shot(ref: SampleRef, max_seconds: float = 3.0,
                  library: SampleLibrary | None = None) -> np.ndarray:
    """Load, mono-fold and resample one catalog percussion asset."""
    library = library or SampleLibrary()
    path = library.resolve(ref)
    audio, rate = _load_sample(str(path), max_seconds)
    if rate != SR:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=SR,
                                 res_type="soxr_hq")
    audio = np.asarray(audio, dtype=np.float32)
    # Remove leading digital silence while retaining a tiny pre-transient margin.
    active = np.flatnonzero(np.abs(audio) > max(float(np.abs(audio).max()) * 0.002, 1e-5))
    if active.size:
        audio = audio[max(0, int(active[0]) - int(0.003 * SR)):]
    release = min(int(0.02 * SR), len(audio))
    if release:
        audio[-release:] *= np.linspace(1.0, 0.0, release)
    peak = float(np.abs(audio).max())
    if not peak:
        return audio
    active_audio = audio[np.abs(audio) > peak * 0.01]
    rms = float(np.sqrt(np.mean(active_audio.astype(np.float64) ** 2))) \
        if active_audio.size else peak
    gain = min(1.0 / peak, 0.16 / max(rms, 1e-8))
    asset = library.asset(ref)
    if asset.spectral_centroid and asset.spectral_centroid > 4200:
        gain *= 10 ** (-3.0 / 20)
    return audio * gain


def build(name: str, velocities: tuple[int, ...] | None = None) -> Instrument:
    """Resolve an instrument name from a BedSpec into a usable instrument."""
    if name == "synth":
        return SynthInstrument()
    aliases = {
        "piano": "salamander",
        "marimba": "vsco-marimba",
        "glockenspiel": "vsco-glockenspiel",
        "strings": "vsco-strings",
    }
    pack_name = aliases.get(name, name)
    if pack_name in sample_packs.PACKS:
        return SampledInstrument(sample_packs.PACKS[pack_name], velocities)
    raise ValueError(f"Unknown instrument '{name}'. "
                     f"Try 'synth', 'piano', 'marimba', 'glockenspiel', "
                     f"'strings', or one of: "
                     f"{', '.join(sample_packs.PACKS)}")
