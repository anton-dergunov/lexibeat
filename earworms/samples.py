"""Downloadable, manifest-driven instrument sample packs.

Each pack describes the source URL and playback metadata for every sample. The
manifest is intentionally explicit: Salamander and VSCO use unrelated filename
schemes, and audio rendering should not have to reverse-engineer either one.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

SEMITONES = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
             "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def midi(note: str) -> int:
    """Convert a scientific-pitch name such as ``F#3`` to MIDI 54."""
    name = note[:2] if len(note) > 2 and note[1] == "#" else note[:1]
    octave = int(note[len(name):])
    return SEMITONES[name] + (octave + 1) * 12


@dataclass(frozen=True)
class Sample:
    filename: str
    remote_path: str
    midi_note: int
    velocity: int = 1
    articulation: str = "natural"


@dataclass(frozen=True)
class SamplePack:
    name: str
    license: str
    attribution: str
    homepage: str
    base_url: str
    samples: tuple[Sample, ...]
    layer_count: int = 1
    default_velocities: tuple[int, ...] | None = None

    def entries(self, velocities: tuple[int, ...] | None = None) -> list[Sample]:
        wanted = velocities if velocities is not None else self.default_velocities
        if wanted is None:
            return list(self.samples)
        return [sample for sample in self.samples if sample.velocity in wanted]

    def filenames(self, velocities: tuple[int, ...] | None = None) -> list[str]:
        return [sample.filename for sample in self.entries(velocities)]


_SALAMANDER_ROOTS = tuple(
    f"{name}{octave}"
    for octave in range(0, 9)
    for name in ("A", "C", "D#", "F#")
    if not (octave == 0 and name != "A") and not (octave == 8 and name != "C")
)
_SALAMANDER_SAMPLES = tuple(
    Sample(f"{root}v{velocity}.flac", f"{root}v{velocity}.flac",
           midi(root), velocity, "piano")
    for velocity in range(1, 17)
    for root in _SALAMANDER_ROOTS
)

SALAMANDER = SamplePack(
    name="salamander",
    license="CC-BY 3.0",
    attribution="Salamander Grand Piano (Yamaha C5) by Alexander Holm",
    homepage="https://archive.org/details/SalamanderGrandPianoV3",
    base_url="https://raw.githubusercontent.com/sfzinstruments/"
             "SalamanderGrandPiano/master/Samples/",
    samples=_SALAMANDER_SAMPLES,
    layer_count=16,
    default_velocities=(5, 11),
)

_VSCO_BASE = "https://raw.githubusercontent.com/sgossner/VSCO-2-CE/master/"


def _vsco_sample(path: str, note: str, velocity: int = 1,
                 articulation: str = "natural") -> Sample:
    return Sample(Path(path).name, path, midi(note), velocity, articulation)


VSCO_MARIMBA = SamplePack(
    name="vsco-marimba",
    license="CC0-1.0",
    attribution="VSCO 2 Community Edition by Versilian Studios",
    homepage="https://github.com/sgossner/VSCO-2-CE",
    base_url=_VSCO_BASE,
    samples=tuple(
        _vsco_sample(f"Percussion/Marimba/Marimba_hit_Outrigger_{note}_loud_01.wav",
                     note, articulation="hit")
        for note in ("F1", "B2", "C2", "F3", "B4", "C4", "C6")
    ),
)

VSCO_GLOCKENSPIEL = SamplePack(
    name="vsco-glockenspiel",
    license="CC0-1.0",
    attribution="VSCO 2 Community Edition by Versilian Studios",
    homepage="https://github.com/sgossner/VSCO-2-CE",
    base_url=_VSCO_BASE,
    samples=tuple(
        _vsco_sample(f"Percussion/Glock/glock_medium_{note}.wav", note,
                     articulation="medium")
        for note in ("G4", "C5", "G5", "C6", "G6", "C7")
    ),
)

_STRING_NOTES = ("A2", "A3", "B2", "B4", "C4", "D3", "D5", "E4",
                 "F#3", "G2", "G4")
VSCO_STRINGS = SamplePack(
    name="vsco-strings",
    license="CC0-1.0",
    attribution="VSCO 2 Community Edition by Versilian Studios",
    homepage="https://github.com/sgossner/VSCO-2-CE",
    base_url=_VSCO_BASE,
    samples=tuple(
        _vsco_sample(f"Strings/Violin Section/susVib/VlnEns_susVib_{note}_v{v}.wav",
                     note, v, "sustain-vibrato")
        for v in (1, 2) for note in _STRING_NOTES
    ),
    layer_count=2,
)

PACKS = {pack.name: pack for pack in (
    SALAMANDER, VSCO_MARIMBA, VSCO_GLOCKENSPIEL, VSCO_STRINGS,
)}
PACK_GROUPS = {"vsco": ("vsco-marimba", "vsco-glockenspiel", "vsco-strings")}


def cache_dir() -> Path:
    root = Path(os.environ.get("EARWORMS_CACHE",
                               Path.home() / ".cache" / "earworms"))
    return root / "samples"


def pack_dir(pack: SamplePack) -> Path:
    return cache_dir() / pack.name


def missing(pack: SamplePack,
            velocities: tuple[int, ...] | None = None) -> list[Sample]:
    target = pack_dir(pack)
    return [entry for entry in pack.entries(velocities)
            if not (target / entry.filename).exists()]


def download(pack: SamplePack, velocities: tuple[int, ...] | None = None,
             quiet: bool = False) -> Path:
    """Fetch uncached samples. Safe to re-run; partial files are discarded."""
    target = pack_dir(pack)
    target.mkdir(parents=True, exist_ok=True)
    todo = missing(pack, velocities)
    if not todo:
        return target

    if not quiet:
        print(f"Downloading {len(todo)} samples for '{pack.name}' "
              f"({pack.license} — {pack.attribution})")
    for i, entry in enumerate(todo, 1):
        url = pack.base_url + urllib.request.quote(entry.remote_path)
        tmp = target / (entry.filename + ".part")
        try:
            with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as f:
                f.write(response.read())
        except (urllib.error.URLError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Could not download {entry.filename}: {exc}") from exc
        tmp.rename(target / entry.filename)
        if not quiet:
            print(f"\r  {i}/{len(todo)} {entry.filename:<40}", end="", flush=True)
    if not quiet:
        print(f"\r  {len(todo)} files into {target}" + " " * 20)
    return target


def download_target(name: str, quiet: bool = False) -> list[Path]:
    names = PACK_GROUPS.get(name, (name,))
    return [download(PACKS[pack_name], quiet=quiet) for pack_name in names]


def is_available(pack: SamplePack,
                 velocities: tuple[int, ...] | None = None) -> bool:
    return not missing(pack, velocities)
