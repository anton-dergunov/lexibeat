"""Downloadable instrument sample packs.

Packs are fetched once into a user cache directory and reused. Only the
velocity layers actually needed are downloaded — the full Salamander library is
748 MB, but two layers are ~93 MB and plenty for a sparse background part.
"""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# Salamander names sharps with '#', which has to be escaped in a URL.
NOTE_FILE = re.compile(r"^(?P<name>[A-G]#?)(?P<octave>-?\d+)v(?P<velocity>\d+)\.flac$")
SEMITONES = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6,
             "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def midi_of(filename: str) -> tuple[int, int] | None:
    """Map 'D#3v7.flac' to (MIDI note, velocity layer)."""
    m = NOTE_FILE.match(filename)
    if not m:
        return None
    note = SEMITONES[m.group("name")] + (int(m.group("octave")) + 1) * 12
    return note, int(m.group("velocity"))


@dataclass(frozen=True)
class SamplePack:
    name: str
    license: str
    attribution: str
    homepage: str
    base_url: str
    roots: tuple[str, ...]  # note names present in the library
    velocities: tuple[int, ...]  # which velocity layers to fetch by default
    layer_count: int = 16  # how many the library actually contains

    def filenames(self, velocities: tuple[int, ...] | None = None) -> list[str]:
        return [f"{root}v{v}.flac"
                for v in (velocities or self.velocities) for root in self.roots]


# Sampled in minor thirds from the lowest A, so any note is at most 1.5
# semitones from a real recording.
_SALAMANDER_ROOTS = tuple(
    f"{name}{octave}"
    for octave in range(0, 9)
    for name in ("A", "C", "D#", "F#")
    if not (octave == 0 and name != "A") and not (octave == 8 and name != "C")
)

SALAMANDER = SamplePack(
    name="salamander",
    license="CC-BY 3.0",
    attribution="Salamander Grand Piano (Yamaha C5) by Alexander Holm",
    homepage="https://archive.org/details/SalamanderGrandPianoV3",
    base_url="https://raw.githubusercontent.com/sfzinstruments/"
             "SalamanderGrandPiano/master/Samples/",
    roots=_SALAMANDER_ROOTS,
    velocities=(5, 11),  # one soft layer, one firmer
)

PACKS = {p.name: p for p in (SALAMANDER,)}


def cache_dir() -> Path:
    root = Path(os.environ.get("EARWORMS_CACHE",
                               Path.home() / ".cache" / "earworms"))
    return root / "samples"


def pack_dir(pack: SamplePack) -> Path:
    return cache_dir() / pack.name


def missing(pack: SamplePack, velocities: tuple[int, ...] | None = None) -> list[str]:
    target = pack_dir(pack)
    return [f for f in pack.filenames(velocities) if not (target / f).exists()]


def download(pack: SamplePack, velocities: tuple[int, ...] | None = None,
             quiet: bool = False) -> Path:
    """Fetch any samples not already cached. Safe to re-run; resumes by file."""
    target = pack_dir(pack)
    target.mkdir(parents=True, exist_ok=True)
    todo = missing(pack, velocities)
    if not todo:
        return target

    if not quiet:
        print(f"Downloading {len(todo)} samples for '{pack.name}' "
              f"({pack.license} — {pack.attribution})")
    for i, name in enumerate(todo, 1):
        url = pack.base_url + urllib.request.quote(name)
        tmp = target / (name + ".part")
        try:
            with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as f:
                f.write(r.read())
        except urllib.error.URLError as exc:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Could not download {name}: {exc}") from exc
        tmp.rename(target / name)  # rename last, so a partial file is never cached
        if not quiet:
            print(f"\r  {i}/{len(todo)} {name:<14}", end="", flush=True)
    if not quiet:
        print(f"\r  {len(todo)} files into {target}" + " " * 20)
    return target


def is_available(pack: SamplePack, velocities: tuple[int, ...] | None = None) -> bool:
    return not missing(pack, velocities)
