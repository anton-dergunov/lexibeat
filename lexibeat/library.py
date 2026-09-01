"""Tiered, explicitly managed sample-library storage and catalog.

Normal music rendering never downloads data.  This module is used by the
``sample_library.py`` maintenance command and by resolved ``SampleRef`` values
stored in new BedSpec JSON files.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .paths import BUNDLED_ROOT, configured_bundle_root
from .sfz import parse as parse_sfz


EXTERNAL_DEFAULT = Path("/Volumes/EXTSSD_SAND/downloaded_music/lexibeat-library")
GB = 1_000_000_000
EXTERNAL_WARN = 450 * GB
EXTERNAL_LIMIT = 500 * GB
LOCAL_WARN = 45 * GB
LOCAL_LIMIT = 50 * GB
AUDIO_SUFFIXES = {".wav", ".flac", ".aif", ".aiff", ".ogg"}


@dataclass(frozen=True)
class SampleCollection:
    id: str
    name: str
    repository: str
    license: str
    attribution: str
    estimated_bytes: int
    categories: tuple[str, ...]
    safe_default: bool = True


@dataclass(frozen=True)
class SampleRef:
    collection: str
    asset_id: str
    sha256: str = ""


@dataclass(frozen=True)
class InstrumentZoneRef:
    """One resolved multisample zone stored independently of physical paths."""

    sample: SampleRef
    root_note: int
    lo_note: int = 0
    hi_note: int = 127
    lo_velocity: int = 0
    hi_velocity: int = 127
    gain_db: float = 0.0
    round_robin: int = 0
    articulation: str = "natural"


@dataclass(frozen=True)
class InstrumentRef:
    """A complete, serializable multisample instrument."""

    name: str
    zones: tuple[InstrumentZoneRef, ...]


@dataclass(frozen=True)
class SampleAsset:
    collection: str
    asset_id: str
    sha256: str
    relative_path: str
    license: str
    category: str
    articulation: str = ""
    midi_note: int | None = None
    velocity_low: int | None = None
    velocity_high: int | None = None
    round_robin: int | None = None
    bpm: float | None = None
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    peak: float | None = None
    rms: float | None = None
    spectral_centroid: float | None = None
    transient_score: float | None = None
    quarantined: bool = False

    @property
    def ref(self) -> SampleRef:
        return SampleRef(self.collection, self.asset_id, self.sha256)


COLLECTIONS: dict[str, SampleCollection] = {
    "vcsl": SampleCollection(
        "vcsl", "Versilian Community Sample Library",
        "https://github.com/sgossner/VCSL.git", "CC0-1.0",
        "Versilian Studios LLC", 18 * GB,
        ("pitched", "orchestral", "percussion", "texture")),
    "vsco2": SampleCollection(
        "vsco2", "VSCO 2 Community Edition",
        "https://github.com/sgossner/VSCO-2-CE.git", "CC0-1.0",
        "Versilian Studios LLC", 4 * GB,
        ("pitched", "orchestral", "percussion")),
    "freepats-world": SampleCollection(
        "freepats-world", "FreePats World Percussion",
        "https://github.com/freepats/world-percussion.git", "CC0-1.0",
        "FreePats contributors", 20_000_000, ("percussion",)),
    "freepats-guitar": SampleCollection(
        "freepats-guitar", "FreePats Spanish Classical Guitar",
        "https://github.com/freepats/spanish-classical-guitar.git", "CC0-1.0",
        "Roberto and FreePats contributors", 30_000_000, ("pitched",)),
    "karoryfer-fashionbass": SampleCollection(
        "karoryfer-fashionbass", "Karoryfer Fashionbass",
        "https://github.com/sfzinstruments/karoryfer.fashionbass.git", "CC0-1.0",
        "Karoryfer Samples", 500_000_000, ("pitched",)),
    "stargate": SampleCollection(
        "stargate", "Stargate Sample Pack",
        "https://github.com/stargatedaw/stargate-sample-pack.git", "CC0-1.0",
        "Stargate sample-pack contributors", 3 * GB,
        ("percussion", "pitched", "texture")),
}
LIBRARY_TARGETS = {
    "library-core": ("freepats-world", "freepats-guitar",
                     "karoryfer-fashionbass", "stargate", "vsco2", "vcsl"),
}


def external_root() -> Path:
    return Path(os.environ.get("LEXIBEAT_LIBRARY_ROOT", EXTERNAL_DEFAULT))


def local_root() -> Path:
    configured = os.environ.get("LEXIBEAT_CACHE")
    if configured:
        return Path(configured)
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home / "lexibeat"


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*")
               if item.is_file() and not item.is_symlink())


def _sha256(path: Path, block: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def _category(path: Path) -> str:
    value = str(path).lower()
    if any(word in value for word in ("kick", "bass drum", "cajon", "conga",
                                       "bongo", "darbuka", "snare", "hat",
                                       "cymbal", "shaker", "tamb", "clap",
                                       "clave", "castanet", "percussion", "drum")):
        return "percussion"
    if any(word in value for word in ("loop", "groove", "rhythm")):
        return "loop"
    if any(word in value for word in ("ambien", "texture", "drone", "field")):
        return "texture"
    return "pitched"


def _note_from_name(path: Path) -> int | None:
    from .samples import midi
    match = re.search(r"(?<![A-Za-z])([A-Ga-g](?:#|b)?-?\d)(?!\d)", path.stem)
    if not match or "b" in match.group(1):
        return None
    try:
        return midi(match.group(1).upper())
    except (ValueError, KeyError):
        return None


def _velocity_from_name(path: str) -> int:
    match = re.search(r"(?:^|[_ -])(?:vl|v)(\d+)(?:[_ .-]|$)", path.lower())
    if match:
        return int(match.group(1))
    dynamic = re.search(r"(?:^|[_ -])(pp|p|mf|f|ff)(?:[_ .-]|$)", path.lower())
    return {"pp": 1, "p": 2, "mf": 3, "f": 4, "ff": 5}.get(
        dynamic.group(1) if dynamic else "", 1)


def infer_round_robin(path: str | Path) -> int | None:
    """Return a zero-based take number from an explicit filename marker."""
    stem = Path(path).stem.lower()
    match = re.search(
        r"(?:^|[_ .-])(?:rr|round[_ -]?robin|take)(\d+)(?=$|[_ .-])",
        stem,
    )
    return max(int(match.group(1)) - 1, 0) if match else None


def infer_articulation(path: str | Path) -> str:
    """Infer a conservative articulation label without crossing bank folders."""
    value = str(path).replace("\\", "/").lower()
    labels = (
        (("pizz", "pizzicato"), "pizzicato"),
        (("spic", "spiccato"), "spiccato"),
        (("stacc", " stac", "_stac"), "staccato"),
        (("susnv", "non-vibrato", "nonvibrato"), "sustain-non-vibrato"),
        (("susvib", "sustain vibrato", "vibrato"), "sustain-vibrato"),
        (("trem", "tremolo"), "tremolo"),
        (("muted", " mute"), "muted"),
        (("bowed", "arco"), "bowed"),
        (("pluck", "finger"), "plucked"),
        (("hard mallet", "/hard/", "_hard_"), "hard-mallet"),
        (("soft mallet", "/soft/", "_soft_"), "soft-mallet"),
        (("sustain", "sustains", "_sus_", "suslong"), "sustain"),
        ((" hit", "_hit", "-hit"), "hit"),
    )
    return next((label for markers, label in labels
                 if any(marker in value for marker in markers)), "natural")


def infer_microphone(path: str | Path) -> str:
    """Return a microphone/render perspective when the name declares one."""
    stem = Path(path).stem.lower()
    for label in ("main", "room", "player", "close", "mid", "sum",
                  "stereo", "mono", "outrigger"):
        if re.search(rf"(?:^|[_ .-]){label}(?=$|[_ .-])", stem):
            return label
    return "default"


def round_robin_group_key(asset: SampleAsset) -> tuple[str, str, str, str]:
    """Identify takes that differ only by an explicit round-robin marker."""
    relative = Path(asset.relative_path)
    stem = re.sub(
        r"(?:^|[_ .-])(?:rr|round[_ -]?robin|take)\d+(?=$|[_ .-])",
        "_rr", relative.stem.lower(),
    )
    return (asset.collection, relative.parent.as_posix().lower(), stem,
            infer_articulation(asset.relative_path))


def instrument_refs(assets: list[SampleAsset], *, min_notes: int = 6,
                    include: tuple[str, ...] | None = None,
                    include_sustained_strings: bool = False) -> list[InstrumentRef]:
    """Group catalog samples into playable directory-level multisample banks.

    Raw VCSL and VSCO collections do not always ship SFZ mappings. Their stable
    directory layout and pitch-bearing filenames still provide enough metadata
    to construct conservative note/velocity zones without treating isolated
    recordings as complete instruments.
    """
    wanted = include or ("piano", "harp", "marimba", "glock", "vibraphone",
                         "kalimba", "mbira", "nyunga", "psaltery", "ocarina",
                         "harmonica", "recorder", "flute", "oboe", "clarinet",
                         "bassoon", "strumstick", "guitar", "fashionbass",
                         "harpsichord", "organ", "celesta", "xylophone",
                         "violin", "viola", "cello", "contrabass")
    excluded = ("release", "/rel", "noise", "demo", "loop")
    groups: dict[tuple[str, str, str, str], list[SampleAsset]] = defaultdict(list)
    for asset in assets:
        parent = Path(asset.relative_path).parent.as_posix()
        lowered = parent.lower()
        searchable = f"{asset.collection}/{parent}".lower()
        string_front = any(word in lowered for word in
                           ("violin", "viola", "cello", "contrabass"))
        string_articulation = infer_articulation(asset.relative_path)
        safe_sustained_string = (
            include_sustained_strings and string_articulation in
            ("bowed", "natural", "sustain", "sustain-non-vibrato")
        )
        if (asset.midi_note is None or
                asset.duration_seconds is None or not 0.08 <= asset.duration_seconds <= 24 or
                not any(word in searchable for word in wanted) or
                any(word in lowered for word in excluded) or
                (string_front and not safe_sustained_string and
                 not any(word in lowered for word in ("pizz", "spic")))):
            continue
        articulation = asset.articulation or infer_articulation(asset.relative_path)
        groups[(asset.collection, parent, articulation,
                infer_microphone(asset.relative_path))].append(asset)

    instruments: list[InstrumentRef] = []
    for (collection, parent, articulation, microphone), values in sorted(groups.items()):
        notes = sorted({asset.midi_note for asset in values if asset.midi_note is not None})
        if len(notes) < min_notes:
            continue
        by_note_velocity: dict[tuple[int, int], list[SampleAsset]] = defaultdict(list)
        for asset in sorted(values, key=lambda row: row.relative_path):
            assert asset.midi_note is not None
            key = (asset.midi_note, _velocity_from_name(asset.relative_path))
            by_note_velocity[key].append(asset)
        velocity_values = sorted({velocity for _, velocity in by_note_velocity})
        velocity_centers = {
            value: (64 if len(velocity_values) == 1 else
                    round(index * 127 / (len(velocity_values) - 1)))
            for index, value in enumerate(velocity_values)
        }
        zones: list[InstrumentZoneRef] = []
        for (note, velocity), layer_assets in by_note_velocity.items():
            note_index = notes.index(note)
            lo_note = 0 if note_index == 0 else (notes[note_index - 1] + note + 1) // 2
            hi_note = 127 if note_index == len(notes) - 1 else (note + notes[note_index + 1]) // 2
            velocity_index = velocity_values.index(velocity)
            center = velocity_centers[velocity]
            lo_velocity = (0 if velocity_index == 0 else
                           (velocity_centers[velocity_values[velocity_index - 1]] +
                            center + 1) // 2)
            hi_velocity = (127 if velocity_index == len(velocity_values) - 1 else
                           (center + velocity_centers[
                               velocity_values[velocity_index + 1]]) // 2)
            ordered_assets = sorted(layer_assets, key=lambda row: row.relative_path)
            explicit = [
                asset for asset in ordered_assets
                if (asset.round_robin is not None or
                    infer_round_robin(asset.relative_path) is not None)
            ]
            # Ambiguous same-layer siblings are not automatically alternate takes.
            # Retain all files only when every sibling declares an RR index.
            selected_assets = (ordered_assets if len(explicit) >= 2 and
                               len(explicit) == len(ordered_assets)
                               else ordered_assets[:1])
            used_round_robins: set[int] = set()
            for asset in selected_assets:
                inferred = (asset.round_robin if asset.round_robin is not None
                            else infer_round_robin(asset.relative_path))
                round_robin = inferred if inferred is not None else 0
                while round_robin in used_round_robins:
                    round_robin += 1
                used_round_robins.add(round_robin)
                zones.append(InstrumentZoneRef(
                    asset.ref, note, lo_note, hi_note,
                    max(0, min(127, lo_velocity)),
                    max(0, min(127, hi_velocity)),
                    round_robin=round_robin, articulation=articulation))
        name = f"{collection}:{parent}#{articulation}@{microphone}"
        instruments.append(InstrumentRef(name, tuple(sorted(
            zones, key=lambda zone: (zone.root_note, zone.lo_velocity,
                                    zone.round_robin, zone.sample.asset_id)))))
    return instruments


def timbre_clusters(assets: list[SampleAsset]) -> dict[tuple[str, str], str]:
    """Cluster catalog descriptors with transparent standardized thresholds."""
    if not assets:
        return {}
    rows = np.asarray([
        [asset.spectral_centroid if asset.spectral_centroid is not None else np.nan,
         asset.transient_score if asset.transient_score is not None else np.nan,
         np.log1p(asset.duration_seconds) if asset.duration_seconds is not None else np.nan]
        for asset in assets
    ], dtype=np.float64)
    medians = np.nanmedian(rows, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    rows = np.where(np.isnan(rows), medians, rows)
    means = rows.mean(axis=0)
    scales = rows.std(axis=0)
    scales = np.where(scales > 1e-9, scales, 1.0)
    standardized = (rows - means) / scales

    def label(value: float, low: str, middle: str, high: str) -> str:
        return low if value < -0.55 else high if value > 0.55 else middle

    result: dict[tuple[str, str], str] = {}
    for asset, values in zip(assets, standardized, strict=True):
        brightness = label(values[0], "dark", "balanced", "bright")
        attack = label(values[1], "soft", "rounded", "crisp")
        length = label(values[2], "short", "medium", "long")
        result[(asset.collection, asset.asset_id)] = \
            f"{brightness}/{attack}/{length}"
    return result


def _asset_family(asset: SampleAsset) -> str:
    value = asset.relative_path.lower()
    families = (
        "piano", "guitar", "bass", "harp", "marimba", "vibraphone",
        "xylophone", "glockenspiel", "mbira", "kalimba", "strings",
        "winds", "organ", "percussion", "texture",
    )
    aliases = {
        "bass": ("fashionbass", "contrabass", "bass guitar"),
        "strings": ("violin", "viola", "cello", "string"),
        "winds": ("flute", "recorder", "clarinet", "oboe", "bassoon",
                  "ocarina", "harmonica"),
        "percussion": ("drum", "snare", "kick", "shaker", "cajon",
                       "conga", "bongo", "darbuka", "clave", "tamb"),
        "texture": ("ambient", "texture", "drone", "field"),
    }
    for family in families:
        markers = aliases.get(family, (family,))
        if any(marker in value for marker in markers):
            return family
    return asset.category


def _rejection_reasons(asset: SampleAsset) -> tuple[str, ...]:
    reasons: list[str] = []
    if not asset.license:
        reasons.append("missing_license")
    if asset.quarantined:
        reasons.append("quarantined")
    if asset.category == "pitched" and asset.midi_note is None:
        reasons.append("missing_pitch_metadata")
    if asset.duration_seconds is None:
        reasons.append("missing_duration")
    elif not 0.015 <= asset.duration_seconds <= 24.0:
        reasons.append("unsuitable_duration")
    if asset.peak is not None and asset.peak <= 1e-6:
        reasons.append("silence")
    return tuple(reasons)


class SampleLibrary:
    def __init__(self, external: Path | None = None, local: Path | None = None,
                 *, use_bundled: bool | None = None):
        self.external = Path(external or external_root())
        self.local = Path(local or local_root())
        self.use_bundled = ((external is None and local is None)
                            if use_bundled is None else use_bundled)
        self._external_explicit = external is not None or bool(
            os.environ.get("LEXIBEAT_LIBRARY_ROOT"))

    @property
    def local_catalog_path(self) -> Path:
        return self.local / "catalog.sqlite3"

    @property
    def bundled_catalog_path(self) -> Path:
        return BUNDLED_ROOT / "catalog.sqlite3"

    @property
    def catalog_path(self) -> Path:
        """Prefer a mutable local catalog, then the shipped read-only catalog."""
        return (self.local_catalog_path
                if self.local_catalog_path.exists() or not self.use_bundled
                else self.bundled_catalog_path)

    @property
    def uses_bundled_catalog(self) -> bool:
        """Whether catalog rows come from the immutable production bundle."""
        return (self.use_bundled and self.bundled_catalog_path.exists()
                and self.catalog_path == self.bundled_catalog_path)

    def expansion_policy(self) -> dict:
        """Return explicit candidate-bundle policy, or an empty control policy."""
        if not self.uses_bundled_catalog:
            return {}
        path = BUNDLED_ROOT / "manifest.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8")).get(
            "expansion_policy", {})

    def ensure_roots(self, *, require_external: bool = False) -> None:
        self.local.mkdir(parents=True, exist_ok=True)
        (self.local / "samples").mkdir(exist_ok=True)
        if self.external.exists() or self._external_explicit or require_external:
            try:
                self.external.mkdir(parents=True, exist_ok=True)
                (self.external / "collections").mkdir(exist_ok=True)
            except OSError as exc:
                if require_external:
                    raise FileNotFoundError(
                        f"External sample tier is unavailable at {self.external}; "
                        "attach the configured library volume.") from exc

    def status(self) -> dict:
        external = directory_size(self.external)
        local = directory_size(self.local)
        return {
            "external": {"path": str(self.external), "bytes": external,
                         "available": self.external.exists(),
                         "warning": external >= EXTERNAL_WARN,
                         "limit_bytes": EXTERNAL_LIMIT},
            "local": {"path": str(self.local), "bytes": local,
                      "warning": local >= LOCAL_WARN, "limit_bytes": LOCAL_LIMIT},
            "bundled": {"path": "assets/production-core/v1",
                        "bytes": directory_size(BUNDLED_ROOT),
                        "available": self.bundled_catalog_path.exists()},
        }

    def _check_budget(self, tier: str, additional: int = 0) -> None:
        root, limit = ((self.external, EXTERNAL_LIMIT) if tier == "external"
                       else (self.local, LOCAL_LIMIT))
        used = directory_size(root)
        if used + additional > limit:
            raise RuntimeError(
                f"{tier} sample budget would exceed {limit / GB:.0f} GB: "
                f"{used / GB:.2f} GB used + {additional / GB:.2f} GB requested")

    def collection_path(self, collection: str) -> Path:
        return self.external / "collections" / collection

    def download(self, collection: str) -> Path:
        """Explicitly shallow-clone or update a configured source collection."""
        if collection not in COLLECTIONS:
            raise ValueError(f"Unknown collection '{collection}'.")
        self.ensure_roots(require_external=True)
        spec = COLLECTIONS[collection]
        target = self.collection_path(collection)
        if target.exists():
            subprocess.run(["git", "-C", str(target), "pull", "--ff-only"], check=True)
            return target
        self._check_budget("external", spec.estimated_bytes)
        staging = self.external / ".staging" / f"{collection}.partial"
        staging.parent.mkdir(exist_ok=True)
        if staging.exists():
            subprocess.run(["git", "-C", str(staging), "fetch", "--depth", "1",
                            "origin"], check=True)
            subprocess.run(["git", "-C", str(staging), "reset", "--hard",
                            "origin/HEAD"], check=True)
        else:
            subprocess.run(["git", "clone", "--depth", "1", spec.repository,
                            str(staging)], check=True)
        self._check_budget("external")
        staging.rename(target)
        return target

    def _connect(self, *, write: bool = False) -> sqlite3.Connection:
        target = self.local_catalog_path if write else self.catalog_path
        if not write and target == self.bundled_catalog_path and target.exists():
            return sqlite3.connect(f"file:{target}?mode=ro", uri=True)
        self.ensure_roots()
        if (write and self.use_bundled and not self.local_catalog_path.exists()
                and self.bundled_catalog_path.exists()):
            shutil.copy2(self.bundled_catalog_path, self.local_catalog_path)
        target = self.local_catalog_path if write else self.catalog_path
        db = sqlite3.connect(target)
        db.execute("""CREATE TABLE IF NOT EXISTS collections (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, repository TEXT NOT NULL,
            revision TEXT NOT NULL, license TEXT NOT NULL, attribution TEXT NOT NULL,
            indexed_at REAL NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS assets (
            collection TEXT NOT NULL, asset_id TEXT NOT NULL, sha256 TEXT NOT NULL,
            relative_path TEXT NOT NULL, license TEXT NOT NULL, category TEXT NOT NULL,
            articulation TEXT NOT NULL, midi_note INTEGER, velocity_low INTEGER,
            velocity_high INTEGER, round_robin INTEGER, bpm REAL,
            duration_seconds REAL, sample_rate INTEGER, channels INTEGER,
            peak REAL, rms REAL, spectral_centroid REAL, transient_score REAL,
            quarantined INTEGER NOT NULL, indexed_at REAL NOT NULL,
            PRIMARY KEY(collection, asset_id))""")
        db.execute("""CREATE TABLE IF NOT EXISTS sfz_documents (
            collection TEXT NOT NULL, relative_path TEXT NOT NULL,
            zone_count INTEGER NOT NULL, unsupported_json TEXT NOT NULL,
            PRIMARY KEY(collection, relative_path))""")
        return db

    def _analyze(self, path: Path, deep: bool) -> dict:
        try:
            info = sf.info(path)
        except (RuntimeError, OSError):
            return {}
        result = {"duration_seconds": float(info.duration),
                  "sample_rate": int(info.samplerate), "channels": int(info.channels)}
        if not deep:
            return result
        try:
            audio, rate = sf.read(path, frames=min(info.frames, info.samplerate * 12),
                                  dtype="float32", always_2d=True)
        except (RuntimeError, OSError):
            return result
        mono = audio.mean(axis=1)
        if not len(mono):
            return result
        peak = float(np.abs(mono).max())
        rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
        window = np.hanning(len(mono))
        spectrum = np.abs(np.fft.rfft(mono * window))
        freqs = np.fft.rfftfreq(len(mono), 1 / rate)
        centroid = float(np.sum(freqs * spectrum) / max(np.sum(spectrum), 1e-12))
        hop = max(rate // 200, 1)
        envelope = np.maximum.reduceat(np.abs(mono), np.arange(0, len(mono), hop))
        transient = float(np.max(np.maximum(np.diff(envelope, prepend=0), 0)) /
                          max(float(envelope.max()), 1e-9))
        result.update(peak=peak, rms=rms, spectral_centroid=centroid,
                      transient_score=transient)
        return result

    def index(self, collection: str | None = None, *, deep: bool = False,
              progress: Callable[[str, int, int], None] | None = None) -> int:
        """Index audio and SFZ mappings from one collection or all installed ones."""
        installed = [collection] if collection else [key for key in COLLECTIONS
                                                      if self.collection_path(key).exists()]
        db = self._connect(write=True)
        count = 0
        try:
            for name in installed:
                if name not in COLLECTIONS:
                    raise ValueError(f"Unknown collection '{name}'.")
                root = self.collection_path(name)
                if not root.exists():
                    raise FileNotFoundError(
                        f"Collection '{name}' is not installed at {root}.")
                spec = COLLECTIONS[name]
                revision = subprocess.run(
                    ["git", "-C", str(root), "rev-parse", "HEAD"],
                    check=False, capture_output=True, text=True).stdout.strip()
                db.execute("INSERT OR REPLACE INTO collections VALUES (?,?,?,?,?,?,?)",
                           (spec.id, spec.name, spec.repository, revision,
                            spec.license, spec.attribution, time.time()))
                paths = [path for path in root.rglob("*")
                         if path.is_file() and ".git" not in path.parts]
                total_audio = sum(path.suffix.lower() in AUDIO_SUFFIXES
                                  for path in paths)
                processed_audio = 0
                for path in paths:
                    relative = path.relative_to(root).as_posix()
                    if path.suffix.lower() == ".sfz":
                        document = parse_sfz(path)
                        db.execute("INSERT OR REPLACE INTO sfz_documents VALUES (?,?,?,?)",
                                   (name, relative, len(document.zones),
                                    json.dumps(document.unsupported_opcodes)))
                        continue
                    if path.suffix.lower() not in AUDIO_SUFFIXES:
                        continue
                    processed_audio += 1
                    if progress and (processed_audio == 1 or
                                     processed_audio % 100 == 0 or
                                     processed_audio == total_audio):
                        progress(name, processed_audio, total_audio)
                    metadata = self._analyze(path, deep)
                    digest = _sha256(path)
                    asset = SampleAsset(
                        collection=name, asset_id=digest[:20], sha256=digest,
                        relative_path=relative, license=spec.license,
                        category=_category(path),
                        articulation=infer_articulation(relative),
                        midi_note=_note_from_name(path),
                        round_robin=infer_round_robin(relative),
                        quarantined=(not bool(spec.license) or
                                     "demo" in relative.lower()), **metadata)
                    values = asdict(asset)
                    db.execute("""INSERT OR REPLACE INTO assets VALUES
                        (:collection,:asset_id,:sha256,:relative_path,:license,:category,
                         :articulation,:midi_note,:velocity_low,:velocity_high,
                         :round_robin,:bpm,:duration_seconds,:sample_rate,:channels,
                         :peak,:rms,:spectral_centroid,:transient_score,:quarantined,
                         :indexed_at)""", {**values, "quarantined": int(asset.quarantined),
                                            "indexed_at": time.time()})
                    count += 1
                db.commit()
        finally:
            db.close()
        return count

    def assets(self, *, category: str | None = None,
               collections: tuple[str, ...] | None = None,
               usable_only: bool = True) -> list[SampleAsset]:
        db = self._connect()
        clauses, args = [], []
        if category:
            clauses.append("category = ?")
            args.append(category)
        if collections:
            clauses.append(f"collection IN ({','.join('?' for _ in collections)})")
            args.extend(collections)
        if usable_only:
            clauses.append("quarantined = 0")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = db.execute("SELECT collection,asset_id,sha256,relative_path,license,"
                          "category,articulation,midi_note,velocity_low,velocity_high,"
                          "round_robin,bpm,duration_seconds,sample_rate,channels,peak,rms,"
                          "spectral_centroid,transient_score,quarantined FROM assets" + where +
                          " ORDER BY collection,relative_path,asset_id",
                          args).fetchall()
        db.close()
        return [SampleAsset(*row[:-1], quarantined=bool(row[-1])) for row in rows]

    def collection_metadata(self, ids: set[str] | None = None) -> list[dict]:
        """Return the pinned provenance recorded when collections were indexed."""
        db = self._connect()
        clauses, args = "", []
        if ids:
            clauses = f" WHERE id IN ({','.join('?' for _ in ids)})"
            args.extend(sorted(ids))
        rows = db.execute(
            "SELECT id,name,repository,revision,license,attribution,indexed_at "
            "FROM collections" + clauses + " ORDER BY id", args).fetchall()
        db.close()
        return [{"id": row[0], "name": row[1], "repository": row[2],
                 "revision": row[3], "license": row[4],
                 "attribution": row[5], "indexed_at": row[6]} for row in rows]

    def asset(self, ref: SampleRef) -> SampleAsset:
        matches = [asset for asset in self.assets(
            collections=(ref.collection,), usable_only=False)
            if asset.asset_id == ref.asset_id]
        if not matches:
            raise FileNotFoundError(
                f"Unknown sample {ref.collection}:{ref.asset_id}; index its collection.")
        asset = matches[0]
        if ref.sha256 and asset.sha256 != ref.sha256:
            raise RuntimeError(f"Checksum mismatch for {ref.collection}:{ref.asset_id}.")
        return asset

    def resolve(self, ref: SampleRef) -> Path:
        db = self._connect()
        row = db.execute("SELECT relative_path,sha256 FROM assets WHERE collection=? "
                         "AND asset_id=?", (ref.collection, ref.asset_id)).fetchone()
        db.close()
        if not row:
            raise FileNotFoundError(
                f"Unknown sample {ref.collection}:{ref.asset_id}; index its collection.")
        suffix = Path(row[0]).suffix.lower()
        local = self.local / "samples" / ref.collection / f"{ref.asset_id}{suffix}"
        legacy_local = self.local / "samples" / ref.collection / ref.asset_id
        if local.exists():
            return local
        if legacy_local.exists():
            return legacy_local
        bundled = BUNDLED_ROOT / "samples" / ref.collection / f"{ref.asset_id}{suffix}"
        if self.use_bundled and bundled.exists():
            return bundled
        if self.uses_bundled_catalog:
            raise FileNotFoundError(
                "The production sample bundle is incomplete: "
                f"missing {bundled}. Repair or reattach the mounted bundle.")
        path = self.collection_path(ref.collection) / row[0]
        if not path.exists():
            raise FileNotFoundError(
                f"Sample collection '{ref.collection}' is offline. Attach the external "
                f"SSD or promote the asset into {self.local / 'samples'}.")
        if ref.sha256 and ref.sha256 != row[1]:
            raise RuntimeError(f"Checksum mismatch for {ref.collection}:{ref.asset_id}.")
        return path

    def is_promoted(self, asset: SampleAsset) -> bool:
        """Return whether an asset can resolve without the external tier."""
        suffix = Path(asset.relative_path).suffix.lower()
        base = self.local / "samples" / asset.collection / asset.asset_id
        bundled = BUNDLED_ROOT / "samples" / asset.collection / f"{asset.asset_id}{suffix}"
        return (base.with_suffix(suffix).exists() or base.exists()
                or (self.use_bundled and bundled.exists()))

    def promote(self, refs: list[SampleRef]) -> list[Path]:
        self.ensure_roots()
        promoted: list[Path] = []
        for ref in refs:
            source = self.resolve(ref)
            self._check_budget("local", source.stat().st_size)
            asset = self.asset(ref)
            target = (self.local / "samples" / ref.collection /
                      f"{ref.asset_id}{Path(asset.relative_path).suffix.lower()}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                legacy = self.local / "samples" / ref.collection / ref.asset_id
                if source == legacy:
                    legacy.rename(target)
                    promoted.append(target)
                    continue
                tmp = target.with_suffix(".partial")
                shutil.copy2(source, tmp)
                if ref.sha256 and _sha256(tmp) != ref.sha256:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f"Promotion checksum failed for {ref.asset_id}.")
                tmp.rename(target)
            promoted.append(target)
        return promoted

    def migrate_legacy_promotions(self) -> dict[str, list[str]]:
        """Rename old extensionless promotions or remove verified duplicates.

        Only 20-character SHA-derived asset names are considered. A redundant
        file is removed solely when its catalog-resolved, suffixed counterpart
        exists and both payloads have the same SHA-256 digest.
        """
        samples = self.local / "samples"
        result: dict[str, list[str]] = {"renamed": [], "removed_duplicates": []}
        if not samples.exists():
            return result
        db = self._connect()
        try:
            for legacy in sorted(samples.glob("*/*")):
                if (not legacy.is_file() or legacy.suffix or
                        not re.fullmatch(r"[0-9a-f]{20}", legacy.name)):
                    continue
                collection = legacy.parent.name
                row = db.execute(
                    "SELECT relative_path FROM assets WHERE collection=? AND asset_id=?",
                    (collection, legacy.name)).fetchone()
                if not row:
                    continue
                suffix = Path(row[0]).suffix.lower()
                if not suffix:
                    continue
                target = legacy.with_suffix(suffix)
                if not target.exists():
                    legacy.rename(target)
                    result["renamed"].append(str(target))
                elif _sha256(legacy) == _sha256(target):
                    legacy.unlink()
                    result["removed_duplicates"].append(str(legacy))
                else:
                    raise RuntimeError(
                        f"Legacy promotion differs from its suffixed copy: {legacy}")
        finally:
            db.close()
        return result

    def verify(self, collection: str | None = None) -> dict:
        assets = self.assets(collections=(collection,) if collection else None,
                             usable_only=False)
        missing = 0
        mismatched = 0
        for asset in assets:
            path = self.collection_path(asset.collection) / asset.relative_path
            if not path.exists():
                missing += 1
            elif _sha256(path) != asset.sha256:
                mismatched += 1
        return {"assets": len(assets), "missing": missing,
                "checksum_mismatches": mismatched, "ok": not missing and not mismatched}

    def report(self) -> dict:
        db = self._connect()
        rows = db.execute("SELECT collection,category,COUNT(*),SUM(COALESCE(duration_seconds,0)) "
                          "FROM assets GROUP BY collection,category ORDER BY collection,category")
        groups = [{"collection": row[0], "category": row[1], "assets": row[2],
                   "duration_seconds": row[3]} for row in rows]
        unsupported = [{"collection": row[0], "sfz": row[1], "zones": row[2],
                        "unsupported": json.loads(row[3])}
                       for row in db.execute("SELECT * FROM sfz_documents") if row[3] != "[]"]
        total, quarantined, unique = db.execute(
            "SELECT COUNT(*),SUM(quarantined),COUNT(DISTINCT sha256) FROM assets"
        ).fetchone()
        db.close()
        all_assets = self.assets(usable_only=False)
        usable_assets = [asset for asset in all_assets if not asset.quarantined]
        clusters = timbre_clusters(usable_assets)
        instruments = instrument_refs(usable_assets)

        def counts(values) -> dict[str, int]:
            result: dict[str, int] = defaultdict(int)
            for value in values:
                result[str(value)] += 1
            return dict(sorted(result.items()))

        explicit_round_robins = [asset for asset in usable_assets
                                 if (asset.round_robin is not None or
                                     infer_round_robin(asset.relative_path) is not None)]
        round_robin_groups: dict[tuple[str, str, str, str], list[SampleAsset]] = \
            defaultdict(list)
        for asset in explicit_round_robins:
            round_robin_groups[round_robin_group_key(asset)].append(asset)
        coherent_round_robins = [values for values in round_robin_groups.values()
                                 if len(values) > 1]

        bank_rows = []
        mapped_ids: set[tuple[str, str]] = set()
        for instrument in instruments:
            for zone in instrument.zones:
                mapped_ids.add((zone.sample.collection, zone.sample.asset_id))
            by_layer: dict[tuple[int, int, int], int] = defaultdict(int)
            for zone in instrument.zones:
                by_layer[(zone.root_note, zone.lo_velocity,
                          zone.hi_velocity)] += 1
            bank_rows.append({
                "name": instrument.name,
                "zones": len(instrument.zones),
                "register": [min(zone.root_note for zone in instrument.zones),
                             max(zone.root_note for zone in instrument.zones)],
                "velocity_layers": len({(zone.lo_velocity, zone.hi_velocity)
                                        for zone in instrument.zones}),
                "max_round_robins": max(by_layer.values()),
                "articulations": sorted({zone.articulation
                                         for zone in instrument.zones}),
            })

        percussion_pool = {
            (asset.collection, asset.asset_id) for asset in usable_assets
            if asset.category == "percussion" and
            asset.duration_seconds is not None and
            0.015 <= asset.duration_seconds <= 3.0
        }
        production_pool = mapped_ids | percussion_pool
        promoted = {(asset.collection, asset.asset_id) for asset in usable_assets
                    if self.is_promoted(asset)}
        rejection_counts: dict[str, int] = defaultdict(int)
        rejected_assets = []
        for asset in all_assets:
            reasons = list(_rejection_reasons(asset))
            if (not reasons and
                    (asset.collection, asset.asset_id) not in production_pool):
                reasons.append("not_mapped_to_production_pool")
            if reasons:
                for reason in reasons:
                    rejection_counts[reason] += 1
                rejected_assets.append({
                    "collection": asset.collection,
                    "asset_id": asset.asset_id,
                    "relative_path": asset.relative_path,
                    "reasons": reasons,
                })
        return {"storage": self.status(), "collections": self.collection_metadata(),
                "assets": {"total": total, "quarantined": quarantined or 0,
                           "unique_sha256": unique}, "groups": groups,
                "coverage": {
                    "families": counts(_asset_family(asset)
                                       for asset in usable_assets),
                    "articulations": counts(
                        asset.articulation or infer_articulation(asset.relative_path)
                        for asset in usable_assets),
                    "registers": counts(
                        "unknown" if asset.midi_note is None else
                        "low" if asset.midi_note < 48 else
                        "middle" if asset.midi_note < 72 else "high"
                        for asset in usable_assets),
                    "timbre_clusters": counts(clusters.values()),
                    "round_robin_assets": len(explicit_round_robins),
                    "round_robin_groups": len(coherent_round_robins),
                    "max_round_robins": max(
                        (len(values) for values in coherent_round_robins), default=1),
                },
                "instrument_banks": bank_rows,
                "utilization": {
                    "production_pool_assets": len(production_pool),
                    "mapped_instrument_assets": len(mapped_ids),
                    "percussion_pool_assets": len(percussion_pool),
                    "promoted_assets": len(promoted),
                    "promoted_not_in_production_pool": [
                        {"collection": collection, "asset_id": asset_id}
                        for collection, asset_id in sorted(promoted - production_pool)
                    ],
                },
                "rejections": {
                    "counts": dict(sorted(rejection_counts.items())),
                    "assets": rejected_assets,
                },
                "sfz_with_unsupported_opcodes": unsupported}
