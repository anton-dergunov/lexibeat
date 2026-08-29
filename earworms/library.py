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
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .sfz import parse as parse_sfz


EXTERNAL_DEFAULT = Path("/Volumes/EXTSSD_SAND/downloaded_music/earworms-library")
LOCAL_DEFAULT = Path("/Users/anton/downloaded_music/earworms-cache")
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
    "stargate": SampleCollection(
        "stargate", "Stargate Sample Pack",
        "https://github.com/stargatedaw/stargate-sample-pack.git", "CC0-1.0",
        "Stargate sample-pack contributors", 3 * GB,
        ("percussion", "pitched", "texture")),
    "open-samples": SampleCollection(
        "open-samples", "Open Samples",
        "https://github.com/pumodi/open-samples.git",
        "Open Samples Permissive Use Public License v2",
        "Jeffrey Brice and Open Samples contributors", 120 * GB,
        ("pitched", "percussion", "keyboard", "synth", "texture"), False),
}
LIBRARY_TARGETS = {
    "library-core": ("freepats-world", "stargate", "vsco2", "vcsl"),
    "library-full": tuple(COLLECTIONS),
}


def external_root() -> Path:
    return Path(os.environ.get("EARWORMS_LIBRARY_ROOT", EXTERNAL_DEFAULT))


def local_root() -> Path:
    return Path(os.environ.get("EARWORMS_CACHE", LOCAL_DEFAULT))


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


class SampleLibrary:
    def __init__(self, external: Path | None = None, local: Path | None = None):
        self.external = Path(external or external_root())
        self.local = Path(local or local_root())

    @property
    def catalog_path(self) -> Path:
        return self.local / "catalog.sqlite3"

    def ensure_roots(self) -> None:
        self.external.mkdir(parents=True, exist_ok=True)
        self.local.mkdir(parents=True, exist_ok=True)
        (self.external / "collections").mkdir(exist_ok=True)
        (self.local / "samples").mkdir(exist_ok=True)

    def status(self) -> dict:
        external = directory_size(self.external)
        local = directory_size(self.local)
        return {
            "external": {"path": str(self.external), "bytes": external,
                         "warning": external >= EXTERNAL_WARN,
                         "limit_bytes": EXTERNAL_LIMIT},
            "local": {"path": str(self.local), "bytes": local,
                      "warning": local >= LOCAL_WARN, "limit_bytes": LOCAL_LIMIT},
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
        self.ensure_roots()
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

    def _connect(self) -> sqlite3.Connection:
        self.ensure_roots()
        db = sqlite3.connect(self.catalog_path)
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

    def index(self, collection: str | None = None, *, deep: bool = False) -> int:
        """Index audio and SFZ mappings from one collection or all installed ones."""
        installed = [collection] if collection else [key for key in COLLECTIONS
                                                      if self.collection_path(key).exists()]
        db = self._connect()
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
                for path in root.rglob("*"):
                    if not path.is_file() or ".git" in path.parts:
                        continue
                    relative = path.relative_to(root).as_posix()
                    if path.suffix.lower() == ".sfz":
                        document = parse_sfz(path)
                        db.execute("INSERT OR REPLACE INTO sfz_documents VALUES (?,?,?,?)",
                                   (name, relative, len(document.zones),
                                    json.dumps(document.unsupported_opcodes)))
                        continue
                    if path.suffix.lower() not in AUDIO_SUFFIXES:
                        continue
                    metadata = self._analyze(path, deep)
                    digest = _sha256(path)
                    asset = SampleAsset(
                        collection=name, asset_id=digest[:20], sha256=digest,
                        relative_path=relative, license=spec.license,
                        category=_category(path), midi_note=_note_from_name(path),
                        quarantined=not bool(spec.license), **metadata)
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
                          "spectral_centroid,transient_score,quarantined FROM assets" + where,
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
        local = self.local / "samples" / ref.collection / ref.asset_id
        if local.exists():
            return local
        db = self._connect()
        row = db.execute("SELECT relative_path,sha256 FROM assets WHERE collection=? "
                         "AND asset_id=?", (ref.collection, ref.asset_id)).fetchone()
        db.close()
        if not row:
            raise FileNotFoundError(
                f"Unknown sample {ref.collection}:{ref.asset_id}; index its collection.")
        path = self.collection_path(ref.collection) / row[0]
        if not path.exists():
            raise FileNotFoundError(
                f"Sample collection '{ref.collection}' is offline. Attach the external "
                f"SSD or promote the asset into {self.local / 'samples'}.")
        if ref.sha256 and ref.sha256 != row[1]:
            raise RuntimeError(f"Checksum mismatch for {ref.collection}:{ref.asset_id}.")
        return path

    def promote(self, refs: list[SampleRef]) -> list[Path]:
        self.ensure_roots()
        promoted: list[Path] = []
        for ref in refs:
            source = self.resolve(ref)
            self._check_budget("local", source.stat().st_size)
            target = self.local / "samples" / ref.collection / ref.asset_id
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                tmp = target.with_suffix(".partial")
                shutil.copy2(source, tmp)
                if ref.sha256 and _sha256(tmp) != ref.sha256:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f"Promotion checksum failed for {ref.asset_id}.")
                tmp.rename(target)
            promoted.append(target)
        return promoted

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
        return {"storage": self.status(), "collections": self.collection_metadata(),
                "assets": {"total": total, "quarantined": quarantined or 0,
                           "unique_sha256": unique}, "groups": groups,
                "sfz_with_unsupported_opcodes": unsupported}
