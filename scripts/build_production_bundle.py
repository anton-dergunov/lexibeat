#!/usr/bin/env python3
"""Build the checksum-locked LexiBeat production sample bundle.

This command performs no downloads. It copies only checksum-verified promoted
assets from supported redistributable collections and the explicit named packs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexibeat.library import COLLECTIONS, external_root, local_root
from lexibeat.samples import PACKS, cache_dir as pack_cache_dir


def sha256(source: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def verified_copy(source: Path, target: Path, expected: str) -> None:
    if sha256(source) != expected:
        raise RuntimeError(f"Source checksum mismatch: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256(target) == expected:
        return
    temporary = target.with_name(target.name + ".partial")
    shutil.copy2(source, temporary)
    if sha256(temporary) != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Copied checksum mismatch: {target}")
    temporary.replace(target)


def copy_license_files(library_root: Path, output: Path) -> list[dict]:
    rows: list[dict] = []
    for collection_id, collection in COLLECTIONS.items():
        source_root = library_root / "collections" / collection_id
        candidates = sorted(
            item for item in source_root.iterdir()
            if item.is_file() and item.name.lower().startswith(("license", "copying"))
        )
        if not candidates:
            raise FileNotFoundError(f"No upstream license file for {collection_id}")
        source = candidates[0]
        target = output / "licenses" / f"{collection_id}{source.suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes().rstrip(b"\r\n") + b"\n")
        target.chmod(0o644)
        rows.append({
            "collection": collection_id,
            "license": collection.license,
            "attribution": collection.attribution,
            "repository": collection.repository,
            "file": target.relative_to(output).as_posix(),
        })
    return rows


def build_catalog_bundle(cache_root: Path, output: Path) -> tuple[list[dict], int]:
    source_catalog = cache_root / "catalog.sqlite3"
    if not source_catalog.exists():
        raise FileNotFoundError(f"Missing source catalog: {source_catalog}")
    source_db = sqlite3.connect(source_catalog)
    source_db.row_factory = sqlite3.Row
    selected: list[dict] = []
    total_bytes = 0
    try:
        for collection_id in sorted(COLLECTIONS):
            promoted = cache_root / "samples" / collection_id
            if not promoted.exists():
                continue
            for source in sorted(item for item in promoted.iterdir() if item.is_file()):
                asset_id = source.stem
                row = source_db.execute(
                    "SELECT * FROM assets WHERE collection=? AND asset_id=?",
                    (collection_id, asset_id),
                ).fetchone()
                if row is None:
                    continue
                values = dict(row)
                if values["license"] != COLLECTIONS[collection_id].license:
                    raise RuntimeError(f"Conflicting license for {collection_id}:{asset_id}")
                if values["quarantined"]:
                    raise RuntimeError(f"Quarantined asset cannot be bundled: {asset_id}")
                target = output / "samples" / collection_id / source.name
                verified_copy(source, target, values["sha256"])
                total_bytes += target.stat().st_size
                selected.append({
                    "collection": collection_id,
                    "asset_id": asset_id,
                    "sha256": values["sha256"],
                    "relative_path": values["relative_path"],
                    "bundle_path": target.relative_to(output).as_posix(),
                    "bytes": target.stat().st_size,
                    "license": values["license"],
                })
    finally:
        source_db.close()

    if not selected:
        raise RuntimeError("No promoted catalog assets were found")
    catalog_target = output / "catalog.sqlite3"
    temporary = output / "catalog.sqlite3.partial"
    temporary.unlink(missing_ok=True)
    shutil.copy2(source_catalog, temporary)
    bundle_db = sqlite3.connect(temporary)
    try:
        bundle_db.execute(
            "CREATE TEMP TABLE bundled_assets(collection TEXT, asset_id TEXT, "
            "PRIMARY KEY(collection, asset_id))"
        )
        bundle_db.executemany(
            "INSERT INTO bundled_assets VALUES (?, ?)",
            [(row["collection"], row["asset_id"]) for row in selected],
        )
        bundle_db.execute(
            "DELETE FROM assets WHERE NOT EXISTS "
            "(SELECT 1 FROM bundled_assets b WHERE b.collection=assets.collection "
            "AND b.asset_id=assets.asset_id)"
        )
        bundle_db.execute(
            "DELETE FROM collections WHERE id NOT IN "
            "(SELECT DISTINCT collection FROM assets)"
        )
        bundle_db.execute("DELETE FROM sfz_documents")
        bundle_db.commit()
        bundle_db.execute("VACUUM")
    finally:
        bundle_db.close()
    temporary.replace(catalog_target)
    return selected, total_bytes


def build_named_packs(pack_cache: Path, output: Path) -> tuple[list[dict], int]:
    rows: list[dict] = []
    total_bytes = 0
    for pack_id, pack in sorted(PACKS.items()):
        for entry in pack.entries():
            source = pack_cache / "samples" / pack_id / entry.filename
            if not source.exists():
                raise FileNotFoundError(f"Missing named-pack sample: {source}")
            digest = sha256(source)
            target = output / "packs" / pack_id / entry.filename
            verified_copy(source, target, digest)
            total_bytes += target.stat().st_size
            rows.append({
                "collection": f"pack:{pack_id}",
                "asset_id": entry.filename,
                "sha256": digest,
                "bundle_path": target.relative_to(output).as_posix(),
                "bytes": target.stat().st_size,
                "license": pack.license,
                "attribution": pack.attribution,
                "homepage": pack.homepage,
            })
    return rows, total_bytes


def catalog_collections(catalog_path: Path) -> list[dict]:
    db = sqlite3.connect(f"file:{catalog_path}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT id,name,repository,revision,license,attribution "
            "FROM collections ORDER BY id"
        ).fetchall()
    finally:
        db.close()
    return [
        {
            "id": row[0], "name": row[1], "repository": row[2],
            "revision": row[3], "license": row[4], "attribution": row[5],
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root", type=Path, default=local_root(),
    )
    parser.add_argument(
        "--pack-cache", type=Path, default=pack_cache_dir().parent,
    )
    parser.add_argument(
        "--library-root", type=Path, default=external_root(),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("assets/production-core/v1"),
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    catalog_assets, catalog_bytes = build_catalog_bundle(args.cache_root, args.out)
    named_assets, named_bytes = build_named_packs(args.pack_cache, args.out)
    licenses = copy_license_files(args.library_root, args.out)
    manifest = {
        "schema_version": 1,
        "bundle": "lexibeat-production-core",
        "version": "1",
        "catalog_assets": catalog_assets,
        "named_pack_assets": named_assets,
        "collections": catalog_collections(args.out / "catalog.sqlite3"),
        "licenses": licenses,
        "asset_count": len(catalog_assets) + len(named_assets),
        "total_audio_bytes": catalog_bytes + named_bytes,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Built {manifest['asset_count']} checksum-verified assets "
        f"({manifest['total_audio_bytes'] / 1_000_000_000:.2f} GB) in {args.out}"
    )


if __name__ == "__main__":
    main()
