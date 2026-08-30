#!/usr/bin/env python3
"""Inspect and verify the bundled LexiBeat production samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from lexibeat.library import BUNDLED_ROOT


def _sha256(source: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing production bundle manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [*manifest["catalog_assets"], *manifest["named_pack_assets"]]
    missing: list[str] = []
    mismatched: list[str] = []
    pointer_only: list[str] = []
    total_bytes = 0
    for row in rows:
        relative = row["bundle_path"]
        source = root / relative
        if not source.exists():
            missing.append(relative)
            continue
        total_bytes += source.stat().st_size
        if source.stat().st_size < 512 and source.read_bytes().startswith(
                b"version https://git-lfs.github.com/spec/v1"):
            pointer_only.append(relative)
            continue
        if _sha256(source) != row["sha256"]:
            mismatched.append(relative)

    catalog_path = root / "catalog.sqlite3"
    catalog_assets = 0
    if catalog_path.exists() and not pointer_only:
        db = sqlite3.connect(f"file:{catalog_path}?mode=ro", uri=True)
        try:
            catalog_assets = int(db.execute("SELECT COUNT(*) FROM assets").fetchone()[0])
        finally:
            db.close()
    expected_catalog = len(manifest["catalog_assets"])
    return {
        "bundle": manifest["bundle"],
        "version": manifest["version"],
        "assets": len(rows),
        "bytes": total_bytes,
        "missing": missing,
        "checksum_mismatches": mismatched,
        "lfs_pointers_without_content": pointer_only,
        "catalog_assets": catalog_assets,
        "expected_catalog_assets": expected_catalog,
        "ok": (
            not missing
            and not mismatched
            and not pointer_only
            and catalog_assets == expected_catalog
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "verify"), default="verify")
    parser.add_argument("--root", type=Path, default=BUNDLED_ROOT)
    args = parser.parse_args()
    result = verify(args.root)
    print(json.dumps(result, indent=2))
    if args.command == "verify" and not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
