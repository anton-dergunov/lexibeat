"""Build an explicit checksum-locked candidate bundle from listener decisions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

from .library import SampleLibrary
from .paths import BUNDLED_ROOT


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_copy(source: Path, target: Path, expected: str, *, link: bool) -> None:
    if _sha256(source) != expected:
        raise RuntimeError(f"Source checksum mismatch: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    else:
        shutil.copy2(source, target)
    if _sha256(target) != expected:
        raise RuntimeError(f"Candidate-bundle checksum mismatch: {target}")


def _clip_banks(workspace: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in ("auditions/manifest.json",
                     "secondary-auditions/manifest.json"):
        path = workspace / relative
        if not path.exists():
            raise FileNotFoundError(f"Missing audition manifest: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for clip in manifest["clips"]:
            clip_id = str(clip.get("clip_id", clip["number"]))
            if not clip_id.startswith("S"):
                clip_id = f"{int(clip_id):02d}"
            result[clip_id.upper()] = clip["bank"]
    return result


def accepted_expansion_policy(
    workspace: Path,
    caution_gains: dict[str, float],
) -> tuple[dict, list[dict]]:
    """Record the all-accepted listener decision and conservative cautions."""
    primary = json.loads(
        (workspace / "candidate-manifest.json").read_text(encoding="utf-8"))
    secondary = json.loads(
        (workspace / "secondary-candidate-manifest.json").read_text(
            encoding="utf-8"))
    clip_banks = _clip_banks(workspace)
    unknown = sorted(set(caution_gains) - set(clip_banks))
    if unknown:
        raise ValueError(f"Unknown caution clip IDs: {', '.join(unknown)}")
    gains_by_bank = {clip_banks[key]: value
                     for key, value in caution_gains.items()}
    accepted = []
    for bank in [*primary["banks"], *secondary["banks"]]:
        listener_gain = float(gains_by_bank.get(bank["name"], 0.0))
        accepted.append({
            **bank,
            "listener_decision": "keep-with-caution" if listener_gain < 0 else "keep",
            "listener_gain_db": listener_gain,
        })
    policy = {
        "schema_version": 1,
        "name": "library-expansion-v2-candidate",
        "listener_decision": "accept-all-auditioned-banks",
        "include_sustained_strings": True,
        "accepted_banks": accepted,
        "caution_clip_gains_db": dict(sorted(caution_gains.items())),
        "production_status": "candidate-awaiting-paired-speech-test",
    }
    assets = [*primary["assets"], *secondary["assets"]]
    return policy, assets


def build_candidate_bundle(
    workspace: Path,
    output: Path,
    caution_gains: dict[str, float],
) -> dict:
    """Merge production v1 with accepted expansion assets without replacing v1."""
    if output.exists():
        raise FileExistsError(
            f"Candidate bundle already exists at {output}; choose a new path.")
    source_catalog = workspace / "catalog.sqlite3"
    if not source_catalog.exists():
        raise FileNotFoundError(f"Missing expansion catalog: {source_catalog}")
    production_manifest_path = BUNDLED_ROOT / "manifest.json"
    if not production_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing production bundle manifest: {production_manifest_path}")
    production = json.loads(production_manifest_path.read_text(encoding="utf-8"))
    policy, expansion_assets = accepted_expansion_policy(
        workspace, {key.upper(): value for key, value in caution_gains.items()})
    production_by_key = {
        (row["collection"], row["asset_id"]): row
        for row in production["catalog_assets"]
    }
    expansion_by_key = {
        (row["collection"], row["asset_id"]): row
        for row in expansion_assets
    }
    selected_keys = set(production_by_key) | set(expansion_by_key)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output.name}-", suffix=".partial", dir=output.parent))
    external = SampleLibrary(
        external=None, local=workspace, use_bundled=False)
    catalog_rows: list[dict] = []
    catalog_bytes = 0
    collections: list[dict] = []
    try:
        catalog_path = staging / "catalog.sqlite3"
        shutil.copy2(source_catalog, catalog_path)
        db = sqlite3.connect(catalog_path)
        db.row_factory = sqlite3.Row
        try:
            db.execute(
                "CREATE TEMP TABLE selected_assets(collection TEXT, asset_id TEXT, "
                "PRIMARY KEY(collection, asset_id))")
            db.executemany("INSERT INTO selected_assets VALUES (?, ?)",
                           sorted(selected_keys))
            found = db.execute(
                "SELECT COUNT(*) FROM assets WHERE EXISTS "
                "(SELECT 1 FROM selected_assets s WHERE s.collection=assets.collection "
                "AND s.asset_id=assets.asset_id)").fetchone()[0]
            if found != len(selected_keys):
                raise RuntimeError(
                    f"Catalog contains {found} of {len(selected_keys)} selected assets.")
            db.execute(
                "DELETE FROM assets WHERE NOT EXISTS "
                "(SELECT 1 FROM selected_assets s WHERE s.collection=assets.collection "
                "AND s.asset_id=assets.asset_id)")
            db.execute(
                "DELETE FROM collections WHERE id NOT IN "
                "(SELECT DISTINCT collection FROM assets)")
            db.execute("DELETE FROM sfz_documents")
            db.commit()
            rows = db.execute(
                "SELECT collection,asset_id,sha256,relative_path,license "
                "FROM assets ORDER BY collection,relative_path,asset_id").fetchall()
            collections = [{
                "id": row[0], "name": row[1], "repository": row[2],
                "revision": row[3], "license": row[4],
                "attribution": row[5],
            } for row in db.execute(
                "SELECT id,name,repository,revision,license,attribution "
                "FROM collections ORDER BY id").fetchall()]
        finally:
            db.close()

        for row in rows:
            key = (row["collection"], row["asset_id"])
            production_row = production_by_key.get(key)
            if production_row:
                source = BUNDLED_ROOT / production_row["bundle_path"]
                link = True
            else:
                source = (external.collection_path(row["collection"]) /
                          row["relative_path"])
                link = False
            suffix = Path(row["relative_path"]).suffix.lower()
            target = (staging / "samples" / row["collection"] /
                      f"{row['asset_id']}{suffix}")
            _verified_copy(source, target, row["sha256"], link=link)
            size = target.stat().st_size
            catalog_bytes += size
            catalog_rows.append({
                "collection": row["collection"], "asset_id": row["asset_id"],
                "sha256": row["sha256"], "relative_path": row["relative_path"],
                "bundle_path": target.relative_to(staging).as_posix(),
                "bytes": size, "license": row["license"],
            })

        named_rows = []
        named_bytes = 0
        for row in production["named_pack_assets"]:
            source = BUNDLED_ROOT / row["bundle_path"]
            target = staging / row["bundle_path"]
            _verified_copy(source, target, row["sha256"], link=True)
            named_rows.append(row)
            named_bytes += target.stat().st_size
        if (BUNDLED_ROOT / "licenses").exists():
            shutil.copytree(BUNDLED_ROOT / "licenses", staging / "licenses")
        manifest = {
            "schema_version": 2,
            "bundle": "lexibeat-production-core",
            "version": "2-candidate",
            "based_on": production["version"],
            "catalog_assets": catalog_rows,
            "named_pack_assets": named_rows,
            "collections": collections,
            "licenses": production["licenses"],
            "asset_count": len(catalog_rows) + len(named_rows),
            "total_audio_bytes": catalog_bytes + named_bytes,
            "expansion_policy": policy,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        staging.replace(output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
