"""Build an explicit checksum-locked candidate bundle from listener decisions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

from .instrument_roles import FINAL_SELECTION_WEIGHTS
from .library import SampleLibrary
from .library_audit import wave3_family
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


def accepted_wave3_policy(
    workspace: Path,
    base_policy: dict,
    family_gains: dict[str, float],
    *,
    keep_families: set[str] | None = None,
    reject_families: set[str] | None = None,
) -> tuple[dict, list[dict]]:
    """Merge the listener-approved Wave 3 subset with the accepted v2 policy."""
    proposal = json.loads(
        (workspace / "wave3" / "candidate-manifest.json").read_text(
            encoding="utf-8"))
    audition = json.loads(
        (workspace / "wave3" / "auditions" / "manifest.json").read_text(
            encoding="utf-8"))
    proposed_names = {bank["name"] for bank in proposal["banks"]}
    auditioned_names = {clip["bank"] for clip in audition["clips"]}
    if proposed_names != auditioned_names:
        missing = sorted(proposed_names - auditioned_names)
        extra = sorted(auditioned_names - proposed_names)
        raise ValueError(
            f"Wave 3 proposal/audition mismatch; missing={missing}, extra={extra}")
    families = {bank["family"] for bank in proposal["banks"]}
    finalized = keep_families is not None or reject_families is not None
    if finalized:
        keep_families = set(keep_families or ())
        reject_families = set(reject_families or ())
        overlap = sorted(keep_families & reject_families)
        unknown_decisions = sorted((keep_families | reject_families) - families)
        missing_decisions = sorted(families - keep_families - reject_families)
        if overlap or unknown_decisions or missing_decisions:
            raise ValueError(
                "Wave 3 family decisions must partition the proposal; "
                f"overlap={overlap}, unknown={unknown_decisions}, "
                f"missing={missing_decisions}")
    else:
        keep_families = set(families)
        reject_families = set()
    unknown = sorted(set(family_gains) - keep_families)
    if unknown:
        raise ValueError(
            f"Unknown or rejected Wave 3 caution families: {', '.join(unknown)}")

    wave3_banks = []
    for bank in proposal["banks"]:
        if bank["family"] not in keep_families:
            continue
        listener_gain = float(family_gains.get(bank["family"], 0.0))
        wave3_banks.append({
            **bank,
            "listener_decision": (
                "keep-with-caution" if listener_gain < 0 else "keep"),
            "listener_gain_db": listener_gain,
            "selection_weight": (
                FINAL_SELECTION_WEIGHTS.get(bank["family"], 1.0)
                if finalized else
                0.35 if bank["family"] == "harpsichord" else 1.0),
        })
    accepted = list(base_policy.get("accepted_banks", []))
    accepted_names = {bank["name"] for bank in accepted}
    accepted.extend(bank for bank in wave3_banks
                    if bank["name"] not in accepted_names)
    policy = {
        "schema_version": 1,
        "name": ("library-expansion-v3-final" if finalized
                 else "library-expansion-v3-candidate"),
        "listener_decision": ("accept-selected-wave3-families" if finalized
                              else "accept-all-wave3-auditioned-banks"),
        "include_sustained_strings": bool(
            base_policy.get("include_sustained_strings", True)),
        "accepted_banks": accepted,
        "accepted_wave3_families": sorted(keep_families),
        "rejected_wave3_families": sorted(reject_families),
        "caution_family_gains_db": dict(sorted(family_gains.items())),
        "production_status": ("listener-approved" if finalized
                              else "candidate-awaiting-wave3-speech-test"),
    }
    accepted_refs = {
        (ref["collection"], ref["asset_id"])
        for bank in wave3_banks for ref in bank["asset_refs"]
    }
    assets = [
        asset for asset in proposal["assets"]
        if (asset["collection"], asset["asset_id"]) in accepted_refs
    ]
    return policy, assets


def build_candidate_bundle(
    workspace: Path,
    output: Path,
    caution_gains: dict[str, float],
    *,
    base_bundle: Path | None = None,
    wave3: bool = False,
    family_gains: dict[str, float] | None = None,
    keep_families: set[str] | None = None,
    reject_families: set[str] | None = None,
) -> dict:
    """Merge production v1 with accepted expansion assets without replacing v1."""
    if output.exists():
        raise FileExistsError(
            f"Candidate bundle already exists at {output}; choose a new path.")
    source_catalog = workspace / "catalog.sqlite3"
    if not source_catalog.exists():
        raise FileNotFoundError(f"Missing expansion catalog: {source_catalog}")
    base_root = base_bundle or BUNDLED_ROOT
    production_manifest_path = base_root / "manifest.json"
    if not production_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing production bundle manifest: {production_manifest_path}")
    production = json.loads(production_manifest_path.read_text(encoding="utf-8"))
    if wave3:
        policy, expansion_assets = accepted_wave3_policy(
            workspace, production.get("expansion_policy", {}), family_gains or {},
            keep_families=keep_families, reject_families=reject_families)
    else:
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
    if reject_families:
        production_by_key = {
            key: row for key, row in production_by_key.items()
            if wave3_family(row["relative_path"]) not in reject_families
        }
        expansion_by_key = {
            key: row for key, row in expansion_by_key.items()
            if wave3_family(row["relative_path"]) not in reject_families
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
                source = base_root / production_row["bundle_path"]
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
            source = base_root / row["bundle_path"]
            target = staging / row["bundle_path"]
            _verified_copy(source, target, row["sha256"], link=True)
            named_rows.append(row)
            named_bytes += target.stat().st_size
        if (base_root / "licenses").exists():
            shutil.copytree(base_root / "licenses", staging / "licenses")
        manifest = {
            "schema_version": 2,
            "bundle": "lexibeat-production-core",
            "version": ("3-final" if wave3 and keep_families is not None
                        else "3-candidate" if wave3 else "2-candidate"),
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
