#!/usr/bin/env python3
"""Manage the explicit tiered sample library used by LexiBeat."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from lexibeat.library import (COLLECTIONS, LIBRARY_TARGETS, SampleLibrary,
                             SampleRef, external_root, instrument_refs)
from lexibeat.library_audit import (audit_markdown, build_expansion_audit,
                                    build_secondary_manifest,
                                    render_bank_audition)
from lexibeat.library_bundle import build_candidate_bundle


def _refs(value) -> list[SampleRef]:
    found: list[SampleRef] = []
    if isinstance(value, dict):
        if {"collection", "asset_id"}.issubset(value):
            found.append(SampleRef(value["collection"], value["asset_id"],
                                   value.get("sha256", "")))
        else:
            for child in value.values():
                found.extend(_refs(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_refs(child))
    return list({(ref.collection, ref.asset_id): ref for ref in found}.values())


def _markdown(report: dict) -> str:
    lines = ["# LexiBeat sample-library report", "", "## Storage", ""]
    for tier, row in report["storage"].items():
        state = "available" if row.get("available", True) else "offline"
        limit = (f" / {row['limit_bytes'] / 1e9:.0f} GB"
                 if row.get("limit_bytes") else "")
        lines.append(f"- {tier}: {row['bytes'] / 1e9:.2f} GB{limit} "
                     f"({state}) — `{row['path']}`")
    assets = report["assets"]
    lines += ["", "## Provenance and licenses", "",
              "| Collection | Revision | License | Attribution |",
              "|---|---|---|---|"]
    for row in report["collections"]:
        lines.append(f"| {row['name']} | `{row['revision'][:12]}` | "
                     f"{row['license']} | {row['attribution']} |")
    lines += ["", "## Indexed assets", "",
              f"{assets['total']} catalog records, {assets['unique_sha256']} unique "
              f"SHA-256 payloads, and {assets['quarantined']} quarantined records.", "",
              "| Collection | Category | Assets | Duration |",
              "|---|---:|---:|---:|"]
    for row in report["groups"]:
        lines.append(f"| {row['collection']} | {row['category']} | {row['assets']} | "
                     f"{row['duration_seconds'] / 60:.1f} min |")
    unsupported = report["sfz_with_unsupported_opcodes"]
    coverage = report["coverage"]
    utilization = report["utilization"]
    lines += ["", "## Production coverage", "",
              f"{len(report['instrument_banks'])} coherent instrument banks; "
              f"{coverage['round_robin_groups']} round-robin groups with up to "
              f"{coverage['max_round_robins']} takes.", "",
              f"{utilization['production_pool_assets']} assets are eligible for "
              "production pools, including "
              f"{utilization['mapped_instrument_assets']} mapped instrument assets "
              f"and {utilization['percussion_pool_assets']} percussion assets.", "",
              "### Articulations", ""]
    lines.extend(f"- {name}: {count}" for name, count in
                 coverage["articulations"].items())
    lines += ["", "### Timbre clusters", ""]
    lines.extend(f"- {name}: {count}" for name, count in
                 coverage["timbre_clusters"].items())
    rejection_counts = report["rejections"]["counts"]
    lines += ["", "## Rejections", ""]
    if rejection_counts:
        lines.extend(f"- {name}: {count}" for name, count in
                     rejection_counts.items())
    else:
        lines.append("No catalog records were rejected.")
    lines += ["", "## SFZ compatibility", "",
              f"{len(unsupported)} documents use unsupported opcodes; inspect the JSON "
              "report before treating those instruments as fully supported.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("migrate-promotions")
    download = sub.add_parser("download")
    download.add_argument("target", choices=sorted(COLLECTIONS | LIBRARY_TARGETS.keys()))
    index = sub.add_parser("index")
    index.add_argument("collection", nargs="?", choices=sorted(COLLECTIONS))
    index.add_argument("--deep", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("collection", nargs="?", choices=sorted(COLLECTIONS))
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--out", type=Path)
    audit = sub.add_parser(
        "audit-expansion",
        help="compare the complete external library with the production bundle")
    audit.add_argument("--workspace", type=Path,
                       default=Path("out/library-expansion"))
    audit.add_argument("--refresh-index", action="store_true",
                       help="deep-index the attached external collections first")
    audit.add_argument("--target-gb", type=float, default=10.0,
                       help="maximum proposed payload; whole banks are never split")
    audition = sub.add_parser(
        "audition-expansion",
        help="render isolated listening probes from an expansion proposal")
    audition.add_argument("--workspace", type=Path,
                          default=Path("out/library-expansion"))
    audition.add_argument("--out-dir", type=Path)
    audition.add_argument("--wave", choices=("primary", "secondary"),
                          default="primary")
    integrate = sub.add_parser(
        "integrate-expansion",
        help="build a separate candidate bundle from accepted audition banks")
    integrate.add_argument("--workspace", type=Path,
                           default=Path("out/library-expansion"))
    integrate.add_argument("--out", type=Path,
                           default=Path("out/library-expansion/candidate-v2"))
    integrate.add_argument("--accept-all", action="store_true",
                           help="confirm that every auditioned bank was accepted")
    integrate.add_argument(
        "--caution", action="append", default=[], metavar="CLIP_ID:GAIN_DB",
        help="record a retained clip with conservative gain, e.g. S21:-5")
    promote = sub.add_parser("promote")
    promote.add_argument("bed_specs", type=Path, nargs="+")
    playlist = sub.add_parser("playlist")
    playlist.add_argument("collection", choices=sorted(COLLECTIONS))
    playlist.add_argument("--category", choices=("pitched", "percussion", "loop",
                                                  "texture"))
    playlist.add_argument("--limit", type=int)
    playlist.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    library = SampleLibrary()
    if args.command == "status":
        result = library.status()
    elif args.command == "migrate-promotions":
        result = library.migrate_legacy_promotions()
    elif args.command == "download":
        names = LIBRARY_TARGETS.get(args.target, (args.target,))
        result = {name: str(library.download(name)) for name in names}
    elif args.command == "index":
        result = {"indexed": library.index(args.collection, deep=args.deep)}
    elif args.command == "verify":
        result = library.verify(args.collection)
    elif args.command == "promote":
        refs = []
        for path in args.bed_specs:
            refs.extend(_refs(json.loads(path.read_text(encoding="utf-8"))))
        refs = list({(ref.collection, ref.asset_id): ref for ref in refs}.values())
        result = {"promoted": [str(path) for path in library.promote(refs)],
                  "refs": [asdict(ref) for ref in refs]}
    elif args.command == "playlist":
        assets = library.assets(category=args.category,
                                collections=(args.collection,))
        if args.limit is not None:
            assets = assets[:args.limit]
        paths = [library.resolve(asset.ref) for asset in assets]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("#EXTM3U\n" + "\n".join(str(path) for path in paths) +
                            "\n", encoding="utf-8")
        result = {"playlist": str(args.out), "collection": args.collection,
                  "category": args.category, "files": len(paths)}
    elif args.command == "audit-expansion":
        if args.target_gb <= 0:
            raise ValueError("--target-gb must be positive")
        args.workspace.mkdir(parents=True, exist_ok=True)
        external_library = SampleLibrary(
            external=external_root(), local=args.workspace, use_bundled=False)
        if args.refresh_index:
            indexed = external_library.index(
                deep=True,
                progress=lambda name, current, total: print(
                    f"Indexing {name}: {current:,}/{total:,}", flush=True),
            )
        elif not external_library.local_catalog_path.exists():
            raise FileNotFoundError(
                f"No external audit catalog at {external_library.local_catalog_path}; "
                "rerun with --refresh-index.")
        else:
            indexed = None
        external_assets = external_library.assets(usable_only=False)
        production_assets = SampleLibrary().assets(usable_only=False)
        sizes = {}
        for asset in external_assets:
            path = external_library.collection_path(
                asset.collection) / asset.relative_path
            sizes[(asset.collection, asset.asset_id)] = (
                path.stat().st_size if path.exists() else 0)
        instruments = instrument_refs(
            [asset for asset in external_assets if not asset.quarantined],
            include_sustained_strings=True)
        expansion, proposal = build_expansion_audit(
            external_assets, production_assets, instruments, sizes,
            target_bytes=int(args.target_gb * 1_000_000_000))
        audit_json = args.workspace / "audit.json"
        proposal_json = args.workspace / "candidate-manifest.json"
        report_md = args.workspace / "report.md"
        audit_json.write_text(json.dumps(expansion, indent=2) + "\n",
                              encoding="utf-8")
        proposal_json.write_text(json.dumps(proposal, indent=2) + "\n",
                                 encoding="utf-8")
        report_md.write_text(audit_markdown(expansion, proposal), encoding="utf-8")
        result = {
            "indexed": indexed,
            "external_assets": expansion["catalogs"]["external_assets"],
            "production_assets": expansion["catalogs"]["production_assets"],
            "bank_counts": expansion["bank_counts"],
            "proposed_banks": len(proposal["banks"]),
            "proposed_assets": len(proposal["assets"]),
            "proposed_bytes": proposal["selected_bytes"],
            "copied_audio": False,
            "audit": str(audit_json),
            "candidate_manifest": str(proposal_json),
            "report": str(report_md),
        }
    elif args.command == "audition-expansion":
        primary_path = args.workspace / "candidate-manifest.json"
        if not primary_path.exists():
            raise FileNotFoundError(
                f"Missing {primary_path}; run audit-expansion first.")
        external_library = SampleLibrary(
            external=external_root(), local=args.workspace, use_bundled=False)
        assets = external_library.assets()
        primary = json.loads(primary_path.read_text(encoding="utf-8"))
        if args.wave == "secondary":
            audit_path = args.workspace / "audit.json"
            if not audit_path.exists():
                raise FileNotFoundError(
                    f"Missing {audit_path}; run audit-expansion first.")
            sizes = {}
            for asset in assets:
                source = external_library.collection_path(
                    asset.collection) / asset.relative_path
                sizes[(asset.collection, asset.asset_id)] = (
                    source.stat().st_size if source.exists() else 0)
            proposal = build_secondary_manifest(
                json.loads(audit_path.read_text(encoding="utf-8")), primary,
                assets, SampleLibrary().assets(usable_only=False), sizes)
            manifest_path = args.workspace / "secondary-candidate-manifest.json"
            manifest_path.write_text(json.dumps(proposal, indent=2) + "\n",
                                     encoding="utf-8")
        else:
            proposal = primary
            manifest_path = primary_path
        instruments = instrument_refs(
            assets, include_sustained_strings=True)
        instruments_by_name = {instrument.name: instrument
                               for instrument in instruments}
        default_dir = ("secondary-auditions" if args.wave == "secondary"
                       else "auditions")
        out_dir = args.out_dir or args.workspace / default_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for number, bank in enumerate(proposal["banks"], 1):
            clip_id = (f"S{number:02d}" if args.wave == "secondary"
                       else str(number))
            slug = re.sub(r"[^a-z0-9]+", "-", bank["family"].lower()).strip("-")
            filename_number = (f"s{number:02d}" if args.wave == "secondary"
                               else f"{number:02d}")
            path = out_dir / f"{filename_number}-{slug}.wav"
            audio = render_bank_audition(
                bank, external_library, instruments_by_name)
            sf.write(path, audio, 44_100)
            rows.append({
                "number": number, "clip_id": clip_id, "file": path.name,
                "bank": bank["name"],
                "family": bank["family"], "role": bank.get("role"),
                "status": bank["status"],
                "warnings": bank["review_warnings"],
                "mapping_limitations": bank.get("mapping_limitations", []),
                "seconds": len(audio) / 44_100,
                "peak": float(np.abs(audio).max()),
                "finite": bool(np.isfinite(audio).all()),
            })
        audition_manifest = {
            "schema_version": 1, "source_proposal": str(manifest_path),
            "wave": args.wave,
            "purpose": "isolated pre-promotion speech-safety screening",
            "clips": rows,
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(audition_manifest, indent=2) + "\n", encoding="utf-8")
        ratings_path = out_dir / "ratings.csv"
        if not ratings_path.exists():
            with ratings_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "clip_id", "file", "family", "bank",
                    "piercing_or_high_pitch_1_5", "attack_distraction_1_5",
                    "perceived_level_consistency_1_5", "naturalness_1_5",
                    "background_suitability_1_5", "keep", "notes",
                ])
                for row in rows:
                    writer.writerow([
                        row["clip_id"], row["file"], row["family"], row["bank"],
                        "", "", "", "", "", "", "",
                    ])
        wave_title = "Secondary (Wave 2)" if args.wave == "secondary" else "Primary"
        guide = [
            f"# {wave_title} expansion-bank audition guide", "",
            "These are isolated, level-preserving probes read directly from the "
            "external library. No samples have been promoted.", "",
            "Listen to the complete set independently before comparing individual "
            "banks. Reject any piercing high note, bottle-like or cosmic effect, "
            "uneven perceived level, distracting attack, or sound that would pull "
            "attention away from spoken words.", "",
            "A low score means the problem is strong; a high score means the bank is "
            "safe or suitable. Record the decision in `ratings.csv`.", "",
            "| # | File | Family/role | Status | Bank | Automated flags |",
            "|---:|---|---|---|---|---|",
        ]
        for row in rows:
            family = row["family"] + (f"/{row['role']}" if row["role"] else "")
            flags = [*row["warnings"], *row["mapping_limitations"]]
            guide.append(
                f"| {row['clip_id']} | `{row['file']}` | {family} | "
                f"{row['status']} | `{row['bank']}` | {', '.join(flags) or 'none'} |")
        (out_dir / "listening-guide.md").write_text(
            "\n".join(guide) + "\n", encoding="utf-8")
        result = {
            "wave": args.wave, "clips": len(rows), "out_dir": str(out_dir),
            "finite": all(row["finite"] for row in rows),
            "max_peak": max((row["peak"] for row in rows), default=0.0),
            "copied_audio": False,
        }
    elif args.command == "integrate-expansion":
        if not args.accept_all:
            raise ValueError(
                "integrate-expansion requires --accept-all; partial decisions "
                "must be recorded explicitly before building a bundle.")
        caution_gains = {}
        for value in args.caution:
            try:
                clip_id, gain = value.rsplit(":", 1)
                caution_gains[clip_id.upper()] = float(gain)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid --caution '{value}'; expected CLIP_ID:GAIN_DB.") from exc
        if any(gain >= 0 for gain in caution_gains.values()):
            raise ValueError("Caution gains must be negative decibel values.")
        manifest = build_candidate_bundle(
            args.workspace, args.out, caution_gains)
        policy = manifest["expansion_policy"]
        result = {
            "bundle": str(args.out), "version": manifest["version"],
            "accepted_banks": len(policy["accepted_banks"]),
            "caution_banks": sum(
                bank["listener_decision"] == "keep-with-caution"
                for bank in policy["accepted_banks"]),
            "catalog_assets": len(manifest["catalog_assets"]),
            "asset_count": manifest["asset_count"],
            "total_audio_bytes": manifest["total_audio_bytes"],
            "production_v1_replaced": False,
        }
    else:
        result = library.report()
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
