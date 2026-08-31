#!/usr/bin/env python3
"""Manage the explicit tiered sample library used by LexiBeat."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from lexibeat.library import COLLECTIONS, LIBRARY_TARGETS, SampleLibrary, SampleRef


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
    else:
        result = library.report()
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
