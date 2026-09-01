"""Read-only ranking and speech-safety policy for sample-bundle expansion."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import median

import numpy as np

from .library import (InstrumentRef, SampleAsset, infer_articulation,
                      infer_microphone, SampleLibrary, SampleRef)
from .instruments import CatalogMultiSampleInstrument, CatalogSampleInstrument, SR

TARGET_NORMALIZED_RMS_DB = -22.0
SAFE_REGISTERS = {
    "piano": (48, 88),
    "strings": (43, 84),
    "guitar": (40, 84),
    "bass": (24, 60),
}
FAMILY_PRIORITY = {"piano": 4, "strings": 3, "guitar": 2, "bass": 1}
MIN_SAFE_NOTES = {"piano": 16, "strings": 8, "guitar": 16, "bass": 7}
MAX_NOTE_GAP = {"piano": 4, "strings": 7, "guitar": 5, "bass": 5}

ATTENTION_WORDS = (
    "bottle", "cosmic", "crystal", "chime", "glock", "bell", "whistle",
    "xylophone", "vibraphone", "recorder", "ocarina", "fx", "effect",
    "synth", "electrophone", "tx81z", "fm piano",
)
EXCLUDED_WORDS = ("release", "/rel", "noise", "demo", "loop")

WAVE3_SAFE_REGISTERS = {
    "accordion": (48, 79),
    "recorder": (48, 79),
    "ocarina": (60, 79),
    "flute": (60, 84),
    "clarinet": (50, 82),
    "oboe": (58, 81),
    "bassoon": (34, 67),
    "harmonica": (55, 79),
    "harp": (40, 84),
    "plucked-string": (40, 84),
    "organ": (36, 79),
    "harpsichord": (40, 79),
    "lamellophone": (48, 79),
    "marimba": (43, 84),
}


def _db(value: float | None) -> float | None:
    if value is None or value <= 1e-12:
        return None
    return 20.0 * math.log10(value)


def _percentile(values: list[float], amount: float) -> float | None:
    return float(np.percentile(values, amount)) if values else None


def _family(name: str) -> str:
    value = name.lower()
    if any(word in value for word in ("electrophone", "tx81z", "fm piano")):
        return "electronic"
    if "piano" in value:
        return "piano"
    if "harp" in value:
        return "other"
    if "fashionbass" in value or "contrabass" in value or "bass guitar" in value:
        return "bass"
    if any(word in value for word in ("violin", "viola", "cello", "strings")):
        return "strings"
    if "guitar" in value:
        return "guitar"
    return "other"


def _largest_gap(notes: list[int]) -> int:
    return max((right - left for left, right in zip(notes, notes[1:])), default=0)


def _normalized_rms_db(asset: SampleAsset) -> float | None:
    if asset.rms is None or asset.peak is None or asset.rms <= 0 or asset.peak <= 0:
        return None
    return _db(asset.rms / asset.peak)


def _register_gain(assets: list[SampleAsset]) -> float:
    middle = [_normalized_rms_db(asset) for asset in assets
              if asset.midi_note is not None and 48 <= asset.midi_note < 72]
    high = [_normalized_rms_db(asset) for asset in assets
            if asset.midi_note is not None and asset.midi_note >= 72]
    middle = [value for value in middle if value is not None]
    high = [value for value in high if value is not None]
    if not middle or not high:
        return 0.0
    excess = median(high) - median(middle)
    return round(-min(max(excess, 0.0), 6.0), 2)


def _asset_metrics(assets: list[SampleAsset]) -> dict:
    normalized = [_normalized_rms_db(asset) for asset in assets]
    normalized = [value for value in normalized if value is not None]
    centroids = [asset.spectral_centroid for asset in assets
                 if asset.spectral_centroid is not None]
    transients = [asset.transient_score for asset in assets
                  if asset.transient_score is not None]
    median_rms = median(normalized) if normalized else None
    spread = ((_percentile(normalized, 90) or 0.0) -
              (_percentile(normalized, 10) or 0.0)) if normalized else None
    return {
        "median_normalized_rms_db": round(median_rms, 2) if median_rms is not None else None,
        "normalized_rms_p90_p10_db": round(spread, 2) if spread is not None else None,
        "spectral_centroid_p90_hz": round(_percentile(centroids, 90), 1)
        if centroids else None,
        "transient_p90": round(_percentile(transients, 90), 3)
        if transients else None,
        "recommended_bank_gain_db": round(min(
            0.0, TARGET_NORMALIZED_RMS_DB - median_rms), 2)
        if median_rms is not None else 0.0,
        "recommended_high_register_gain_db": _register_gain(assets),
    }


def evaluate_instrument_bank(
    instrument: InstrumentRef,
    assets_by_key: dict[tuple[str, str], SampleAsset],
    production_sha256: set[str],
    asset_sizes: dict[tuple[str, str], int],
) -> dict:
    """Score one coherent bank without accepting it into production."""
    family = _family(instrument.name)
    keys = list(dict.fromkeys(
        (zone.sample.collection, zone.sample.asset_id) for zone in instrument.zones
    ))
    assets = [assets_by_key[key] for key in keys if key in assets_by_key]
    new_assets = [asset for asset in assets if asset.sha256 not in production_sha256]
    register = SAFE_REGISTERS.get(family, (0, 127))
    safe_zones = [zone for zone in instrument.zones
                  if register[0] <= zone.root_note <= register[1]]
    safe_notes = sorted({zone.root_note for zone in safe_zones})
    velocity_layers = len({(zone.lo_velocity, zone.hi_velocity)
                           for zone in safe_zones})
    rr_counts: dict[tuple[int, int, int], int] = defaultdict(int)
    for zone in safe_zones:
        rr_counts[(zone.root_note, zone.lo_velocity, zone.hi_velocity)] += 1
    max_round_robins = max(rr_counts.values(), default=1)
    metrics = _asset_metrics(assets)
    name = instrument.name.lower()
    reasons: list[str] = []
    warnings: list[str] = []
    limitations: list[str] = []

    if family not in FAMILY_PRIORITY:
        reasons.append("outside_natural_priority_families")
    if not new_assets:
        reasons.append("already_in_production_bundle")
    if any(word in name for word in EXCLUDED_WORDS):
        reasons.append("release_noise_demo_or_loop")
    if family == "electronic":
        reasons.append("electronic_approximation_not_a_natural_expansion")
    if any(word in name for word in ATTENTION_WORDS):
        reasons.append("attention_grabbing_timbre_family")
    if family in FAMILY_PRIORITY:
        if len(safe_notes) < MIN_SAFE_NOTES[family]:
            if len(safe_notes) < 6:
                reasons.append("insufficient_safe_register_notes")
            else:
                warnings.append("limited_safe_register_coverage")
        if _largest_gap(safe_notes) > MAX_NOTE_GAP[family]:
            if _largest_gap(safe_notes) > 12:
                reasons.append("excessive_transposition_gap")
            else:
                warnings.append("wide_transposition_gap_requires_listening")
    if any((asset.rms or 0.0) <= 1e-7 for asset in assets):
        reasons.append("silent_or_near_silent_asset")

    spread = metrics["normalized_rms_p90_p10_db"]
    centroid = metrics["spectral_centroid_p90_hz"]
    transient = metrics["transient_p90"]
    high_assets = [asset for asset in assets if (asset.midi_note or 0) >= 72]
    piercing = [asset for asset in high_assets
                if (asset.spectral_centroid or 0.0) >= 5000]
    if high_assets and len(piercing) / len(high_assets) >= 0.15:
        warnings.append("high_register_brightness_requires_listening")
    if centroid is not None and centroid >= 5500:
        warnings.append("bright_spectral_tail_requires_listening")
    if transient is not None and transient >= 0.75:
        warnings.append("sharp_attack_may_compete_with_consonants")
    if spread is not None and spread >= 14:
        warnings.append("large_perceived_level_spread_requires_zone_gains")
    if family == "strings" and "vibrato" in name and "non-vibrato" not in name:
        warnings.append("vibrato_may_be_too_foreground_for_repeated_background")
    if "spic" in name:
        warnings.append("short_string_attack_requires_speech_masking_test")
    if velocity_layers < 2:
        limitations.append("single_velocity_layer")
    if max_round_robins < 2:
        limitations.append("no_multi_take_round_robin")

    status = "rejected" if reasons else "review" if warnings else "candidate"
    priority = FAMILY_PRIORITY.get(family, 0)
    score = (priority * 1000 + min(len(safe_notes), 100) * 8 +
             min(velocity_layers, 5) * 45 + min(max_round_robins, 4) * 25 -
             len(warnings) * 35)
    return {
        "kind": "pitched",
        "name": instrument.name,
        "family": family,
        "priority": priority,
        "score": score,
        "status": status,
        "rejection_reasons": reasons,
        "review_warnings": warnings,
        "mapping_limitations": limitations,
        "safe_register": list(register),
        "source_register": [min((zone.root_note for zone in instrument.zones), default=0),
                            max((zone.root_note for zone in instrument.zones), default=0)],
        "safe_note_count": len(safe_notes),
        "largest_safe_note_gap": _largest_gap(safe_notes),
        "velocity_layers": velocity_layers,
        "max_round_robins": max_round_robins,
        "articulations": sorted({zone.articulation for zone in safe_zones}),
        "assets": len(assets),
        "new_assets": len(new_assets),
        "new_bytes": sum(asset_sizes.get((asset.collection, asset.asset_id), 0)
                         for asset in new_assets),
        "metrics": metrics,
        "_asset_keys": keys,
    }


def _percussion_role(value: str) -> str:
    value = value.lower()
    if any(word in value for word in ("bass drum", "cajon", "frame drum",
                                      "low conga", "low bongo")):
        return "low"
    if any(word in value for word in ("shaker", "tamb", "cabasa", "brush")):
        return "high"
    return "mid"


def evaluate_percussion_groups(
    assets: list[SampleAsset],
    production_sha256: set[str],
    asset_sizes: dict[tuple[str, str], int],
) -> list[dict]:
    groups: dict[tuple[str, str, str, str], list[SampleAsset]] = defaultdict(list)
    for asset in assets:
        if asset.category != "percussion" or asset.duration_seconds is None:
            continue
        parent = Path(asset.relative_path).parent.as_posix()
        groups[(asset.collection, parent,
                asset.articulation or infer_articulation(asset.relative_path),
                infer_microphone(asset.relative_path))].append(asset)
    rows = []
    organic_words = (
        "cajon", "bongo", "conga", "darbuka", "frame drum", "hand drum",
        "log drum", "wood block", "woodblock", "clave", "shaker", "cabasa",
        "brush", "tambourine",
    )
    foreground_words = (
        "snare", "marching", "metal", "cymbal", "gong", "tenor", "tom",
        "clap", "sticks", "misc", "giant",
    )
    for (collection, parent, articulation, microphone), values in sorted(groups.items()):
        new_assets = [asset for asset in values if asset.sha256 not in production_sha256]
        if not new_assets:
            continue
        name = f"{collection}:{parent}#{articulation}@{microphone}"
        role = _percussion_role(name)
        metrics = _asset_metrics(values)
        reasons: list[str] = []
        warnings: list[str] = []
        lowered = name.lower()
        if len(values) < 3:
            reasons.append("isolated_hits_not_a_coherent_group")
        if any(not 0.015 <= (asset.duration_seconds or 0) <= 3.0 for asset in values):
            reasons.append("unsuitable_percussion_duration")
        if any(word in name.lower() for word in ATTENTION_WORDS):
            reasons.append("attention_grabbing_timbre_family")
        if not any(word in lowered for word in organic_words):
            reasons.append("not_a_supported_soft_organic_percussion_role")
        if any(word in lowered for word in foreground_words):
            reasons.append("foreground_or_aggressive_percussion_family")
        transient = metrics["transient_p90"]
        centroid = metrics["spectral_centroid_p90_hz"]
        if transient is not None and transient >= 0.78:
            warnings.append("sharp_transient_requires_speech_masking_test")
        if role == "high" and centroid is not None and centroid >= 6500:
            warnings.append("bright_high_percussion_requires_low_gain_test")
        spread = metrics["normalized_rms_p90_p10_db"]
        if spread is not None and spread >= 14:
            warnings.append("large_perceived_level_spread_requires_gain_mapping")
        status = "rejected" if reasons else "review" if warnings else "candidate"
        rows.append({
            "kind": "percussion", "name": name, "family": "percussion",
            "role": role, "priority": 1, "score": 500 + min(len(values), 40) * 4 -
            len(warnings) * 35, "status": status,
            "rejection_reasons": reasons, "review_warnings": warnings,
            "assets": len(values), "new_assets": len(new_assets),
            "new_bytes": sum(asset_sizes.get((asset.collection, asset.asset_id), 0)
                             for asset in new_assets),
            "articulations": [articulation], "metrics": metrics,
            "_asset_keys": [(asset.collection, asset.asset_id) for asset in values],
        })
    return rows


def _public_row(row: dict) -> dict:
    result = {key: value for key, value in row.items() if not key.startswith("_")}
    result["asset_refs"] = [
        {"collection": collection, "asset_id": asset_id}
        for collection, asset_id in row.get("_asset_keys", [])
    ]
    return result


def build_expansion_audit(
    external_assets: list[SampleAsset],
    production_assets: list[SampleAsset],
    instruments: list[InstrumentRef],
    asset_sizes: dict[tuple[str, str], int],
    *,
    target_bytes: int,
) -> tuple[dict, dict]:
    """Compare catalogs and emit an audit plus a no-copy promotion proposal."""
    production_sha256 = {asset.sha256 for asset in production_assets}
    assets_by_key = {(asset.collection, asset.asset_id): asset
                     for asset in external_assets}
    rows = [evaluate_instrument_bank(
        instrument, assets_by_key, production_sha256, asset_sizes)
        for instrument in instruments]
    rows.extend(evaluate_percussion_groups(
        external_assets, production_sha256, asset_sizes))
    rows.sort(key=lambda row: (-row["priority"], -row["score"], row["name"]))

    candidates = [row for row in rows if row["status"] in ("candidate", "review")]
    selected: list[dict] = []
    selected_sha: set[str] = set(production_sha256)
    selected_bytes = 0

    # Cover each priority family and percussion role before filling by score.
    coverage_order = [
        ("piano", None), ("strings", None), ("guitar", None), ("bass", None),
        ("percussion", "low"), ("percussion", "mid"), ("percussion", "high"),
    ]
    ordered: list[dict] = []
    for family, role in coverage_order:
        match = next((row for row in candidates if row["family"] == family and
                      (role is None or row.get("role") == role)), None)
        if match and match not in ordered:
            ordered.append(match)
    limits = {
        ("piano", None): 3,
        ("strings", None): 4,
        ("guitar", None): 2,
        ("bass", None): 2,
        ("percussion", "low"): 3,
        ("percussion", "mid"): 3,
        ("percussion", "high"): 3,
    }
    selected_per_group: dict[tuple[str, str | None], int] = defaultdict(int)
    for row in ordered:
        selected_per_group[(row["family"], row.get("role"))] += 1
    for row in candidates:
        group = (row["family"], row.get("role"))
        limit = limits.get(group, limits.get((row["family"], None), 0))
        if row in ordered or selected_per_group[group] >= limit:
            continue
        ordered.append(row)
        selected_per_group[group] += 1

    manifest_assets: list[dict] = []
    for row in ordered:
        additions = []
        for key in row["_asset_keys"]:
            asset = assets_by_key.get(key)
            if not asset or asset.sha256 in selected_sha:
                continue
            additions.append(asset)
        addition_bytes = sum(asset_sizes.get(
            (asset.collection, asset.asset_id), 0) for asset in additions)
        if selected and selected_bytes + addition_bytes > target_bytes:
            continue
        if not additions:
            continue
        selected.append(row)
        selected_bytes += addition_bytes
        for asset in additions:
            key = (asset.collection, asset.asset_id)
            selected_sha.add(asset.sha256)
            manifest_assets.append({
                **asdict(asset.ref), "relative_path": asset.relative_path,
                "license": asset.license, "bytes": asset_sizes.get(key, 0),
            })
        if selected_bytes >= target_bytes:
            break

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["status"]] += 1
    audit = {
        "schema_version": 1,
        "policy": {
            "priority": ["piano", "strings", "guitar", "bass", "percussion"],
            "target_normalized_rms_db": TARGET_NORMALIZED_RMS_DB,
            "safe_registers": {key: list(value) for key, value in SAFE_REGISTERS.items()},
            "attention_words": list(ATTENTION_WORDS),
            "note": ("Descriptor gates are conservative screening, not automatic "
                     "production acceptance; every selected bank still needs the "
                     "paired speech listening test."),
        },
        "catalogs": {
            "external_assets": len(external_assets),
            "external_bytes": sum(asset_sizes.values()),
            "production_assets": len(production_assets),
            "new_payloads": len({asset.sha256 for asset in external_assets} -
                                production_sha256),
        },
        "bank_counts": dict(sorted(counts.items())),
        "banks": [_public_row(row) for row in rows],
    }
    manifest = {
        "schema_version": 1,
        "status": "proposed-not-promoted",
        "target_bytes": target_bytes,
        "selected_bytes": selected_bytes,
        "banks": [_public_row(row) for row in selected],
        "assets": manifest_assets,
        "requires_paired_speech_listening": True,
    }
    return audit, manifest


def build_secondary_manifest(
    audit: dict,
    primary_manifest: dict,
    external_assets: list[SampleAsset],
    production_assets: list[SampleAsset],
    asset_sizes: dict[tuple[str, str], int],
) -> dict:
    """Select the remaining unique reviewable banks for a second audition wave."""
    assets_by_key = {(asset.collection, asset.asset_id): asset
                     for asset in external_assets}
    primary_names = {bank["name"] for bank in primary_manifest["banks"]}
    occupied_sha256 = {asset.sha256 for asset in production_assets}
    seen_signatures: set[frozenset[str]] = set()

    for bank in primary_manifest["banks"]:
        signature = frozenset(
            assets_by_key[(ref["collection"], ref["asset_id"])].sha256
            for ref in bank["asset_refs"]
            if (ref["collection"], ref["asset_id"]) in assets_by_key
        )
        if signature:
            seen_signatures.add(signature)
            occupied_sha256.update(signature)

    banks: list[dict] = []
    manifest_assets: list[dict] = []
    skipped_aliases: list[str] = []
    selected_bytes = 0
    for bank in audit["banks"]:
        if (bank["status"] not in ("candidate", "review") or
                bank["name"] in primary_names):
            continue
        values = [assets_by_key[(ref["collection"], ref["asset_id"])]
                  for ref in bank["asset_refs"]
                  if (ref["collection"], ref["asset_id"]) in assets_by_key]
        signature = frozenset(asset.sha256 for asset in values)
        additions = [asset for asset in values
                     if asset.sha256 not in occupied_sha256]
        if not signature or signature in seen_signatures or not additions:
            skipped_aliases.append(bank["name"])
            continue
        seen_signatures.add(signature)
        banks.append(bank)
        for asset in additions:
            if asset.sha256 in occupied_sha256:
                continue
            occupied_sha256.add(asset.sha256)
            key = (asset.collection, asset.asset_id)
            size = asset_sizes.get(key, 0)
            selected_bytes += size
            manifest_assets.append({
                **asdict(asset.ref), "relative_path": asset.relative_path,
                "license": asset.license, "bytes": size,
            })

    return {
        "schema_version": 1,
        "status": "secondary-proposed-not-promoted",
        "source_primary_status": primary_manifest["status"],
        "selected_bytes": selected_bytes,
        "banks": banks,
        "assets": manifest_assets,
        "skipped_audio_aliases": skipped_aliases,
        "requires_paired_speech_listening": True,
    }


def _wave3_family(name: str) -> str | None:
    lowered = name.lower()
    for family in ("accordion", "recorder", "ocarina", "flute", "clarinet", "oboe",
                   "bassoon", "harmonica", "organ", "harpsichord",
                   "marimba"):
        if family in lowered:
            return family
    if "harp" in lowered and "harpsichord" not in lowered:
        return "harp"
    if any(word in lowered for word in ("guitar", "strumstick")):
        return "plucked-string"
    if any(word in lowered for word in ("kalimba", "mbira", "nyunga")):
        return "lamellophone"
    return None


def evaluate_wave3_bank(
    instrument: InstrumentRef,
    assets_by_key: dict[tuple[str, str], SampleAsset],
    baseline_sha256: set[str],
    asset_sizes: dict[tuple[str, str], int],
) -> dict | None:
    """Evaluate a broader natural bank with warnings rather than narrow taste gates."""
    family = _wave3_family(instrument.name)
    if family is None:
        return None
    keys = list(dict.fromkeys(
        (zone.sample.collection, zone.sample.asset_id) for zone in instrument.zones
    ))
    assets = [assets_by_key[key] for key in keys if key in assets_by_key]
    new_assets = [asset for asset in assets if asset.sha256 not in baseline_sha256]
    lo, hi = WAVE3_SAFE_REGISTERS[family]
    safe_zones = [zone for zone in instrument.zones if lo <= zone.root_note <= hi]
    safe_notes = sorted({zone.root_note for zone in safe_zones})
    velocity_layers = len({(zone.lo_velocity, zone.hi_velocity)
                           for zone in safe_zones})
    rr_counts: dict[tuple[int, int, int], int] = defaultdict(int)
    for zone in safe_zones:
        rr_counts[(zone.root_note, zone.lo_velocity, zone.hi_velocity)] += 1
    max_round_robins = max(rr_counts.values(), default=1)
    metrics = _asset_metrics(assets)
    lowered = instrument.name.lower()
    reasons: list[str] = []
    warnings: list[str] = []
    limitations: list[str] = []
    if not new_assets:
        reasons.append("already_in_candidate_v2")
    if any(word in lowered for word in EXCLUDED_WORDS):
        reasons.append("release_noise_demo_or_loop")
    if any(word in lowered for word in
           ("electrophone", "tx81z", "fm piano", "synth")):
        reasons.append("electronic_bank_not_needed_for_natural_wave")
    if len(safe_notes) < 5:
        reasons.append("insufficient_middle_register_coverage")
    largest_gap = _largest_gap(safe_notes)
    if largest_gap > 12:
        reasons.append("excessive_transposition_gap")
    elif largest_gap > 7:
        warnings.append("wide_transposition_gap_requires_listening")
    if any((asset.rms or 0.0) <= 1e-7 for asset in assets):
        reasons.append("silent_or_near_silent_asset")
    centroid = metrics["spectral_centroid_p90_hz"]
    transient = metrics["transient_p90"]
    spread = metrics["normalized_rms_p90_p10_db"]
    if centroid is not None and centroid >= 6000:
        warnings.append("bright_upper_partial_requires_role_gain")
    if transient is not None and transient >= 0.78:
        warnings.append("sharp_attack_requires_speech_masking_test")
    if spread is not None and spread >= 14:
        warnings.append("large_perceived_level_spread_requires_zone_gains")
    if "vib" in lowered:
        warnings.append("vibrato_is_expressive_and_should_remain_occasional")
    if family in ("recorder", "ocarina", "flute"):
        warnings.append("keep_melody_in_middle_register_by_default")
    if velocity_layers < 2:
        limitations.append("single_velocity_layer")
    if max_round_robins < 2:
        limitations.append("no_multi_take_round_robin")
    status = "rejected" if reasons else "review" if warnings else "candidate"
    score = (min(len(safe_notes), 100) * 10 + min(velocity_layers, 5) * 40 +
             min(max_round_robins, 4) * 25 - len(warnings) * 20)
    return {
        "kind": "pitched", "name": instrument.name, "family": family,
        "priority": 1, "score": score, "status": status,
        "rejection_reasons": reasons, "review_warnings": warnings,
        "mapping_limitations": limitations, "safe_register": [lo, hi],
        "source_register": [
            min((zone.root_note for zone in instrument.zones), default=0),
            max((zone.root_note for zone in instrument.zones), default=0),
        ],
        "safe_note_count": len(safe_notes),
        "largest_safe_note_gap": largest_gap,
        "velocity_layers": velocity_layers,
        "max_round_robins": max_round_robins,
        "articulations": sorted({zone.articulation for zone in safe_zones}),
        "assets": len(assets), "new_assets": len(new_assets),
        "new_bytes": sum(asset_sizes.get((asset.collection, asset.asset_id), 0)
                         for asset in new_assets),
        "metrics": metrics, "_asset_keys": keys,
    }


def build_wave3_expansion(
    external_assets: list[SampleAsset],
    baseline_assets: list[SampleAsset],
    instruments: list[InstrumentRef],
    asset_sizes: dict[tuple[str, str], int],
) -> tuple[dict, dict]:
    """Select every technically usable, checksum-distinct broader natural bank."""
    baseline_sha256 = {asset.sha256 for asset in baseline_assets}
    assets_by_key = {(asset.collection, asset.asset_id): asset
                     for asset in external_assets}
    rows = [row for instrument in instruments
            if (row := evaluate_wave3_bank(
                instrument, assets_by_key, baseline_sha256, asset_sizes)) is not None]
    rows.sort(key=lambda row: (-row["score"], row["family"], row["name"]))
    selected = []
    seen_signatures: set[frozenset[str]] = set()
    selected_sha256 = set(baseline_sha256)
    manifest_assets = []
    selected_bytes = 0
    aliases = []
    for row in rows:
        if row["status"] not in ("candidate", "review"):
            continue
        values = [assets_by_key[key] for key in row["_asset_keys"]
                  if key in assets_by_key]
        signature = frozenset(asset.sha256 for asset in values)
        additions = [asset for asset in values
                     if asset.sha256 not in selected_sha256]
        if not signature or signature in seen_signatures or not additions:
            aliases.append(row["name"])
            continue
        seen_signatures.add(signature)
        selected.append(row)
        for asset in additions:
            if asset.sha256 in selected_sha256:
                continue
            selected_sha256.add(asset.sha256)
            key = (asset.collection, asset.asset_id)
            size = asset_sizes.get(key, 0)
            selected_bytes += size
            manifest_assets.append({
                **asdict(asset.ref), "relative_path": asset.relative_path,
                "license": asset.license, "bytes": size,
            })
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["status"]] += 1
    audit = {
        "schema_version": 1,
        "policy": {
            "direction": "broader-natural-instruments",
            "safe_registers": {key: list(value)
                               for key, value in WAVE3_SAFE_REGISTERS.items()},
            "approach": ("Include all technically usable natural banks; retain "
                         "brightness, vibrato and attack as gain/role warnings rather "
                         "than aesthetic rejection reasons."),
            "preserve_existing_electronic_leads": True,
        },
        "catalogs": {
            "external_assets": len(external_assets),
            "baseline_assets": len(baseline_assets),
            "new_payloads": len({asset.sha256 for asset in external_assets} -
                                baseline_sha256),
        },
        "bank_counts": dict(sorted(counts.items())),
        "banks": [_public_row(row) for row in rows],
    }
    manifest = {
        "schema_version": 1,
        "status": "wave3-proposed-not-promoted",
        "selected_bytes": selected_bytes,
        "banks": [_public_row(row) for row in selected],
        "assets": manifest_assets,
        "skipped_audio_aliases": aliases,
        "requires_role_based_listening": True,
    }
    return audit, manifest


def wave3_markdown(audit: dict, manifest: dict) -> str:
    lines = [
        "# Wave 3 broader natural-instrument audit", "",
        "This wave intentionally accepts a wider range of instrumental color. "
        "Warnings control register, gain and frequency of use; they are not "
        "automatic aesthetic rejections.", "",
        f"- Baseline candidate-v2 catalog: {audit['catalogs']['baseline_assets']:,} assets.",
        f"- Broader banks evaluated: {len(audit['banks'])}.",
        f"- Audition banks: {len(manifest['banks'])}.",
        f"- New payload: {len(manifest['assets']):,} assets, "
        f"{manifest['selected_bytes'] / 1e9:.2f} GB.",
        f"- Checksum aliases removed: {len(manifest['skipped_audio_aliases'])}.", "",
        "The existing TX81Z/FM and synthesized lead colors remain available; Wave 3 "
        "adds natural recordings rather than replacing those occasional colors.", "",
        "| # | Status | Family | Bank | Safe notes | Layers | RR | Size | Flags |",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for number, row in enumerate(manifest["banks"], 1):
        flags = [*row["review_warnings"], *row["mapping_limitations"]]
        lines.append(
            f"| W3-{number:02d} | {row['status']} | {row['family']} | "
            f"`{row['name']}` | {row['safe_note_count']} | "
            f"{row['velocity_layers']} | {row['max_round_robins']} | "
            f"{row['new_bytes'] / 1e6:.1f} MB | {', '.join(flags) or 'none'} |")
    return "\n".join(lines) + "\n"


def audit_markdown(audit: dict, manifest: dict) -> str:
    catalogs = audit["catalogs"]
    lines = [
        "# Natural sample-library expansion audit", "",
        "This is a read-only proposal. No audio was copied or promoted.", "",
        "## Catalog comparison", "",
        f"- External catalog: {catalogs['external_assets']:,} assets, "
        f"{catalogs['external_bytes'] / 1e9:.2f} GB.",
        f"- Production bundle: {catalogs['production_assets']:,} assets.",
        f"- New unique payloads: {catalogs['new_payloads']:,}.",
        f"- Proposed candidate payload: {manifest['selected_bytes'] / 1e9:.2f} GB "
        f"across {len(manifest['banks'])} coherent banks (target ceiling: "
        f"{manifest['target_bytes'] / 1e9:.2f} GB).", "",
        ("The shortlist is intentionally below the storage ceiling because no more "
         "complete banks passed the current natural-instrument and speech-safety "
         "screening."), "",
        "Descriptor gates are screening rules only. Candidate banks still require "
        "independent-set listening followed by paired speech A/B testing.", "",
        "## Proposed banks", "",
        "| Status | Priority | Family/role | Bank | Safe notes | Layers | RR | Size | Flags |",
        "|---|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for row in manifest["banks"]:
        family = row["family"] + (f"/{row['role']}" if row.get("role") else "")
        flags = [*row["review_warnings"], *row.get("mapping_limitations", [])]
        warnings = ", ".join(flags) or "none"
        lines.append(
            f"| {row['status']} | {row['priority']} | {family} | `{row['name']}` | "
            f"{row.get('safe_note_count', '—')} | {row.get('velocity_layers', '—')} | "
            f"{row.get('max_round_robins', '—')} | {row['new_bytes'] / 1e6:.1f} MB | "
            f"{warnings} |"
        )
    lines += ["", "## Screening totals", ""]
    lines.extend(f"- {status}: {count}" for status, count in
                 audit["bank_counts"].items())
    lines += ["", "## Required next gate", "",
              "Render the proposed banks in the control composition policy, cap them "
              "to their safe registers, apply the recommended conservative gains, "
              "and reject any piercing, bottle-like, cosmic, masking, or pulse-obscuring "
              "result before creating a candidate production bundle.", ""]
    return "\n".join(lines)


def _evenly_spaced(values: list[int], count: int) -> list[int]:
    if len(values) <= count:
        return values
    indexes = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[index] for index in indexes]


def _sequence(clips: list[np.ndarray], gap: int) -> np.ndarray:
    if not clips:
        return np.zeros(SR, dtype=np.float32)
    parts = []
    silence = np.zeros(gap, dtype=np.float32)
    for clip in clips:
        parts.extend((clip.astype(np.float32, copy=False), silence))
    return np.concatenate(parts)


def render_bank_audition(
    bank: dict,
    library: SampleLibrary,
    instruments_by_name: dict[str, InstrumentRef],
) -> np.ndarray:
    """Render a level-preserving isolated probe for one shortlisted bank."""
    bank_gain = 10 ** (float(bank["metrics"]["recommended_bank_gain_db"]) / 20)
    high_gain = 10 ** (
        float(bank["metrics"]["recommended_high_register_gain_db"]) / 20)
    if bank["kind"] == "pitched":
        instrument = instruments_by_name[bank["name"]]
        lo, hi = bank["safe_register"]
        notes = sorted({zone.root_note for zone in instrument.zones
                        if lo <= zone.root_note <= hi})
        notes = _evenly_spaced(notes, 8)
        phrase_indexes = (0, 2, 1, 4, 3, 5, 2, 0)
        phrase = [notes[min(index, len(notes) - 1)] for index in phrase_indexes]
        renderer = CatalogMultiSampleInstrument(instrument, library)
        clips = []
        for index, note in enumerate([*notes, *phrase]):
            audio = renderer.render(note, 0.42, 0.62, variation=index)
            gain = bank_gain * (high_gain if note >= 72 else 1.0)
            clips.append(audio * gain)
        mono = _sequence(clips, int(0.13 * SR))
    else:
        refs = [SampleRef(row["collection"], row["asset_id"])
                for row in sorted(
                    bank["asset_refs"],
                    key=lambda value: (value["collection"], value["asset_id"]),
                )[:8]]
        clips = [CatalogSampleInstrument(ref, library).render(
            60, 0.34, 0.42) * bank_gain for ref in refs]
        mono = _sequence([*clips, *clips[:4]], int(0.11 * SR))
    if not np.isfinite(mono).all():
        raise ValueError(f"Non-finite audition audio for {bank['name']}.")
    peak = float(np.abs(mono).max()) if len(mono) else 0.0
    if peak > 0.969:
        mono = mono * (0.969 / peak)
    return np.column_stack((mono, mono)).astype(np.float32)
