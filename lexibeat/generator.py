"""Candidate generation and selection behind the LexiBeat public API."""

from __future__ import annotations

import hashlib
import json
import math
import random
import secrets
import threading
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import samples as sample_packs
from .api import (
    BedFingerprint,
    MusicGenerationResult,
    MusicRequest,
    SampleUsage,
)
from .bedspec import ENGINE_VERSION, BedSpec
from .instrument_roles import (FINAL_ACCEPTED_FAMILIES,
                               apply_final_wave3_role_profile)
from .library import (
    COLLECTIONS,
    InstrumentRef,
    SampleAsset,
    SampleLibrary,
    SampleRef,
    infer_articulation,
    instrument_refs,
)
from .music import Grid, render_bed, render_stems
from .profiles import GenerationProfile, ListenerPolicy, get_profile
from .quality import (
    Candidate,
    evaluate_preview,
    fingerprint_distance,
    instrument_families,
    motif_features,
    preference_score,
)

SEED_STEP = 104_729
ProgressCallback = Callable[[float, str], None]
CancelCheck = Callable[[], bool]


class GenerationCancelledError(RuntimeError):
    """Raised at a safe boundary when a caller cancels candidate generation."""


_inventory_lock = threading.Lock()
_bundled_inventory_cache: dict[
    tuple[str, int, int, int], tuple[tuple[SampleAsset, ...], tuple[InstrumentRef, ...]]
] = {}

_ORNAMENT_WORDS = (
    "sleigh", "jingle", "bell", "cowbell", "chime", "cymbal", "triangle",
    "musicbox", "music box", "roll",
)


def _sha256(source: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def sample_refs(spec: BedSpec) -> list[SampleRef]:
    refs = [lane.sample for lane in spec.phrase.percussion if lane.sample]
    for instrument in (
        spec.phrase.lead_instrument,
        spec.phrase.pad_instrument,
        spec.phrase.bass_instrument,
    ):
        if instrument:
            refs.extend(zone.sample for zone in instrument.zones)
    unique = {(ref.collection, ref.asset_id): ref for ref in refs}
    return list(unique.values())


def named_pack_names(spec: BedSpec) -> set[str]:
    aliases = {
        "piano": "salamander",
        "marimba": "vsco-marimba",
        "glockenspiel": "vsco-glockenspiel",
        "strings": "vsco-strings",
    }
    names: set[str] = set()
    if not spec.phrase.lead_instrument:
        name = aliases.get(spec.lead.instrument)
        if name:
            names.add(name)
    if not spec.phrase.pad_instrument:
        name = aliases.get(spec.pad.instrument)
        if name:
            names.add(name)
    return names


def _matching(assets: list[SampleAsset], words: tuple[str, ...]) -> list[SampleAsset]:
    return [
        asset for asset in assets
        if any(word in asset.relative_path.lower() for word in words)
    ]


def _choose(rng: np.random.Generator, values: list[SampleAsset]) -> SampleAsset | None:
    return values[int(rng.integers(0, len(values)))] if values else None


def _choose_across_collections(
    rng: np.random.Generator,
    values: list[SampleAsset],
) -> SampleAsset | None:
    by_collection: dict[str, list[SampleAsset]] = {}
    for asset in values:
        by_collection.setdefault(asset.collection, []).append(asset)
    if not by_collection:
        return None
    names = sorted(by_collection)
    collection = names[int(rng.integers(0, len(names)))]
    return _choose(rng, by_collection[collection])


def _instrument_matches(name: str, word: str) -> bool:
    lowered = name.lower()
    if "/bowed" in lowered:
        return False
    if word == "harp":
        return "harp" in lowered and "harpsichord" not in lowered
    if word == "psaltery":
        return "psaltery" in lowered and "/pluck" in lowered
    return word in lowered


def _role_assets(assets: list[SampleAsset], role: str, *,
                 expanded: bool = False) -> list[SampleAsset]:
    words = {
        "low": (("kick", "bass drum", "bassdrum", "low tom", "bass cajon",
                 "cajon", "frame drum", "low conga", "low bongo")
                if expanded else
                ("kick", "bass drum", "bassdrum", "low tom", "bass cajon")),
        "mid": (
            "snare", "rim", "wood", "clave", "castanet", "clap", "stick",
            "cardboard", "porcelain", "darbuka", "bongo", "conga", "cajon",
        ),
        "high": ("shaker", "maraca", "hat", "tamb", "brush", "key"),
    }[role]
    matched = _matching(assets, words)
    if role == "low":
        aggressive = (
            "hardstyle", "rawstyle", "distkit", "synthkit", "x0xproc",
            "sdbkit", "sub-a",
        )
        return [
            asset for asset in matched
            if (asset.spectral_centroid is None or asset.spectral_centroid < 2400)
            and not any(word in asset.relative_path.lower() for word in aggressive)
        ]
    if role == "high":
        return [
            asset for asset in matched
            if asset.spectral_centroid is None or asset.spectral_centroid > 1300
        ]
    return matched


def enrich_with_catalog_samples(
    spec: BedSpec,
    assets: list[SampleAsset],
    instruments: list[InstrumentRef],
    seed: int,
    *,
    palette: str = "hybrid",
    expansion_policy: dict | None = None,
    listener: ListenerPolicy | None = None,
) -> None:
    """Resolve safe catalog choices without consulting network availability."""
    if not assets or palette == "electronic":
        return
    rng = np.random.default_rng(seed * 7919 + 17)
    expansion_policy = expansion_policy or {}
    accepted_banks = {
        bank["name"]: bank for bank in expansion_policy.get("accepted_banks", [])
    }
    asset_policy = {
        (ref["collection"], ref["asset_id"]): bank
        for bank in accepted_banks.values() for ref in bank["asset_refs"]
    }
    short = [
        asset for asset in assets
        if asset.category == "percussion"
        and asset.duration_seconds is not None
        and 0.015 <= asset.duration_seconds <= 3.0
        and not any(word in asset.relative_path.lower() for word in _ORNAMENT_WORDS)
    ]
    roles = {role: _role_assets(
        short, role, expanded=bool(expansion_policy))
        for role in ("low", "mid", "high")}
    percussion_probability = 0.86 if palette == "acoustic" else 0.66
    for lane in spec.phrase.percussion:
        if rng.random() > percussion_probability:
            continue
        role = lane.role or (
            "low" if "kick" in lane.sound
            else "high" if any(name in lane.sound for name in ("shaker", "hat"))
            else "mid"
        )
        asset = _choose_across_collections(rng, roles[role])
        if asset:
            lane.sample = asset.ref
            lane.articulation = (asset.articulation or
                                 infer_articulation(asset.relative_path))
            lane.sound = f"sample:{asset.collection}"
            bank = asset_policy.get((asset.collection, asset.asset_id))
            if bank:
                gain_db = (float(bank["metrics"]["recommended_bank_gain_db"]) +
                           float(bank.get("listener_gain_db", 0.0)))
                lane.level *= 10 ** (gain_db / 20)

    preferences = {
        "piano": ("piano",),
        "marimba": (
            "marimba", "vibraphone", "xylophone", "kalimba", "mbira", "nyunga",
        ),
        "glockenspiel": (
            "glockenspiel", "celesta", "concert harp", "folk harp", "psaltery",
        ),
        "synth": (
            "recorder", "flute", "oboe", "clarinet", "strumstick", "guitar",
            "harp", "organ", "ocarina", "harmonica", "bassoon", "accordion",
            "harpsichord", "kalimba", "marimba", "pizz", "spic", "mbira",
            "nyunga", "psaltery",
        ),
    }
    family_preferences = {
        "sunlit-acoustic": ("guitar", "psaltery", "pizz", "harp", "mbira"),
        "gentle-movement": ("piano", "vibraphone", "ocarina", "organ", "pizz"),
        "playful-plucked": (
            "mbira", "nyunga", "kalimba", "marimba", "strumstick",
            "psaltery", "guitar", "pizz",
        ),
        "bright-pastoral": (
            "ocarina", "pizz", "psaltery",
        ),
    }
    preferred_words = preferences.get(spec.lead.instrument, ())
    if spec.phrase.family in family_preferences and rng.random() < 0.76:
        preferred_words = family_preferences[spec.phrase.family]
    compatible_by_word = {
        word: [
            instrument for instrument in instruments
            if _instrument_matches(instrument.name, word)
            and "contrabass" not in instrument.name.lower()
        ]
        for word in preferred_words
    }
    compatible_by_word = {
        word: values for word, values in compatible_by_word.items() if values
    }
    compatible = [
        instrument for values in compatible_by_word.values() for instrument in values
    ]
    probability = (
        0.90 if palette == "acoustic"
        else 0.50 if spec.lead.instrument == "piano"
        else 0.76 if spec.phrase.family in family_preferences
        else 0.64
    )
    if compatible and rng.random() < probability:
        if spec.phrase.family in family_preferences:
            timbres = sorted(compatible_by_word)
            choices = compatible_by_word[
                timbres[int(rng.integers(0, len(timbres)))]
            ]
        else:
            by_collection: dict[str, list[InstrumentRef]] = {}
            for instrument in compatible:
                collection = instrument.name.split(":", 1)[0]
                by_collection.setdefault(collection, []).append(instrument)
            collection = sorted(by_collection)[int(rng.integers(0, len(by_collection)))]
            choices = by_collection[collection]
        weights = np.asarray([
            float(accepted_banks.get(instrument.name, {}).get(
                "selection_weight", 1.0))
            for instrument in choices
        ], dtype=np.float64)
        weights /= weights.sum()
        chosen: InstrumentRef | None = choices[int(rng.choice(len(choices), p=weights))]
        if listener and listener.approved_catalog_only and chosen.name not in accepted_banks:
            # From a stream of its own, so every draw after this one — the pad, the bass — is the
            # draw the bed would have made anyway, and only the lead differs.
            approved = sorted((instrument for instrument in compatible
                               if instrument.name in accepted_banks),
                              key=lambda instrument: instrument.name)
            chosen = (approved[int(np.random.default_rng(seed * 6007 + 71).integers(
                0, len(approved)))] if approved else None)
        if chosen is not None:
            spec.phrase.lead_instrument = chosen
            selected_bank = accepted_banks.get(chosen.name, {})
            selected_family = selected_bank.get("family")
            if selected_family in FINAL_ACCEPTED_FAMILIES:
                apply_final_wave3_role_profile(spec, selected_family, chosen)
            else:
                articulation = chosen.zones[0].articulation
                for event in spec.phrase.lead:
                    event.articulation = articulation
                    event.midi_note = _fit_instrument_note(event.midi_note, chosen)

    sustained_strings = [
        instrument for instrument in instruments
        if instrument.name in accepted_banks and
        accepted_banks[instrument.name]["family"] == "strings" and
        any(word in instrument.name.lower()
            for word in ("#bowed", "#sustain", "sustain-non-vibrato"))
    ]
    pad_rng = np.random.default_rng(seed * 3571 + 101)
    pad_probability = 0.46 if palette == "acoustic" else 0.28
    if sustained_strings and pad_rng.random() < pad_probability:
        choices = sorted(sustained_strings, key=lambda value: value.name)
        spec.phrase.pad_instrument = choices[
            int(pad_rng.integers(0, len(choices)))]
        articulation = spec.phrase.pad_instrument.zones[0].articulation
        for event in spec.phrase.chords:
            event.articulation = articulation
            event.midi_notes = [
                _fit_instrument_note(note, spec.phrase.pad_instrument)
                for note in event.midi_notes
            ]

    natural_basses = [
        instrument for instrument in instruments
        if ("fashionbass" in instrument.name.lower() or
            (instrument.name in accepted_banks and
             accepted_banks[instrument.name]["family"] == "bass"))
    ]
    bass_probability = 0.58 if palette == "acoustic" else 0.34
    if natural_basses and rng.random() < bass_probability:
        spec.phrase.bass_instrument = natural_basses[
            int(rng.integers(0, len(natural_basses)))
        ]
        articulation = spec.phrase.bass_instrument.zones[0].articulation
        for event in spec.phrase.bass:
            event.articulation = articulation
            event.midi_note = _fit_instrument_note(
                event.midi_note, spec.phrase.bass_instrument)


def apply_listener_policy(spec: BedSpec, listener: ListenerPolicy) -> None:
    """The post-resolution half of a `ListenerPolicy`: fifths and the lead's ceiling.

    Both only move notes that show the defect, and draw nothing, so a bed without it is unchanged.
    """
    if listener.diatonic_fifths:
        steps = spec.scale_steps()
        in_scale = {(spec.root + step) % 12 for step in steps}
        per_bar = spec.steps_per_bar

        def fifth_of(step: int) -> int:
            degree = spec.progression[(step // per_bar) % len(spec.progression)]
            return (spec.chord_root(degree) + 7) % 12

        # Only the chord's own fifth, and only where it leaves the scale: the passing bass
        # deliberately walks chromatically, and those notes are not this defect. A perfect fifth
        # that is off-key is always a semitone above the scale's own, diminished one.
        def diatonic(note: int, step: int) -> int:
            return note - 1 if note % 12 == fifth_of(step) and note % 12 not in in_scale else note

        for chord in spec.phrase.chords:
            chord.midi_notes = [diatonic(note, chord.step) for note in chord.midi_notes]
        for event in (*spec.phrase.bass, *spec.phrase.lead):
            event.midi_note = diatonic(event.midi_note, event.step)
    if listener.lead_register_cap is not None:
        for event in spec.phrase.lead:
            while event.midi_note > listener.lead_register_cap:
                event.midi_note -= 12


def _fit_instrument_note(note: int, instrument: InstrumentRef) -> int:
    """Octave-fold a resolved event into the explicitly mapped safe zone."""
    lo = min(zone.lo_note for zone in instrument.zones)
    hi = max(zone.hi_note for zone in instrument.zones)
    while note < lo and note + 12 <= hi:
        note += 12
    while note > hi and note - 12 >= lo:
        note -= 12
    return min(max(note, lo), hi)


def _apply_expansion_instrument_policy(
    instruments: list[InstrumentRef], policy: dict,
) -> list[InstrumentRef]:
    """Apply audited registers and gains before choices enter a BedSpec."""
    by_name = {bank["name"]: bank for bank in policy.get("accepted_banks", [])}
    resolved = []
    for instrument in instruments:
        bank = by_name.get(instrument.name)
        if not bank:
            resolved.append(instrument)
            continue
        safe_register = bank.get("safe_register")
        lo, hi = safe_register if safe_register else (0, 127)
        bank_gain = (float(bank["metrics"]["recommended_bank_gain_db"]) +
                     float(bank.get("listener_gain_db", 0.0)))
        high_gain = float(
            bank["metrics"].get("recommended_high_register_gain_db", 0.0))
        zones = []
        for zone in instrument.zones:
            if not lo <= zone.root_note <= hi:
                continue
            gain_db = zone.gain_db + bank_gain
            if zone.root_note >= 72:
                gain_db += high_gain
            zones.append(replace(
                zone, lo_note=max(zone.lo_note, lo), hi_note=min(zone.hi_note, hi),
                gain_db=gain_db))
        if zones:
            resolved.append(InstrumentRef(instrument.name, tuple(zones)))
    return resolved


def _apply_request(spec: BedSpec, request: MusicRequest, profile: GenerationProfile) -> None:
    if request.energy == "calm":
        spec.bpm = max(56.0, spec.bpm * 0.94)
        spec.drums.level *= 0.90
        spec.lead.level *= 0.95
        spec.pad.cutoff_base *= 0.92
    elif request.energy == "bright":
        spec.bpm = min(104.0, spec.bpm * 1.04)
        spec.lead.level *= 1.04
        spec.pad.cutoff_base *= 1.10

    if request.rhythm == "sparse":
        spec.phrase.percussion = spec.phrase.percussion[:2]
        spec.drums.level *= 0.88
    elif request.rhythm == "groovy":
        spec.drums.level = min(spec.drums.level * 1.06, 0.62)

    spec.swing = min(spec.swing, profile.max_swing)
    spec.phrase.palette = request.palette
    if request.palette == "electronic":
        spec.pad.instrument = "synth"
        spec.lead.instrument = "synth"
        spec.phrase.pad_instrument = None
        spec.phrase.lead_instrument = None
        spec.phrase.bass_instrument = None
        for lane in spec.phrase.percussion:
            lane.sample = None


def _safe_assets(library: SampleLibrary) -> list[SampleAsset]:
    if not library.catalog_path.exists():
        return []
    safe_collections = tuple(
        name for name, source in COLLECTIONS.items() if source.safe_default
    )
    assets = library.assets(collections=safe_collections)
    # The bundle builder removes every catalog row whose checksum-verified file
    # was not copied. Trust that immutable catalog instead of issuing thousands
    # of remote existence checks against an attached bucket.
    if library.uses_bundled_catalog:
        return assets
    if not library.external.exists():
        assets = [asset for asset in assets if library.is_promoted(asset)]
    return assets


def _safe_inventory(
    library: SampleLibrary,
) -> tuple[list[SampleAsset], list[InstrumentRef], bool]:
    """Load safe assets and instruments, caching immutable bundled metadata."""
    if not library.uses_bundled_catalog:
        assets = _safe_assets(library)
        return assets, instrument_refs(assets), False
    catalog = library.catalog_path
    stat = catalog.stat()
    policy = library.expansion_policy()
    policy_digest = int.from_bytes(hashlib.sha256(
        json.dumps(policy, sort_keys=True).encode("utf-8")).digest()[:8])
    key = (str(catalog.resolve()), stat.st_mtime_ns, stat.st_size, policy_digest)
    with _inventory_lock:
        cached = _bundled_inventory_cache.get(key)
        if cached is not None:
            return list(cached[0]), list(cached[1]), True
        assets = _safe_assets(library)
        instruments = instrument_refs(
            assets,
            include_sustained_strings=bool(
                policy.get("include_sustained_strings", False)))
        instruments = _apply_expansion_instrument_policy(instruments, policy)
        _bundled_inventory_cache.clear()
        _bundled_inventory_cache[key] = (tuple(assets), tuple(instruments))
    return assets, instruments, False


def build_bed(
    family: str,
    bed_seed: int,
    request: MusicRequest,
    profile: GenerationProfile,
    assets: list[SampleAsset],
    instruments: list[InstrumentRef],
    *,
    expansion_policy: dict | None = None,
    listener: ListenerPolicy | None = None,
) -> BedSpec:
    """One candidate, exactly as the pool builds it — before it is rendered or scored.

    The listening tools rebuild a winning candidate through this with one switch flipped, so a pair
    differs in that switch and nothing else; resolving twice would let the switch re-rank the pool
    and hand back a different bed entirely.
    """
    listener = listener if listener is not None else profile.listener
    spec = BedSpec.from_style(family, bed_seed)
    spec.engine_version = ENGINE_VERSION
    spec.profile_version = profile.name
    _apply_request(spec, request, profile)
    if assets and request.palette != "electronic":
        enrich_with_catalog_samples(
            spec, assets, instruments, bed_seed, palette=request.palette,
            expansion_policy=expansion_policy, listener=listener,
        )
    apply_listener_policy(spec, listener)
    return spec


def build_candidates(
    count: int,
    multiplier: int,
    seed: int,
    assets: list[SampleAsset],
    instruments: list[InstrumentRef] | None = None,
    families: tuple[str, ...] | None = None,
    *,
    request: MusicRequest | None = None,
    profile: GenerationProfile | None = None,
    pool_size: int | None = None,
    stop_after_valid: int | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
    expansion_policy: dict | None = None,
    listener: ListenerPolicy | None = None,
) -> tuple[list[Candidate], list[dict]]:
    """Build and validate a deterministic candidate pool."""
    request = request or MusicRequest(seed=seed)
    profile = profile or get_profile(request.profile)
    listener = listener if listener is not None else profile.listener
    families = families or profile.families
    resolved_pool_size = (pool_size if pool_size is not None
                          else max(count * multiplier, len(families) * 2))
    candidates: list[Candidate] = []
    rejected: list[dict] = []
    for index in range(resolved_pool_size):
        if cancel_check and cancel_check():
            raise GenerationCancelledError("Music resolution cancelled.")
        if progress_callback:
            if stop_after_valid is not None:
                accepted_target = max(stop_after_valid, 1)
                progress_callback(
                    index / max(resolved_pool_size, 1),
                    f"Testing candidate {index + 1} of {resolved_pool_size} · "
                    f"{len(candidates)} of {accepted_target} accepted",
                )
            else:
                progress_callback(index / max(resolved_pool_size, 1),
                                  f"Analyzing candidate {index + 1} of {resolved_pool_size}")
        family = families[index % len(families)]
        bed_seed = (seed + index * SEED_STEP) % (2 ** 64)
        spec = build_bed(family, bed_seed, request, profile, assets, instruments or [],
                         expansion_policy=expansion_policy, listener=listener)
        grid = Grid.from_spec(spec)
        if 20 * grid.bar + 1.0 > 90.0:
            rejected.append({
                "family": family,
                "seed": bed_seed,
                "reason": "speech schedule exceeds 90 seconds",
            })
            continue
        try:
            bars = max(spec.phrase.loop_bars, 4)
            stems = render_stems(spec, bars, cancel_check=cancel_check)
            preview = sum(stems.values(), np.zeros_like(next(iter(stems.values()))))
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            rejected.append({"family": family, "seed": bed_seed, "reason": str(exc)})
            continue
        quality, fingerprint = evaluate_preview(preview, stems, spec, profile)
        if not quality.accepted:
            rejected.append({
                "family": family,
                "seed": bed_seed,
                "reason": "; ".join(quality.rejection_reasons),
            })
            continue
        collections = tuple(sorted(
            {ref.collection for ref in sample_refs(spec)}
            | {f"pack:{name}" for name in named_pack_names(spec)}
        ))
        candidates.append(Candidate(
            family=family,
            seed=bed_seed,
            spec=spec,
            features=np.asarray(fingerprint.audio_features),
            preview_seconds=len(preview) / 44_100,
            sample_collections=collections,
            fingerprint=fingerprint,
            quality=quality,
        ))
        if stop_after_valid is not None and len(candidates) >= stop_after_valid:
            break
    if progress_callback:
        progress_callback(1.0, "Candidate analysis complete")
    if len(candidates) < count:
        incomplete_bundle = next(
            (row["reason"] for row in rejected
             if "production sample bundle is incomplete" in row["reason"].lower()),
            None,
        )
        if incomplete_bundle:
            raise RuntimeError(incomplete_bundle)
        raise RuntimeError(
            f"Only {len(candidates)} valid candidates for requested count {count}; "
            "install the production sample bundle, choose the electronic palette, "
            "or inspect the rejection report."
        )
    return candidates, rejected


def select_balanced(
    candidates: list[Candidate],
    count: int,
    families: tuple[str, ...] | None = None,
) -> list[Candidate]:
    """Round-robin families while maximizing feature and motif distance."""
    if count < 1:
        raise ValueError("count must be positive")
    if count > len(candidates):
        raise ValueError(f"Requested {count} candidates from {len(candidates)}.")
    family_order = families or tuple(dict.fromkeys(row.family for row in candidates))
    by_family = {
        family: [index for index, row in enumerate(candidates) if row.family == family]
        for family in family_order
    }

    def candidate_score(index: int) -> float:
        report = candidates[index].quality
        return report.score if report is not None else preference_score(
            candidates[index].spec)

    def candidate_fingerprint(index: int) -> BedFingerprint:
        row = candidates[index]
        if row.fingerprint is not None:
            return row.fingerprint
        phrase = row.spec.phrase
        return BedFingerprint(
            family=phrase.family,
            audio_features=tuple(float(value) for value in row.features),
            motif_features=tuple(float(value) for value in motif_features(row.spec)),
            instrument_families=instrument_families(row.spec),
        )

    selected: list[int] = []
    while len(selected) < count:
        progressed = False
        for family in family_order:
            available = [index for index in by_family.get(family, ())
                         if index not in selected]
            if not available or len(selected) >= count:
                continue
            if not selected:
                chosen = max(
                    available,
                    key=lambda index: (candidate_score(index),
                                       -candidates[index].seed),
                )
            else:
                chosen = max(
                    available,
                    key=lambda index: (
                        min(
                            fingerprint_distance(
                                candidate_fingerprint(index),
                                candidate_fingerprint(prior),
                            )
                            for prior in selected
                        ) + 0.35 * candidate_score(index),
                        -candidates[index].seed,
                    ),
                )
            selected.append(chosen)
            progressed = True
        if not progressed:
            break

    def coverage(indexes: list[int]) -> set[str]:
        tags: set[str] = set()
        for index in indexes:
            row = candidates[index]
            phrase = row.spec.phrase
            if "freepats-guitar" in row.sample_collections:
                tags.add("classical-guitar")
            if phrase.bass_instrument and "fashionbass" in \
                    phrase.bass_instrument.name.lower():
                tags.add("natural-bass")
            lead_name = (phrase.lead_instrument.name.lower()
                         if phrase.lead_instrument else "")
            if any(word in lead_name for word in (
                "mbira", "nyunga", "psaltery", "ocarina", "harmonica",
                "/pizz", "/spic",
            )):
                tags.add("expanded-front")
        return tags

    while True:
        current = coverage(selected)
        best: tuple[tuple[int, float, int], int, int] | None = None
        for position, old_index in enumerate(selected):
            family = candidates[old_index].family
            for new_index in by_family.get(family, ()):
                if new_index in selected:
                    continue
                proposal = [*selected]
                proposal[position] = new_index
                gained = len(coverage(proposal)) - len(current)
                if gained <= 0:
                    continue
                score = (gained, candidate_score(new_index), -candidates[new_index].seed)
                if best is None or score > best[0]:
                    best = (score, position, new_index)
        if best is None:
            break
        selected[best[1]] = best[2]

    return [candidates[index] for index in selected]


def _manifest(spec: BedSpec, library: SampleLibrary) -> tuple[SampleUsage, ...]:
    rows: list[SampleUsage] = []
    for ref in sample_refs(spec):
        asset = library.asset(ref)
        collection = COLLECTIONS[asset.collection]
        rows.append(SampleUsage(
            collection=asset.collection,
            asset_id=asset.asset_id,
            sha256=asset.sha256,
            license=asset.license,
            attribution=collection.attribution,
            relative_path=asset.relative_path,
        ))
    for name in sorted(named_pack_names(spec)):
        pack = sample_packs.PACKS[name]
        for entry in pack.entries():
            path = sample_packs.pack_dir(pack) / entry.filename
            if not path.exists():
                continue
            digest = _sha256(path)
            rows.append(SampleUsage(
                collection=f"pack:{name}",
                asset_id=entry.filename,
                sha256=digest,
                license=pack.license,
                attribution=pack.attribution,
                relative_path=entry.filename,
            ))
    return tuple(rows)


def _novelty(candidate: Candidate, avoid: Sequence[BedFingerprint]) -> float:
    if not avoid or candidate.fingerprint is None:
        return 0.0
    return min(fingerprint_distance(candidate.fingerprint, prior) for prior in avoid)


def resolve_request(
    request: MusicRequest,
    *,
    avoid_fingerprints: Sequence[BedFingerprint] = (),
    library: SampleLibrary | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
    listener: ListenerPolicy | None = None,
) -> MusicGenerationResult:
    """`listener` overrides the profile's policy; the listening tools use it to A/B a switch."""
    request.validated()
    profile = get_profile(request.profile)
    seed = request.seed if request.seed is not None else secrets.randbits(64)
    resolved_request = replace(request, seed=seed)
    family_rng = random.Random(seed ^ 0x4C45584942454154)
    family = request.family if request.family != "auto" else family_rng.choice(
        profile.families
    )
    library = library or SampleLibrary()
    if request.palette != "electronic":
        if progress_callback:
            progress_callback(0.0, "Opening the production sample catalog")
        expansion_policy = library.expansion_policy()
        assets, instruments, inventory_cached = _safe_inventory(library)
        if progress_callback:
            action = "Using cached" if inventory_cached else "Loaded"
            progress_callback(
                0.0,
                f"{action} sample inventory: {len(assets):,} samples, "
                f"{len(instruments)} instruments")
    else:
        assets, instruments = [], []
        expansion_policy = {}
    candidates, _ = build_candidates(
        1,
        1,
        seed,
        assets,
        instruments,
        (family,),
        request=resolved_request,
        profile=profile,
        pool_size=profile.candidate_count * profile.candidate_attempt_multiplier,
        stop_after_valid=profile.candidate_count,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        expansion_policy=expansion_policy,
        listener=listener,
    )
    ranked = sorted(
        candidates,
        key=lambda row: ((row.quality.score if row.quality else 0.0)
                         + 0.20 * _novelty(row, avoid_fingerprints),
                         -row.seed),
        reverse=True,
    )
    tier_size = max(1, math.ceil(len(ranked) * profile.top_tier_fraction))
    top_tier = ranked[:tier_size]
    selection_rng = random.Random(seed ^ 0x50524F4455435449)
    chosen = top_tier[selection_rng.randrange(len(top_tier))]
    assert chosen.fingerprint is not None
    assert chosen.quality is not None
    return MusicGenerationResult(
        request=resolved_request,
        bed_spec=chosen.spec,
        fingerprint=chosen.fingerprint,
        quality=chosen.quality,
        sample_manifest=_manifest(chosen.spec, library),
        engine_version=ENGINE_VERSION,
        profile_version=profile.name,
    )


def render_resolved(spec: BedSpec, *, duration_seconds: float,
                    progress_callback: ProgressCallback | None = None,
                    cancel_check: CancelCheck | None = None) -> np.ndarray:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be a positive finite number")
    grid = Grid.from_spec(spec)
    bars = max(1, round(max(duration_seconds - 1.0, grid.bar) / grid.bar))
    try:
        return render_bed(spec, bars, progress_callback=progress_callback,
                          cancel_check=cancel_check)
    except InterruptedError as exc:
        raise GenerationCancelledError("Music render cancelled.") from exc
