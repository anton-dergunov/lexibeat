#!/usr/bin/env python3
"""Generate a maximally varied, reproducible listening set of music beds."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from compare_gemini_batched import split_on_long_silences
from lexibeat.arrange import PATTERNS
from lexibeat.api import MusicRequest
from lexibeat.bedspec import TIMBRE_PALETTES, BedSpec
from lexibeat.emotion import NEUTRAL
from lexibeat.generator import (
    ENGINE_VERSION,
    _apply_request,
    build_candidates,
    enrich_with_catalog_samples,
    named_pack_names as _named_pack_names,
    sample_refs as _refs,
    select_balanced,
)
from lexibeat.library import COLLECTIONS, SampleAsset, SampleLibrary, instrument_refs
from lexibeat.mix import mix_stems
from lexibeat.music import Grid, SR, render_stems
from lexibeat.profiles import BROAD_FAMILIES, POSITIVE_FAMILIES, get_profile
from lexibeat.quality import Candidate, evaluate_preview
from lexibeat import samples as sample_packs
from lexibeat.voice import Prosody, Speaker, fit


FAMILIES = BROAD_FAMILIES
ITEMS = (("el cava", "sparkling wine"), ("la salchicha", "sausage"))
DEFAULT_MODEL = "gemini-3.1-flash-tts-preview"


def _speech_manifest_path(out_dir: Path) -> Path:
    return out_dir / "speech" / "segments.json"


def _load_shared_speech(out_dir: Path) -> tuple[dict[tuple[int, str, int], np.ndarray], dict] | None:
    path = _speech_manifest_path(out_dir)
    if not path.exists():
        return None
    metadata = json.loads(path.read_text(encoding="utf-8"))
    segments: dict[tuple[int, str, int], np.ndarray] = {}
    for row in metadata["segments"]:
        audio, rate = sf.read(path.parent / row["path"], dtype="float32")
        if rate != SR:
            raise RuntimeError(f"Cached speech segment has unexpected rate {rate}.")
        segments[(row["repeat"], row["language"], row["item"])] = audio
    return segments, metadata


def _synthesize_backend(out_dir: Path, backend: str, model: str | None,
                        seed: int) -> tuple[dict[tuple[int, str, int], np.ndarray], dict]:
    speech_dir = out_dir / "speech"
    speech_dir.mkdir(parents=True, exist_ok=True)
    speaker = Speaker(backend=backend, model=model, voice_seed=seed)
    segments: dict[tuple[int, str, int], np.ndarray] = {}
    rows: list[dict] = []
    try:
        if backend.startswith("gemini"):
            for rep in range(3):
                prosody = Prosody.for_repeat(rep, speaker.prosody_strength)
                for lang in ("es", "en"):
                    phrases = [item[0 if lang == "es" else 1] for item in ITEMS]
                    audio = speaker.say("\n[long pause]\n".join(phrases), lang,
                                        prosody, NEUTRAL)
                    pieces, split = split_on_long_silences(audio, len(phrases),
                                                           sample_rate=SR)
                    for item_index, piece in enumerate(pieces):
                        segments[(rep, lang, item_index)] = piece
                    rows.append({"repeat": rep, "language": lang,
                                 "phrases": phrases, "split": split,
                                 "synthesis": speaker.stats[-1]})
        else:
            for rep in range(3):
                prosody = Prosody.for_repeat(rep, speaker.prosody_strength)
                for lang in ("es", "en"):
                    for item_index, item in enumerate(ITEMS):
                        audio = speaker.say(item[0 if lang == "es" else 1], lang,
                                            prosody, NEUTRAL)
                        segments[(rep, lang, item_index)] = audio
                        rows.append({"repeat": rep, "language": lang,
                                     "item": item_index,
                                     "synthesis": speaker.stats[-1]})
    finally:
        speaker.close()
    saved: list[dict] = []
    for (rep, lang, item), audio in sorted(segments.items()):
        name = f"rep-{rep + 1}-{lang}-item-{item + 1}.wav"
        sf.write(speech_dir / name, audio, SR)
        saved.append({"repeat": rep, "language": lang, "item": item,
                      "path": name, "duration_seconds": len(audio) / SR})
    metadata = {"backend": backend, "model": model, "items": ITEMS,
                "segments": saved, "requests": rows}
    _speech_manifest_path(out_dir).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return segments, metadata


def shared_speech(out_dir: Path, backend: str, model: str | None,
                  fallback: str, seed: int) -> tuple[dict, dict]:
    cached = _load_shared_speech(out_dir)
    if cached:
        return cached
    try:
        return _synthesize_backend(out_dir, backend, model, seed)
    except Exception as exc:
        if backend == fallback:
            raise
        warnings.warn(f"{backend} shared speech failed ({exc}); using {fallback}.")
        segments, metadata = _synthesize_backend(out_dir, fallback, None, seed)
        metadata["fallback_from"] = backend
        metadata["fallback_reason"] = str(exc)
        _speech_manifest_path(out_dir).write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return segments, metadata


def _speech_track(spec: BedSpec, segments: dict) -> tuple[np.ndarray, int, list[dict]]:
    grid = Grid.from_spec(spec)
    total_bars = max(20, int(round(74.0 / grid.bar)))
    while total_bars * grid.bar + 1 < 60:
        total_bars += 1
    while total_bars * grid.bar + 1 > 90 and total_bars > 20:
        total_bars -= 1
    if total_bars * grid.bar + 1 > 90:
        raise RuntimeError("The fixed speech schedule cannot fit within 90 seconds.")
    speech = np.zeros(grid.samples(total_bars * grid.bar) + SR, dtype=np.float32)
    events: list[dict] = []
    bar = 2
    for item_index, item in enumerate(ITEMS):
        for kind, rep in PATTERNS["retrieval"]:
            if kind in ("gap", "rest"):
                bar += 1
                continue
            audio = segments[(rep, kind, item_index)]
            target = grid.bar * 0.92
            fitted = fit(audio, target) if len(audio) / SR > target else audio
            at = grid.samples(grid.bar_start(bar))
            speech[at:at + len(fitted)] += fitted
            events.append({"bar": bar, "seconds": grid.bar_start(bar),
                           "language": kind, "repeat": rep, "item": item_index,
                           "text": item[0 if kind == "es" else 1],
                           "duration_seconds": len(fitted) / SR,
                           "fit_applied": len(fitted) != len(audio)})
            bar += 1
    return speech, total_bars, events


def _validate(audio: np.ndarray, spec: BedSpec, events: list[dict]) -> dict:
    duration = len(audio) / SR
    peak = float(np.abs(audio).max())
    finite = bool(np.isfinite(audio).all())
    downbeats = all(abs(event["seconds"] / Grid.from_spec(spec).bar -
                        round(event["seconds"] / Grid.from_spec(spec).bar)) < 1e-7
                    for event in events)
    speech_within_bars = all(event["duration_seconds"] <=
                             Grid.from_spec(spec).bar + 1e-7 for event in events)
    return {"sample_rate": SR, "channels": int(audio.shape[1]),
            "duration_seconds": duration, "peak": peak, "finite": finite,
            "speech_on_downbeats": downbeats,
            "speech_within_bars": speech_within_bars,
            "valid": finite and audio.ndim == 2 and audio.shape[1] == 2 and
                     60 <= duration <= 90 and peak <= 0.97 and downbeats and
                     speech_within_bars}


def _constrain_peak(audio: np.ndarray, ceiling: float = 0.969) -> np.ndarray:
    """Leave headroom so a float32 WAV also remains strictly below 0.97."""
    peak = float(np.abs(audio).max())
    if peak > ceiling:
        return (audio * (ceiling / peak)).astype(np.float32)
    return audio.astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--candidate-multiplier", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("out/music-bakeoff"))
    parser.add_argument("--sample-policy", choices=("none", "safe"),
                        default="safe")
    parser.add_argument("--family-profile", choices=("broad", "positive"),
                        default="broad")
    parser.add_argument("--palettes", nargs="+", choices=TIMBRE_PALETTES,
                        default=["hybrid"],
                        help="one or more palettes to cover in the candidate pool")
    parser.add_argument("--voice-backend", choices=("gemini-vertex", "gemini",
                                                     "chatterbox", "none"),
                        default="gemini-vertex")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fallback-backend", default="chatterbox")
    parser.add_argument("--speech-cache-from", type=Path,
                        help="reuse a validated speech/ directory from another bake-off")
    parser.add_argument("--replay-manifest", type=Path,
                        help="re-resolve the exact family/seed pairs from an earlier bake-off")
    args = parser.parse_args()
    if args.count < 1 or args.candidate_multiplier < 1:
        raise SystemExit("--count and --candidate-multiplier must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    library = SampleLibrary()
    assets: list[SampleAsset] = []
    if args.sample_policy != "none" and library.catalog_path.exists():
        assets = library.assets(collections=tuple(COLLECTIONS))
        if not library.external.exists():
            assets = [asset for asset in assets if library.is_promoted(asset)]
            warnings.warn(
                f"External sample tier is offline; candidate generation is limited to "
                f"{len(assets)} locally promoted catalog assets.", RuntimeWarning)
    families = POSITIVE_FAMILIES if args.family_profile == "positive" else BROAD_FAMILIES
    instruments = instrument_refs(assets)
    profile_name = ("production-v1" if args.family_profile == "positive"
                    else "exploration-v1")
    requests = [MusicRequest(seed=args.seed, profile=profile_name, palette=palette)
                for palette in args.palettes]
    if args.replay_manifest:
        source_manifest = json.loads(args.replay_manifest.read_text(encoding="utf-8"))
        source_clips = source_manifest["clips"]
        if len(source_clips) != args.count:
            raise ValueError(
                f"Replay manifest has {len(source_clips)} clips; expected {args.count}.")
        print(f"Replaying {len(source_clips)} exact family/seed pairs using "
              f"{len(assets)} catalog assets and {len(instruments)} instruments…",
              flush=True)
        selected = []
        rejected = []
        for row in source_clips:
            bed_seed = int(row["seed"])
            spec = BedSpec.from_style(row["family"], bed_seed)
            spec.engine_version = ENGINE_VERSION
            spec.profile_version = profile_name
            profile = get_profile(profile_name)
            palette = row.get("phrase", {}).get("palette", args.palettes[0])
            request = MusicRequest(
                seed=args.seed, profile=profile_name, palette=palette)
            _apply_request(spec, request, profile)
            enrich_with_catalog_samples(
                spec, assets, instruments, bed_seed, palette=request.palette)
            preview_stems = render_stems(
                spec, max(spec.phrase.loop_bars if spec.phrase else 4, 4))
            preview = sum(preview_stems.values(),
                          np.zeros_like(next(iter(preview_stems.values()))))
            quality, fingerprint = evaluate_preview(
                preview, preview_stems, spec, profile)
            collections = tuple(sorted(
                {ref.collection for ref in _refs(spec)} |
                {f"pack:{name}" for name in _named_pack_names(spec)}))
            selected.append(Candidate(
                row["family"], bed_seed, spec,
                np.asarray(fingerprint.audio_features), len(preview) / SR,
                collections, fingerprint=fingerprint, quality=quality))
            if not quality.accepted:
                rejected.append({
                    "family": row["family"], "seed": bed_seed,
                    "reason": "; ".join(quality.rejection_reasons),
                    "retained_for_paired_replay": True,
                })
        pool_size = len(source_clips)
    else:
        pool_size = max(args.count * args.candidate_multiplier, len(families) * 2)
        per_palette = max(len(families), math.ceil(pool_size / len(requests)))
        print(f"Building {per_palette * len(requests)} candidates across "
              f"{len(families)} families using {len(assets)} catalog assets and "
              f"{len(instruments)} multisample instruments and "
              f"{len(requests)} palettes…", flush=True)
        candidates = []
        rejected = []
        for palette_index, request in enumerate(requests):
            palette_seed = args.seed + palette_index * 1_000_003
            palette_candidates, palette_rejected = build_candidates(
                1, 1, palette_seed, assets, instruments, families,
                request=request, pool_size=per_palette)
            candidates.extend(palette_candidates)
            rejected.extend({**row, "palette": request.palette}
                            for row in palette_rejected)
        selected = select_balanced(candidates, args.count, families)
    refs = list({(ref.collection, ref.asset_id): ref
                 for candidate in selected for ref in _refs(candidate.spec)}.values())
    if refs:
        library.promote(refs)

    speech_segments: dict = {}
    speech_metadata: dict = {"backend": "none"}
    if args.voice_backend != "none":
        print("Preparing one shared bilingual voice set…", flush=True)
        if args.speech_cache_from:
            source = args.speech_cache_from / "speech"
            if not (source / "segments.json").exists():
                raise FileNotFoundError(f"No reusable speech manifest at {source}.")
            shutil.copytree(source, args.out_dir / "speech", dirs_exist_ok=True)
        speech_segments, speech_metadata = shared_speech(
            args.out_dir, args.voice_backend, args.model,
            args.fallback_backend, args.seed)
        if args.speech_cache_from:
            speech_metadata = dict(speech_metadata)
            speech_metadata["reused_from"] = str(args.speech_cache_from)

    manifest_rows: list[dict] = []
    for number, candidate in enumerate(selected, 1):
        spec = copy.deepcopy(candidate.spec)
        grid = Grid.from_spec(spec)
        if speech_segments:
            speech, total_bars, events = _speech_track(spec, speech_segments)
            stems = render_stems(spec, total_bars)
            depths = {name: getattr(spec, name).duck_db for name in stems}
            track = mix_stems(stems, speech, depths)
        else:
            total_bars = max(spec.phrase.loop_bars, round(74 / grid.bar))
            stems = render_stems(spec, total_bars)
            track = sum(stems.values(), np.zeros_like(next(iter(stems.values()))))
            peak = float(np.abs(track).max())
            if peak:
                track = track / peak * 0.7
            events = []
        track = _constrain_peak(track)
        validation = _validate(track, spec, events)
        if not validation["valid"]:
            raise RuntimeError(f"Validation failed for {candidate.family}/{candidate.seed}: "
                               f"{validation}")
        stem = f"{number:02d}-{candidate.family}-s{candidate.seed}"
        wav_path = args.out_dir / f"{stem}.wav"
        sf.write(wav_path, track, SR)
        spec.to_json(args.out_dir / f"{stem}.bed.json")
        row = {"number": number, "file": wav_path.name, "family": candidate.family,
               "seed": candidate.seed, "bpm": spec.bpm,
               "palette": spec.phrase.palette if spec.phrase else "legacy",
               "meter": f"{spec.beats_per_bar}/{spec.beat_unit}",
               "scale": spec.scale, "phrase": asdict(spec.phrase),
               "sample_refs": [asdict(ref) for ref in _refs(spec)],
               "sample_collections": candidate.sample_collections,
               "features": candidate.features.tolist(), "events": events,
               "validation": validation}
        manifest_rows.append(row)
        print(f"[{number:02d}/{args.count}] {candidate.family:<11} "
              f"{spec.bpm:>3g} BPM -> {wav_path.name}", flush=True)

    used_assets = [asdict(library.asset(ref)) for ref in refs]
    used_collections = {asset["collection"] for asset in used_assets}
    named_pack_names = sorted({name for candidate in selected
                               for name in _named_pack_names(candidate.spec)})
    named_packs = [{"id": name, "license": sample_packs.PACKS[name].license,
                    "attribution": sample_packs.PACKS[name].attribution,
                    "homepage": sample_packs.PACKS[name].homepage,
                    "files": sample_packs.PACKS[name].filenames()}
                   for name in named_pack_names]
    manifest = {"schema_version": 1, "seed": args.seed, "count": args.count,
                "candidate_multiplier": args.candidate_multiplier,
                "sample_policy": args.sample_policy,
                "family_profile": args.family_profile,
                "palettes": args.palettes,
                "replay_manifest": str(args.replay_manifest) if args.replay_manifest else None,
                "storage": library.status(),
                "voice": speech_metadata, "items": ITEMS,
                "sample_assets": used_assets,
                "sample_collections": library.collection_metadata(used_collections),
                "named_sample_packs": named_packs,
                "rejected_candidates": rejected, "clips": manifest_rows}
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.out_dir / "ratings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["number", "file", "family", "palette",
                         "distinctiveness_1_5",
                         "pleasantness_1_5", "rhythmic_usefulness_1_5",
                         "speech_clarity_1_5", "repetitiveness_1_5", "keep", "notes"])
        for row in manifest_rows:
            writer.writerow([row["number"], row["file"], row["family"],
                             row["palette"], "", "", "",
                             "", "", "", ""])
    guide = [
        "# Music bake-off listening guide", "",
        ("All clips use `el cava — sparkling wine` and `la salchicha — sausage` "
         "with the same three-repetition retrieval pattern and shared speech. "
         "Rate the music, not the vocabulary." if speech_segments else
         "These clips contain music only. Rate natural repetition, musical identity, "
         "pulse clarity, balance, and any sample artifacts."), "",
        (("Listen for distinct musical identity, pleasant repetition, a useful pulse, "
          "clear pronunciation, unobtrusive transitions, and any sample or alignment "
          "artifacts." if speech_segments else
          "Listen for distinct musical identity, pleasant repetition, a useful pulse, "
          "coherent articulation and any repeated-sample or microphone-switching "
          "artifacts.") + " Record scores in `ratings.csv`."), "",
        "| # | File | Family | Palette | BPM | Meter | Harmony | Bass | Motif | Sources |",
        "|---:|---|---|---|---:|---|---|---|---|---|",
    ]
    for row in manifest_rows:
        phrase = row["phrase"]
        sources = ", ".join(row["sample_collections"]) or "synth"
        guide.append(f"| {row['number']} | `{row['file']}` | {row['family']} | "
                     f"{row['palette']} | {row['bpm']:g} | {row['meter']} | "
                     f"{phrase['harmony_texture']} | {phrase['bass_timbre']} | "
                     f"{phrase['motif_grammar']} | {sources} |")
    (args.out_dir / "listening-guide.md").write_text(
        "\n".join(guide) + "\n", encoding="utf-8")
    print(f"Wrote {args.count} validated clips and comparison files to {args.out_dir}")


if __name__ == "__main__":
    main()
