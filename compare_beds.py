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
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from compare_gemini_batched import split_on_long_silences
from earworms.arrange import PATTERNS
from earworms.bedspec import BedSpec
from earworms.emotion import NEUTRAL
from earworms.library import (InstrumentRef, SampleAsset, SampleLibrary, SampleRef,
                              instrument_refs)
from earworms.mix import mix_stems
from earworms.music import Grid, SR, render_stems
from earworms import samples as sample_packs
from earworms.voice import Prosody, Speaker, fit


BROAD_FAMILIES = ("meditative", "organic", "acoustic", "nocturnal", "sunlit",
                  "lofi-wide")
POSITIVE_FAMILIES = ("meditative", "organic", "acoustic", "sunlit", "radiant",
                     "acoustic-flow", "playful-minimal", "warm-motion",
                     "bright-organic", "gentle-game", "sunlit-acoustic",
                     "gentle-movement", "playful-plucked", "bright-pastoral")
FAMILIES = BROAD_FAMILIES  # compatibility for callers importing the original pool
ITEMS = (("el cava", "sparkling wine"), ("la salchicha", "sausage"))
DEFAULT_MODEL = "gemini-3.1-flash-tts-preview"


@dataclass
class Candidate:
    family: str
    seed: int
    spec: BedSpec
    features: np.ndarray
    preview_seconds: float
    sample_collections: tuple[str, ...]


def _audio_features(audio: np.ndarray, spec: BedSpec) -> np.ndarray:
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    mono = np.asarray(mono, dtype=np.float64)
    peak = max(float(np.abs(mono).max()), 1e-9)
    rms = float(np.sqrt(np.mean(mono ** 2)))
    decimated = mono[::max(len(mono) // (SR * 8), 1)]
    spectrum = np.abs(np.fft.rfft(decimated * np.hanning(len(decimated))))
    freqs = np.fft.rfftfreq(len(decimated), 1 / (SR * len(mono) /
                                                  max(len(decimated), 1)))
    total = max(float(spectrum.sum()), 1e-12)
    centroid = float(np.sum(freqs * spectrum) / total)
    high = float(spectrum[freqs >= 2500].sum() / total)
    hop = max(SR // 100, 1)
    envelope = np.maximum.reduceat(np.abs(mono), np.arange(0, len(mono), hop))
    changes = np.maximum(np.diff(envelope, prepend=0), 0)
    onset_density = float(np.count_nonzero(changes > np.percentile(changes, 90)) /
                          max(len(mono) / SR, 1))
    phrase = spec.phrase
    downbeat_hits = 0
    if phrase.percussion:
        downbeat_hits = sum(
            phrase.percussion[0].pattern[bar * spec.steps_per_bar] == "x"
            for bar in range(phrase.loop_bars))
    return np.array([
        spec.bpm / 100, spec.beats_per_bar / 5, spec.swing,
        rms / peak, math.log10(max(centroid, 1)) / 4, high, onset_density / 10,
        len(phrase.chords) / max(phrase.loop_bars * spec.steps_per_bar, 1),
        len(phrase.bass) / max(phrase.loop_bars * spec.steps_per_bar, 1),
        len(phrase.lead) / max(phrase.loop_bars * spec.steps_per_bar, 1),
        len(phrase.percussion) / 4,
        downbeat_hits / max(phrase.loop_bars, 1),
    ], dtype=np.float64)


def _preference_score(candidate: Candidate) -> float:
    """Transparent prior distilled from the first 30 human ratings."""
    spec, phrase = candidate.spec, candidate.spec.phrase
    straight = 1.0 - min(spec.swing / 0.08, 1.0)
    downbeats = 0.0
    if phrase and phrase.percussion:
        downbeats = sum(
            phrase.percussion[0].pattern[bar * spec.steps_per_bar] == "x"
            for bar in range(phrase.loop_bars)) / phrase.loop_bars
    restrained_drums = 1.0 - min(abs(spec.drums.level - 0.55) / 0.2, 1.0)
    positive = 1.0 if spec.scale in ("major", "lydian") else 0.55
    clarity = _metrical_clarity(spec)
    return (0.28 * straight + 0.23 * downbeats + 0.2 * restrained_drums +
            0.12 * positive + 0.17 * clarity)


def _metrical_clarity(spec: BedSpec) -> float:
    """Score a stable low anchor and penalize bar-boundary collisions."""
    phrase = spec.phrase
    if not phrase or not phrase.percussion:
        return 1.0
    lane = phrase.percussion[0]
    steps = spec.steps_per_bar
    downbeats = sum(lane.pattern[bar * steps] == "x"
                    for bar in range(phrase.loop_bars))
    collisions = 0
    for boundary in range(steps, len(lane.pattern), steps):
        collisions += lane.pattern[boundary - 1:boundary + 2].count("x") > 1
    anchor = downbeats / max(phrase.loop_bars, 1)
    return max(0.0, anchor - collisions / max(phrase.loop_bars, 1) * 0.55)


def _motif_fingerprint(spec: BedSpec, length: int = 12) -> np.ndarray:
    """Describe melodic intervals and onset gaps independently of key/timbre."""
    phrase = spec.phrase
    if not phrase or not phrase.lead:
        return np.zeros(length * 2, dtype=np.float64)
    events = phrase.lead[:length + 1]
    intervals = np.diff([event.midi_note for event in events]) / 12.0
    gaps = np.diff([event.step for event in events]) / max(spec.steps_per_bar, 1)
    result = np.zeros(length * 2, dtype=np.float64)
    result[:min(length, len(intervals))] = intervals[:length]
    result[length:length + min(length, len(gaps))] = gaps[:length]
    return result


def select_balanced(candidates: list[Candidate], count: int,
                    families: tuple[str, ...] | None = None) -> list[Candidate]:
    """Round-robin families while maximizing distance from prior selections."""
    if count < 1:
        raise ValueError("count must be positive")
    if count > len(candidates):
        raise ValueError(f"Requested {count} candidates from a pool of {len(candidates)}.")
    matrix = np.stack([candidate.features for candidate in candidates])
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-9] = 1.0
    normalized = (matrix - mean) / scale
    motifs = np.stack([_motif_fingerprint(candidate.spec) for candidate in candidates])
    family_order = families or tuple(dict.fromkeys(
        candidate.family for candidate in candidates))
    by_family = {family: [index for index, candidate in enumerate(candidates)
                          if candidate.family == family] for family in family_order}
    selected: list[int] = []
    while len(selected) < count:
        made_progress = False
        for family in family_order:
            available = [index for index in by_family[family] if index not in selected]
            if not available or len(selected) >= count:
                continue
            if not selected:
                chosen = max(available, key=lambda index: (
                    _preference_score(candidates[index]) +
                    0.12 * float(np.linalg.norm(normalized[index])),
                    -candidates[index].seed))
            else:
                chosen = max(available, key=lambda index: (
                    min(float(np.linalg.norm(normalized[index] - normalized[prior])) +
                        0.7 * float(np.linalg.norm(motifs[index] - motifs[prior]))
                        for prior in selected) + 0.35 * _preference_score(candidates[index]),
                    -candidates[index].seed))
            selected.append(chosen)
            made_progress = True
        if not made_progress:
            break
    # The targeted CC0 additions and newly exposed organic instruments should
    # be audibly represented, not merely present in the catalog. Preserve the
    # one-per-family balance while maximizing these broad coverage tags.
    def coverage(indexes: list[int]) -> set[str]:
        tags: set[str] = set()
        for index in indexes:
            row = candidates[index]
            phrase = row.spec.phrase
            if "freepats-guitar" in row.sample_collections:
                tags.add("classical-guitar")
            if phrase and phrase.bass_instrument and "fashionbass" in \
                    phrase.bass_instrument.name.lower():
                tags.add("natural-bass")
            lead_name = phrase.lead_instrument.name.lower() \
                if phrase and phrase.lead_instrument else ""
            if any(word in lead_name for word in
                   ("mbira", "nyunga", "psaltery", "ocarina", "harmonica",
                    "/pizz", "/spic")):
                tags.add("expanded-front")
        return tags

    while True:
        current = coverage(selected)
        best_score: tuple[int, float, int, int] | None = None
        best_replacement: tuple[int, int] | None = None
        for position, old_index in enumerate(selected):
            family = candidates[old_index].family
            for new_index in by_family[family]:
                if new_index in selected:
                    continue
                proposal = [*selected]
                proposal[position] = new_index
                gained = len(coverage(proposal)) - len(current)
                if gained <= 0:
                    continue
                score = (gained, _preference_score(candidates[new_index]),
                         -candidates[new_index].seed, new_index)
                if best_score is None or score > best_score:
                    best_score = score
                    best_replacement = (position, new_index)
        if best_replacement is None:
            break
        selected[best_replacement[0]] = best_replacement[1]
    return [candidates[index] for index in selected]


def _matching(assets: list[SampleAsset], words: tuple[str, ...]) -> list[SampleAsset]:
    return [asset for asset in assets
            if any(word in asset.relative_path.lower() for word in words)]


def _choose(rng: np.random.Generator, values: list[SampleAsset]) -> SampleAsset | None:
    return values[int(rng.integers(0, len(values)))] if values else None


def _choose_across_collections(rng: np.random.Generator,
                               values: list[SampleAsset]) -> SampleAsset | None:
    """Give each source equal weight before choosing one of its assets."""
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


_ORNAMENT_WORDS = ("sleigh", "jingle", "bell", "cowbell", "chime", "cymbal",
                   "triangle", "musicbox", "music box", "roll")


def _role_assets(assets: list[SampleAsset], role: str) -> list[SampleAsset]:
    """Conservatively map one-shots to musical roles using names and spectra."""
    words = {
        "low": ("kick", "bass drum", "bassdrum", "low tom", "bass cajon"),
        "mid": ("snare", "rim", "wood", "clave", "castanet", "clap", "stick",
                "cardboard", "porcelain", "darbuka", "bongo", "conga", "cajon"),
        "high": ("shaker", "maraca", "hat", "tamb", "brush", "key"),
    }[role]
    matched = _matching(assets, words)
    if role == "low":
        aggressive = ("hardstyle", "rawstyle", "distkit", "synthkit", "x0xproc",
                      "sdbkit", "sub-a")
        return [asset for asset in matched
                if (asset.spectral_centroid is None or asset.spectral_centroid < 2400)
                and not any(word in asset.relative_path.lower() for word in aggressive)]
    if role == "high":
        return [asset for asset in matched if asset.spectral_centroid is None or
                asset.spectral_centroid > 1300]
    return matched


def enrich_with_catalog_samples(spec: BedSpec, assets: list[SampleAsset],
                                instruments: list[InstrumentRef], seed: int) -> None:
    """Resolve catalog assets into a phrase without consulting cache availability."""
    if spec.phrase is None or not assets:
        return
    rng = np.random.default_rng(seed * 7919 + 17)
    short = [asset for asset in assets if asset.category == "percussion" and
             asset.duration_seconds is not None and 0.015 <= asset.duration_seconds <= 3.0 and
             not any(word in asset.relative_path.lower() for word in _ORNAMENT_WORDS)]
    roles = {role: _role_assets(short, role) for role in ("low", "mid", "high")}
    for lane_index, lane in enumerate(spec.phrase.percussion):
        if rng.random() > 0.66:
            continue
        role = lane.role or ("low" if "kick" in lane.sound else "high" if any(
            name in lane.sound for name in ("shaker", "hat")) else "mid")
        asset = _choose_across_collections(rng, roles[role])
        if asset:
            lane.sample = asset.ref
            lane.sound = f"sample:{asset.collection}"

    preferences = {
        "piano": ("piano",),
        "marimba": ("marimba", "vibraphone", "xylophone", "kalimba", "mbira",
                    "nyunga"),
        "glockenspiel": ("glockenspiel", "celesta", "concert harp", "folk harp",
                          "psaltery"),
        "synth": ("recorder", "flute", "oboe", "clarinet", "strumstick",
                  "guitar", "harp", "organ", "ocarina", "harmonica", "bassoon",
                  "pizz", "spic", "mbira", "nyunga", "psaltery"),
    }
    family_preferences = {
        "sunlit-acoustic": ("guitar", "psaltery", "pizz", "harp", "mbira"),
        "gentle-movement": ("piano", "vibraphone", "ocarina", "recorder", "pizz"),
        "playful-plucked": ("mbira", "nyunga", "kalimba", "psaltery", "guitar",
                            "pizz"),
        "bright-pastoral": ("ocarina", "harmonica", "flute", "recorder", "pizz",
                            "psaltery"),
    }
    preferred_words = preferences.get(spec.lead.instrument, ())
    if spec.phrase.family in family_preferences and rng.random() < 0.76:
        preferred_words = family_preferences[spec.phrase.family]
    compatible_by_word = {
        word: [instrument for instrument in instruments
               if _instrument_matches(instrument.name, word) and
               "contrabass" not in instrument.name.lower()]
        for word in preferred_words
    }
    compatible_by_word = {word: values for word, values in compatible_by_word.items()
                          if values}
    compatible = [instrument for values in compatible_by_word.values()
                  for instrument in values]
    # Preserve the dedicated Salamander/VSCO instruments frequently, while
    # exercising complete catalog instruments instead of isolated note files.
    catalog_probability = (0.5 if spec.lead.instrument == "piano" else
                           0.76 if spec.phrase.family in family_preferences else 0.64)
    if compatible and rng.random() < catalog_probability:
        if spec.phrase.family in family_preferences:
            timbres = sorted(compatible_by_word)
            choices = compatible_by_word[timbres[int(rng.integers(0, len(timbres)))]]
        else:
            by_collection: dict[str, list[InstrumentRef]] = {}
            for instrument in compatible:
                collection = instrument.name.split(":", 1)[0]
                by_collection.setdefault(collection, []).append(instrument)
            collection = sorted(by_collection)[int(rng.integers(0, len(by_collection)))]
            choices = by_collection[collection]
        spec.phrase.lead_instrument = choices[int(rng.integers(0, len(choices)))]

    natural_basses = [instrument for instrument in instruments
                      if "fashionbass" in instrument.name.lower()]
    if natural_basses and rng.random() < 0.34:
        spec.phrase.bass_instrument = natural_basses[
            int(rng.integers(0, len(natural_basses)))]


def _refs(spec: BedSpec) -> list[SampleRef]:
    if spec.phrase is None:
        return []
    refs = [lane.sample for lane in spec.phrase.percussion if lane.sample]
    refs.extend(ref for ref in (spec.phrase.lead_sample, spec.phrase.pad_sample) if ref)
    for instrument in (spec.phrase.lead_instrument, spec.phrase.pad_instrument,
                       spec.phrase.bass_instrument):
        if instrument:
            refs.extend(zone.sample for zone in instrument.zones)
    return list({(ref.collection, ref.asset_id): ref for ref in refs}.values())


def _named_pack_names(spec: BedSpec) -> set[str]:
    if spec.phrase is None:
        return set()
    aliases = {"piano": "salamander", "marimba": "vsco-marimba",
               "glockenspiel": "vsco-glockenspiel", "strings": "vsco-strings"}
    names: set[str] = set()
    if not spec.phrase.lead_instrument and not spec.phrase.lead_sample:
        name = aliases.get(spec.lead.instrument)
        if name:
            names.add(name)
    if not spec.phrase.pad_instrument and not spec.phrase.pad_sample:
        name = aliases.get(spec.pad.instrument)
        if name:
            names.add(name)
    return names


def build_candidates(count: int, multiplier: int, seed: int,
                     assets: list[SampleAsset], instruments: list[InstrumentRef] | None = None,
                     families: tuple[str, ...] = FAMILIES) -> tuple[list[Candidate], list[dict]]:
    pool_size = max(count * multiplier, len(families) * 2)
    candidates: list[Candidate] = []
    rejected: list[dict] = []
    for index in range(pool_size):
        family = families[index % len(families)]
        bed_seed = seed + index * 104729
        spec = BedSpec.from_style(family, bed_seed)
        if assets:
            enrich_with_catalog_samples(spec, assets, instruments or [], bed_seed)
        grid = Grid.from_spec(spec)
        minimum_lesson = 20 * grid.bar + 1.0
        if minimum_lesson > 90.0:
            rejected.append({"family": family, "seed": bed_seed,
                             "reason": "speech schedule exceeds 90 seconds"})
            continue
        try:
            bars = max(spec.phrase.loop_bars if spec.phrase else 4, 4)
            stems = render_stems(spec, bars)
            preview = sum(stems.values(), np.zeros_like(next(iter(stems.values()))))
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            rejected.append({"family": family, "seed": bed_seed,
                             "reason": str(exc)})
            continue
        if not np.isfinite(preview).all() or not float(np.abs(preview).max()):
            rejected.append({"family": family, "seed": bed_seed,
                             "reason": "empty or non-finite preview"})
            continue
        preview_rms = float(np.sqrt(np.mean(preview.astype(np.float64) ** 2)))
        drum_rms = float(np.sqrt(np.mean(
            stems["drums"].astype(np.float64) ** 2)))
        drum_share = drum_rms / max(preview_rms, 1e-9)
        if drum_share > 0.60:
            rejected.append({"family": family, "seed": bed_seed,
                             "reason": f"percussion dominance {drum_share:.3f}"})
            continue
        collections = tuple(sorted(
            {ref.collection for ref in _refs(spec)} |
            {f"pack:{name}" for name in _named_pack_names(spec)}))
        candidates.append(Candidate(family, bed_seed, spec,
                                    _audio_features(preview, spec),
                                    len(preview) / SR, collections))
    if len(candidates) < count:
        raise RuntimeError(f"Only {len(candidates)} valid candidates for --count {count}; "
                           "install referenced instrument packs or increase the pool.")
    return candidates, rejected


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
    parser.add_argument("--sample-policy", choices=("none", "safe", "all"),
                        default="safe")
    parser.add_argument("--family-profile", choices=("broad", "positive"),
                        default="broad")
    parser.add_argument("--voice-backend", choices=("gemini-vertex", "gemini",
                                                     "chatterbox", "none"),
                        default="gemini-vertex")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fallback-backend", default="chatterbox")
    parser.add_argument("--speech-cache-from", type=Path,
                        help="reuse a validated speech/ directory from another bake-off")
    args = parser.parse_args()
    if args.count < 1 or args.candidate_multiplier < 1:
        raise SystemExit("--count and --candidate-multiplier must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    library = SampleLibrary()
    assets: list[SampleAsset] = []
    if args.sample_policy != "none" and library.catalog_path.exists():
        collections = None if args.sample_policy == "all" else tuple(
            name for name, source in __import__("earworms.library", fromlist=["COLLECTIONS"])
            .COLLECTIONS.items() if source.safe_default)
        assets = library.assets(collections=collections)
        if not library.external.exists():
            assets = [asset for asset in assets if library.is_promoted(asset)]
            warnings.warn(
                f"External sample tier is offline; candidate generation is limited to "
                f"{len(assets)} locally promoted catalog assets.", RuntimeWarning)
    families = POSITIVE_FAMILIES if args.family_profile == "positive" else BROAD_FAMILIES
    instruments = instrument_refs(assets)
    print(f"Building {args.count * args.candidate_multiplier} candidates across "
          f"{len(families)} families using {len(assets)} catalog assets and "
          f"{len(instruments)} multisample instruments…", flush=True)
    candidates, rejected = build_candidates(args.count, args.candidate_multiplier,
                                            args.seed, assets, instruments, families)
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
        writer.writerow(["number", "file", "family", "distinctiveness_1_5",
                         "pleasantness_1_5", "rhythmic_usefulness_1_5",
                         "speech_clarity_1_5", "repetitiveness_1_5", "keep", "notes"])
        for row in manifest_rows:
            writer.writerow([row["number"], row["file"], row["family"], "", "", "",
                             "", "", "", ""])
    guide = [
        "# Music bake-off listening guide", "",
        "All clips use `el cava — sparkling wine` and `la salchicha — sausage` "
        "with the same three-repetition retrieval pattern and shared Gemini speech. "
        "Rate the music, not the vocabulary.", "",
        "Listen for distinct musical identity, pleasant repetition, a useful pulse, "
        "clear pronunciation, unobtrusive transitions, and any sample or alignment "
        "artifacts. Record scores in `ratings.csv`.", "",
        "| # | File | Family | BPM | Meter | Harmony | Bass | Percussion | Sources |",
        "|---:|---|---|---:|---|---|---|---|---|",
    ]
    for row in manifest_rows:
        phrase = row["phrase"]
        percussion = ", ".join(lane["pattern"] for lane in phrase["percussion"])
        sources = ", ".join(row["sample_collections"]) or "synth"
        guide.append(f"| {row['number']} | `{row['file']}` | {row['family']} | "
                     f"{row['bpm']:g} | {row['meter']} | "
                     f"{phrase['harmony_texture']} | {phrase['bass_timbre']} | "
                     f"{percussion} | {sources} |")
    (args.out_dir / "listening-guide.md").write_text(
        "\n".join(guide) + "\n", encoding="utf-8")
    print(f"Wrote {args.count} validated clips and comparison files to {args.out_dir}")


if __name__ == "__main__":
    main()
