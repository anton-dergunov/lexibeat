#!/usr/bin/env python3
"""Compare six-call Gemini TTS batching against per-utterance synthesis.

Each provider request contains all vocabulary items for one language and one
repeat style, separated by Gemini 3.1's documented ``[long pause]`` tag. The
continuous PCM response is split locally at its longest silent regions and then
reassembled into the ordinary item/repetition/language listening order.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from lexibeat.emotion import NEUTRAL
from lexibeat.music import SR
from lexibeat.vocab import load
from lexibeat.voice import Prosody, Speaker

VOCAB_DIR = Path("/Users/anton/obsidian/Languages/Spanish/Vocabulary")
MODELS = (
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-tts",
    "gemini-2.5-flash-lite-preview-tts",
    "gemini-2.5-pro-tts",
)


def split_on_long_silences(
    audio: np.ndarray,
    expected: int,
    *,
    sample_rate: int = SR,
    min_silence_seconds: float = 0.35,
) -> tuple[list[np.ndarray], dict]:
    """Split at the expected-1 strongest silent gaps, with strict validation."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if expected < 1 or not audio.size or not np.all(np.isfinite(audio)):
        raise ValueError("Batched audio must be finite, non-empty, and expected > 0.")
    if expected == 1:
        return [audio], {"top_db": None, "pause_seconds": [], "confidence": None}

    best: tuple[float, list[float], list[int], int] | None = None
    frame_length = max(32, int(round(0.025 * sample_rate)))
    hop_length = max(8, int(round(0.003 * sample_rate)))
    for top_db in (20, 25, 30, 35, 40, 45, 50):
        intervals = librosa.effects.split(
            audio, top_db=top_db, frame_length=frame_length,
            hop_length=hop_length)
        if len(intervals) < expected:
            continue
        gaps = [
            (int(intervals[index][0] - intervals[index - 1][1]),
             int((intervals[index][0] + intervals[index - 1][1]) // 2))
            for index in range(1, len(intervals))
            if intervals[index][0] > intervals[index - 1][1]
        ]
        if len(gaps) < expected - 1:
            continue
        ranked = sorted(gaps, reverse=True)
        chosen = ranked[:expected - 1]
        pause_seconds = [gap / sample_rate for gap, _ in chosen]
        shortest = min(pause_seconds)
        next_gap = (ranked[expected - 1][0] / sample_rate
                    if len(ranked) >= expected else 0.0)
        confidence = shortest / max(next_gap, 1.0 / sample_rate)
        score = shortest + min(confidence, 10.0) * 0.01
        if best is None or score > best[0]:
            best = (score, pause_seconds,
                    sorted(midpoint for _, midpoint in chosen), top_db)

    if best is None or min(best[1]) < min_silence_seconds:
        observed = 0.0 if best is None else min(best[1])
        raise RuntimeError(
            f"Could not find {expected - 1} reliable pause boundaries; "
            f"shortest candidate was {observed:.3f}s.")

    _, pauses, boundaries, top_db = best
    edges = [0, *boundaries, len(audio)]
    segments: list[np.ndarray] = []
    for start, end in zip(edges, edges[1:]):
        segment, _ = librosa.effects.trim(audio[start:end], top_db=32)
        segment = np.asarray(segment, dtype=np.float32)
        if len(segment) < int(0.08 * sample_rate):
            raise RuntimeError("A batched Gemini segment was empty or implausibly short.")
        segments.append(segment)
    ranked_pauses = sorted(pauses)
    intervals = librosa.effects.split(
        audio, top_db=top_db, frame_length=frame_length,
        hop_length=hop_length)
    all_gaps = sorted((int(intervals[index][0] - intervals[index - 1][1]) /
                       sample_rate for index in range(1, len(intervals))),
                      reverse=True)
    next_gap = all_gaps[expected - 1] if len(all_gaps) >= expected else 0.0
    return segments, {
        "top_db": top_db,
        "pause_seconds": ranked_pauses,
        "shortest_pause_seconds": min(ranked_pauses),
        "next_unselected_gap_seconds": next_gap,
        "confidence": min(ranked_pauses) / max(next_gap, 1.0 / sample_rate),
        "segment_seconds": [len(segment) / sample_rate for segment in segments],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--words", type=int, default=10)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", choices=MODELS,
                        default="gemini-3.1-flash-tts-preview")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("out/hosted-tts-batched"))
    parser.add_argument("--min-silence-seconds", type=float, default=0.35)
    args = parser.parse_args()
    if args.words < 1 or args.reps < 1:
        raise SystemExit("--words and --reps must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    items = load([VOCAB_DIR], mode="words", limit=args.words, seed=args.seed)
    speaker = Speaker(backend="gemini-vertex", model=args.model,
                      voice_seed=args.seed)
    batches: dict[tuple[int, str], list[np.ndarray]] = {}
    batch_stats: list[dict] = []
    started = time.perf_counter()
    try:
        for rep in range(args.reps):
            prosody = Prosody.for_repeat(rep, speaker.prosody_strength)
            for lang in ("es", "en"):
                phrases = [item.source if lang == "es" else item.target
                           for item in items]
                transcript = "\n[long pause]\n".join(phrases)
                print(f"[{rep + 1}/{args.reps}] {lang}: {len(phrases)} phrases",
                      flush=True)
                audio = speaker.say(transcript, lang, prosody, NEUTRAL)
                raw_path = raw_dir / f"rep-{rep + 1}-{lang}.wav"
                sf.write(raw_path, audio, SR)
                segments, split_stats = split_on_long_silences(
                    audio, len(phrases), sample_rate=SR,
                    min_silence_seconds=args.min_silence_seconds)
                batches[(rep, lang)] = segments
                batch_stats.append({
                    "repeat": rep + 1,
                    "language": lang,
                    "phrases": phrases,
                    "prosody": asdict(prosody),
                    "raw_path": str(raw_path),
                    "split": split_stats,
                    "synthesis": speaker.stats[-1],
                })
    finally:
        speaker.close()

    gap = np.zeros(int(0.35 * SR), dtype=np.float32)
    long_gap = np.zeros(int(0.9 * SR), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for item_index in range(len(items)):
        for rep in range(args.reps):
            for lang in ("es", "en"):
                pieces.extend((batches[(rep, lang)][item_index], gap))
        pieces.append(long_gap)
    track = np.concatenate(pieces)
    peak = float(np.abs(track).max())
    if peak > 0:
        track = track / peak * 0.95
    stem = args.model.replace("gemini-", "").replace("-tts-preview", "")
    wav_path = args.out_dir / f"gemini-batched-{stem}.wav"
    sf.write(wav_path, track, SR)
    estimated_cost = sum(float(row["synthesis"].get("controls", {}).get(
        "estimated_cost_usd") or 0.0) for row in batch_stats)
    stats = {
        "schema_version": 1,
        "experiment": "gemini-long-pause-batching",
        "model": args.model,
        "provider_calls": len(batch_stats),
        "items": len(items),
        "repetitions": args.reps,
        "expected_utterances": len(items) * args.reps * 2,
        "duration_seconds": len(track) / SR,
        "finite": bool(np.isfinite(track).all()),
        "peak": float(np.abs(track).max()),
        "estimated_provider_cost_usd": estimated_cost,
        "elapsed_seconds": time.perf_counter() - started,
        "output": str(wav_path),
        "batches": batch_stats,
        "warning": (
            "Segmentation is heuristic: listen for omitted, merged, translated, "
            "or spoken pause tags before using individual segments."),
    }
    stats_path = wav_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
    (args.out_dir / "listening-guide.md").write_text(
        "# Batched Gemini TTS listening guide\n\n"
        "Compare the reconstructed WAV with the six files under `raw/`.\n\n"
        "Listen for:\n\n"
        "- all ten phrases appearing exactly once and in the saved order;\n"
        "- `[long pause]` never being spoken aloud;\n"
        "- clean segment starts and endings with no clipped phonemes;\n"
        "- correct Spanish stress and clear English;\n"
        "- consistent delivery within a batch and useful variation across repeats;\n"
        "- merged words, hallucinated speech, or unusually long retained silence.\n\n"
        "Silence confidence is structural evidence, not transcript verification.\n",
        encoding="utf-8")
    print(f"Wrote {wav_path}: {len(batch_stats)} provider calls, "
          f"estimated ${estimated_cost:.6f}")


if __name__ == "__main__":
    main()
