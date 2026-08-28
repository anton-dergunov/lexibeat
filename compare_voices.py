#!/usr/bin/env python3
"""Render the same words through several voice setups so they can be compared.

    uv run compare_voices.py --words 4

Writes one file per configuration into out/compare/, each containing every
repeat of every word back to back — the repeats are the thing under test, since
that is where the old pipeline started to sound mechanical.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from earworms.emotion import for_item
from earworms.music import SR
from earworms.vocab import load
from earworms.voice import Prosody, Speaker, ensure_reference

VOCAB_DIR = Path("/Users/anton/obsidian/Languages/Spanish/Vocabulary")
OUT = Path("out/compare")

# name -> kwargs for Speaker
CONFIGS: dict[str, dict] = {
    "kokoro-full": dict(backend="kokoro", prosody_strength=1.0),
    "kokoro-gentle": dict(backend="kokoro", prosody_strength=0.5),
    "kokoro-flat": dict(backend="kokoro", prosody_strength=0.0),
    "chatterbox-kokoro-ref": dict(backend="chatterbox"),
    "chatterbox-paulina-ref": dict(backend="chatterbox", ref_audio="say:Paulina"),
}


def render(name: str, kwargs: dict, items, reps: int, emotions: bool) -> dict:
    if kwargs.get("ref_audio", "").startswith("say:"):
        kwargs = {**kwargs, "ref_audio": str(ensure_reference(kwargs["ref_audio"]))}

    started = time.time()
    speaker = Speaker(backend=kwargs.pop("backend"), **kwargs)
    load_time = time.time() - started

    pieces, count, synth_seconds = [], 0, 0.0
    gap = np.zeros(int(0.35 * SR), dtype=np.float32)
    long_gap = np.zeros(int(0.9 * SR), dtype=np.float32)

    for item in items:
        emotion = for_item(item.source, item.emoji, enabled=emotions)
        for rep in range(reps):
            prosody = Prosody.for_repeat(rep, speaker.prosody_strength)
            prosody = prosody.with_emotion(emotion, speaker.prosody_strength)
            for lang, text in (("es", item.source), ("en", item.target)):
                t0 = time.time()
                audio = speaker.say(text, lang, prosody, emotion)
                synth_seconds += time.time() - t0
                count += 1
                pieces += [audio, gap]
        pieces.append(long_gap)

    track = np.concatenate(pieces)
    peak = np.abs(track).max()
    if peak > 0:
        track = track / peak * 0.95
    path = OUT / f"{name}.wav"
    sf.write(path, track, SR)
    return {"name": name, "path": path, "load": load_time,
            "per_utterance": synth_seconds / max(count, 1),
            "audio": len(track) / SR, "count": count}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--words", type=int, default=3)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--configs", nargs="+", default=list(CONFIGS),
                   choices=list(CONFIGS))
    p.add_argument("--no-emotion", action="store_true")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    items = load([VOCAB_DIR], mode="words", limit=args.words, seed=args.seed)
    print("Words:", ", ".join(
        f"{i.emoji}{i.source} ({for_item(i.source, i.emoji).name})" for i in items))
    print()

    rows = []
    for name in args.configs:
        print(f"→ {name}", flush=True)
        try:
            rows.append(render(name, dict(CONFIGS[name]), items, args.reps,
                               not args.no_emotion))
        except Exception as exc:
            print(f"   failed: {exc}")

    print(f"\n{'config':<24} {'load':>7} {'per utt':>9} {'audio':>7}  file")
    for r in rows:
        print(f"{r['name']:<24} {r['load']:>6.1f}s {r['per_utterance']:>8.2f}s "
              f"{r['audio']:>6.1f}s  {r['path']}")


if __name__ == "__main__":
    main()
