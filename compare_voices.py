#!/usr/bin/env python3
"""Render the same words through several voice setups so they can be compared.

    uv run compare_voices.py --words 4

Writes one file per configuration into out/compare/, each containing every
repeat of every word back to back — the repeats are the thing under test, since
that is where the old pipeline started to sound mechanical.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from earworms.emotion import for_item
from earworms.music import SR
from earworms.vocab import load
from earworms.voice import Prosody, Speaker

VOCAB_DIR = Path("/Users/anton/obsidian/Languages/Spanish/Vocabulary")
OUT = Path("out/compare")

# name -> kwargs for Speaker
CONFIGS: dict[str, dict] = {
    "kokoro-full": dict(backend="kokoro", prosody_strength=1.0),
    "kokoro-gentle": dict(backend="kokoro", prosody_strength=0.5),
    "kokoro-flat": dict(backend="kokoro", prosody_strength=0.0),
    "chatterbox-bilingual": dict(
        backend="chatterbox",
        ref_audios={"es": "say:Paulina", "en": "say:Daniel"},
    ),
    "chatterbox-shared-paulina": dict(
        backend="chatterbox", ref_audio="say:Paulina"),
    "gemini": dict(backend="gemini"),
    "gemini-vertex-31": dict(
        backend="gemini-vertex", model="gemini-3.1-flash-tts-preview"),
    "gemini-vertex-25-flash": dict(
        backend="gemini-vertex", model="gemini-2.5-flash-tts"),
    "gemini-vertex-25-lite": dict(
        backend="gemini-vertex", model="gemini-2.5-flash-lite-preview-tts"),
    "gemini-vertex-25-pro": dict(
        backend="gemini-vertex", model="gemini-2.5-pro-tts"),
    "cloudflare-aura2": dict(backend="cloudflare-aura2"),
    # Cloudflare currently rejects MeloTTS lang="es"; retain a useful English
    # diagnostic without silently substituting a different model for Spanish.
    "cloudflare-melotts": dict(backend="cloudflare-melotts", languages=("en",)),
}


def render(name: str, kwargs: dict, items, reps: int, emotions: bool,
           out_dir: Path) -> dict:
    started = time.time()
    languages = tuple(kwargs.pop("languages", ("es", "en")))
    speaker = Speaker(backend=kwargs.pop("backend"), **kwargs)
    load_time = time.time() - started

    pieces, count, synth_seconds = [], 0, 0.0
    gap = np.zeros(int(0.35 * SR), dtype=np.float32)
    long_gap = np.zeros(int(0.9 * SR), dtype=np.float32)

    for item_number, item in enumerate(items, 1):
        print(f"   [{item_number}/{len(items)}] {item.source} — {item.target}",
              flush=True)
        emotion = for_item(item.source, item.emoji, enabled=emotions)
        for rep in range(reps):
            prosody = Prosody.for_repeat(rep, speaker.prosody_strength)
            prosody = prosody.with_emotion(emotion, speaker.prosody_strength)
            texts = {"es": item.source, "en": item.target}
            for lang in languages:
                text = texts[lang]
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
    path = out_dir / f"{name}.wav"
    sf.write(path, track, SR)
    estimated_cost = sum(float(row.get("controls", {}).get(
        "estimated_cost_usd") or 0.0) for row in speaker.stats)
    result = {"name": name, "path": str(path), "load": load_time,
              "per_utterance": synth_seconds / max(count, 1),
              "audio": len(track) / SR, "count": count,
              "languages": list(languages),
              "estimated_provider_cost_usd": estimated_cost,
              "seed_warning": ("Hosted APIs do not honor voice_seed."
                               if name.startswith(("gemini", "cloudflare-"))
                               else None),
              "utterances": speaker.stats}
    (out_dir / f"{name}.stats.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    speaker.close()
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--words", type=int, default=3)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--configs", nargs="+", default=list(CONFIGS),
                   choices=list(CONFIGS))
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--no-emotion", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    items = load([VOCAB_DIR], mode="words", limit=args.words, seed=args.seed)
    print("Words:", ", ".join(
        f"{i.emoji}{i.source} ({for_item(i.source, i.emoji).name})" for i in items))
    print()

    rows = []
    failures = 0
    for name in args.configs:
        print(f"→ {name}", flush=True)
        try:
            rows.append(render(name, dict(CONFIGS[name]), items, args.reps,
                               not args.no_emotion, args.out_dir))
        except Exception as exc:
            print(f"   failed: {exc}")
            failures += 1

    print(f"\n{'config':<24} {'load':>7} {'per utt':>9} {'audio':>7} "
          f"{'est. cost':>10}  file")
    for r in rows:
        print(f"{r['name']:<24} {r['load']:>6.1f}s {r['per_utterance']:>8.2f}s "
              f"{r['audio']:>6.1f}s ${r['estimated_provider_cost_usd']:>9.5f}  "
              f"{r['path']}")
    comparison_path = args.out_dir / "comparison.json"
    existing_rows = []
    if comparison_path.exists():
        try:
            existing_rows = json.loads(comparison_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing_rows = []
    combined = {row["name"]: row for row in existing_rows}
    combined.update({row["name"]: row for row in rows})
    comparison_path.write_text(
        json.dumps(list(combined.values()), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    listening = [
        "# Hosted TTS listening guide", "",
        f"Compare {len(items)} vocabulary items with {args.reps} repetitions each.", "",
        "Listen for:", "",
        "- correct Spanish stress and accent;",
        "- clear, natural English;",
        "- subtle but audible variation between repetitions;",
        "- added, omitted, translated, or hallucinated speech;",
        "- timing fits or audible speed/pitch post-processing;",
        "- whether the voice stays engaging without becoming theatrical.", "",
        "Structural success does not establish transcript faithfulness. Hosted APIs "
        "do not honor `voice_seed`.",
    ]
    (args.out_dir / "listening-guide.md").write_text(
        "\n".join(listening) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(f"{failures}/{len(args.configs)} configurations failed")


if __name__ == "__main__":
    main()
