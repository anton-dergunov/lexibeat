#!/usr/bin/env python3
"""Generate an Earworms-style language-learning track from a vocabulary list.

    uv run generate.py --words 12 --out out/spanish.wav

Everything runs locally: the music bed is synthesised, the speech comes from
Kokoro-82M. See DESIGN.md for the reasoning and for the alternatives that were
considered at each stage.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import soundfile as sf

from earworms.arrange import PATTERNS, arrange, render_speech
from earworms.mix import mix
from earworms.music import SR, Grid, render_bed
from earworms.vocab import load
from earworms.voice import DEFAULT_VOICES, Speaker

VOCAB_DIR = Path("/Users/anton/obsidian/Languages/Spanish/Vocabulary")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vocab", type=Path, nargs="+", default=[VOCAB_DIR],
                   help="markdown files or directories of vocabulary notes")
    p.add_argument("--words", type=int, default=12, help="how many items to teach")
    p.add_argument("--mode", choices=["words", "phrases", "mixed"], default="words",
                   help="teach headwords, example sentences, or both")
    p.add_argument("--pattern", choices=sorted(PATTERNS), default="retrieval",
                   help="'retrieval' leaves a silent bar to recall the answer in")
    p.add_argument("--bpm", type=float, default=80.0)
    p.add_argument("--seed", type=int, default=7, help="selection and music seed")
    p.add_argument("--out", type=Path, default=Path("out/lesson.wav"))
    p.add_argument("--voice-es", default=DEFAULT_VOICES["es"])
    p.add_argument("--voice-en", default=DEFAULT_VOICES["en"])
    p.add_argument("--duck-db", type=float, default=5.0,
                   help="how far the music drops while someone is speaking")
    p.add_argument("--speech-lufs", type=float, default=-16.0)
    p.add_argument("--music-lufs", type=float, default=-26.0)
    p.add_argument("--dry-run", action="store_true",
                   help="print the selected items and the timing plan, generate nothing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    grid = Grid(bpm=args.bpm)
    slots = len(PATTERNS[args.pattern])

    items = load(args.vocab, mode=args.mode, limit=args.words, seed=args.seed)
    if not items:
        raise SystemExit("No vocabulary items found — check --vocab.")

    minutes = (len(items) * slots + 4) * grid.bar / 60
    print(f"{len(items)} items · {args.bpm:g} BPM · bar {grid.bar:.2f}s · "
          f"{slots} bars each · ~{minutes:.1f} min · pattern '{args.pattern}'")

    if args.dry_run:
        for i, item in enumerate(items, 1):
            at = (2 + (i - 1) * slots) * grid.bar
            print(f"  {int(at)//60:02d}:{int(at)%60:02d}  {item.source} — {item.target}")
        return

    started = time.time()
    print("Synthesising speech…")
    speaker = Speaker({"es": args.voice_es, "en": args.voice_en})
    events, total_bars = arrange(items, speaker, grid, pattern=args.pattern)
    speech = render_speech(events, total_bars, grid)

    print("Rendering music bed…")
    bed = render_bed(grid, total_bars, seed=args.seed)

    print("Mixing…")
    track = mix(bed, speech, duck_db=args.duck_db, speech_lufs=args.speech_lufs,
                music_lufs=args.music_lufs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, track, SR)

    # A sidecar tracklist makes it possible to skip to a word while listening.
    listing = args.out.with_suffix(".txt")
    lines = [f"{args.out.name} — {len(items)} items, {args.bpm:g} BPM, "
             f"pattern '{args.pattern}'", ""]
    for i, item in enumerate(items):
        at = (2 + i * slots) * grid.bar
        lines.append(f"{int(at)//60:02d}:{int(at)%60:02d}  {item.source} — {item.target}")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nWrote {args.out} ({len(track)/SR/60:.1f} min) "
          f"and {listing.name} in {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
