#!/usr/bin/env python3
"""Generate an Earworms-style language-learning track from a vocabulary list.

    uv run generate.py --words 12 --out out/spanish.wav

Everything runs locally: the music bed is synthesised from a parameter set, the
speech comes from Kokoro-82M or, on Apple Silicon, from Chatterbox via
mlx-audio. See DESIGN.md for the reasoning and the alternatives considered.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import soundfile as sf

from earworms.arrange import PATTERNS, arrange, render_speech
from earworms.bedspec import STYLES, BedSpec
from earworms.mix import mix
from earworms.music import SR, Grid, render_bed
from earworms.samples import PACKS, download
from earworms.vocab import load
from earworms.voice import DEFAULT_MLX_MODEL, DEFAULT_VOICES, Speaker

VOCAB_DIR = Path("/Users/anton/obsidian/Languages/Spanish/Vocabulary")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    src = p.add_argument_group("vocabulary")
    src.add_argument("--vocab", type=Path, nargs="+", default=[VOCAB_DIR],
                     help="markdown files or directories of vocabulary notes")
    src.add_argument("--words", type=int, default=12, help="how many items to teach")
    src.add_argument("--mode", choices=["words", "phrases", "mixed"], default="words",
                     help="teach headwords, example sentences, or both")
    src.add_argument("--seed", type=int, default=7, help="item selection seed")

    lesson = p.add_argument_group("lesson shape")
    lesson.add_argument("--pattern", choices=sorted(PATTERNS), default="retrieval",
                        help="'retrieval' leaves a silent bar to recall the answer in")

    music = p.add_argument_group("music bed")
    music.add_argument("--bed-style", choices=sorted(STYLES), default="yoga")
    music.add_argument("--bed-seed", type=int, default=None,
                       help="varies the bed independently of item selection")
    music.add_argument("--bed-spec", type=Path,
                       help="load a saved bed spec JSON instead of a style")
    music.add_argument("--bpm", type=float, default=None,
                       help="override the tempo the style chose")
    music.add_argument("--instrument", default=None,
                       help="lead instrument: 'synth', 'piano'")
    music.add_argument("--bed-only", action="store_true",
                       help="render just the music, with no speech")
    music.add_argument("--download-samples", nargs="?", const="salamander",
                       choices=sorted(PACKS), help="fetch an instrument sample pack")

    voice = p.add_argument_group("voice")
    voice.add_argument("--backend", choices=["kokoro", "chatterbox"], default="kokoro",
                       help="'chatterbox' is far better and far slower (Apple Silicon)")
    voice.add_argument("--model", default=DEFAULT_MLX_MODEL)
    voice.add_argument("--ref-audio", help="voice to clone for the chatterbox backend")
    voice.add_argument("--voice-es", default=DEFAULT_VOICES["es"])
    voice.add_argument("--voice-en", default=DEFAULT_VOICES["en"])
    voice.add_argument("--prosody-strength", type=float, default=1.0,
                       help="0 disables per-repeat variation, 1 is the default")
    voice.add_argument("--no-emotion", action="store_true",
                       help="ignore the emoji in the notes and read everything neutrally")

    out = p.add_argument_group("output")
    out.add_argument("--out", type=Path, default=Path("out/lesson.wav"))
    out.add_argument("--duck-db", type=float, default=5.0,
                     help="how far the music drops while someone is speaking")
    out.add_argument("--speech-lufs", type=float, default=-16.0)
    out.add_argument("--music-lufs", type=float, default=-26.0)
    out.add_argument("--dry-run", action="store_true",
                     help="print the selected items and the timing plan only")
    return p.parse_args()


def build_spec(args: argparse.Namespace) -> tuple[BedSpec, str]:
    """Resolve the bed parameters, and a label describing where they came from."""
    if args.bed_spec:
        spec = BedSpec.from_json(args.bed_spec)
        label = args.bed_spec.name
    else:
        seed = args.bed_seed if args.bed_seed is not None else args.seed
        spec = BedSpec.from_style(args.bed_style, seed)
        label = args.bed_style
    if args.bpm:
        spec.bpm = args.bpm
    if args.instrument:
        spec.lead.instrument = args.instrument
    return spec, label


def main() -> None:
    args = parse_args()

    if args.download_samples:
        download(PACKS[args.download_samples])
        return

    spec, bed_label = build_spec(args)
    grid = Grid.from_spec(spec)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.bed_only:
        bars = 24
        started = time.time()
        sf.write(args.out, render_bed(spec, bars), SR)
        spec.to_json(args.out.with_suffix(".bed.json"))
        print(f"{bed_label} · {spec.bpm:g} BPM · {spec.scale} · "
              f"lead {spec.lead.instrument} · {bars * grid.bar:.0f}s "
              f"in {time.time()-started:.1f}s -> {args.out}")
        return

    slots = len(PATTERNS[args.pattern])
    items = load(args.vocab, mode=args.mode, limit=args.words, seed=args.seed)
    if not items:
        raise SystemExit("No vocabulary items found — check --vocab.")

    minutes = (len(items) * slots + 4) * grid.bar / 60
    print(f"{len(items)} items · {spec.bpm:g} BPM · bar {grid.bar:.2f}s · "
          f"{slots} bars each · ~{minutes:.1f} min · pattern '{args.pattern}' · "
          f"bed '{bed_label}' · voice '{args.backend}'")

    if args.dry_run:
        for i, item in enumerate(items, 1):
            at = (2 + (i - 1) * slots) * grid.bar
            print(f"  {int(at)//60:02d}:{int(at)%60:02d}  {item.emoji or ' '} "
                  f"{item.source} — {item.target}")
        return

    started = time.time()
    print("Synthesising speech…")
    speaker = Speaker({"es": args.voice_es, "en": args.voice_en},
                      backend=args.backend, model=args.model,
                      ref_audio=args.ref_audio,
                      prosody_strength=args.prosody_strength)
    events, total_bars = arrange(items, speaker, grid, pattern=args.pattern,
                                 emotions=not args.no_emotion)
    speech = render_speech(events, total_bars, grid)

    print("Rendering music bed…")
    bed = render_bed(spec, total_bars)

    print("Mixing…")
    track = mix(bed, speech, duck_db=args.duck_db, speech_lufs=args.speech_lufs,
                music_lufs=args.music_lufs)
    sf.write(args.out, track, SR)
    spec.to_json(args.out.with_suffix(".bed.json"))

    # A sidecar tracklist makes it possible to skip to a word while listening.
    listing = args.out.with_suffix(".txt")
    lines = [f"{args.out.name} — {len(items)} items, {spec.bpm:g} BPM, "
             f"pattern '{args.pattern}', bed '{bed_label}', "
             f"voice '{args.backend}'", ""]
    for i, item in enumerate(items):
        at = (2 + i * slots) * grid.bar
        lines.append(f"{int(at)//60:02d}:{int(at)%60:02d}  {item.emoji or ' '} "
                     f"{item.source} — {item.target}")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nWrote {args.out} ({len(track)/SR/60:.1f} min), {listing.name} "
          f"and {args.out.with_suffix('.bed.json').name} in {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
