#!/usr/bin/env python3
"""Generate an Earworms-style language-learning track from a vocabulary list.

    uv run generate.py --words 12 --out out/spanish.wav

The music bed is synthesised locally. Speech can come from the preferred local
Chatterbox backend, fast Kokoro fallback, experimental local models, or explicit
hosted Gemini and Cloudflare backends. See DESIGN.md for the trade-offs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from earworms.arrange import PATTERNS, arrange, render_speech
from earworms.bedspec import STYLES, BedSpec
from earworms.mix import mix_stems
from earworms.music import SR, Grid, render_bed, render_stems
from earworms.samples import PACKS, PACK_GROUPS, download_target
from earworms.vocab import load
from earworms.voice import CAPABILITIES, DEFAULT_MODELS, Speaker

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
    music.add_argument("--meter", choices=["3/4", "4/4", "5/4"],
                       help="override the style's time signature")
    music.add_argument("--chord-extension",
                       choices=["none", "seventh", "add9", "ninth"],
                       help="override the style's chord colour")
    music.add_argument("--instrument",
                       choices=["synth", "piano", "marimba", "glockenspiel"],
                       help="override the sparse melodic instrument")
    music.add_argument("--pad-instrument", choices=["synth", "strings"],
                       help="override the sustained background instrument")
    music.add_argument("--bed-only", action="store_true",
                       help="render just the music, with no speech")
    music.add_argument("--download-samples", nargs="?", const="salamander",
                       choices=sorted(PACKS | PACK_GROUPS),
                       help="fetch one sample pack, or 'vsco' for all VSCO packs")

    voice = p.add_argument_group("voice")
    voice.add_argument("--backend", choices=["kokoro", "chatterbox", "gemini",
                                              "gemini-vertex",
                                              "cloudflare-aura2",
                                              "cloudflare-melotts", "indextts25",
                                              "voxcpm2", "qwen3", "tada", "fish-s2"],
                       default="chatterbox",
                       help="Chatterbox is the quality default; Kokoro is the fast fallback")
    voice.add_argument("--model", default=None,
                       help="override the backend-specific default model")
    voice.add_argument("--ref-audio", help="voice to clone for the chatterbox backend")
    voice.add_argument("--ref-audio-es",
                       help="Spanish reference path or descriptor such as say:Paulina")
    voice.add_argument("--ref-audio-en",
                       help="English reference path or descriptor such as say:Daniel")
    voice.add_argument("--ref-text-es", help="transcript for a custom Spanish reference")
    voice.add_argument("--ref-text-en", help="transcript for a custom English reference")
    voice.add_argument("--voice-es", default=None)
    voice.add_argument("--voice-en", default=None)
    voice.add_argument("--voice-seed", type=int, default=None,
                       help="TTS sampling seed (defaults to --seed)")
    voice.add_argument("--prosody-strength", type=float, default=1.0,
                       help="0 disables per-repeat variation, 1 is the default")
    voice.add_argument("--no-emotion", action="store_true",
                       help="ignore the emoji in the notes and read everything neutrally")

    out = p.add_argument_group("output")
    out.add_argument("--out", type=Path, default=Path("out/lesson.wav"))
    out.add_argument("--duck-db", type=float, default=None,
                     help="override every layer's configured speech-duck depth")
    out.add_argument("--speech-lufs", type=float, default=-16.0)
    out.add_argument("--music-lufs", type=float, default=-26.0)
    out.add_argument("--dry-run", action="store_true",
                     help="print the selected items and the timing plan only")
    out.add_argument("--stats-json", type=Path,
                     help="write detailed timing, control and audio statistics")
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
    if args.meter:
        numerator, denominator = args.meter.split("/")
        spec.beats_per_bar = int(numerator)
        spec.beat_unit = int(denominator)
    if args.chord_extension:
        spec.chord_extension = args.chord_extension
    if args.instrument:
        spec.lead.instrument = args.instrument
    if args.pad_instrument:
        spec.pad.instrument = args.pad_instrument
    return spec, label


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _write_stats(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def main() -> None:
    args = parse_args()
    process_started = time.perf_counter()
    cpu_started = time.process_time()

    if args.download_samples:
        download_target(args.download_samples)
        return

    spec, bed_label = build_spec(args)
    grid = Grid.from_spec(spec)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.bed_only:
        bars = 24
        started = time.time()
        sf.write(args.out, render_bed(spec, bars), SR)
        spec.to_json(args.out.with_suffix(".bed.json"))
        print(f"{bed_label} · {spec.bpm:g} BPM · "
              f"{spec.beats_per_bar}/{spec.beat_unit} · {spec.scale} · "
              f"pad {spec.pad.instrument} · lead {spec.lead.instrument} · "
              f"{bars * grid.bar:.0f}s "
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

    started = time.perf_counter()
    print("Synthesising speech…")
    voices = {key: value for key, value in {
        "es": args.voice_es, "en": args.voice_en}.items() if value}
    speaker_started = time.perf_counter()
    speaker = Speaker(voices or None, backend=args.backend, model=args.model,
                      ref_audio=args.ref_audio,
                      ref_audios={"es": args.ref_audio_es,
                                  "en": args.ref_audio_en},
                      ref_texts={key: value for key, value in {
                          "es": args.ref_text_es, "en": args.ref_text_en}.items()
                          if value},
                      prosody_strength=args.prosody_strength,
                      voice_seed=args.voice_seed if args.voice_seed is not None
                      else args.seed)
    speaker_init_seconds = time.perf_counter() - speaker_started
    speech_started = time.perf_counter()
    events, total_bars = arrange(items, speaker, grid, pattern=args.pattern,
                                 emotions=not args.no_emotion)
    speech = render_speech(events, total_bars, grid)
    speech_seconds = time.perf_counter() - speech_started
    speaker.close()

    print("Rendering music bed…")
    music_started = time.perf_counter()
    stems = render_stems(spec, total_bars)
    music_seconds = time.perf_counter() - music_started

    print("Mixing…")
    mix_started = time.perf_counter()
    depths = {name: getattr(spec, name).duck_db for name in stems}
    if args.duck_db is not None:
        depths = {name: args.duck_db for name in stems}
    track = mix_stems(stems, speech, depths, speech_lufs=args.speech_lufs,
                      music_lufs=args.music_lufs)
    sf.write(args.out, track, SR)
    spec.to_json(args.out.with_suffix(".bed.json"))
    mix_seconds = time.perf_counter() - mix_started

    # A sidecar tracklist makes it possible to skip to a word while listening.
    listing = args.out.with_suffix(".txt")
    lines = [f"{args.out.name} — {len(items)} items, {spec.bpm:g} BPM, "
             f"pattern '{args.pattern}', bed '{bed_label}', "
             f"voice '{args.backend}'", ""]
    capabilities = CAPABILITIES[args.backend]
    if capabilities.experimental:
        lines += [f"Experimental model: {getattr(speaker.backend, 'model_id', args.model)}",
                  f"License: {capabilities.license}",
                  *[f"Warning: {warning}" for warning in capabilities.warnings], ""]
    for i, item in enumerate(items):
        at = (2 + i * slots) * grid.bar
        lines.append(f"{int(at)//60:02d}:{int(at)%60:02d}  {item.emoji or ' '} "
                     f"{item.source} — {item.target}")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")

    total_seconds = time.perf_counter() - process_started
    if args.stats_json:
        utterance_audio = sum(float(row["duration_seconds"]) for row in speaker.stats)
        generation_seconds = sum(float(row["generation_seconds"]) for row in speaker.stats)
        estimated_cost = sum(float(row.get("controls", {}).get(
            "estimated_cost_usd") or 0.0) for row in speaker.stats)
        reference_seconds = sum(float(row.get("controls", {}).get(
            "reference_preparation_seconds", 0.0)) for row in speaker.stats)
        peak_mlx = max((int(row["peak_memory_bytes"] or 0)
                        for row in speaker.stats), default=0)
        stats = {
            "schema_version": 1,
            "success": True,
            "backend": args.backend,
            "model_id": getattr(speaker.backend, "model_id",
                                args.model or DEFAULT_MODELS.get(args.backend, "")),
            "capabilities": asdict(capabilities),
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "cpu_count": os.cpu_count(),
            },
            "timing": {
                "backend_init_seconds": speaker_init_seconds,
                "model_load_seconds": float(getattr(speaker.backend, "load_seconds", 0.0)),
                "speech_stage_seconds": speech_seconds,
                "model_generation_seconds": generation_seconds,
                "reference_preparation_seconds": reference_seconds,
                "music_render_seconds": music_seconds,
                "mix_write_seconds": mix_seconds,
                "total_seconds": total_seconds,
                "cpu_seconds": time.process_time() - cpu_started,
                "speech_rtf": generation_seconds / utterance_audio
                if utterance_audio else None,
            },
            "memory": {
                "process_peak_rss_bytes": _peak_rss_bytes(),
                "mlx_peak_bytes": peak_mlx or None,
            },
            "audio": {
                "path": str(args.out), "sample_rate": SR,
                "channels": int(track.shape[1]) if track.ndim == 2 else 1,
                "duration_seconds": len(track) / SR,
                "peak": float(np.abs(track).max()),
                "finite": bool(np.isfinite(track).all()),
            },
            "estimated_provider_cost_usd": estimated_cost,
            "utterances": speaker.stats,
        }
        _write_stats(args.stats_json, stats)
    print(f"\nWrote {args.out} ({len(track)/SR/60:.1f} min), {listing.name} "
          f"and {args.out.with_suffix('.bed.json').name} in "
          f"{time.perf_counter()-started:.0f}s")


if __name__ == "__main__":
    main()
