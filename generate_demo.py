#!/usr/bin/env python3
"""Generate synchronized LexiBeat README WAV and MP4 demonstrations."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from lexibeat.arrange import render_speech
from lexibeat.demo import (
    DEMO_VIDEO_CRF,
    PersistentSpeaker,
    arrange_demo,
    audio_summary,
    build_timeline,
    encode_visual_track,
    load_demo_config,
    mux_audio,
    resolve_demo_specs,
    resolve_font,
    write_tracklist,
)
from lexibeat.mix import mix_stems
from lexibeat.music import SR, Grid, render_stems
from lexibeat.voice import DEFAULT_MODELS, Speaker

DEFAULT_CONFIG = Path(__file__).parent / "demo" / "readme_demo.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=Path("out/readme-demo"))
    parser.add_argument("--backend", choices=("gemini", "gemini-vertex"),
                        default="gemini")
    parser.add_argument("--model", default="gemini-3.1-flash-tts-preview")
    parser.add_argument("--font", type=Path)
    parser.add_argument("--refresh-speech", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_demo_config(args.config)
    specs = resolve_demo_specs(config)
    first_spec = next(iter(specs.values()))
    grid = Grid.from_spec(first_spec)
    font = resolve_font(args.font)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(config.items)} items · {len(config.variants)} beds · "
          f"{grid.bpm:g} BPM · {grid.beats_per_bar}/{grid.beat_unit} · "
          f"voice {args.backend}/{args.model}")
    started = time.perf_counter()
    speaker = Speaker(backend=args.backend, model=args.model, voice_seed=7)
    cached = PersistentSpeaker(
        speaker, args.out_dir / "speech-cache", refresh=args.refresh_speech)
    try:
        print("Synthesizing or loading speech takes…")
        events, total_bars = arrange_demo(config, cached, grid)
        speech = render_speech(events, total_bars, grid)
    finally:
        cached.close()
    if not len(speech) or not np.isfinite(speech).all():
        raise RuntimeError("Speech renderer produced invalid audio.")

    speech_path = args.out_dir / "shared-speech.wav"
    sf.write(speech_path, speech, SR, subtype="PCM_16")
    timeline = build_timeline(config.items, events, grid, total_bars,
                              config.pattern)
    timeline_payload = {
        "schema_version": 1,
        "title": config.title,
        "pattern": config.pattern,
        "bpm": grid.bpm,
        "meter": f"{grid.beats_per_bar}/{grid.beat_unit}",
        "total_bars": total_bars,
        "duration_seconds": len(speech) / SR,
        "items": timeline,
    }
    _write_json(args.out_dir / "timeline.json", timeline_payload)
    _write_json(args.out_dir / "speech.stats.json", {
        "backend": args.backend,
        "model_id": getattr(speaker.backend, "model_id",
                            args.model or DEFAULT_MODELS[args.backend]),
        "audio": audio_summary(speech),
        "utterances": speaker.stats,
    })

    wav_paths: dict[str, Path] = {}
    variant_stats: dict[str, dict] = {}
    for variant in config.variants:
        name = variant.name
        spec = specs[name]
        print(f"Rendering and mixing {name}…")
        variant_started = time.perf_counter()
        stems = render_stems(spec, total_bars)
        depths = {stem_name: getattr(spec, stem_name).duck_db
                  for stem_name in stems}
        track = mix_stems(stems, speech, depths)
        summary = audio_summary(track)
        if track.ndim != 2 or track.shape[1] != 2 or not summary["finite"] or \
                summary["peak"] > 0.97001:
            raise RuntimeError(f"{name} produced invalid or over-limit audio.")
        wav_path = args.out_dir / f"{name}.wav"
        sf.write(wav_path, track, SR, subtype="PCM_16")
        spec.to_json(args.out_dir / f"{name}.bed.json")
        write_tracklist(args.out_dir / f"{name}.txt", name, config,
                        timeline, spec)
        variant_stats[name] = {
            "variant": asdict(variant),
            "audio": summary,
            "render_seconds": time.perf_counter() - variant_started,
        }
        wav_paths[name] = wav_path

    duration = max(float(sf.info(path).duration) for path in wav_paths.values())
    visual_path = args.out_dir / ".shared-visual.mp4"
    print("Encoding shared synchronized visuals…")
    encode_visual_track(config.title, timeline, duration, grid, visual_path,
                        font_path=font)
    for variant in config.variants:
        name = variant.name
        print(f"Muxing {name}.mp4…")
        mp4_path = args.out_dir / f"{name}.mp4"
        mux_audio(visual_path, wav_paths[name], mp4_path)
        variant_stats[name]["video"] = {
            "path": str(mp4_path),
            "bytes": mp4_path.stat().st_size,
            "video_crf": DEMO_VIDEO_CRF,
        }
        _write_json(args.out_dir / f"{name}.stats.json", variant_stats[name])
    visual_path.unlink(missing_ok=True)
    print(f"Completed {len(config.variants)} demo variants in "
          f"{time.perf_counter()-started:.1f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
