"""A video of a finished loop: its MP3, and the cards its cues describe, under a size budget.

    uv run --extra video-demo python -m scripts.demos.render_loop_video \\
        --audio out/format-demos/radio-lesson.mp3 --timeline out/format-demos/radio-lesson.json \\
        --out out/format-demos/radio-lesson.mp4 --title "LexiBeat · Radio lesson"

Nothing is rendered again: the audio is the loop's own, and the text is its `cues`, shown card by
card by `group` (`lexibeat.demo.cue_frames`). The timeline JSON is a loop result's `cues`, `items`
and `bpm`. `--max-mb` is a ceiling, since GitHub refuses a larger attachment: the audio is
stepped down first, then the picture, until the file fits.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import soundfile as sf

from lexibeat.demo import DEMO_VIDEO_CRF, cue_frames, encode_visual_track, mux_audio
from lexibeat.music import Grid

AUDIO_BITRATES = (96_000, 80_000, 64_000)
CRF_STEPS = (DEMO_VIDEO_CRF, DEMO_VIDEO_CRF + 4, DEMO_VIDEO_CRF + 8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="LexiBeat")
    parser.add_argument("--max-mb", type=float, default=10.0)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args(argv)

    data = json.loads(args.timeline.read_text(encoding="utf-8"))
    missing = [key for key in ("cues", "items", "bpm") if key not in data]
    if missing:
        raise SystemExit(f"{args.timeline} has no {', '.join(missing)}: it is not a loop result.")
    duration = sf.info(str(args.audio)).duration
    grid = Grid(bpm=float(data["bpm"]), beats_per_bar=4, beat_unit=4)
    frame = cue_frames(args.title, data["cues"], data["items"], duration, grid,
                       font_path=args.font)
    limit = args.max_mb * 1024 * 1024
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for crf in CRF_STEPS:
            visual = Path(tmp) / f"visual-{crf}.mp4"
            print(f"Encoding the picture at CRF {crf}…", flush=True)
            encode_visual_track(args.title, [], duration, grid, visual, frame=frame, crf=crf)
            for bitrate in AUDIO_BITRATES:
                mux_audio(visual, args.audio, args.out, bitrate=bitrate)
                size = args.out.stat().st_size
                print(f"  {bitrate // 1000} kbps audio: {size / 1024 / 1024:.2f} MB", flush=True)
                if size <= limit:
                    print(f"{args.out}: {size / 1024 / 1024:.2f} MB, {duration / 60:.1f} min")
                    return 0
    raise SystemExit(f"{args.out} is still over {args.max_mb} MB at the smallest settings.")


if __name__ == "__main__":
    raise SystemExit(main())
