"""Mix word sounds into a real loop, to hear what the end result would be.

    uv run python experiments/sfx_stable_audio/demo_mix.py      # from the repository root
    open experiments/sfx_stable_audio/out/demo/index.html

Unlike the rest of this experiment, this runs in the repository's own environment: it takes the
README demo's speech and timeline, renders the demo's bed with LexiBeat, and mixes the sounds in
with LexiBeat's own loudness, ducking and limiter, so what you hear is what a render would give.

Each sound is trimmed of its silence, starts on a beat and ends just before its word's first
utterance, in the gap after the previous word. The bed ducks under it as it does under speech. Two
versions: dry (trimmed and levelled only), and glued (filtered, lightly compressed and given a
little room), to hear whether processing does what prompt steering could not.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile as sf
import soxr
from pedalboard import Compressor, HighpassFilter, LowpassFilter, Pedalboard, Reverb

from lexibeat.demo import load_demo_config, resolve_demo_specs
from lexibeat.mix import duck_envelope, limit, loudness_normalize
from lexibeat.music import SR, Grid, render_stems

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEMO = ROOT / "out" / "readme-demo"  # the README demo's own render: speech and timeline
OUT = HERE / "out" / "demo"
BED = "acoustic"
SPEECH_LUFS, MUSIC_LUFS, OUTPUT_LUFS = -16.0, -26.0, -16.0  # as mix_stems
SFX_LUFS = -30.0  # well under the voice: at -20 the listener found them too loud
SFX_DUCK_DB = 4.0
MAX_LEAD_BEATS = 6  # how far before its word a sound may start
GAP_SECONDS = 0.25  # between the end of a sound and its word

# Timeline item -> sound, each one a clip the listener rated well.
SOUNDS = {
    "¡Qué susto!": ("thunder", "expand/10_la-tormenta_plain_1.wav", "model"),
    "la pesadilla": ("wolf howl", "expand/04_el-lobo_plain_1.wav", "model"),
    "estar envuelto en sus pensamientos": ("clock ticking", "catalogue/08_reloj_fsd50k_1.mp3", "FSD50K"),
    "hacer ilusión": ("cork pop", "catalogue/22_descorchar-una-botella_freesound_1.mp3", "Freesound"),
    "espectacular": ("applause", "catalogue/10_aplaudir_esc50_1.mp3", "ESC-50"),
    "¿arrancamos?": ("train", "catalogue/05_tren_freesound_1.mp3", "Freesound"),
    "¡Que cante!": ("nightingale", "catalogue/15_el-ruisenor_fsd50k_1.mp3", "FSD50K"),
    "descansar": ("yawn", "expand/18_el-bostezo_plain_1.wav", "model"),
}
GLUE = Pedalboard([HighpassFilter(140), LowpassFilter(6500),
                   Compressor(threshold_db=-24, ratio=3, attack_ms=10, release_ms=150),
                   Reverb(room_size=0.35, damping=0.6, wet_level=0.18, dry_level=0.85, width=0.8)])


def load(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, always_2d=True, dtype="float32")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return soxr.resample(audio, rate, SR) if rate != SR else audio


def trim(audio: np.ndarray, floor_db: float = -40.0) -> np.ndarray:
    """Cut leading and trailing silence: frames more than floor_db under the loudest frame."""
    frame = SR // 100
    n = len(audio) // frame
    rms = np.sqrt((audio[: n * frame] ** 2).mean(axis=1).reshape(n, frame).mean(axis=1) + 1e-12)
    loud = np.flatnonzero(20 * np.log10(rms / rms.max()) > floor_db)
    return audio[max(loud[0] - 2, 0) * frame: (loud[-1] + 3) * frame]


def fade(audio: np.ndarray, seconds: float) -> np.ndarray:
    audio = audio.copy()
    fade_in, fade_out = int(0.01 * SR), int(min(0.3, seconds / 3) * SR)
    audio[:fade_in] *= np.linspace(0, 1, fade_in)[:, None]
    audio[-fade_out:] *= np.linspace(1, 0, fade_out)[:, None] ** 2
    return audio


def place(timeline: dict, beat: float, glued: bool) -> tuple[np.ndarray, list[dict]]:
    n = round(timeline["duration_seconds"] * SR)
    track = np.zeros((n, 2), dtype=np.float32)
    placed = []
    for item in timeline["items"]:
        if item["source"] not in SOUNDS:
            continue
        label, path, source = SOUNDS[item["source"]]
        word = item["utterances"][0]["start"]
        previous = max((u["end"] for other in timeline["items"] if other["end"] <= item["start"]
                        for u in other["utterances"]), default=0.0)
        # Start on a beat, at most MAX_LEAD_BEATS early, and never over the previous word.
        room = word - GAP_SECONDS - previous - 0.2
        beats = min(MAX_LEAD_BEATS, math.floor(room / beat))
        start = word - beats * beat
        length = beats * beat - GAP_SECONDS
        clip = trim(load(HERE / "out" / path))[: int(length * SR)]
        if glued:
            clip = GLUE(clip.T.copy(), SR).T[: len(clip)]
        clip = fade(loudness_normalize(clip, SFX_LUFS, SR), len(clip) / SR)
        at = int(start * SR)
        track[at: at + len(clip)] += clip[: n - at]
        placed.append({"word": item["source"], "meaning": item["target"], "sound": label,
                       "source": source, "start": round(start, 2),
                       "seconds": round(len(clip) / SR, 2)})
    return track, placed


def mix(stems: dict[str, np.ndarray], depths: dict[str, float], speech: np.ndarray,
        sfx: np.ndarray) -> np.ndarray:
    """mix_stems, with the sounds as a second voice the bed makes room for."""
    n = len(speech)
    padded = {k: np.pad(v, ((0, max(n - len(v), 0)), (0, 0)))[:n] for k, v in stems.items()}
    unducked = sum(padded.values())
    music_gain = 10 ** ((MUSIC_LUFS - pyloudnorm.Meter(SR).integrated_loudness(unducked)) / 20)
    voice = loudness_normalize(np.stack([speech, speech], axis=1), SPEECH_LUFS, SR)
    sfx_duck = duck_envelope(sfx.mean(axis=1), SFX_DUCK_DB, sr=SR)[:, None]
    bed = sum(stem * music_gain * duck_envelope(speech, depths.get(name, 5.0), sr=SR)[:, None]
              for name, stem in padded.items()) * sfx_duck
    return limit(loudness_normalize(bed + voice + sfx, OUTPUT_LUFS, SR), sr=SR).astype(np.float32)


def write_page(placed: list[dict], files: dict[str, str]) -> None:
    rows = "".join(
        f"<tr><td><button data-t='{p['start']}'>{int(p['start'] // 60)}:{p['start'] % 60:04.1f}</button></td>"
        f"<td><b>{html.escape(p['word'])}</b><small>{html.escape(p['meaning'])}</small></td>"
        f"<td>{html.escape(p['sound'])}<small>{html.escape(p['source'])} · {p['seconds']} s</small></td></tr>"
        for p in placed)
    players = "".join(
        f"<section><h2>{html.escape(label)}</h2><audio controls preload='metadata' src='{name}'></audio></section>"
        for label, name in files.items())
    OUT.joinpath("index.html").write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sounds in a loop</title>
<style>
:root {{ --fg: #1d1d1f; --bg: #fbfbfd; --muted: #6e6e73; --line: #e0e0e5; --accent: #0a66c2; }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{ --fg: #f5f5f7; --bg: #161617; --muted: #a1a1a6; --line: #333336; --accent: #6cb4ff; }}
}}
:root[data-theme="dark"] {{ --fg: #f5f5f7; --bg: #161617; --muted: #a1a1a6; --line: #333336; --accent: #6cb4ff; }}
body {{ font: 15px/1.45 system-ui, -apple-system, sans-serif; color: var(--fg); background: var(--bg);
       margin: 0; padding: 2rem 16px 4rem; }}
main {{ max-width: 760px; margin: 0 auto; }}
p, small {{ color: var(--muted); }}
small {{ display: block; font-size: .8rem; }}
h2 {{ font-size: 1rem; margin: 1rem 0 .3rem; }}
audio {{ width: 100%; }}
section.active h2 {{ color: var(--accent); }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1.5rem; }}
td {{ border-top: 1px solid var(--line); padding: .5rem .4rem; vertical-align: top; }}
button {{ font: inherit; font-variant-numeric: tabular-nums; color: var(--accent); background: none;
          border: 1px solid var(--line); border-radius: 6px; padding: .15rem .5rem; cursor: pointer; }}
</style></head><body><main>
<h1>Sounds in a loop</h1>
<p>The README demo's words over the "{BED}" bed, with a sound before {len(placed)} of its 13 words:
more than the product would use, so there is plenty to judge. Each sound starts on a beat and ends
just before its word; the bed ducks under it. <b>Dry</b> is each sound trimmed and levelled.
<b>Glued</b> is the same sound filtered, lightly compressed and given a little room.</p>
{players}
<table>{rows}</table>
<p>A time jumps the last player you used to a second before that sound.</p>
</main>
<script>
const players = [...document.querySelectorAll("audio")];
let current = players[0];
for (const p of players) p.addEventListener("play", () => {{
  for (const q of players) if (q !== p) q.pause();
  current = p;
  for (const s of document.querySelectorAll("section")) s.classList.toggle("active", s.contains(p));
}});
for (const b of document.querySelectorAll("button[data-t]")) b.addEventListener("click", () => {{
  current.currentTime = Math.max(0, Number(b.dataset.t) - 1);
  current.play();
}});
</script>
</body></html>
""")


def main() -> None:
    timeline = json.loads((DEMO / "timeline.json").read_text())
    speech, rate = sf.read(DEMO / "shared-speech.wav", dtype="float32")
    assert rate == SR
    spec = resolve_demo_specs(load_demo_config(ROOT / "examples" / "readme_demo" / "full.json"))[BED]
    grid = Grid.from_spec(spec)
    if abs(grid.bpm - timeline["bpm"]) > 1e-6:
        raise SystemExit(f"bed at {grid.bpm} BPM, speech at {timeline['bpm']}")
    stems = render_stems(spec, timeline["total_bars"])
    depths = {name: getattr(spec, name).duck_db for name in stems}
    OUT.mkdir(parents=True, exist_ok=True)

    files = {}
    for label, glued in (("Dry", False), ("Glued", True)):
        sfx, placed = place(timeline, 60 / grid.bpm, glued)
        track = mix(stems, depths, speech, sfx)
        assert np.isfinite(track).all() and np.abs(track).max() <= 0.9701
        name = f"{label.lower()}.mp3"
        sf.write(OUT / name, track, SR, format="MP3")
        files[label] = name
        print(f"{label:<6} peak {np.abs(track).max():.3f}  {len(track) / SR:.1f} s  -> {OUT / name}")
    OUT.joinpath("placements.json").write_text(json.dumps(placed, indent=2, ensure_ascii=False) + "\n")
    write_page(placed, files)
    for p in placed:
        print(f"  {p['start']:7.2f}s  {p['sound']:<14} {p['seconds']:4.2f}s  before {p['word']}")
    print(f"listen: {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
