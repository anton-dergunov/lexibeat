"""Render a round of prompt variants side by side, for listening.

    uv run python compare.py rounds/steer.json
    uv run python compare.py rounds/expand.json
    open out/steer/index.html

A round is rows (words) by columns (prompt variants), each cell rendered `takes` times. Take k of a
row uses seed `seed + 1000 * k`, so take 1 of a row reproduces the first round's clip for the same
prompt. Every clip is matched to one loudness, so a comparison hears character, not level.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile as sf
import torch

from generate import MODEL, check, sync

HERE = Path(__file__).resolve().parent
TARGET_LUFS = -20.0
PEAK = 10 ** (-1 / 20)


def slug(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")


def level(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict]:
    """Match the clip to TARGET_LUFS, backing off only where that would push the peak past -1 dBFS."""
    meter = pyloudnorm.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio.T)
    gain = 10 ** ((TARGET_LUFS - loudness) / 20)
    gain = min(gain, PEAK / float(np.abs(audio).max()))
    audio = audio * gain
    final = meter.integrated_loudness(audio.T)
    peak_db = 20 * np.log10(float(np.abs(audio).max()))
    power = np.abs(np.fft.rfft(audio.mean(0))) ** 2
    freqs = np.fft.rfftfreq(audio.shape[-1], 1 / sample_rate)
    return audio, {
        "lufs": round(final, 1),
        # Peak-to-loudness ratio. Kept in the report, not the page: a sparse sound (two woofs, a few
        # crackles) scores high however gentle it is, so it misreads as harshness.
        "plr_db": round(peak_db - final, 1),
        # Spectral centroid: where the energy sits; lower is darker and warmer.
        "centroid_hz": int(float((freqs * power).sum() / power.sum())),
    }


def write_page(out: Path, round_: dict, cells: dict) -> None:
    columns = round_["columns"]
    head = "".join(
        f"<th>{html.escape(c['label'])}<small>{html.escape(c.get('note', ''))}</small></th>"
        for c in columns)
    body = []
    for row in round_["rows"]:
        tds = []
        for column in columns:
            takes = cells.get((row["word"], column["id"]))
            if not takes:
                tds.append("<td></td>")
                continue
            players = "".join(
                f"""<div class="take"><span>{k}</span><audio controls preload="none" src="{html.escape(t['file'])}"></audio>
<small>{t['centroid_hz'] / 1000:.1f} kHz{' · ' + ', '.join(t['problems']) if t['problems'] else ''}{' · ' + t['caption'] if t.get('caption') else ''}</small></div>"""
                for k, t in enumerate(takes, 1))
            # A take's caption is trusted HTML (a source link); a prompt is text.
            prompt = row.get("prompts", {}).get(column["id"])
            prompt = f'<p class="prompt">{html.escape(prompt)}</p>' if prompt else ""
            tds.append(f'<td><div class="takes">{players}</div>{prompt}</td>')
        seconds = f" · {row['seconds']} s" if "seconds" in row else ""
        body.append(f"""<tr><th scope="row">{html.escape(row['word'])}<small>{html.escape(row['meaning'])}{seconds}</small></th>{''.join(tds)}</tr>""")
    out.joinpath("index.html").write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(round_['title'])}</title>
<style>
:root {{ --fg: #1d1d1f; --bg: #fbfbfd; --card: #ffffff; --muted: #6e6e73; --line: #e0e0e5; --accent: #0a66c2; }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{ --fg: #f5f5f7; --bg: #161617; --card: #1f1f21; --muted: #a1a1a6; --line: #333336; --accent: #6cb4ff; }}
}}
:root[data-theme="dark"] {{ --fg: #f5f5f7; --bg: #161617; --card: #1f1f21; --muted: #a1a1a6; --line: #333336; --accent: #6cb4ff; }}
* {{ box-sizing: border-box; }}
body {{ font: 15px/1.45 system-ui, -apple-system, sans-serif; color: var(--fg); background: var(--bg);
       margin: 0; padding: 2rem 16px 4rem; }}
main {{ max-width: 1400px; margin: 0 auto; }}
h1 {{ margin: 0 0 .4rem; font-size: 1.6rem; }}
.intro {{ color: var(--muted); max-width: 70ch; margin: 0 0 1.5rem; }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: separate; border-spacing: 0; width: 100%; min-width: {220 + 290 * len(columns)}px; }}
th, td {{ text-align: left; vertical-align: top; padding: .8rem .7rem; border-top: 1px solid var(--line); }}
thead th {{ border-top: 0; position: sticky; top: 0; background: var(--bg); z-index: 1; }}
small {{ display: block; color: var(--muted); font-weight: 400; font-size: .8rem; margin-top: .15rem; }}
tbody th {{ width: 180px; font-size: 1.05rem; }}
.takes {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); column-gap: 1rem; }}
.take {{ display: grid; grid-template-columns: 1.1rem 1fr; align-items: center; column-gap: .3rem; margin-bottom: .45rem; }}
.take > span {{ color: var(--muted); font-size: .8rem; font-variant-numeric: tabular-nums; }}
.take small {{ grid-column: 2; margin-top: 0; }}
audio {{ width: 100%; height: 34px; }}
audio.playing {{ outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 18px; }}
small a {{ color: inherit; }}
.prompt {{ color: var(--muted); font-size: .82rem; margin: .3rem 0 0; }}
.legend {{ color: var(--muted); font-size: .82rem; margin-top: 1.2rem; max-width: 80ch; }}
</style></head><body><main>
<h1>{html.escape(round_['title'])}</h1>
<p class="intro">{html.escape(round_.get('intro', ''))}</p>
<div class="scroll"><table><thead><tr><th></th>{head}</tr></thead><tbody>
{''.join(body)}
</tbody></table></div>
<p class="legend">All clips at {TARGET_LUFS:.0f} LUFS unless the −1 dBFS peak ceiling holds one lower. <b>kHz</b> is
the spectral centroid: lower is darker and warmer. {round_.get('legend', f'Stable Audio 3 {MODEL}, fp32 on MPS, 8 steps.')}
Starting one clip pauses the others.</p>
</main>
<script>
const players = [...document.querySelectorAll("audio")];
for (const p of players) {{
  p.addEventListener("play", () => {{
    for (const q of players) if (q !== p) {{ q.pause(); q.currentTime = 0; }}
    p.classList.add("playing");
  }});
  for (const e of ["pause", "ended"]) p.addEventListener(e, () => p.classList.remove("playing"));
}}
</script>
</body></html>
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("round", type=Path)
    parser.add_argument("--device", choices=("mps", "cpu"),
                        default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--page-only", action="store_true", help="rebuild index.html from report.json")
    args = parser.parse_args()

    round_ = json.loads(args.round.read_text())
    out = HERE / "out" / args.round.stem
    if args.page_only:
        report = json.loads(out.joinpath("report.json").read_text())
        write_page(out, round_, {(row["word"], column): clips for row in report["rows"]
                                 for column, clips in row["clips"].items() if clips})
        return
    for row in round_["rows"]:
        if row["seconds"] != int(row["seconds"]):
            raise SystemExit(f"{row['word']}: ask for whole seconds (see README)")
    out.mkdir(parents=True, exist_ok=True)

    from stable_audio_3 import StableAudioModel

    model = StableAudioModel.from_pretrained(MODEL, device=args.device)
    sample_rate = model.model.sample_rate
    model.generate(prompt="warm-up", duration=2, steps=args.steps, seed=0)

    cells: dict[tuple[str, str], list[dict]] = {}
    for index, row in enumerate(round_["rows"], 1):
        for column in round_["columns"]:
            prompt = row["prompts"].get(column["id"])
            if prompt is None:
                continue
            for take in range(round_.get("takes", 1)):
                seed = row["seed"] + 1000 * take
                sync(args.device)
                start = time.perf_counter()
                audio = model.generate(prompt=prompt, duration=row["seconds"], steps=args.steps,
                                       seed=seed)
                sync(args.device)
                seconds = time.perf_counter() - start
                audio, metrics = level(audio[0].float().cpu().numpy(), sample_rate)
                name = f"{index:02d}_{slug(row['word'])}_{column['id']}_{take + 1}.wav"
                sf.write(out / name, audio.T, sample_rate, subtype="PCM_16")
                clip = {"file": name, "seed": seed, "generate_s": round(seconds, 2), **metrics,
                        "problems": check(audio, row["seconds"], sample_rate)}
                cells.setdefault((row["word"], column["id"]), []).append(clip)
                print(f"{name:<44} {seconds:5.2f}s  {metrics['lufs']:6.1f} LUFS  "
                      f"PLR {metrics['plr_db']:4.1f}  {metrics['centroid_hz']:5d} Hz  "
                      f"{' '.join(clip['problems']) or 'ok'}")

    report = {"round": args.round.stem, "model": MODEL, "device": args.device, "steps": args.steps,
              "target_lufs": TARGET_LUFS,
              "rows": [{**row, "clips": {c["id"]: cells.get((row["word"], c["id"]), [])
                                         for c in round_["columns"]}} for row in round_["rows"]]}
    out.joinpath("report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    write_page(out, round_, cells)
    print(f"listen: {out / 'index.html'}")


if __name__ == "__main__":
    main()
