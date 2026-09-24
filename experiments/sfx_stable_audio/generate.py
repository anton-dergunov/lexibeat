"""Generate the word sounds with Stable Audio 3 Small SFX, and measure what it costs.

    uv run python generate.py                       # every prompt, best local device
    uv run python generate.py --device cpu --only gato,tren
    uv run python generate.py --half                # fp16 weights (the library itself uses fp32 off CUDA)

Writes `out/<device>[-fp16]/NN_word.wav`, `report.json` and an `index.html` to listen from.
"""

from __future__ import annotations

import argparse
import html
import json
import platform
import resource
import threading
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
MODEL = "small-sfx"
PEAK = 10 ** (-1 / 20)  # -1 dBFS: normalized, and under the repository's 0.97 ceiling
GB = 1024**3


class RssSampler:
    """The process's peak resident memory, sampled in the background, resettable per phase."""

    def __init__(self, interval: float = 0.02):
        self.process = psutil.Process()
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(self.interval)

    def start(self) -> RssSampler:
        self._thread.start()
        return self

    def take_peak(self) -> int:
        """The peak since the last call, which also starts the next phase."""
        peak = max(self.peak, self.process.memory_info().rss)
        self.peak = 0
        return peak

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def mps_bytes(device: str) -> int | None:
    # The driver figure includes the allocator's cache, so after a generation it is that
    # generation's high-water mark rather than what happens to be live now.
    return torch.mps.driver_allocated_memory() if device == "mps" else None


def generate(model, item: dict, device: str, steps: int) -> tuple[np.ndarray, float]:
    sync(device)
    start = time.perf_counter()
    audio = model.generate(prompt=item["prompt"], duration=item["seconds"], steps=steps,
                           seed=item["seed"])
    sync(device)
    return audio[0].float().cpu().numpy(), time.perf_counter() - start


def check(audio: np.ndarray, seconds: float, sample_rate: int) -> list[str]:
    problems = []
    if not np.isfinite(audio).all():
        problems.append("non-finite samples")
    if abs(audio.shape[-1] / sample_rate - seconds) > 0.1:
        problems.append(f"length {audio.shape[-1] / sample_rate:.2f}s")
    if np.abs(audio).max() > 0.97:
        problems.append("peak above 0.97")
    return problems


def write_page(out: Path, report: dict) -> None:
    rows = "\n".join(
        f"""<tr><td><b>{html.escape(s['word'])}</b><br><span>{html.escape(s['meaning'])}</span></td>
<td>{html.escape(s['prompt'])}</td><td>{s['seconds']}s<br><span>{s['generate_s']:.2f}s to make</span></td>
<td><audio controls preload="none" src="{html.escape(s['file'])}"></audio></td></tr>"""
        for s in report["sounds"])
    out.joinpath("index.html").write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Word sounds</title>
<style>
:root {{ color-scheme: light dark; --fg: #1d1d1f; --bg: #fbfbfd; --muted: #6e6e73; --line: #d2d2d7; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg: #f5f5f7; --bg: #161617; --muted: #a1a1a6; --line: #3a3a3c; }} }}
body {{ font: 15px/1.45 system-ui, sans-serif; color: var(--fg); background: var(--bg);
       max-width: 960px; margin: 2rem auto; padding: 0 16px; }}
table {{ border-collapse: collapse; width: 100%; }}
td {{ border-top: 1px solid var(--line); padding: .6rem .5rem; vertical-align: top; }}
span, p {{ color: var(--muted); }}
audio {{ width: 220px; }}
</style></head><body>
<h1>Word sounds</h1>
<p>Stable Audio 3 {report['model']} on {report['device']} ({report['dtype']}), {report['steps']} steps,
loaded in {report['load_s']:.1f}s; mean {report['mean_rtf']:.2f}× real time.</p>
<table>{rows}</table>
</body></html>
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", choices=("mps", "cpu"),
                        default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--half", action="store_true", help="cast the weights to fp16")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--only", help="comma-separated words to generate")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    items = json.loads(HERE.joinpath("prompts.json").read_text())
    # Measured, not guessed: the same prompt and seed at 2.4, 2.5 or 3.5 s comes out as broadband
    # noise, and at 2.0, 3.0 or 2.9999 s as the sound. Ask for whole seconds and trim afterwards.
    for item in items:
        if abs(item["seconds"] - round(item["seconds"])) > 1e-3:
            raise SystemExit(f"{item['word']}: {item['seconds']}s is not a whole number of seconds")
    if args.only:
        wanted = set(args.only.split(","))
        items = [item for item in items if item["word"] in wanted]
    out = args.out or HERE / "out" / (args.device + ("-fp16" if args.half else ""))
    out.mkdir(parents=True, exist_ok=True)

    from stable_audio_3 import StableAudioModel

    rss = RssSampler().start()
    start = time.perf_counter()
    if args.half:
        # Cast on the CPU and only then move, so no fp32 copy ever reaches the device's allocator.
        model = StableAudioModel.from_pretrained(MODEL, device="cpu")
        model.model.to(torch.float16).to(args.device)
        model.device = args.device
    else:
        model = StableAudioModel.from_pretrained(MODEL, device=args.device)
    sync(args.device)
    load_s = time.perf_counter() - start
    load_rss = rss.take_peak()
    sample_rate = model.model.sample_rate
    print(f"loaded {MODEL} on {args.device} in {load_s:.1f}s, rss {load_rss / GB:.2f} GB")

    # Kernels compile and caches fill on the first call; it is not what a render would pay.
    _, warmup_s = generate(model, {**items[0], "seed": 0}, args.device, args.steps)
    print(f"warm-up {warmup_s:.2f}s")
    rss.take_peak()

    sounds = []
    for index, item in enumerate(items, 1):
        audio, seconds = generate(model, item, args.device, args.steps)
        peak = float(np.abs(audio).max())
        audio = audio * (PEAK / peak) if peak > 0 else audio
        name = f"{index:02d}_{item['word']}.wav"
        sf.write(out / name, audio.T, sample_rate, subtype="PCM_16")
        sound = {**item, "file": name, "generate_s": round(seconds, 3),
                 "rtf": round(seconds / item["seconds"], 3), "raw_peak": round(peak, 4),
                 "channels": int(audio.shape[0]), "rss_peak_gb": round(rss.take_peak() / GB, 3),
                 "problems": check(audio, item["seconds"], sample_rate)}
        mps = mps_bytes(args.device)
        if mps is not None:
            sound["mps_driver_gb"] = round(mps / GB, 3)
        sounds.append(sound)
        print(f"{name:<20} {seconds:6.2f}s  rtf {sound['rtf']:.2f}  "
              f"rss {sound['rss_peak_gb']:.2f} GB  {' '.join(sound['problems']) or 'ok'}")
    rss.stop()

    report = {
        "model": MODEL, "device": args.device, "dtype": "fp16" if args.half else "fp32",
        "steps": args.steps, "sample_rate": sample_rate,
        "machine": f"{platform.machine()} {platform.mac_ver()[0] or platform.system()}",
        "torch": torch.__version__, "stable_audio_3": version("stable-audio-3"),
        "load_s": round(load_s, 2), "load_rss_gb": round(load_rss / GB, 3),
        "warmup_s": round(warmup_s, 2),
        "mean_generate_s": round(float(np.mean([s["generate_s"] for s in sounds])), 3),
        "mean_rtf": round(float(np.mean([s["rtf"] for s in sounds])), 3),
        "rss_peak_gb": round(max([load_rss] + [s["rss_peak_gb"] * GB for s in sounds]) / GB, 3),
        # ru_maxrss is bytes on macOS: the whole run's high-water mark, as the kernel saw it.
        "ru_maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GB, 3),
        "sounds": sounds,
    }
    if args.device == "mps":
        report["mps_driver_peak_gb"] = max(s["mps_driver_gb"] for s in sounds)
    out.joinpath("report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_page(out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "sounds"}, indent=2))
    print(f"listen: {out / 'index.html'}")


if __name__ == "__main__":
    main()
