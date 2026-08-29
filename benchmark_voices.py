#!/usr/bin/env python3
"""Render a comparable one-item lesson through the experimental TTS backends."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from earworms.bedspec import BedSpec
from earworms.voice import CAPABILITIES, DEFAULT_MODELS, model_cache_root

BACKENDS = ("indextts25", "voxcpm2", "qwen3", "tada", "fish-s2")
VOCAB_DIR = Path("/Users/anton/obsidian/Languages/Spanish/Vocabulary")


def directory_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for filename in files:
            try:
                total += (Path(root) / filename).stat().st_size
            except FileNotFoundError:
                pass
    return total


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=False, capture_output=True,
                              text=True, timeout=15).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unavailable"


def pressure_snapshot() -> dict[str, Any]:
    raw = command_output(["memory_pressure", "-Q"])
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", raw)
    swap = command_output(["sysctl", "-n", "vm.swapusage"])
    swap_match = re.search(r"used\s*=\s*([\d.]+)([KMGTP])", swap)
    units = {"K": 2**10, "M": 2**20, "G": 2**30,
             "T": 2**40, "P": 2**50}
    return {
        "free_percent": int(match.group(1)) if match else None,
        "memory_pressure": raw,
        "vm_stat": command_output(["vm_stat"]),
        "swap_usage": swap,
        "swap_used_bytes": (int(float(swap_match.group(1)) *
                                units[swap_match.group(2)])
                            if swap_match else None),
    }


def hardware_snapshot() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "chip": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memory_bytes": command_output(["sysctl", "-n", "hw.memsize"]),
    }


def prefetch(backends: list[str]) -> dict[str, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    downloads: dict[str, dict[str, Any]] = {}
    for backend in backends:
        model = DEFAULT_MODELS[backend]
        print(f"Downloading/resolving {backend}: {model}", flush=True)
        started = time.perf_counter()
        try:
            resolved = Path(snapshot_download(model))
            downloads[backend] = {
                "success": True,
                "model_id": model,
                "resolved_path": str(resolved),
                "resolved_revision": resolved.name,
                "seconds": time.perf_counter() - started,
                "bytes": directory_bytes(resolved),
            }
        except Exception as exc:
            downloads[backend] = {
                "success": False, "model_id": model,
                "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return downloads


def run_backend(backend: str, args: argparse.Namespace, bed: Path,
                download: dict[str, Any], hardware: dict[str, Any]) -> dict[str, Any]:
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("Run 'uv sync --extra experimental-tts' first.") from exc

    output = args.out_dir / f"{backend}.wav"
    stats_path = args.out_dir / f"{backend}.stats.json"
    log_path = args.out_dir / f"{backend}.log"
    command = [
        "uv", "run", "--extra", "experimental-tts", "generate.py",
        "--backend", backend,
        "--words", str(args.words), "--seed", str(args.seed),
        "--voice-seed", str(args.voice_seed), "--pattern", "retrieval",
        "--bed-spec", str(bed), "--out", str(output),
        "--stats-json", str(stats_path), "--vocab", *map(str, args.vocab),
    ]
    if download.get("success") and download.get("resolved_path"):
        command += ["--model", str(download["resolved_path"])]
    env = os.environ.copy()
    env["HF_HOME"] = str(model_cache_root() / "huggingface")
    env["UV_CACHE_DIR"] = str(model_cache_root() / "uv")
    before = pressure_snapshot()
    lowest_free = before.get("free_percent")
    peak_tree_rss = 0
    observed_tree_cpu = 0.0
    started = time.perf_counter()
    print(f"\n→ {backend}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   text=True, env=env)
        root = psutil.Process(process.pid)
        last_pressure = 0.0
        while process.poll() is None:
            try:
                members = [root, *root.children(recursive=True)]
                tree_rss = sum(member.memory_info().rss for member in members
                               if member.is_running())
                peak_tree_rss = max(peak_tree_rss, tree_rss)
                tree_cpu = sum(sum(member.cpu_times()[:2]) for member in members
                               if member.is_running())
                observed_tree_cpu = max(observed_tree_cpu, tree_cpu)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            if time.monotonic() - last_pressure >= 2.0:
                sample = pressure_snapshot().get("free_percent")
                if sample is not None:
                    lowest_free = sample if lowest_free is None else min(lowest_free, sample)
                last_pressure = time.monotonic()
            time.sleep(0.1)
        return_code = process.wait()
    elapsed = time.perf_counter() - started
    after = pressure_snapshot()
    before_swap = before.get("swap_used_bytes")
    after_swap = after.get("swap_used_bytes")
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        stats = {"schema_version": 1, "backend": backend, "success": False}
    stats.update({
        "success": return_code == 0 and output.exists(),
        "exit_code": return_code,
        "download": download,
        "hardware": hardware,
        "system_memory": {
            "before": before, "after": after,
            "lowest_free_percent": lowest_free,
            "peak_process_tree_rss_bytes": peak_tree_rss,
            "observed_process_tree_cpu_seconds": observed_tree_cpu,
            "swap_delta_bytes": (after_swap - before_swap
                                 if before_swap is not None and
                                 after_swap is not None else None),
        },
        "benchmark_wall_seconds": elapsed,
        "log_path": str(log_path),
        "license": CAPABILITIES[backend].license,
    })
    if return_code != 0:
        stats["error"] = f"generate.py exited {return_code}; see {log_path}"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
    return stats


def write_comparison(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    combined = out_dir / "comparison.json"
    combined.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    lines = [
        "# Experimental TTS bake-off", "",
        "| Backend | Result | Load | Synthesis | Total | Audio | Peak RSS | MLX peak | Lowest free | Swap Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        timing = row.get("timing", {})
        audio = row.get("audio", {})
        memory = row.get("memory", {})
        system = row.get("system_memory", {})
        def seconds(value: Any) -> str:
            return f"{float(value):.1f}s" if value is not None else "—"
        def gib(value: Any) -> str:
            return f"{int(value) / 2**30:.2f} GiB" if value else "—"
        lines.append(
            f"| {row['backend']} | {'ok' if row.get('success') else 'failed'} | "
            f"{seconds(timing.get('model_load_seconds'))} | "
            f"{seconds(timing.get('model_generation_seconds'))} | "
            f"{seconds(row.get('benchmark_wall_seconds'))} | "
            f"{seconds(audio.get('duration_seconds'))} | "
            f"{gib(system.get('peak_process_tree_rss_bytes'))} | "
            f"{gib(memory.get('mlx_peak_bytes'))} | "
            f"{system.get('lowest_free_percent', '—')}% | "
            f"{gib(system.get('swap_delta_bytes'))} |"
        )
    lines += [
        "",
        "A successful render means only that the file passed structural and "
        "numerical validation. It does not certify pronunciation, transcript "
        "faithfulness, or freedom from hallucinated trailing speech.",
    ]
    if (out_dir / "listening-notes.md").exists():
        lines += ["", "See [listening-notes.md](listening-notes.md) for the "
                  "human listening assessment of these exact artifacts."]
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS,
                        default=list(BACKENDS))
    parser.add_argument("--words", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--voice-seed", type=int, default=7007)
    parser.add_argument("--bed-style", default="yoga")
    parser.add_argument("--vocab", type=Path, nargs="+", default=[VOCAB_DIR])
    parser.add_argument("--out-dir", type=Path, default=Path("out/tts-bakeoff"))
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Resolve every snapshot into Earworms' explicit cache, including the
    # untimed prefetch phase.  Child renders receive the resolved local path.
    cache_root = model_cache_root()
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("UV_CACHE_DIR", str(cache_root / "uv"))
    shared_bed = args.out_dir / "shared.bed.json"
    BedSpec.from_style(args.bed_style, args.seed).to_json(shared_bed)
    hardware = hardware_snapshot()
    downloads = ({backend: {"skipped": True, "model_id": DEFAULT_MODELS[backend]}
                  for backend in args.backends}
                 if args.skip_download else prefetch(args.backends))
    rows = [run_backend(backend, args, shared_bed, downloads[backend], hardware)
            for backend in args.backends]
    combined = {row["backend"]: row for row in rows}
    for backend in BACKENDS:
        previous = args.out_dir / f"{backend}.stats.json"
        if backend not in combined and previous.exists():
            combined[backend] = json.loads(previous.read_text(encoding="utf-8"))
    write_comparison(args.out_dir, [combined[name] for name in BACKENDS
                                    if name in combined])
    succeeded = sum(bool(row.get("success")) for row in rows)
    print(f"\n{succeeded}/{len(rows)} requested review tracks succeeded -> {args.out_dir}")
    if succeeded != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
