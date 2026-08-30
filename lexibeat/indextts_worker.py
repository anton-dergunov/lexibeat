# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "mlx-indextts[v25] @ git+https://github.com/vanch007/mlx-indextts2.git@a7666367b8551656a2029ad75f259cb5e4936b3b",
# ]
# ///
"""Persistent JSON-lines worker for the dependency-isolated IndexTTS 2.5 port."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


def emit(payload: dict) -> None:
    sys.__stdout__.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.cache / "huggingface"))

    try:
        with contextlib.redirect_stdout(sys.stderr):
            import mlx.core as mx
            from huggingface_hub import snapshot_download
            from mlx_indextts.runtime import GenerateOptions, TTSRuntime

            model_path = (str(Path(args.model).resolve()) if Path(args.model).exists()
                          else snapshot_download(args.model))
            runtime = TTSRuntime(memory_limit_gb=12.0, quantize="8")
            if hasattr(mx, "reset_peak_memory"):
                mx.reset_peak_memory()
            runtime.load(model_path)
        emit({"ready": True, "model_path": model_path})
    except Exception as exc:
        emit({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return

    speaker_dir = args.cache / "indextts25-speakers"
    speaker_dir.mkdir(parents=True, exist_ok=True)
    speaker_cache: dict[str, str] = {}

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "close":
                break
            reference = str(request["reference"])
            reference_started = time.perf_counter()
            reference_key = hashlib.sha256(
                (reference + str(Path(reference).stat().st_mtime_ns)).encode()
            ).hexdigest()[:20]
            if reference_key not in speaker_cache:
                cached = speaker_dir / f"{reference_key}.npz"
                if not cached.exists():
                    with contextlib.redirect_stdout(sys.stderr):
                        runtime.save_speaker(reference, str(cached), profile="v25",
                                             model=model_path)
                speaker_cache[reference_key] = str(cached)
            reference_seconds = time.perf_counter() - reference_started

            factor = float(request.get("duration_factor") or 1.0)
            target = request.get("target_seconds")
            output = str(request["output"])
            passes = 1
            started = time.perf_counter()

            def generate(duration_factor: float) -> dict:
                options = GenerateOptions(
                    language=request["lang"],
                    emotion=request["emotion"],
                    emo_alpha=1.0,
                    duration_factor=duration_factor,
                    seed=request.get("seed"),
                    diffusion_steps=16,
                    max_tokens=500,
                    denoise_ref_audio=False,
                    denoise_emotion_ref_audio=False,
                )
                with contextlib.redirect_stdout(sys.stderr):
                    return runtime.generate(
                        text=request["text"], ref_audio=speaker_cache[reference_key],
                        output_path=output, profile="v25", model=model_path,
                        options=options,
                    )

            result = generate(factor)
            first_duration = float(result["duration_s"])
            if target and first_duration > float(target):
                factor = max(0.5, min(2.0, factor * float(target) / first_duration))
                result = generate(factor)
                passes = 2
            elapsed = time.perf_counter() - started
            emit({
                "generation_seconds": elapsed,
                "passes": passes,
                "peak_memory_bytes": int(mx.get_peak_memory()),
                "controls": {
                    "emotion_vector": request["emotion"],
                    "emo_alpha": 1.0,
                    "duration_factor": factor,
                    "first_duration_seconds": first_duration,
                    "final_duration_seconds": result["duration_s"],
                    "reference_preparation_seconds": reference_seconds,
                },
            })
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit({"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
