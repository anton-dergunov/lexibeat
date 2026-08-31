"""Gradio entry point for the LexiBeat Hugging Face Space."""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import spaces
except ImportError:  # The package exists in ZeroGPU Spaces, not local installs.
    class _LocalSpaces:
        @staticmethod
        def GPU(function=None, **_kwargs):
            def decorate(callback):
                return callback
            return decorate(function) if function is not None else decorate

    spaces = _LocalSpaces()

import gradio as gr


def _configure_attached_sample_bucket() -> None:
    """Use the conventional read-only bucket mount when it is available."""
    if os.environ.get("LEXIBEAT_BUNDLE_ROOT"):
        return
    mount_root = Path(os.environ.get("LEXIBEAT_BUCKET_MOUNT", "/data"))
    mounted = mount_root / "lexibeat-production-core" / "v1"
    if mounted.joinpath("catalog.sqlite3").is_file():
        os.environ["LEXIBEAT_BUNDLE_ROOT"] = str(mounted)


_configure_attached_sample_bucket()

from lexibeat.explorer import (
    ArtifactStore,
    ExplorerConfig,
    SampleService,
    explorer_schema,
)
from lexibeat.explorer_ui import build_demo
from lexibeat.lesson import lesson_gpu_duration, render_lesson_speech

config = ExplorerConfig.from_environment()
artifacts = ArtifactStore(config)
samples = SampleService(config)
palette_choices = explorer_schema(config)["simple"]["palette"]
lesson_palette = "hybrid" if "hybrid" in palette_choices else "electronic"

_hosted_backend = None
if config.hosted:
    vendor_root = Path(__file__).parent / "third_party" / "chatterbox"
    sys.path.insert(0, str(vendor_root))
    from lexibeat.cuda_voice import CudaChatterboxBackend, load_cuda_chatterbox

    _hosted_backend = CudaChatterboxBackend(load_cuda_chatterbox())


@spaces.GPU(duration=lesson_gpu_duration)
def generate_hosted_lesson(rows: object, model: str, state: dict,
                           progress=gr.Progress()) -> dict:
    """The directly registered ZeroGPU boundary for Chatterbox synthesis."""
    if _hosted_backend is None:
        raise RuntimeError("The hosted CUDA voice backend is unavailable.")
    return render_lesson_speech(
        rows, model, state, backend=_hosted_backend, config=config,
        palette=lesson_palette,
        progress=lambda value, message: progress(value, desc=message))


demo = build_demo(
    config, artifacts=artifacts, samples=samples,
    lesson_generate=generate_hosted_lesson if config.hosted else None)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        ssr_mode=False,
    )
