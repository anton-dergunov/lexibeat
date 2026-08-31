"""CUDA Chatterbox adapter used only by the Hugging Face Space."""

from __future__ import annotations

import random
import time

import numpy as np

from .emotion import Emotion
from .voice import (
    CAPABILITIES,
    CHATTERBOX_TEMPERATURE,
    Prosody,
    SynthesisResult,
)


class CudaChatterboxBackend:
    name = "chatterbox"
    capabilities = CAPABILITIES[name]
    model_id = "ResembleAI/chatterbox:multilingual-v3"

    def __init__(self, model: object) -> None:
        self._model = model
        self.sample_rate = int(getattr(model, "sr", 24_000))
        self.load_seconds = 0.0

    def synth(self, text: str, lang: str, prosody: Prosody,
              emotion: Emotion, target_seconds: float | None = None,
              seed: int | None = None) -> SynthesisResult:
        del target_seconds
        import torch

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            np.random.seed(seed)
        exaggeration = float(np.clip(
            emotion.exaggeration + prosody.exaggeration_bias, 0.05, 1.0))
        controls = {
            "language_id": lang,
            "exaggeration": exaggeration,
            "cfg_weight": emotion.cfg_weight,
            "temperature": CHATTERBOX_TEMPERATURE,
        }
        started = time.perf_counter()
        wav = self._model.generate(text, **controls)
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        return SynthesisResult(
            audio, self.sample_rate, time.perf_counter() - started,
            controls, "natural")


def load_cuda_chatterbox():
    """Load weights onto the CUDA-emulated device at Space module startup."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    return ChatterboxMultilingualTTS.from_pretrained("cuda", t3_model="v3")
