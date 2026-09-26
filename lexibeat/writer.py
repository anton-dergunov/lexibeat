"""The writer: the model that writes a programme's lines, injected the way the voice is.

A format that takes text from a writer — examples, remarks, a story, an intro — gets it through a
`Writer`. **The host owns the model call**: it has the credentials, the model chain and the retry
policy, and it returns the model's text. LexiBeat owns the prompt, the parser (`script.py`) and so
what a good line is. `create_service(writer_factory=…)` builds one per render from the
`RenderContext`, exactly as `backend_factory` builds the voice.

`GeminiWriter` is for the command line and the live tests only. It reads `GEMINI_API_KEY`, as the
command line's Gemini voice does, and nothing on the service's integration path constructs it.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol

from .voice import _redact_provider_text


@dataclass(frozen=True)
class WriteRequest:
    """What a writer is asked: one prompt, and what it is for, for a host's own logs and routing."""

    prompt: str
    purpose: str = "programme"


class Writer(Protocol):
    def write(self, request: WriteRequest) -> str: ...


class WriterError(RuntimeError):
    """No model answered. The message keeps each model's own reason."""


# Strongest first, a lite model last. The strong ones are overloaded often: a model that says so
# is waited out a little before the next is asked, and one out of quota is passed over at once.
GEMINI_MODELS = ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash",
                 "gemini-3.5-flash-lite")
_BUSY = ("500", "502", "503", "504", "UNAVAILABLE", "DEADLINE_EXCEEDED")
_OUT_OF_QUOTA = ("429", "RESOURCE_EXHAUSTED")
BUSY_WAITS = (15.0, 30.0)


class GeminiWriter:
    """Text from the Gemini API on `GEMINI_API_KEY`: the command line's writer, never the service's.

    It asks for text, not JSON mode and not a schema. A model that is busy is asked again after a
    short wait, twice; one out of quota, or still busy, gives way to the next; any other refusal is a
    mistake to see, and is raised. `last_model` says which model answered, so a script from a
    fallback model is not mistaken for the best one.
    """

    def __init__(self, models: tuple[str, ...] = GEMINI_MODELS, *, api_key: str | None = None,
                 temperature: float | None = None) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is required for the Gemini writer.")
        from google import genai

        self._client = genai.Client(api_key=key)
        self.models = models
        self.temperature = temperature
        self.last_model: str | None = None
        self.last_seconds: float = 0.0

    def write(self, request: WriteRequest) -> str:
        from google.genai import types

        reasons: list[str] = []
        for model in self.models:
            for wait in (*BUSY_WAITS, None):
                started = time.perf_counter()
                try:
                    answer = self._client.models.generate_content(
                        model=model, contents=request.prompt,
                        config=types.GenerateContentConfig(temperature=self.temperature))
                except Exception as failure:  # the SDK raises one class per status
                    text = _redact_provider_text(str(failure))
                    if any(code in text for code in _BUSY) and wait is not None:
                        self._sleep(wait)
                        continue
                    if any(code in text for code in (*_BUSY, *_OUT_OF_QUOTA)):
                        reasons.append(f"{model}: {text[:200]}")
                        break
                    raise WriterError(f"{model} refused: {text[:400]}") from failure
                self.last_model = model
                self.last_seconds = round(time.perf_counter() - started, 1)
                return answer.text or ""
        raise WriterError("No Gemini model answered. " + " | ".join(reasons))

    @staticmethod
    def _sleep(seconds: float) -> None:
        time.sleep(seconds)
