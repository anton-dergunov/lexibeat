"""Speech synthesis with capability-aware local and hosted backends.

Kokoro and Chatterbox remain the stable paths. Experimental model-side controls,
provider metadata, and any local post-processing are recorded separately.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import librosa
import numpy as np

from .emotion import NEUTRAL, Emotion
from .music import SR

KOKORO_SR = 24000
LANGS = {"es": "e", "en": "a"}
DEFAULT_VOICES = {"es": "ef_dora", "en": "af_heart"}
CHATTERBOX_LANGS = {"es": "es", "en": "en"}
DEFAULT_MLX_MODEL = "mlx-community/chatterbox-multilingual-v3"
DEFAULT_MODELS = {
    "chatterbox": DEFAULT_MLX_MODEL,
    "gemini": "gemini-3.1-flash-tts-preview",
    "gemini-vertex": "gemini-3.1-flash-tts-preview",
    "cloudflare-aura2": "@cf/deepgram/aura-2-{lang}",
    "cloudflare-melotts": "@cf/myshell-ai/melotts",
    "indextts25": "vanch007/mlx-indextts2-2.5-8bit",
    "voxcpm2": "mlx-community/VoxCPM2-4bit",
    "qwen3": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit",
    "tada": "HumeAI/mlx-tada-3b",
    "fish-s2": "mlx-community/fish-audio-s2-pro-8bit",
}
TADA_TOKENIZER_MODEL = "gafiatulin/tada-3b-ml-mlx"
QWEN_VOICES = {"es": "Serena", "en": "Ryan"}
GEMINI_VOICES = {"es": "Sulafat", "en": "Achird"}
GEMINI_LOCALES = {"es": "es-US", "en": "en-GB"}
AURA2_VOICES = {"es": "aquila", "en": "luna"}
GEMINI_FREE_TIER_INTERVALS = {
    "gemini-3.1-flash-tts-preview": 20.5,
    "gemini-2.5-flash-preview-tts": 20.5,
}

REFERENCE_TEXTS = {
    "es": ("Buenos días. Hoy vamos a practicar algunas palabras nuevas. "
           "Escucha con calma: primero en español, después en inglés. "
           "¿Preparado? Empezamos."),
    "en": ("Good morning. Today we're going to practise a few new words. "
           "Listen calmly: first in Spanish, then in English. "
           "Ready? Let's begin."),
}
DEFAULT_REFERENCES = {"es": "say:Paulina", "en": "say:Daniel"}


@dataclass(frozen=True)
class BackendCapabilities:
    emotion: str
    rate: str
    voice: str
    languages: tuple[str, ...] = ("es", "en")
    experimental: bool = False
    license: str = ""
    warnings: tuple[str, ...] = ()


CAPABILITIES = {
    "kokoro": BackendCapabilities("post-process", "native", "preset"),
    "chatterbox": BackendCapabilities("exaggeration", "unsupported", "clone"),
    "gemini": BackendCapabilities(
        "instruction", "instruction", "preset", experimental=True,
        license="Google Gemini API Additional Terms",
        warnings=("Preview API; output is nondeterministic and voice_seed is not supported.",),
    ),
    "gemini-vertex": BackendCapabilities(
        "instruction", "instruction", "preset", experimental=True,
        license="Google Cloud and Vertex AI terms",
        warnings=("Hosted output is nondeterministic and voice_seed is not supported.",),
    ),
    "cloudflare-aura2": BackendCapabilities(
        "post-process", "post-process", "preset", experimental=True,
        license="Cloudflare Workers AI and Deepgram terms",
        warnings=("The hosted model has no explicit prosody or seed control; "
                  "gentle variation is applied locally.",),
    ),
    "cloudflare-melotts": BackendCapabilities(
        "post-process", "post-process", "unsupported", languages=("en",),
        experimental=True,
        license="Cloudflare Workers AI and MeloTTS terms",
        warnings=("The hosted API exposes only text and language; Cloudflare's "
                  "current deployment rejects Spanish (AiError 8002).",),
    ),
    "indextts25": BackendCapabilities(
        "8-float vector", "model duration factor", "clone", experimental=True,
        license="bilibili Model Use License Agreement",
        warnings=("Duration shaping is model-side but not sample-exact.",),
    ),
    "voxcpm2": BackendCapabilities(
        "instruction", "instruction", "clone", experimental=True,
        license="Apache-2.0",
        warnings=("Pace and pitch are qualitative natural-language controls.",),
    ),
    "qwen3": BackendCapabilities(
        "instruction", "instruction", "preset", experimental=True,
        license="Apache-2.0",
        warnings=("No native Spanish preset or reference cloning in CustomVoice.",),
    ),
    "tada": BackendCapabilities(
        "unsupported", "unsupported", "clone", experimental=True,
        license="Llama 3.2 Community License",
        warnings=("TADA contributes stochastic prosody, not semantic emotion or rate control.",),
    ),
    "fish-s2": BackendCapabilities(
        "inline tags and instruction", "instruction", "clone", experimental=True,
        license="Fish Audio Research License",
        warnings=("The MLX speed argument is post-hoc and is intentionally not used.",),
    ),
}


@dataclass(frozen=True)
class Prosody:
    speed: float = 1.0
    semitones: float = 0.0
    gain_db: float = 0.0
    exaggeration_bias: float = 0.0

    TABLE = (
        (1.00, 0.00, 0.0, 0.00),
        (0.96, -0.35, -0.8, -0.06),
        (1.02, +0.40, +0.6, +0.10),
        (0.98, +0.15, -0.3, +0.03),
    )

    @classmethod
    def for_repeat(cls, index: int, strength: float = 1.0) -> "Prosody":
        speed, semitones, gain, exaggeration = cls.TABLE[index % len(cls.TABLE)]
        return cls(speed=1.0 + (speed - 1.0) * strength,
                   semitones=semitones * strength,
                   gain_db=gain * strength,
                   exaggeration_bias=exaggeration * strength)

    def with_emotion(self, emotion: Emotion, strength: float = 1.0) -> "Prosody":
        return replace(
            self,
            speed=self.speed * (1.0 + (emotion.speed_bias - 1.0) * strength),
            semitones=self.semitones + emotion.pitch_bias * strength,
        )


@dataclass
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    generation_seconds: float
    controls: dict[str, Any] = field(default_factory=dict)
    timing_mode: str = "natural"
    warnings: list[str] = field(default_factory=list)
    peak_memory_bytes: int | None = None
    passes: int = 1

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("audio")
        return data


class Backend(Protocol):
    name: str
    sample_rate: int
    capabilities: BackendCapabilities
    load_seconds: float
    model_id: str

    def synth(self, text: str, lang: str, prosody: Prosody,
              emotion: Emotion, target_seconds: float | None = None,
              seed: int | None = None) -> SynthesisResult: ...


def model_cache_root() -> Path:
    root = Path(os.environ.get("LEXIBEAT_CACHE", Path.home() / ".cache" / "lexibeat"))
    path = root / "models"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Read-only CI/sandbox homes still need a harmless location for mocked
        # backend tests. Real runs should set LEXIBEAT_CACHE explicitly.
        path = Path(tempfile.gettempdir()) / "lexibeat-models"
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(path / "huggingface"))
    return path


def _trim(audio: np.ndarray) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(audio, top_db=32)
    return trimmed if trimmed.size else audio


def _peak_mlx_memory() -> int | None:
    if "mlx.core" not in sys.modules:
        return None
    try:
        mx = sys.modules["mlx.core"]
        return int(mx.get_peak_memory())
    except Exception:
        return None


def _seed_mlx(seed: int | None) -> None:
    if seed is None or "mlx.core" not in sys.modules:
        return
    try:
        mx = sys.modules["mlx.core"]
        mx.random.seed(seed)
    except Exception:
        pass


def _collect(results: Any, fallback_rate: int) -> tuple[np.ndarray, int]:
    pieces: list[np.ndarray] = []
    rate = fallback_rate
    for result in results:
        pieces.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
        rate = int(getattr(result, "sample_rate", rate) or rate)
    return (np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32), rate)


class KokoroBackend:
    name = "kokoro"
    capabilities = CAPABILITIES[name]
    model_id = "hexgrad/Kokoro-82M"
    sample_rate = KOKORO_SR

    def __init__(self, voices: dict[str, str] | None = None) -> None:
        from kokoro import KPipeline
        started = time.perf_counter()
        self._voices = {**DEFAULT_VOICES, **(voices or {})}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._pipes = {lang: KPipeline(lang_code=code, repo_id=self.model_id)
                           for lang, code in LANGS.items()}
        self.load_seconds = time.perf_counter() - started

    def synth(self, text: str, lang: str, prosody: Prosody,
              emotion: Emotion, target_seconds: float | None = None,
              seed: int | None = None) -> SynthesisResult:
        del emotion, target_seconds, seed
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunks = list(self._pipes[lang](text, voice=self._voices[lang],
                                            speed=prosody.speed))
        audio = np.concatenate([c.audio.numpy() for c in chunks]).astype(np.float32)
        return SynthesisResult(audio, self.sample_rate,
                               time.perf_counter() - started,
                               {"speed": prosody.speed}, "native-rate")


class _ReferenceBackend:
    def _configure_references(
        self,
        ref_audio: str | None,
        ref_audios: dict[str, str] | None,
        ref_texts: dict[str, str] | None,
    ) -> None:
        sources = dict(DEFAULT_REFERENCES)
        if ref_audio:
            warnings.warn("A shared reference applies one speaker to both languages.",
                          stacklevel=3)
            sources = {"es": ref_audio, "en": ref_audio}
        if ref_audios:
            sources.update({key: value for key, value in ref_audios.items() if value})
        self.ref_audios = {
            lang: str(ensure_reference(source, lang))
            if source.startswith(("say:", "kokoro:")) else str(source)
            for lang, source in sources.items()
        }
        self.ref_texts = {**REFERENCE_TEXTS, **(ref_texts or {})}


class MlxAudioBackend(_ReferenceBackend):
    """Chatterbox via mlx-audio; kept under its historical public class name."""

    name = "chatterbox"
    capabilities = CAPABILITIES[name]

    def __init__(self, model: str = DEFAULT_MLX_MODEL,
                 ref_audio: str | None = None,
                 ref_audios: dict[str, str] | None = None,
                 ref_texts: dict[str, str] | None = None, **_: Any) -> None:
        model_cache_root()
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as exc:
            raise RuntimeError("mlx-audio is unavailable (Apple Silicon only).") from exc
        self.model_id = model
        self._configure_references(ref_audio, ref_audios, ref_texts)
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = load_model(model)
        self.load_seconds = time.perf_counter() - started
        self.sample_rate = int(getattr(self._model, "sample_rate", 24000))

    def synth(self, text: str, lang: str, prosody: Prosody,
              emotion: Emotion, target_seconds: float | None = None,
              seed: int | None = None) -> SynthesisResult:
        del target_seconds
        _seed_mlx(seed)
        kwargs = dict(
            text=text, lang_code=CHATTERBOX_LANGS.get(lang, lang),
            ref_audio=self.ref_audios[lang],
            exaggeration=float(np.clip(
                emotion.exaggeration + prosody.exaggeration_bias, 0.05, 1.0)),
            cfg_weight=emotion.cfg_weight, temperature=0.8,
        )
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, rate = _collect(self._model.generate(**kwargs), self.sample_rate)
        self.sample_rate = rate
        return SynthesisResult(audio, rate, time.perf_counter() - started, kwargs,
                               "natural", peak_memory_bytes=_peak_mlx_memory())


def delivery_instruction(emotion: Emotion, prosody: Prosody) -> str:
    pace = "at a natural pace"
    if prosody.speed < 0.985:
        pace = "slightly slowly and deliberately"
    elif prosody.speed > 1.015:
        pace = "slightly briskly"
    pitch = "with a natural pitch range"
    if prosody.semitones > 0.25:
        pitch = "with a slightly brighter, higher pitch"
    elif prosody.semitones < -0.25:
        pitch = "with a slightly lower, softer pitch"
    return f"Speak in a {emotion.name} but clear tone, {pace}, {pitch}."


def gemini_prompt(text: str, lang: str, emotion: Emotion,
                  prosody: Prosody) -> str:
    """Build restrained director notes without changing the spoken transcript."""
    language = "native Spanish" if lang == "es" else "native English"
    instruction = delivery_instruction(emotion, prosody)
    pause_note = ""
    if "[long pause]" in text:
        pause_note = (
            " Treat every [long pause] tag as a silent timing instruction: "
            "do not speak the tag, and leave a clearly separable long silence."
        )
    return (
        "Generate speech for a short language-learning repetition.\n"
        f"Use {language}. {instruction}\n"
        "Keep the delivery natural, subtle, clear, and gently reinforcing. "
        "Do not sing, spell, translate, paraphrase, add words, or make any "
        "non-verbal sounds. Speak only the transcript between the markers."
        f"{pause_note}\n"
        "<TRANSCRIPT>\n"
        f"{text}\n"
        "</TRANSCRIPT>"
    )


def _decode_audio_file(data: bytes) -> tuple[np.ndarray, int]:
    import soundfile as sf

    try:
        audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception as exc:
        raise RuntimeError("Provider returned malformed or unsupported audio.") from exc
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.reshape(-1)
    if not audio.size or not np.all(np.isfinite(audio)):
        raise RuntimeError("Provider returned empty or non-finite audio.")
    return audio, int(rate)


def _redact_provider_text(value: str) -> str:
    redacted = value
    for name in ("GEMINI_API_KEY", "CLOUDFLARE_ACCOUNT_ID",
                 "CLOUDFLARE_API_TOKEN"):
        secret = os.environ.get(name)
        if secret:
            redacted = redacted.replace(secret, "<redacted>")
    return redacted[:500]


class GeminiBackend:
    name = "gemini"
    capabilities = CAPABILITIES[name]
    sample_rate = 24000

    def __init__(self, model: str = DEFAULT_MODELS[name],
                 voices: dict[str, str] | None = None,
                 vertex: bool = False, **_: Any) -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if vertex and not project:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT is required for the Gemini Vertex backend.")
        if not vertex and not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for the Gemini backend.")
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as exc:
            raise RuntimeError("Install hosted backends with "
                               "'uv sync --extra hosted-tts'.") from exc
        started = time.perf_counter()
        self.vertex = vertex
        self.capabilities = CAPABILITIES[
            "gemini-vertex" if vertex else "gemini"]
        self.project = project if vertex else None
        self.location = (os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
                         if vertex else None)
        self._types = genai_types
        if vertex:
            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=self.location,
                http_options=genai_types.HttpOptions(timeout=180_000),
            )
        else:
            self._client = genai.Client(api_key=api_key)
        self.load_seconds = time.perf_counter() - started
        self.model_id = model
        self.voices = {**GEMINI_VOICES, **(voices or {})}
        # These preview TTS models are currently limited to 3 RPM on the free
        # project used by this repository. The interval also adapts upward when
        # the API supplies a longer Retry-After value.
        self._min_request_interval = (
            0.0 if vertex else GEMINI_FREE_TIER_INTERVALS.get(model, 0.0))
        self._last_request_started: float | None = None

    def _wait_for_request_slot(self) -> None:
        interval = float(getattr(self, "_min_request_interval", 0.0))
        now = time.monotonic()
        previous = getattr(self, "_last_request_started", None)
        delay = 0.0 if previous is None else interval - (now - previous)
        if delay > 0:
            time.sleep(delay)
            now += delay
        self._last_request_started = now

    @staticmethod
    def _retry_after(exc: Exception) -> float | None:
        headers = getattr(getattr(exc, "response", None), "headers", None)
        value = headers.get("retry-after") if headers else None
        try:
            return min(max(float(value), 0.0), 120.0)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _daily_quota_exhausted(exc: Exception) -> bool:
        detail = str(exc).lower().replace("_", "").replace("-", "")
        return any(marker in detail for marker in (
            "requestsperday", "permodelperday", "dailyrequest", "rpd",
        ))

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        value = (getattr(exc, "status_code", None) or
                 getattr(exc, "code", None) or
                 getattr(getattr(exc, "response", None), "status_code", None))
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _generate(self, prompt: str, voice: str, lang: str = "en"):
        last_error: Exception | None = None
        for attempt in range(6):
            self._wait_for_request_slot()
            try:
                if getattr(self, "vertex", False):
                    config = self._types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=self._types.SpeechConfig(
                            language_code=GEMINI_LOCALES[lang],
                            voice_config=self._types.VoiceConfig(
                                prebuilt_voice_config=
                                self._types.PrebuiltVoiceConfig(
                                    voice_name=voice),
                            ),
                        ),
                    )
                    return self._client.models.generate_content(
                        model=self.model_id,
                        contents=prompt,
                        config=config,
                    )
                return self._client.interactions.create(
                    model=self.model_id,
                    input=prompt,
                    response_format={"type": "audio"},
                    generation_config={"speech_config": [{"voice": voice}]},
                    timeout=180.0,
                )
            except Exception as exc:
                last_error = exc
                status = self._status_code(exc)
                if status == 429 and self._daily_quota_exhausted(exc):
                    break
                if status != 429 and (not status or int(status) < 500):
                    break
                if attempt < 5:
                    if status == 429:
                        # The preview free tier commonly allows only a few
                        # requests per minute. Learn that constraint on demand.
                        retry_after = self._retry_after(exc)
                        self._min_request_interval = max(
                            float(getattr(self, "_min_request_interval", 0.0)),
                            retry_after or 20.5,
                        )
                    else:
                        time.sleep(0.5 * 2 ** attempt)
        status = self._status_code(last_error) if last_error else None
        suffix = f" (HTTP {status})" if status else ""
        provider = "Gemini Vertex TTS" if getattr(self, "vertex", False) else "Gemini TTS"
        raise RuntimeError(f"{provider} request failed{suffix}: "
                           f"{_redact_provider_text(str(last_error))}") from last_error

    def synth(self, text: str, lang: str, prosody: Prosody,
              emotion: Emotion, target_seconds: float | None = None,
              seed: int | None = None) -> SynthesisResult:
        del target_seconds
        prompt = gemini_prompt(text, lang, emotion, prosody)
        voice = self.voices[lang]
        started = time.perf_counter()
        interaction = self._generate(prompt, voice, lang)
        if getattr(self, "vertex", False):
            candidates = getattr(interaction, "candidates", None) or []
            content = getattr(candidates[0], "content", None) if candidates else None
            parts = getattr(content, "parts", None) or []
            output = getattr(parts[0], "inline_data", None) if parts else None
        else:
            output = getattr(interaction, "output_audio", None)
        encoded = getattr(output, "data", None)
        if not encoded:
            raise RuntimeError("Gemini response contained no audio data.")
        if isinstance(encoded, bytes):
            raw = encoded
        else:
            try:
                raw = base64.b64decode(str(encoded), validate=True)
            except (ValueError, TypeError) as exc:
                raise RuntimeError("Gemini returned invalid base64 audio data.") from exc
        mime_type = str(getattr(output, "mime_type", None) or "audio/l16")
        rate_match = re.search(r"(?:rate|sample_rate)=(\d+)", mime_type,
                               flags=re.IGNORECASE)
        channels_match = re.search(r"channels=(\d+)", mime_type,
                                   flags=re.IGNORECASE)
        rate = int(getattr(output, "sample_rate", None) or
                   (rate_match.group(1) if rate_match else self.sample_rate))
        channels = int(getattr(output, "channels", None) or
                       (channels_match.group(1) if channels_match else 1))
        base_mime_type = mime_type.split(";", 1)[0].strip().lower()
        if base_mime_type in ("audio/l16", "audio/pcm", "none"):
            if len(raw) % 2:
                raise RuntimeError("Gemini returned malformed PCM16 audio.")
            audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            if channels > 1:
                if len(audio) % channels:
                    raise RuntimeError("Gemini returned malformed multichannel PCM.")
                audio = audio.reshape(-1, channels).mean(axis=1)
        else:
            audio, rate = _decode_audio_file(raw)
        if not audio.size or not np.all(np.isfinite(audio)):
            raise RuntimeError("Gemini returned empty or non-finite audio.")
        duration = len(audio) / rate
        usage = (getattr(interaction, "usage_metadata", None) or
                 getattr(interaction, "usage", None))
        input_tokens = (getattr(usage, "prompt_token_count", None) or
                        getattr(usage, "total_input_tokens", None))
        output_tokens = (getattr(usage, "candidates_token_count", None) or
                         getattr(usage, "total_output_tokens", None))
        prices = {
            "gemini-3.1-flash-tts-preview": (1.0, 20.0),
            "gemini-2.5-flash-preview-tts": (0.5, 10.0),
            "gemini-2.5-flash-tts": (0.5, 10.0),
            "gemini-2.5-flash-lite-preview-tts": (0.5, 10.0),
            "gemini-2.5-pro-preview-tts": (1.0, 20.0),
            "gemini-2.5-pro-tts": (1.0, 20.0),
        }.get(self.model_id)
        estimated_cost = None
        if prices:
            billed_input = int(input_tokens or max(1, len(prompt) // 4))
            billed_output = int(output_tokens or round(duration * 25))
            estimated_cost = (billed_input * prices[0] +
                              billed_output * prices[1]) / 1_000_000
        controls = {
            "model": self.model_id,
            "provider": "vertex-ai" if getattr(self, "vertex", False)
            else "gemini-api",
            "location": getattr(self, "location", None),
            "instruction": delivery_instruction(emotion, prosody),
            "voice": voice,
            "language": lang,
            "characters": len(text),
            "seed_supported": False,
            "requested_seed": seed,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": estimated_cost,
        }
        return SynthesisResult(audio.astype(np.float32), rate,
                               time.perf_counter() - started, controls,
                               "instruction-rate",
                               list(self.capabilities.warnings))


class _CloudflareBackend:
    capabilities: BackendCapabilities

    def _configure_cloudflare(self) -> None:
        self.account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        self.api_token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
        missing = [name for name, value in (
            ("CLOUDFLARE_ACCOUNT_ID", self.account_id),
            ("CLOUDFLARE_API_TOKEN", self.api_token),
        ) if not value]
        if missing:
            raise RuntimeError(f"{', '.join(missing)} required for Cloudflare.")
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Install hosted backends with "
                               "'uv sync --extra hosted-tts'.") from exc
        self._session = requests.Session()

    def _request(self, model: str, payload: dict[str, Any]) -> tuple[bytes, str]:
        endpoint = ("https://api.cloudflare.com/client/v4/accounts/"
                    f"{self.account_id}/ai/run/{model}")
        response = None
        for attempt in range(6):
            try:
                response = self._session.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {self.api_token}"},
                    json=payload,
                    timeout=180.0,
                )
            except Exception as exc:
                if attempt < 5:
                    time.sleep(1.0 * 2 ** attempt)
                    continue
                raise RuntimeError("Cloudflare TTS request failed: "
                                   f"{type(exc).__name__}") from exc
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 5:
                    retry_after = response.headers.get("retry-after")
                    try:
                        delay = min(float(retry_after), 5.0)
                    except (TypeError, ValueError):
                        delay = 1.0 * 2 ** attempt
                    time.sleep(delay)
                    continue
            break
        assert response is not None
        if not response.ok:
            detail = _redact_provider_text(response.text.strip())
            raise RuntimeError(f"Cloudflare TTS HTTP {response.status_code}: "
                               f"{detail or 'empty response body'}")
        content_type = response.headers.get("content-type", "").split(";", 1)[0]
        if content_type.startswith("audio/"):
            return response.content, content_type
        try:
            envelope = response.json()
        except ValueError as exc:
            raise RuntimeError("Cloudflare returned neither audio nor JSON.") from exc
        if isinstance(envelope, dict) and not envelope.get("success", True):
            detail = _redact_provider_text(json.dumps(
                envelope.get("errors", "unknown error"), ensure_ascii=False))
            raise RuntimeError(f"Cloudflare TTS failed: {detail}")
        result = envelope.get("result", envelope) if isinstance(envelope, dict) else envelope
        encoded: Any = result
        if isinstance(result, dict):
            encoded = result.get("audio", result.get("data"))
            content_type = str(result.get("content_type", "audio/mpeg"))
        if isinstance(encoded, list) and encoded:
            encoded = encoded[0]
        if not isinstance(encoded, str):
            raise RuntimeError("Cloudflare response contained no audio data.")
        try:
            return base64.b64decode(encoded.split(",", 1)[-1], validate=True), content_type
        except (ValueError, TypeError) as exc:
            raise RuntimeError("Cloudflare returned invalid base64 audio.") from exc


class CloudflareAura2Backend(_CloudflareBackend):
    name = "cloudflare-aura2"
    capabilities = CAPABILITIES[name]
    sample_rate = 24000

    def __init__(self, model: str = DEFAULT_MODELS[name],
                 voices: dict[str, str] | None = None, **_: Any) -> None:
        if "{lang}" not in model:
            raise ValueError("Cloudflare Aura 2 model must contain '{lang}'.")
        started = time.perf_counter()
        self._configure_cloudflare()
        self.load_seconds = time.perf_counter() - started
        self.model_id = model
        self.voices = {**AURA2_VOICES, **(voices or {})}

    def synth(self, text: str, lang: str, prosody: Prosody,
              emotion: Emotion, target_seconds: float | None = None,
              seed: int | None = None) -> SynthesisResult:
        del target_seconds
        model = self.model_id.format(lang=lang)
        payload = {
            "text": text,
            "speaker": self.voices[lang],
            "encoding": "linear16",
            "container": "wav",
            "sample_rate": self.sample_rate,
        }
        started = time.perf_counter()
        raw, _ = self._request(model, payload)
        audio, rate = _decode_audio_file(raw)
        controls = {
            "model": model,
            "voice": self.voices[lang],
            "characters": len(text),
            "native_prosody_supported": False,
            "requested_emotion": emotion.name,
            "requested_speed": prosody.speed,
            "requested_semitones": prosody.semitones,
            "seed_supported": False,
            "requested_seed": seed,
            "estimated_cost_usd": len(text) / 1000 * 0.03,
        }
        return SynthesisResult(audio, rate, time.perf_counter() - started,
                               controls, "local-post-process",
                               list(self.capabilities.warnings))


class CloudflareMeloBackend(_CloudflareBackend):
    name = "cloudflare-melotts"
    capabilities = CAPABILITIES[name]
    sample_rate = 24000

    def __init__(self, model: str = DEFAULT_MODELS[name],
                 voices: dict[str, str] | None = None, **_: Any) -> None:
        if voices:
            raise ValueError("Cloudflare MeloTTS does not expose voice selection.")
        started = time.perf_counter()
        self._configure_cloudflare()
        self.load_seconds = time.perf_counter() - started
        self.model_id = model
        self._audio_cache: dict[tuple[str, str], tuple[np.ndarray, int, float]] = {}

    def synth(self, text: str, lang: str, prosody: Prosody,
              emotion: Emotion, target_seconds: float | None = None,
              seed: int | None = None) -> SynthesisResult:
        del target_seconds
        if lang != "en":
            raise RuntimeError(
                "Cloudflare MeloTTS currently rejects Spanish with AiError 8002; "
                "the backend is retained as an English-only baseline until the "
                "provider fixes cloudflare/ai#221.")
        started = time.perf_counter()
        cache_key = (text, lang)
        cached = self._audio_cache.get(cache_key)
        if cached is None:
            raw, _ = self._request(self.model_id, {"prompt": text, "lang": lang})
            audio, rate = _decode_audio_file(raw)
            provider_cost = len(audio) / rate / 60 * 0.0002
            self._audio_cache[cache_key] = (audio.copy(), rate, provider_cost)
            cache_hit = False
        else:
            audio, rate, _ = cached
            audio = audio.copy()
            provider_cost = 0.0
            cache_hit = True
        controls = {
            "model": self.model_id,
            "language": lang,
            "characters": len(text),
            "native_prosody_supported": False,
            "requested_emotion": emotion.name,
            "requested_speed": prosody.speed,
            "requested_semitones": prosody.semitones,
            "seed_supported": False,
            "requested_seed": seed,
            "cache_hit": cache_hit,
            "estimated_cost_usd": provider_cost,
        }
        return SynthesisResult(audio, rate, time.perf_counter() - started,
                               controls, "local-post-process",
                               list(self.capabilities.warnings))


class VoxCPM2Backend(_ReferenceBackend):
    name = "voxcpm2"
    capabilities = CAPABILITIES[name]

    def __init__(self, model: str = DEFAULT_MODELS[name], ref_audio: str | None = None,
                 ref_audios: dict[str, str] | None = None,
                 ref_texts: dict[str, str] | None = None, **_: Any) -> None:
        model_cache_root()
        from mlx_audio.tts.utils import load_model
        self.model_id = model
        self._configure_references(ref_audio, ref_audios, ref_texts)
        started = time.perf_counter()
        self._model = load_model(model)
        self.load_seconds = time.perf_counter() - started
        self.sample_rate = int(getattr(self._model, "sample_rate", 48000))

    def synth(self, text: str, lang: str, prosody: Prosody,
              emotion: Emotion, target_seconds: float | None = None,
              seed: int | None = None) -> SynthesisResult:
        del target_seconds
        _seed_mlx(seed)
        instruct = delivery_instruction(emotion, prosody)
        kwargs = dict(text=text, ref_audio=self.ref_audios[lang],
                      ref_text=self.ref_texts[lang], instruct=instruct,
                      inference_timesteps=10, cfg_value=2.0)
        started = time.perf_counter()
        audio, rate = _collect(self._model.generate(**kwargs), self.sample_rate)
        self.sample_rate = rate
        return SynthesisResult(audio, rate, time.perf_counter() - started,
                               {"instruct": instruct, "reference": self.ref_audios[lang]},
                               "instruction-rate", peak_memory_bytes=_peak_mlx_memory())


class Qwen3Backend:
    name = "qwen3"
    capabilities = CAPABILITIES[name]

    def __init__(self, model: str = DEFAULT_MODELS[name],
                 voices: dict[str, str] | None = None, **_: Any) -> None:
        model_cache_root()
        from mlx_audio.tts.utils import load_model
        self.model_id = model
        self.voices = {**QWEN_VOICES, **(voices or {})}
        started = time.perf_counter()
        self._model = load_model(model)
        self.load_seconds = time.perf_counter() - started
        self.sample_rate = int(getattr(self._model, "sample_rate", 24000))

    def synth(self, text: str, lang: str, prosody: Prosody,
              emotion: Emotion, target_seconds: float | None = None,
              seed: int | None = None) -> SynthesisResult:
        del target_seconds
        _seed_mlx(seed)
        instruct = delivery_instruction(emotion, prosody)
        kwargs = dict(text=text, voice=self.voices[lang], instruct=instruct,
                      lang_code="spanish" if lang == "es" else "english",
                      temperature=0.85, top_k=50, top_p=0.95)
        started = time.perf_counter()
        audio, rate = _collect(self._model.generate(**kwargs), self.sample_rate)
        self.sample_rate = rate
        return SynthesisResult(audio, rate, time.perf_counter() - started,
                               {"instruct": instruct, "voice": self.voices[lang]},
                               "instruction-rate", list(self.capabilities.warnings),
                               _peak_mlx_memory())


FISH_TAGS = {
    "happy": "excited", "warm": "delight", "delighted": "delight",
    "emphatic": "emphasis", "surprised": "surprised", "angry": "angry",
    "sad": "sad", "afraid": "fearful", "disgusted": "disgusted",
    "thoughtful": "calm", "calm": "calm", "neutral": "calm",
}


class FishS2Backend(_ReferenceBackend):
    name = "fish-s2"
    capabilities = CAPABILITIES[name]

    def __init__(self, model: str = DEFAULT_MODELS[name], ref_audio: str | None = None,
                 ref_audios: dict[str, str] | None = None,
                 ref_texts: dict[str, str] | None = None, **_: Any) -> None:
        model_cache_root()
        from mlx_audio.tts.utils import load_model
        self.model_id = model
        self._configure_references(ref_audio, ref_audios, ref_texts)
        started = time.perf_counter()
        self._model = load_model(model)
        self.load_seconds = time.perf_counter() - started
        self.sample_rate = int(getattr(self._model, "sample_rate", 44100))

    def _reference_array(self, lang: str):
        from mlx_audio.tts.generate import load_audio
        return load_audio(self.ref_audios[lang], sample_rate=self.sample_rate)

    def synth(self, text: str, lang: str, prosody: Prosody,
              emotion: Emotion, target_seconds: float | None = None,
              seed: int | None = None) -> SynthesisResult:
        del target_seconds
        _seed_mlx(seed)
        tag = FISH_TAGS.get(emotion.name, "calm")
        instruct = delivery_instruction(emotion, prosody)
        kwargs = dict(text=f"[{tag}] {text}", ref_audio=self._reference_array(lang),
                      ref_text=self.ref_texts[lang], instruct=instruct,
                      temperature=0.75, top_p=0.8, top_k=30, speed=1.0)
        started = time.perf_counter()
        audio, rate = _collect(self._model.generate(**kwargs), self.sample_rate)
        self.sample_rate = rate
        return SynthesisResult(audio, rate, time.perf_counter() - started,
                               {"tag": tag, "instruct": instruct, "speed": 1.0},
                               "instruction-rate", list(self.capabilities.warnings),
                               _peak_mlx_memory())


class TadaBackend(_ReferenceBackend):
    name = "tada"
    capabilities = CAPABILITIES[name]

    def __init__(self, model: str = DEFAULT_MODELS[name], ref_audio: str | None = None,
                 ref_audios: dict[str, str] | None = None,
                 ref_texts: dict[str, str] | None = None, **_: Any) -> None:
        model_cache_root()
        try:
            from mlx_tada import TadaForCausalLM
        except ImportError as exc:
            raise RuntimeError("Install experimental backends with "
                               "'uv sync --extra experimental-tts'.") from exc
        self.model_id = model
        self._configure_references(ref_audio, ref_audios, ref_texts)
        started = time.perf_counter()
        # Hume's otherwise self-contained MLX checkpoint hard-codes a gated
        # Meta tokenizer lookup. Use the identical public Llama tokenizer
        # bundled by the TADA MLX conversion, without substituting model weights.
        from huggingface_hub import snapshot_download
        from mlx_tada import model as tada_model
        tokenizer_path = snapshot_download(
            TADA_TOKENIZER_MODEL,
            allow_patterns=["tokenizer.json", "tokenizer_config.json"],
        )
        original_tokenizer_loader = tada_model.AutoTokenizer.from_pretrained
        tada_model.AutoTokenizer.from_pretrained = lambda *_args, **_kwargs: (
            original_tokenizer_loader(tokenizer_path, local_files_only=True))
        try:
            self._model = (TadaForCausalLM.from_weights(model, quantize=4)
                           if Path(model).exists()
                           else TadaForCausalLM.from_pretrained(model, quantize=4))
        finally:
            tada_model.AutoTokenizer.from_pretrained = original_tokenizer_loader
        self.load_seconds = time.perf_counter() - started
        self.sample_rate = 24000
        self._references: dict[str, Any] = {}

    def synth(self, text: str, lang: str, prosody: Prosody,
              emotion: Emotion, target_seconds: float | None = None,
              seed: int | None = None) -> SynthesisResult:
        del target_seconds, emotion
        from mlx_tada import InferenceOptions
        _seed_mlx(seed)
        reference_seconds = 0.0
        if lang not in self._references:
            reference_started = time.perf_counter()
            self._references[lang] = self._model.load_reference(
                self.ref_audios[lang], audio_text=self.ref_texts[lang])
            reference_seconds = time.perf_counter() - reference_started
        variation = float(np.clip(0.9 + prosody.exaggeration_bias, 0.75, 1.05))
        options = InferenceOptions(text_temperature=0.6,
                                   noise_temperature=variation,
                                   acoustic_cfg_scale=1.6,
                                   duration_cfg_scale=1.0,
                                   num_flow_matching_steps=10)
        started = time.perf_counter()
        output = self._model.generate(text, self._references[lang],
                                      inference_options=options)
        return SynthesisResult(np.asarray(output.audio, dtype=np.float32),
                               self.sample_rate, time.perf_counter() - started,
                               {"noise_temperature": variation, "quantize": 4,
                                "reference_preparation_seconds": reference_seconds},
                               "natural", list(self.capabilities.warnings),
                               _peak_mlx_memory())


class IndexTTS25Backend(_ReferenceBackend):
    name = "indextts25"
    capabilities = CAPABILITIES[name]

    def __init__(self, model: str = DEFAULT_MODELS[name], ref_audio: str | None = None,
                 ref_audios: dict[str, str] | None = None,
                 ref_texts: dict[str, str] | None = None, **_: Any) -> None:
        self.model_id = model
        self._configure_references(ref_audio, ref_audios, ref_texts)
        self.sample_rate = 22050
        self._tmp = tempfile.TemporaryDirectory(prefix="lexibeat-indextts-")
        worker = Path(__file__).with_name("indextts_worker.py")
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("uv is required for the isolated IndexTTS worker.")
        env = os.environ.copy()
        env.setdefault("HF_HOME", str(model_cache_root() / "huggingface"))
        started = time.perf_counter()
        self._process = subprocess.Popen(
            [uv, "run", "--script", str(worker), "--model", model,
             "--cache", str(model_cache_root())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, bufsize=1, env=env,
        )
        ready = self._read_response()
        if not ready.get("ready"):
            raise RuntimeError(ready.get("error", "IndexTTS worker failed to start"))
        self.load_seconds = time.perf_counter() - started

    def _read_response(self) -> dict[str, Any]:
        assert self._process.stdout is not None
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError(
                f"IndexTTS worker exited unexpectedly ({self._process.poll()}).")
        return json.loads(line)

    def synth(self, text: str, lang: str, prosody: Prosody,
              emotion: Emotion, target_seconds: float | None = None,
              seed: int | None = None) -> SynthesisResult:
        output = Path(self._tmp.name) / f"{hashlib.sha1(os.urandom(16)).hexdigest()}.wav"
        request = {
            "text": text, "lang": lang, "reference": self.ref_audios[lang],
            "emotion": emotion.vector(), "duration_factor": 1.0 / prosody.speed,
            "target_seconds": target_seconds, "seed": seed, "output": str(output),
        }
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()
        response = self._read_response()
        if response.get("error"):
            raise RuntimeError(response["error"])
        import soundfile as sf
        audio, rate = sf.read(output, dtype="float32")
        self.sample_rate = int(rate)
        return SynthesisResult(np.asarray(audio, dtype=np.float32).reshape(-1),
                               self.sample_rate, float(response["generation_seconds"]),
                               response.get("controls", {}), "model-duration-factor",
                               list(self.capabilities.warnings),
                               response.get("peak_memory_bytes"),
                               int(response.get("passes", 1)))

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process and process.poll() is None:
            try:
                assert process.stdin is not None
                process.stdin.write('{"command":"close"}\n')
                process.stdin.flush()
                process.wait(timeout=10)
            except Exception:
                process.terminate()
        tmp = getattr(self, "_tmp", None)
        if tmp:
            tmp.cleanup()

    def __del__(self) -> None:
        self.close()


def reference_path(name: str = "kokoro-ef_dora", lang: str | None = None) -> Path:
    directory = Path(os.environ.get("LEXIBEAT_CACHE",
                                    Path.home() / ".cache" / "lexibeat")) / "refs"
    stem = name.replace(":", "-")
    if lang:
        stem += f"-{lang}"
    return directory / f"{stem}.wav"


def ensure_reference(source: str = "kokoro:ef_dora", lang: str = "es") -> Path:
    import soundfile as sf
    if lang not in REFERENCE_TEXTS:
        raise ValueError(f"No reference script for language '{lang}'.")
    path = reference_path(source, lang)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    kind, _, voice = source.partition(":")
    if kind == "say":
        with tempfile.TemporaryDirectory() as tmp:
            aiff = f"{tmp}/ref.aiff"
            fallback = "Paulina" if lang == "es" else "Daniel"
            subprocess.run(["say", "-v", voice or fallback, "-o", aiff,
                            REFERENCE_TEXTS[lang]], check=True)
            audio, rate = sf.read(aiff)
        sf.write(path, audio, rate)
    else:
        fallback = DEFAULT_VOICES[lang]
        backend = KokoroBackend({lang: voice or fallback})
        result = backend.synth(REFERENCE_TEXTS[lang], lang, Prosody(), NEUTRAL)
        sf.write(path, result.audio, result.sample_rate)
    return path


def make_backend(name: str, *, voices: dict[str, str] | None = None,
                 model: str | None = None, ref_audio: str | None = None,
                 ref_audios: dict[str, str] | None = None,
                 ref_texts: dict[str, str] | None = None) -> Backend:
    normalized = "chatterbox" if name == "mlx" else name
    if normalized == "kokoro":
        return KokoroBackend(voices)
    classes = {
        "chatterbox": MlxAudioBackend,
        "gemini": GeminiBackend,
        "gemini-vertex": GeminiBackend,
        "cloudflare-aura2": CloudflareAura2Backend,
        "cloudflare-melotts": CloudflareMeloBackend,
        "indextts25": IndexTTS25Backend,
        "voxcpm2": VoxCPM2Backend,
        "qwen3": Qwen3Backend,
        "tada": TadaBackend,
        "fish-s2": FishS2Backend,
    }
    if normalized not in classes:
        raise ValueError(f"Unknown voice backend '{name}'.")
    selected_model = model or DEFAULT_MODELS[normalized]
    if normalized == "gemini-vertex":
        return GeminiBackend(selected_model, voices=voices, vertex=True)
    return classes[normalized](selected_model, voices=voices, ref_audio=ref_audio,
                               ref_audios=ref_audios, ref_texts=ref_texts)


class Speaker:
    def __init__(self, voices: dict[str, str] | None = None, *,
                 backend: str = "chatterbox", model: str | None = None,
                 ref_audio: str | None = None,
                 ref_audios: dict[str, str] | None = None,
                 ref_texts: dict[str, str] | None = None,
                 prosody_strength: float = 1.0,
                 voice_seed: int = 7) -> None:
        normalized = "chatterbox" if backend == "mlx" else backend
        self.capabilities = CAPABILITIES[normalized]
        if self.capabilities.experimental:
            warnings.warn(
                f"Experimental backend '{normalized}' ({self.capabilities.license}): "
                + " ".join(self.capabilities.warnings), stacklevel=2)
        self.backend = make_backend(normalized, voices=voices, model=model,
                                    ref_audio=ref_audio, ref_audios=ref_audios,
                                    ref_texts=ref_texts)
        self.prosody_strength = prosody_strength
        self.voice_seed = voice_seed
        self.post_process_pitch = normalized in {
            "kokoro", "cloudflare-aura2", "cloudflare-melotts"}
        self.post_process_speed = normalized in {
            "cloudflare-aura2", "cloudflare-melotts"}
        self.post_process_gain = normalized in {
            "cloudflare-aura2", "cloudflare-melotts"}
        self._cache: dict[tuple, np.ndarray] = {}
        self.stats: list[dict[str, Any]] = []
        self._call_index = 0

    def say(self, text: str, lang: str, prosody: Prosody = Prosody(),
            emotion: Emotion = NEUTRAL,
            target_seconds: float | None = None) -> np.ndarray:
        key = (text, lang, prosody, emotion, target_seconds)
        if key in self._cache:
            return self._cache[key]
        seed = self.voice_seed + self._call_index
        self._call_index += 1
        result = self.backend.synth(text, lang, prosody, emotion,
                                    target_seconds=target_seconds, seed=seed)
        audio = _trim(result.audio)
        rate = result.sample_rate
        post_process: dict[str, float] = {}
        if self.post_process_speed and abs(prosody.speed - 1.0) > 0.001:
            audio = librosa.effects.time_stretch(audio, rate=prosody.speed)
            post_process["speed"] = prosody.speed
        if self.post_process_pitch and abs(prosody.semitones) > 0.01:
            audio = _pitch_shift(audio, rate, prosody.semitones)
            post_process["semitones"] = prosody.semitones
        if rate != SR:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=SR,
                                     res_type="soxr_hq")
        audio = audio * 10 ** (prosody.gain_db / 20)
        if self.post_process_gain and abs(prosody.gain_db) > 0.01:
            post_process["gain_db"] = prosody.gain_db
        if post_process:
            result.controls["local_post_process"] = post_process
        before_fit = len(audio) / SR
        fitted = False
        if target_seconds is not None and before_fit > target_seconds:
            fitted_audio = fit(audio, target_seconds)
            fitted = len(fitted_audio) != len(audio)
            audio = fitted_audio
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak * 0.9
        audio = audio.astype(np.float32)
        metadata = result.metadata()
        metadata.update({
            "text": text, "lang": lang, "emotion": emotion.name,
            "emotion_vector": emotion.vector(), "seed": seed,
            "target_seconds": target_seconds, "duration_before_fit": before_fit,
            "duration_seconds": len(audio) / SR,
            "post_fit_applied": fitted,
        })
        self.stats.append(metadata)
        self._cache[key] = audio
        return audio

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if close:
            close()


def _pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    try:
        from pedalboard import PitchShift
        return PitchShift(semitones=semitones)(audio, sr)
    except ImportError:
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)


def fit(audio: np.ndarray, max_seconds: float, sr: int = SR) -> np.ndarray:
    limit = int(max_seconds * sr)
    if len(audio) <= limit or limit <= 0:
        return audio
    rate = min(len(audio) / limit, 1.35)
    try:
        from pedalboard import time_stretch
        stretched = time_stretch(audio, sr, stretch_factor=rate).reshape(-1)
    except (ImportError, AttributeError):
        stretched = librosa.effects.time_stretch(audio, rate=rate)
    return stretched[:limit] if len(stretched) > limit else stretched
