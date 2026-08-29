"""Speech synthesis with capability-aware expressive backends.

Kokoro and Chatterbox remain the stable paths. The other backends are local
research integrations whose model-side controls are recorded separately from
post-processing.
"""

from __future__ import annotations

import hashlib
import json
import os
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
    "indextts25": "vanch007/mlx-indextts2-2.5-8bit",
    "voxcpm2": "mlx-community/VoxCPM2-4bit",
    "qwen3": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit",
    "tada": "HumeAI/mlx-tada-3b",
    "fish-s2": "mlx-community/fish-audio-s2-pro-8bit",
}
TADA_TOKENIZER_MODEL = "gafiatulin/tada-3b-ml-mlx"
QWEN_VOICES = {"es": "Serena", "en": "Ryan"}

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
    root = Path(os.environ.get("EARWORMS_CACHE", Path.home() / ".cache" / "earworms"))
    path = root / "models"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Read-only CI/sandbox homes still need a harmless location for mocked
        # backend tests. Real runs should set EARWORMS_CACHE explicitly.
        path = Path(tempfile.gettempdir()) / "earworms-models"
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
        self._tmp = tempfile.TemporaryDirectory(prefix="earworms-indextts-")
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
    directory = Path(os.environ.get("EARWORMS_CACHE",
                                    Path.home() / ".cache" / "earworms")) / "refs"
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
        "indextts25": IndexTTS25Backend,
        "voxcpm2": VoxCPM2Backend,
        "qwen3": Qwen3Backend,
        "tada": TadaBackend,
        "fish-s2": FishS2Backend,
    }
    if normalized not in classes:
        raise ValueError(f"Unknown voice backend '{name}'.")
    selected_model = model or DEFAULT_MODELS[normalized]
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
        self.post_process = normalized == "kokoro"
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
        if self.post_process and abs(prosody.semitones) > 0.01:
            audio = _pitch_shift(audio, rate, prosody.semitones)
        if rate != SR:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=SR,
                                     res_type="soxr_hq")
        audio = audio * 10 ** (prosody.gain_db / 20)
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
