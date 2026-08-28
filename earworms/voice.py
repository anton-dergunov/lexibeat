"""Speech synthesis with per-repetition variation.

Two backends sit behind one interface:

* Kokoro — small, fast, cross-platform, and the default. It exposes only a
  speed control, so variation between repeats is produced afterwards.
* mlx-audio — runs Chatterbox and friends natively on Apple Silicon and takes
  real expressive parameters, so no post-processing is needed at all.

Post-processing is the weaker path on purpose: librosa's pitch shifter is a
teaching-grade phase vocoder that smears transients, which is what made the
later repeats of a word sound mechanical. Where a shift is still applied it now
goes through Rubber Band via pedalboard, and the range is much narrower.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, replace
from typing import Protocol

import librosa
import numpy as np

from .emotion import NEUTRAL, Emotion
from .music import SR

KOKORO_SR = 24000

LANGS = {"es": "e", "en": "a"}
DEFAULT_VOICES = {"es": "ef_dora", "en": "af_heart"}
CHATTERBOX_LANGS = {"es": "es", "en": "en"}
DEFAULT_MLX_MODEL = "mlx-community/chatterbox-multilingual-v3"

# Long enough for the model to characterise each speaker, with varied phrase
# lengths so the cloned delivery is less monotone than a single flat sentence.
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
class Prosody:
    """One reading of a phrase.

    Speed is native to the engine and therefore free of artifacts; pitch and
    gain are applied afterwards and are kept deliberately small.
    """

    speed: float = 1.0
    semitones: float = 0.0
    gain_db: float = 0.0
    exaggeration_bias: float = 0.0  # only the native-control backends use this

    # Neutral, then softer and a shade lower, then brighter and more insistent.
    # The old table reached ±0.9 semitones and sounded synthetic by the third
    # repeat; this tops out near where the old second-to-last variant sat.
    # The last column carries the same arc for backends that shape delivery
    # themselves, so those never need the audio touched afterwards.
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


def _trim(audio: np.ndarray) -> np.ndarray:
    """Strip leading/trailing silence so the onset can be placed on a downbeat."""
    trimmed, _ = librosa.effects.trim(audio, top_db=32)
    return trimmed if trimmed.size else audio


class Backend(Protocol):
    sample_rate: int

    def synth(self, text: str, lang: str, prosody: Prosody,
              emotion: Emotion) -> np.ndarray: ...


class KokoroBackend:
    """Kokoro-82M. Fast, runs anywhere, no native expressive control."""

    name = "kokoro"
    sample_rate = KOKORO_SR

    def __init__(self, voices: dict[str, str] | None = None) -> None:
        from kokoro import KPipeline

        self._voices = {**DEFAULT_VOICES, **(voices or {})}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._pipes = {lang: KPipeline(lang_code=code,
                                           repo_id="hexgrad/Kokoro-82M")
                           for lang, code in LANGS.items()}

    def synth(self, text: str, lang: str, prosody: Prosody,
              emotion: Emotion) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunks = list(self._pipes[lang](text, voice=self._voices[lang],
                                            speed=prosody.speed))
        return np.concatenate([c.audio.numpy() for c in chunks]).astype(np.float32)


class MlxAudioBackend:
    """Chatterbox (and siblings) via mlx-audio, on Apple Silicon.

    Expressive control is native here: exaggeration, guidance weight and
    temperature all shape the delivery, so nothing is done to the audio
    afterwards.
    """

    name = "mlx"

    def __init__(self, model: str = DEFAULT_MLX_MODEL,
                 ref_audio: str | None = None,
                 ref_audios: dict[str, str] | None = None) -> None:
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as exc:  # pragma: no cover - platform dependent
            raise RuntimeError(
                "mlx-audio is unavailable (Apple Silicon only). "
                "Use --backend kokoro.") from exc

        self.model_id = model
        # This checkpoint ships without built-in voice conditioning. Keep a
        # native reference per language so English never inherits a Spanish
        # speaker's accent (and vice versa).
        sources = dict(DEFAULT_REFERENCES)
        if ref_audio:
            warnings.warn(
                "A shared ref_audio applies one speaker to both languages and "
                "may reintroduce an accent mismatch; prefer ref_audios.",
                stacklevel=2,
            )
            sources = {"es": ref_audio, "en": ref_audio}
        if ref_audios:
            sources.update({k: v for k, v in ref_audios.items() if v})
        self.ref_audios = {
            lang: str(ensure_reference(source, lang))
            if source.startswith(("say:", "kokoro:")) else str(source)
            for lang, source in sources.items()
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = load_model(model)
        self.sample_rate = int(getattr(self._model, "sample_rate", 24000))

    def synth(self, text: str, lang: str, prosody: Prosody,
              emotion: Emotion) -> np.ndarray:
        kwargs = dict(
            text=text,
            lang_code=CHATTERBOX_LANGS.get(lang, lang),
            ref_audio=self.ref_audios[lang],
            exaggeration=float(np.clip(
                emotion.exaggeration + prosody.exaggeration_bias, 0.05, 1.0)),
            cfg_weight=emotion.cfg_weight,
            temperature=0.8,
            speed=prosody.speed,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pieces, rate = [], self.sample_rate
            for result in self._model.generate(**kwargs):
                pieces.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
                rate = int(getattr(result, "sample_rate", rate) or rate)
        self.sample_rate = rate
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def reference_path(name: str = "kokoro-ef_dora", lang: str | None = None) -> "Path":
    from pathlib import Path

    directory = Path(os.environ.get("EARWORMS_CACHE",
                                    Path.home() / ".cache" / "earworms")) / "refs"
    stem = name.replace(":", "-")
    if lang:
        stem += f"-{lang}"
    return directory / f"{stem}.wav"


def ensure_reference(source: str = "kokoro:ef_dora", lang: str = "es") -> "Path":
    """Get (or make) the voice-reference clip Chatterbox clones from.

    "kokoro:VOICE" synthesises one locally and works on any platform.
    "say:VOICE" uses the macOS speech synthesiser, which offers Latin American
    Spanish voices such as Paulina that Kokoro does not have.
    """
    import soundfile as sf

    if lang not in REFERENCE_TEXTS:
        raise ValueError(f"No reference script for language '{lang}'.")
    path = reference_path(source, lang)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)

    kind, _, voice = source.partition(":")
    if kind == "say":
        import subprocess
        import tempfile

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
        audio = backend.synth(REFERENCE_TEXTS[lang], lang, Prosody(), NEUTRAL)
        sf.write(path, audio, KOKORO_SR)
    return path


def make_backend(name: str, *, voices: dict[str, str] | None = None,
                 model: str = DEFAULT_MLX_MODEL,
                 ref_audio: str | None = None,
                 ref_audios: dict[str, str] | None = None) -> Backend:
    if name == "kokoro":
        return KokoroBackend(voices)
    if name in ("mlx", "chatterbox"):
        return MlxAudioBackend(model, ref_audio, ref_audios)
    raise ValueError(f"Unknown voice backend '{name}'. Try 'kokoro' or 'chatterbox'.")


class Speaker:
    """Renders utterances, caching repeats of identical requests."""

    def __init__(self, voices: dict[str, str] | None = None, *,
                 backend: str = "chatterbox", model: str = DEFAULT_MLX_MODEL,
                 ref_audio: str | None = None,
                 ref_audios: dict[str, str] | None = None,
                 prosody_strength: float = 1.0) -> None:
        self.backend = make_backend(backend, voices=voices, model=model,
                                    ref_audio=ref_audio, ref_audios=ref_audios)
        self.prosody_strength = prosody_strength
        # Chatterbox already varies delivery natively; touching it afterwards
        # would only add the artifacts we are trying to remove.
        self.post_process = self.backend.name == "kokoro"
        self._cache: dict[tuple, np.ndarray] = {}

    def say(self, text: str, lang: str, prosody: Prosody = Prosody(),
            emotion: Emotion = NEUTRAL) -> np.ndarray:
        key = (text, lang, prosody, emotion)
        if key in self._cache:
            return self._cache[key]

        audio = _trim(self.backend.synth(text, lang, prosody, emotion))
        rate = self.backend.sample_rate

        if self.post_process and abs(prosody.semitones) > 0.01:
            audio = _pitch_shift(audio, rate, prosody.semitones)
        if rate != SR:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=SR,
                                     res_type="soxr_hq")
        audio = audio * 10 ** (prosody.gain_db / 20)

        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.9
        audio = audio.astype(np.float32)
        self._cache[key] = audio
        return audio


def _pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Rubber Band via pedalboard, falling back to librosa if unavailable."""
    try:
        from pedalboard import PitchShift

        return PitchShift(semitones=semitones)(audio, sr)
    except ImportError:  # pragma: no cover
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)


def fit(audio: np.ndarray, max_seconds: float, sr: int = SR) -> np.ndarray:
    """Compress an utterance into its slot if it overruns, preserving pitch.

    Speeding speech up slightly is far less noticeable than letting it drift off
    the beat, so overruns are stretched rather than truncated.
    """
    limit = int(max_seconds * sr)
    if len(audio) <= limit or limit <= 0:
        return audio
    rate = min(len(audio) / limit, 1.35)  # beyond this it sounds comical
    try:
        from pedalboard import time_stretch

        stretched = time_stretch(audio, sr, stretch_factor=rate).reshape(-1)
    except (ImportError, AttributeError):  # pragma: no cover
        stretched = librosa.effects.time_stretch(audio, rate=rate)
    return stretched[:limit] if len(stretched) > limit else stretched
