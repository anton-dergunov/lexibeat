"""Speech synthesis with per-repetition prosody variation.

Kokoro exposes only a speed control, so the "each repeat sounds slightly
different" quality of the original is produced afterwards: a small pitch shift,
a small rate change and a gain tilt per repetition. At background-listening
attention this is hard to tell apart from deliberate re-emphasis.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import librosa
import numpy as np

from .music import SR

KOKORO_SR = 24000

LANGS = {"es": "e", "en": "a"}
DEFAULT_VOICES = {"es": "ef_dora", "en": "af_heart"}


@dataclass(frozen=True)
class Prosody:
    """One reading of a phrase."""

    speed: float = 1.0  # passed to the TTS engine; changes rate, not pitch
    semitones: float = 0.0  # pitch shift applied afterwards
    gain_db: float = 0.0

    @staticmethod
    def for_repeat(index: int) -> "Prosody":
        """Neutral, then softer and lower, then brighter and more insistent."""
        table = [
            Prosody(speed=1.00, semitones=0.0, gain_db=0.0),
            Prosody(speed=0.94, semitones=-0.8, gain_db=-1.0),
            Prosody(speed=1.05, semitones=+0.9, gain_db=+0.8),
            Prosody(speed=0.97, semitones=+0.3, gain_db=-0.4),
        ]
        return table[index % len(table)]


def _trim(audio: np.ndarray) -> np.ndarray:
    """Strip leading/trailing silence so the onset can be placed on a downbeat."""
    trimmed, _ = librosa.effects.trim(audio, top_db=32)
    return trimmed if trimmed.size else audio


class Speaker:
    def __init__(self, voices: dict[str, str] | None = None) -> None:
        from kokoro import KPipeline

        self._voices = {**DEFAULT_VOICES, **(voices or {})}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._pipes = {lang: KPipeline(lang_code=code, repo_id="hexgrad/Kokoro-82M")
                           for lang, code in LANGS.items()}
        self._cache: dict[tuple, np.ndarray] = {}

    def say(self, text: str, lang: str, prosody: Prosody = Prosody()) -> np.ndarray:
        """Render one utterance at the project sample rate, silence trimmed."""
        key = (text, lang, prosody)
        if key in self._cache:
            return self._cache[key]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunks = list(self._pipes[lang](text, voice=self._voices[lang],
                                            speed=prosody.speed))
        audio = np.concatenate([c.audio.numpy() for c in chunks]).astype(np.float32)
        audio = _trim(audio)

        if prosody.semitones:
            audio = librosa.effects.pitch_shift(audio, sr=KOKORO_SR,
                                                n_steps=prosody.semitones)
        audio = librosa.resample(audio, orig_sr=KOKORO_SR, target_sr=SR)
        audio *= 10 ** (prosody.gain_db / 20)

        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.9
        self._cache[key] = audio
        return audio


def fit(audio: np.ndarray, max_seconds: float, sr: int = SR) -> np.ndarray:
    """Compress an utterance into its slot if it overruns, preserving pitch.

    Speeding speech up slightly is far less noticeable than letting it drift off
    the beat, so overruns are stretched rather than truncated.
    """
    limit = int(max_seconds * sr)
    if len(audio) <= limit or limit <= 0:
        return audio
    rate = len(audio) / limit
    if rate > 1.35:  # beyond this it starts to sound comical; let it overhang
        rate = 1.35
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    return stretched[:limit] if len(stretched) > limit else stretched
