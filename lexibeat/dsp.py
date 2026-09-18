"""The four signal operations everything else needs, and the reason librosa is not a dependency.

`librosa.resample(..., res_type="soxr_hq")` called soxr anyway, and `librosa.effects.trim` is a
framed RMS threshold. Between them they were the only reason the music path imported librosa — and
with it numba and llvmlite, most of a 534 MB environment — to render a bed that contains no speech
at all. Calling soxr directly and framing in numpy costs nothing and leaves librosa an optional
dependency of the local voice backends, where a mel spectrogram is genuinely wanted.

Time-stretch and pitch-shift go through pedalboard, with librosa as a lazy fallback for a machine
that has one and not the other.
"""

from __future__ import annotations

import numpy as np
import soxr


def resample(audio: np.ndarray, orig_sr: float, target_sr: float) -> np.ndarray:
    """High-quality rate conversion. ``target_sr`` may be fractional: pitch is shifted by lying
    about the rate, and the ratio is what matters."""
    if abs(float(orig_sr) - float(target_sr)) < 1e-9:
        return np.asarray(audio, dtype=np.float32)
    converted = soxr.resample(np.asarray(audio, dtype=np.float32),
                              float(orig_sr), float(target_sr), quality="HQ")
    return np.asarray(converted, dtype=np.float32)


def trim(audio: np.ndarray, top_db: float = 32.0,
         frame: int = 1024, hop: int = 256) -> np.ndarray:
    """Cut leading and trailing silence. Same framing and ``top_db`` convention as librosa's."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size < frame:
        return audio
    count = 1 + (audio.size - frame) // hop
    windows = np.lib.stride_tricks.as_strided(
        audio, shape=(count, frame),
        strides=(audio.strides[0] * hop, audio.strides[0]))
    power = np.mean(np.square(windows, dtype=np.float64), axis=1)
    reference = float(power.max())
    if reference <= 0:
        return audio
    loud = np.flatnonzero(power >= reference * 10 ** (-top_db / 10.0))
    if not loud.size:
        return audio
    start = int(loud[0]) * hop
    end = min(audio.size, int(loud[-1]) * hop + frame)
    trimmed = audio[start:end]
    return trimmed if trimmed.size else audio


def time_stretch(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Change duration without changing pitch. ``rate > 1`` makes it shorter."""
    try:
        from pedalboard import time_stretch as _stretch
        return np.asarray(_stretch(audio, sr, stretch_factor=rate),
                          dtype=np.float32).reshape(-1)
    except (ImportError, AttributeError):
        import librosa
        return np.asarray(librosa.effects.time_stretch(audio, rate=rate), dtype=np.float32)


def pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    try:
        from pedalboard import PitchShift
        return np.asarray(PitchShift(semitones=semitones)(audio, sr), dtype=np.float32)
    except ImportError:
        import librosa
        return np.asarray(librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones),
                          dtype=np.float32)


def fit(audio: np.ndarray, max_seconds: float, sr: int) -> np.ndarray:
    """Squeeze an over-long utterance into its bar, capped so it never sounds hurried."""
    limit = int(max_seconds * sr)
    if len(audio) <= limit or limit <= 0:
        return audio
    rate = min(len(audio) / limit, 1.35)
    stretched = time_stretch(audio, sr, rate)
    return stretched[:limit] if len(stretched) > limit else stretched
