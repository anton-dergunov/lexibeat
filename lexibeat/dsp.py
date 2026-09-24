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


def _unavailable(operation: str, first: BaseException, second: BaseException) -> RuntimeError:
    """Say what happened to *both* backends, because one of them is not supposed to be installed.

    Only the import is guarded above, and that is the whole of this fix: the `try` used to wrap the
    pedalboard *call* as well, so any failure inside pedalboard fell through to `import librosa` and
    surfaced as `ModuleNotFoundError: No module named 'librosa'` — naming a dependency that was
    removed on purpose and saying nothing about the one that actually broke.
    """
    return RuntimeError(
        f"{operation} needs pedalboard or librosa, and neither answered. "
        f"pedalboard: {first!r}. librosa (an optional fallback, not a dependency of the "
        f"music path): {second!r}."
    )


# Speech is stretched by Rubber Band's offline engine, which keeps a voice's formants. Transients
# are "mixed" rather than the default "crisp": crisp keeps every attack sharp, which on a voice
# squeezed into its bar is what made consonants ring and the vowels between them sound metallic.
SPEECH_TRANSIENTS = "mixed"

# How hard an over-long take is squeezed before the rest is faded instead. It was 1.35 with a hard
# cut after it, and every take long enough to need it came out both hurried and chopped; with a
# tail to fade into, the squeeze can be gentle and the overflow simply heard, quieter.
MAX_SQUEEZE = 1.2
# Without a slot to fade into there is nowhere for the overflow to go, so it squeezes harder.
MAX_SQUEEZE_WITHOUT_SLOT = 1.35
# The tail of a take that runs past its slot — the next downbeat. It stays audible under the next
# voice rather than competing with it: down to TAIL_DUCK_DB over TAIL_DUCK_SECONDS, then a lazy
# fall to TAIL_FLOOR_DB by TAIL_MAX_SECONDS, where it ends with a short cosine so nothing clicks.
TAIL_DUCK_DB = -4.0
TAIL_DUCK_SECONDS = 0.25
TAIL_FLOOR_DB = -10.0
TAIL_MAX_SECONDS = 1.2
TAIL_END_FADE_SECONDS = 0.08


def time_stretch(audio: np.ndarray, sr: int, rate: float, semitones: float = 0.0) -> np.ndarray:
    """Change duration, and optionally pitch, in one pass. ``rate > 1`` makes it shorter.

    Pitch and speed together, rather than a stretch followed by a separate pitch shift: pedalboard's
    `PitchShift` is Rubber Band's real-time engine without formant preservation, and running it
    after a stretch put a voice through two phase vocoders — the textbook metallic voice.
    """
    try:
        from pedalboard import time_stretch as _stretch
    except (ImportError, AttributeError) as missing:
        try:
            import librosa
        except ImportError as absent:
            raise _unavailable("Time-stretching", missing, absent) from missing
        audio = np.asarray(librosa.effects.time_stretch(audio, rate=rate), dtype=np.float32)
        if abs(semitones) > 1e-6:
            audio = np.asarray(librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones),
                               dtype=np.float32)
        return audio
    return np.asarray(_stretch(audio, sr, stretch_factor=rate, pitch_shift_in_semitones=semitones,
                               transient_mode=SPEECH_TRANSIENTS),
                      dtype=np.float32).reshape(-1)


def fade_tail(audio: np.ndarray, slot: int, sr: int) -> np.ndarray:
    """Keep everything up to ``slot`` samples at full level, and let the rest fade under what follows."""
    if len(audio) <= slot:
        return audio
    end = min(len(audio), slot + int(TAIL_MAX_SECONDS * sr))
    t = np.arange(end - slot) / sr
    gain_db = np.empty_like(t)
    ducking = t < TAIL_DUCK_SECONDS
    gain_db[ducking] = TAIL_DUCK_DB * (1 - np.cos(np.pi * t[ducking] / TAIL_DUCK_SECONDS)) / 2
    gain_db[~ducking] = TAIL_DUCK_DB + (TAIL_FLOOR_DB - TAIL_DUCK_DB) * (
        (t[~ducking] - TAIL_DUCK_SECONDS) / (TAIL_MAX_SECONDS - TAIL_DUCK_SECONDS))
    gain = 10 ** (gain_db / 20)
    closing = min(int(TAIL_END_FADE_SECONDS * sr), len(gain))
    if closing:
        gain[-closing:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, closing)))
    out = np.array(audio[:end], dtype=np.float32)
    out[slot:] *= gain.astype(np.float32)
    return out


def fit(audio: np.ndarray, max_seconds: float, sr: int, *,
        slot_seconds: float | None = None) -> np.ndarray:
    """Squeeze an over-long utterance towards ``max_seconds``, gently, and fade what still overflows.

    ``slot_seconds`` is where the next utterance may begin — the next downbeat. Whatever runs past it
    fades under that utterance (`fade_tail`) instead of being cut off mid-word, which is what used to
    happen to "to pee oneself". Without a slot the old contract holds: squeeze harder, then cut.
    """
    limit = int(max_seconds * sr)
    if len(audio) <= limit or limit <= 0:
        return audio
    cap = MAX_SQUEEZE if slot_seconds is not None else MAX_SQUEEZE_WITHOUT_SLOT
    stretched = time_stretch(audio, sr, min(len(audio) / limit, cap))
    if slot_seconds is not None:
        return fade_tail(stretched, int(slot_seconds * sr), sr)
    return stretched[:limit] if len(stretched) > limit else stretched
