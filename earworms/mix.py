"""Balance speech against the bed.

Fixed gains do not survive a change of music, so the balance is set two ways
that both adapt: loudness normalisation to LUFS targets, and sidechain ducking
that pulls the bed down only while someone is speaking.
"""

from __future__ import annotations

import numpy as np
import pyloudnorm as pyln

from .music import SR

CONTROL_SR = 1000  # the ducking envelope is computed at a low rate; it is slow anyway


def _as_stereo(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 2 else np.stack([x, x], axis=1)


def loudness_normalize(audio: np.ndarray, target_lufs: float, sr: int = SR) -> np.ndarray:
    meter = pyln.Meter(sr)
    measured = meter.integrated_loudness(audio)
    if not np.isfinite(measured):
        return audio
    return audio * 10 ** ((target_lufs - measured) / 20)


def _loudness_gain(audio: np.ndarray, target_lufs: float, sr: int = SR) -> float:
    measured = pyln.Meter(sr).integrated_loudness(audio)
    return 1.0 if not np.isfinite(measured) else 10 ** ((target_lufs - measured) / 20)


def _follower(env: np.ndarray, attack_ms: float, release_ms: float) -> np.ndarray:
    """Classic attack/release envelope follower, run at CONTROL_SR."""
    a_att = np.exp(-1.0 / (CONTROL_SR * attack_ms / 1000.0))
    a_rel = np.exp(-1.0 / (CONTROL_SR * release_ms / 1000.0))
    out = np.empty_like(env)
    y = 0.0
    for i, x in enumerate(env):
        coeff = a_att if x > y else a_rel
        y = coeff * y + (1 - coeff) * x
        out[i] = y
    return out


def duck_envelope(speech: np.ndarray, depth_db: float = 5.0,
                  attack_ms: float = 100.0, release_ms: float = 400.0,
                  sr: int = SR) -> np.ndarray:
    """Gain curve for the music: unity when quiet, -depth_db under speech."""
    step = sr // CONTROL_SR
    # Peak-per-block, so a brief consonant still triggers the duck.
    n_blocks = len(speech) // step
    blocks = np.abs(speech[: n_blocks * step]).reshape(n_blocks, step).max(axis=1)

    peak = np.percentile(blocks, 99.5)
    presence = np.clip(blocks / peak * 3.0, 0, 1) if peak > 0 else blocks * 0
    smooth = _follower(presence, attack_ms, release_ms)

    gain = 10 ** (-depth_db * smooth / 20)
    full = np.repeat(gain, step)
    if len(full) < len(speech):
        full = np.concatenate([full, np.full(len(speech) - len(full), full[-1])])
    return full[: len(speech)]


def mix(bed: np.ndarray, speech: np.ndarray, *, speech_lufs: float = -16.0,
        music_lufs: float = -26.0, duck_db: float = 5.0,
        output_lufs: float = -16.0, sr: int = SR) -> np.ndarray:
    """Combine bed and speech into the final stereo track."""
    n = max(len(bed), len(speech))
    bed = np.pad(_as_stereo(bed), ((0, n - len(bed)), (0, 0)))
    speech_mono = np.pad(speech, (0, n - len(speech)))

    bed = loudness_normalize(bed, music_lufs, sr)
    speech_st = loudness_normalize(_as_stereo(speech_mono), speech_lufs, sr)

    gain = duck_envelope(speech_mono, duck_db, sr=sr)[:, None]
    out = bed * gain + speech_st
    out = loudness_normalize(out, output_lufs, sr)
    return limit(out, sr=sr).astype(np.float32)


def mix_stems(stems: dict[str, np.ndarray], speech: np.ndarray,
              duck_depths: dict[str, float], *, speech_lufs: float = -16.0,
              music_lufs: float = -26.0, output_lufs: float = -16.0,
              sr: int = SR) -> np.ndarray:
    """Mix named music stems with a different speech-duck depth per layer."""
    if not stems:
        raise ValueError("At least one music stem is required.")
    n = max([len(speech), *(len(stem) for stem in stems.values())])
    padded = {
        name: np.pad(_as_stereo(stem), ((0, n - len(stem)), (0, 0)))
        for name, stem in stems.items()
    }
    speech_mono = np.pad(speech, (0, n - len(speech)))

    unducked = sum(padded.values(), np.zeros((n, 2), dtype=np.float32))
    music_gain = _loudness_gain(unducked, music_lufs, sr)
    speech_st = loudness_normalize(_as_stereo(speech_mono), speech_lufs, sr)

    bed = np.zeros((n, 2), dtype=np.float64)
    for name, stem in padded.items():
        depth = duck_depths.get(name, 5.0)
        gain = duck_envelope(speech_mono, depth, sr=sr)[:, None]
        bed += stem * music_gain * gain

    out = loudness_normalize(bed + speech_st, output_lufs, sr)
    return limit(out, sr=sr).astype(np.float32)


def limit(audio: np.ndarray, ceiling: float = 0.97, block_ms: float = 5.0,
          release_ms: float = 120.0, sr: int = SR) -> np.ndarray:
    """Block-wise peak limiter.

    Rescaling the whole track by its single loudest sample throws away several dB
    of level for the sake of one transient, so gain is reduced only around the
    peaks and released smoothly.
    """
    step = max(int(sr * block_ms / 1000), 1)
    n_blocks = int(np.ceil(len(audio) / step))
    padded = np.pad(audio, ((0, n_blocks * step - len(audio)), (0, 0)))
    peaks = np.abs(padded.reshape(n_blocks, step, -1)).max(axis=(1, 2))

    need = np.minimum(1.0, ceiling / np.maximum(peaks, 1e-9))
    # Hold the reduction across neighbouring blocks, then let it recover.
    release_blocks = max(int(release_ms / block_ms), 1)
    smoothed = need.copy()
    for i in range(1, n_blocks):
        recovered = smoothed[i - 1] + (1.0 - smoothed[i - 1]) / release_blocks
        smoothed[i] = min(need[i], recovered)
    for i in range(n_blocks - 2, -1, -1):  # look ahead so gain drops before the peak
        smoothed[i] = min(smoothed[i], smoothed[i + 1] + (1.0 - smoothed[i + 1]) / 4)

    # Interpolating between block gains can land above what a block itself needs,
    # so widen each reduction over its neighbours first.
    widened = np.minimum.reduce([smoothed,
                                 np.roll(smoothed, 1), np.roll(smoothed, -1)])
    widened[0] = min(smoothed[0], smoothed[1] if n_blocks > 1 else 1.0)
    widened[-1] = min(smoothed[-1], smoothed[-2] if n_blocks > 1 else 1.0)

    curve = np.interp(np.arange(len(padded)), np.arange(n_blocks) * step + step / 2,
                      widened)
    return np.clip(padded * curve[:, None], -ceiling, ceiling)[: len(audio)]
