"""Cut one spoken word into its syllables, knowing how many there are.

No aligner and no model: a word's syllables are its vowel nuclei, and a nucleus is a peak in the
loudness envelope. Given the count from the syllabifier, take the `n` strongest peaks at least
90 ms apart and cut at the quietest point between each neighbouring pair. That is crude, and it is
measured by ear on the page rather than trusted. It fails where two vowels merge (Spanish
synalepha, "el a-tas-co" is heard as "e-la-tas-co") or where a nucleus is unstressed and weak.
"""

from __future__ import annotations

import numpy as np

FRAME_SECONDS = 0.01
MIN_GAP_SECONDS = 0.09


def envelope(audio: np.ndarray, sr: int) -> np.ndarray:
    frame = max(int(sr * FRAME_SECONDS), 1)
    count = len(audio) // frame
    rms = np.sqrt(np.mean(audio[: count * frame].reshape(count, frame) ** 2, axis=1) + 1e-12)
    kernel = np.hanning(7)
    return np.convolve(rms, kernel / kernel.sum(), mode="same")


def boundaries(audio: np.ndarray, sr: int, count: int) -> list[int]:
    """Sample positions of the `count - 1` cuts, in order. Empty when the word will not split."""
    if count < 2:
        return []
    env = envelope(audio, sr)
    frame = max(int(sr * FRAME_SECONDS), 1)
    maxima = [i for i in range(1, len(env) - 1) if env[i] >= env[i - 1] and env[i] > env[i + 1]]
    maxima.sort(key=lambda i: -env[i])
    peaks: list[int] = []
    gap = int(MIN_GAP_SECONDS / FRAME_SECONDS)
    for index in maxima:
        if all(abs(index - chosen) >= gap for chosen in peaks):
            peaks.append(index)
        if len(peaks) == count:
            break
    if len(peaks) < count:
        return []
    peaks.sort()
    return [int((a + int(np.argmin(env[a:b]))) * frame) for a, b in zip(peaks, peaks[1:])]


def split(audio: np.ndarray, sr: int, count: int) -> list[np.ndarray]:
    """The syllables, each with a 6 ms fade at both ends so a cut does not click."""
    cuts = boundaries(audio, sr, count)
    if not cuts:
        return []
    pieces = np.split(audio, cuts)
    fade = int(0.006 * sr)
    out = []
    for piece in pieces:
        piece = piece.astype(np.float32).copy()
        if len(piece) > 2 * fade:
            piece[:fade] *= np.linspace(0, 1, fade)
            piece[-fade:] *= np.linspace(1, 0, fade)
        out.append(piece)
    return out
