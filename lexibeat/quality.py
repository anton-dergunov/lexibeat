"""Audio validation, scoring, and novelty fingerprints for generated beds."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .api import BedFingerprint, QualityReport
from .bedspec import BedSpec
from .music import SR
from .profiles import GenerationProfile


@dataclass
class Candidate:
    family: str
    seed: int
    spec: BedSpec
    features: np.ndarray
    preview_seconds: float
    sample_collections: tuple[str, ...]
    fingerprint: BedFingerprint | None = None
    quality: QualityReport | None = None


def audio_features(audio: np.ndarray, spec: BedSpec) -> np.ndarray:
    """Return transparent, approximately normalized descriptors."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    mono = np.asarray(mono, dtype=np.float64)
    if not len(mono):
        return np.zeros(12, dtype=np.float64)
    peak = max(float(np.abs(mono).max()), 1e-9)
    rms = float(np.sqrt(np.mean(mono ** 2)))
    stride = max(len(mono) // (SR * 8), 1)
    decimated = mono[::stride]
    spectrum = np.abs(np.fft.rfft(decimated * np.hanning(len(decimated))))
    freqs = np.fft.rfftfreq(len(decimated), stride / SR)
    total = max(float(spectrum.sum()), 1e-12)
    centroid = float(np.sum(freqs * spectrum) / total)
    high = float(spectrum[freqs >= 2500].sum() / total)
    hop = max(SR // 100, 1)
    envelope = np.maximum.reduceat(np.abs(mono), np.arange(0, len(mono), hop))
    changes = np.maximum(np.diff(envelope, prepend=0), 0)
    onset_threshold = np.percentile(changes, 90) if len(changes) else 0.0
    onset_density = float(np.count_nonzero(changes > onset_threshold) /
                          max(len(mono) / SR, 1))
    phrase = spec.phrase
    if phrase is None:
        return np.array([
            spec.bpm / 100, spec.beats_per_bar / 5, spec.swing, rms / peak,
            math.log10(max(centroid, 1)) / 4, high, onset_density / 10,
            0, 0, 0, 0, 0,
        ], dtype=np.float64)
    downbeat_hits = 0
    if phrase.percussion:
        lane = phrase.percussion[0]
        downbeat_hits = sum(
            lane.pattern[bar * spec.steps_per_bar] == "x"
            for bar in range(phrase.loop_bars)
        )
    denominator = max(phrase.loop_bars * spec.steps_per_bar, 1)
    return np.array([
        spec.bpm / 100,
        spec.beats_per_bar / 5,
        spec.swing,
        rms / peak,
        math.log10(max(centroid, 1)) / 4,
        high,
        onset_density / 10,
        len(phrase.chords) / denominator,
        len(phrase.bass) / denominator,
        len(phrase.lead) / denominator,
        len(phrase.percussion) / 4,
        downbeat_hits / max(phrase.loop_bars, 1),
    ], dtype=np.float64)


def metrical_clarity(spec: BedSpec) -> float:
    """Score a stable low anchor and penalize bar-boundary collisions."""
    phrase = spec.phrase
    if not phrase or not phrase.percussion:
        return 1.0
    lane = phrase.percussion[0]
    steps = spec.steps_per_bar
    downbeats = sum(
        lane.pattern[bar * steps] == "x" for bar in range(phrase.loop_bars)
    )
    collisions = 0
    for boundary in range(steps, len(lane.pattern), steps):
        collisions += lane.pattern[boundary - 1:boundary + 2].count("x") > 1
    anchor = downbeats / max(phrase.loop_bars, 1)
    return max(0.0, anchor - collisions / max(phrase.loop_bars, 1) * 0.55)


def motif_features(spec: BedSpec, length: int = 12) -> np.ndarray:
    """Describe melodic intervals and onset gaps independently of key/timbre."""
    phrase = spec.phrase
    if not phrase or not phrase.lead:
        return np.zeros(length * 2, dtype=np.float64)
    events = phrase.lead[:length + 1]
    intervals = np.diff([event.midi_note for event in events]) / 12.0
    gaps = np.diff([event.step for event in events]) / max(spec.steps_per_bar, 1)
    result = np.zeros(length * 2, dtype=np.float64)
    result[:min(length, len(intervals))] = intervals[:length]
    result[length:length + min(length, len(gaps))] = gaps[:length]
    return result


def instrument_families(spec: BedSpec) -> tuple[str, ...]:
    phrase = spec.phrase
    values = {spec.pad.instrument, spec.lead.instrument}
    if phrase:
        for instrument in (phrase.pad_instrument, phrase.bass_instrument,
                           phrase.lead_instrument):
            if instrument:
                values.add(instrument.name.split(":", 1)[0])
        values.update(
            lane.sample.collection for lane in phrase.percussion if lane.sample
        )
    return tuple(sorted(values))


def make_fingerprint(audio: np.ndarray, spec: BedSpec) -> BedFingerprint:
    phrase = spec.phrase
    return BedFingerprint(
        family=phrase.family if phrase else "legacy",
        audio_features=tuple(float(value) for value in audio_features(audio, spec)),
        motif_features=tuple(float(value) for value in motif_features(spec)),
        instrument_families=instrument_families(spec),
    )


def preference_score(spec: BedSpec) -> float:
    """Transparent prior distilled from the listening feedback."""
    phrase = spec.phrase
    straight = 1.0 - min(spec.swing / 0.08, 1.0)
    downbeats = 1.0
    if phrase and phrase.percussion:
        downbeats = sum(
            phrase.percussion[0].pattern[bar * spec.steps_per_bar] == "x"
            for bar in range(phrase.loop_bars)
        ) / phrase.loop_bars
    restrained_drums = 1.0 - min(abs(spec.drums.level - 0.55) / 0.2, 1.0)
    positive = 1.0 if spec.scale in ("major", "lydian") else 0.55
    clarity = metrical_clarity(spec)
    return (
        0.28 * straight
        + 0.23 * downbeats
        + 0.20 * restrained_drums
        + 0.12 * positive
        + 0.17 * clarity
    )


def evaluate_preview(
    audio: np.ndarray,
    stems: dict[str, np.ndarray],
    spec: BedSpec,
    profile: GenerationProfile,
) -> tuple[QualityReport, BedFingerprint]:
    reasons: list[str] = []
    finite = bool(len(audio) and np.isfinite(audio).all())
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    preview_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) \
        if len(audio) else 0.0
    drums = stems.get("drums", np.zeros_like(audio))
    drum_rms = float(np.sqrt(np.mean(drums.astype(np.float64) ** 2))) \
        if len(drums) else 0.0
    drum_share = drum_rms / max(preview_rms, 1e-9)
    clarity = metrical_clarity(spec)
    score = preference_score(spec)
    if not finite:
        reasons.append("preview contains non-finite samples")
    if peak <= 0.0:
        reasons.append("preview is silent")
    if spec.swing > profile.max_swing + 1e-9:
        reasons.append(f"swing {spec.swing:.3f} exceeds {profile.max_swing:.3f}")
    if drum_share > profile.max_percussion_share:
        reasons.append(
            f"percussion dominance {drum_share:.3f} exceeds "
            f"{profile.max_percussion_share:.3f}"
        )
    if clarity < 0.72:
        reasons.append(f"metrical clarity {clarity:.3f} is too low")
    if score < profile.min_quality_score:
        reasons.append(
            f"quality score {score:.3f} is below {profile.min_quality_score:.3f}"
        )
    report = QualityReport(
        accepted=not reasons,
        score=score,
        rejection_reasons=tuple(reasons),
        measurements={
            "peak": peak,
            "rms": preview_rms,
            "percussion_rms": drum_rms,
            "percussion_share": drum_share,
            "metrical_clarity": clarity,
            "swing": spec.swing,
        },
    )
    return report, make_fingerprint(audio, spec)


def fingerprint_distance(left: BedFingerprint, right: BedFingerprint) -> float:
    audio = np.asarray(left.audio_features) - np.asarray(right.audio_features)
    motif = np.asarray(left.motif_features) - np.asarray(right.motif_features)
    left_instruments = set(left.instrument_families)
    right_instruments = set(right.instrument_families)
    union = left_instruments | right_instruments
    instrument_distance = 1.0 - len(left_instruments & right_instruments) / max(
        len(union), 1
    )
    family_distance = 0.0 if left.family == right.family else 1.0
    return (
        float(np.linalg.norm(audio))
        + 0.7 * float(np.linalg.norm(motif))
        + 0.5 * instrument_distance
        + 0.25 * family_distance
    )
