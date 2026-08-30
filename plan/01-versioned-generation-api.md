# Step 1: Versioned Generation API

## Outcome

Turn the successful bake-off pipeline into a stable Python library API. A caller
that knows nothing about music should be able to submit a small, high-level
request and receive a safe, varied, reproducible `BedSpec`. The existing
personal-vocabulary application is deliberately out of scope for this step;
this repository will only provide the interface it can consume later.

The current 14-clip result becomes the reference behavior for a named
`production-v1` profile. The profile captures the rules learned during the
listening iterations: clear downbeats, mostly straight rhythm, restrained
percussion, audible but non-dominant natural instruments, and music that leaves
space for speech.

## Public model

Keep two levels of configuration rather than exposing one partially resolved
dictionary:

1. `MusicRequest` expresses user intent with a few stable controls.
2. `BedSpec` remains the complete, immutable, replayable musical description.

Proposed public types:

```python
@dataclass(frozen=True)
class MusicRequest:
    family: str = "auto"
    energy: Literal["calm", "balanced", "bright"] = "balanced"
    rhythm: Literal["sparse", "steady", "groovy"] = "steady"
    palette: Literal["acoustic", "hybrid", "electronic"] = "hybrid"
    seed: int | None = None
    profile: str = "production-v1"


@dataclass(frozen=True)
class BedFingerprint:
    family: str
    audio_features: tuple[float, ...]
    motif_features: tuple[float, ...]
    instrument_families: tuple[str, ...]


@dataclass(frozen=True)
class QualityReport:
    accepted: bool
    score: float
    rejection_reasons: tuple[str, ...]
    measurements: dict[str, float]


@dataclass(frozen=True)
class MusicGenerationResult:
    request: MusicRequest
    bed_spec: BedSpec
    fingerprint: BedFingerprint
    quality: QualityReport
    sample_manifest: tuple[SampleUsage, ...]
    engine_version: str
    profile_version: str
```

The public entry points should be small and usable independently of the CLI:

```python
resolve_music(
    request: MusicRequest,
    *,
    avoid_fingerprints: Sequence[BedFingerprint] = (),
) -> MusicGenerationResult

render_music(result_or_spec, *, duration_seconds, sample_rate=44_100) -> np.ndarray

generate_music(
    request: MusicRequest,
    *,
    duration_seconds: float,
    avoid_fingerprints: Sequence[BedFingerprint] = (),
) -> tuple[np.ndarray, MusicGenerationResult]
```

`resolve_music` must not synthesize speech and must not know about vocabulary.
A future application may define a `LessonRequest` that combines words, voice,
and `MusicRequest`, but that belongs at the application boundary.

## Production resolution algorithm

Do not accept the first randomly generated bed. A production request should use
the successful bake-off logic on a smaller scale:

1. Validate the high-level request and resolve `family="auto"` through the
   profile's balanced family weights.
2. If `seed` is absent, create a new 64-bit seed and return it in the resolved
   result. If it is supplied, all subsequent choices must be deterministic.
3. Generate a small candidate pool, initially six candidates for a single
   result. Derive each candidate seed deterministically from the request seed.
4. Resolve every musical choice and logical sample reference into a `BedSpec`.
5. Render short analysis previews without speech or network access.
6. Reject candidates with missing samples, invalid audio, excessive density,
   weak metrical anchors, excessive percussion, or profile violations.
7. Measure rhythmic, spectral, density, motif, and instrument-palette features.
8. Rank by the `production-v1` quality score. Randomly select within the top
   accepted tier instead of always taking the single highest score.
9. When `avoid_fingerprints` are supplied, include distance from those results
   in selection. Recent-history diversity is therefore optional and caller
   controlled; the engine itself remains stateless.
10. Return the selected `BedSpec`, quality evidence, fingerprint, seed, and
    complete sample manifest.

The default profile should preserve these established constraints:

- Speech-sized bars and deterministic downbeats.
- Mostly straight rhythm with only subtle, bounded swing.
- A strong low-lane anchor on beat one.
- Restrained percussion and metallic accents.
- No aggressive club or hard-style kick samples.
- Music around the established `-26 LUFS` target, with speech mixing handled by
  the existing mixer.
- Two-bar fades, centered bass, finite stereo output, and peak at or below
  `0.97`.
- Safe source/license policy by default, with no implicit downloads.

## Code organization

Extract reusable behavior rather than importing functions from
`compare_beds.py`:

- `earworms/api.py`: public types, validation, and entry points.
- `earworms/profiles.py`: immutable, versioned safe ranges, family weights,
  quality thresholds, and instrument policies.
- `earworms/generator.py`: high-level request resolution and candidate-pool
  orchestration.
- `earworms/quality.py`: preview measurements, hard rejection, scoring,
  fingerprints, and distance calculations.
- `earworms/bedspec.py`: complete serializable musical choices and legacy JSON
  compatibility.
- `earworms/music.py`: rendering only.
- `compare_beds.py`: a thin batch consumer of the public API, retaining
  listening-guide and rating-file production.
- `generate.py`: a thin lesson/CLI consumer of the same API.

The batch selector should remain available for selecting several mutually
different outputs. Single-result generation uses a small quality pool; batch
generation additionally maximizes pairwise distance and coverage.

## Versioning and reproducibility

- Assign an explicit engine schema version to `BedSpec` without breaking older
  JSON. New fields require compatible defaults or a documented migration.
- Record both `engine_version` and `profile_version`. Changing profile ranges in
  the future creates `production-v2`; it must not silently change `v1`.
- A saved `BedSpec` always wins over current profile defaults and must reproduce
  its musical choices from logical sample IDs and checksums.
- Serialize enum values as stable strings and reject unknown values with useful
  messages.
- Keep seed derivation in one documented function and cover it with regression
  tests.

## Asset behavior

- Resolve samples through logical `SampleRef` values, never hard-coded physical
  paths.
- Prefer the locally promoted cache and fall back to the external library.
- If neither contains a required checksum, return the existing actionable
  attach-or-promote error. Never silently substitute a different instrument.
- Ordinary API calls remain offline and never acquire samples implicitly.
- The source package contains code and manifests, not the downloaded cache.
  A separately versioned production asset bundle may be published through
  GitHub Releases or equivalent artifact storage, with checksums and licenses.

## Verification

Add tests for:

- High-level request validation and stable serialization.
- Deterministic `MusicRequest` to `BedSpec` resolution for a fixed seed.
- Different unseeded requests recording different replayable seeds.
- Profile version pinning and legacy `BedSpec` JSON.
- Hard rejection and top-tier randomized selection.
- Optional recent-fingerprint avoidance.
- Batch family, motif, and instrument coverage.
- Identical logical sample resolution from local and external storage.
- An actionable failure when an unpromoted external sample is unavailable.
- No network access during resolve, render, or generate.
- Existing downbeat, bar-boundary, loudness, stereo, finite-value, and peak
  invariants.

Retain the current final 14 seeds as regression fixtures. They need not produce
byte-identical PCM after an intentional engine-version change, but `production-v1`
must preserve their resolved specifications and safety measurements.

## Completion criteria

- External callers can generate and replay a bed without importing a CLI
  module.
- `compare_beds.py` uses the same public pipeline as single generation.
- `MusicRequest()` reliably produces a safe result without musical knowledge.
- A generated result contains enough information to replay, audit, compare,
  package, and display it.
- All existing and new unit tests pass, followed by synth-only and sampled
  integration renders.

