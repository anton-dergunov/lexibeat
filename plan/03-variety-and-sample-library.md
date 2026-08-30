# Step 3: Variety and Sample-Library Refinement

## Outcome

Increase perceptible variety while protecting the quality demonstrated by the
final 14 clips. The current library is already large enough to succeed: roughly
31.2 GB is available externally and 1.81 GB is promoted locally. This step is
therefore driven by measured musical gaps, not by a storage target.

Every improvement must be accessible through the Step 1 API and visible in the
Step 2 explorer. New assets alone do not count as added variety until they are
cataloged, mapped to coherent instruments or percussion roles, included in the
generation grammar, and exercised by tests and listening candidates.

## Workstream 1: Natural sample variation

### Deterministic round robins

Extend sampled-instrument references to resolve round-robin groups and variation
indices. Repeated notes should cycle or deterministically choose among valid
takes instead of repeatedly using the first sample.

- Serialize the resolved round-robin strategy or choices in `BedSpec`.
- Preserve deterministic PCM for a fixed specification.
- Never mix articulations, dynamic layers, or microphones merely because files
  share a directory.
- Fall back only when the instrument mapping explicitly permits it.

This is the highest-value near-term change for reducing machine-like repetition
without changing successful composition rules.

### Velocity and articulation layers

- Exercise existing velocity layers more fully while keeping speech-time
  dynamics restrained.
- Add coherent natural-bass, piano, guitar, mallet, plucked-string, and selected
  orchestral articulations where the catalog mappings are reliable.
- Keep attacks that compete with consonants out of speech bars or duck them more
  strongly.
- Expose articulation and layer choices in the Lab interface.

## Workstream 2: More musical grammars

Add diversity inside the established metrical frame:

- Call-and-response motifs with a stable downbeat and sparse answer.
- Target-contour motifs such as rising, arch, falling, and return-home.
- Restrained chord-tone accents and occasional stepwise connectors.
- Wider but instrument-appropriate piano registers and left/right-hand spacing.
- Natural-bass variants: sustained, root-fifth, chord-tone pulse, restrained
  syncopation, and carefully bounded passing tones.
- Percussion patterns with straight, Euclidean, clave-like, sparse, and gentle
  syncopated structures, while swing remains subtle in production profiles.

Resolved motif notes, onsets, velocities, and articulations remain serialized in
`BedSpec`; no renderer-time compositional randomness is allowed.

Add grammar-distance features so the selection pipeline recognizes two clips
with similar audio spectra but genuinely different melodic or rhythmic ideas.

## Workstream 3: Timbre and palette coverage

Add reusable palette dimensions rather than continually creating narrowly named
styles:

- `airy`
- `earthy`
- `wooden`
- `warm`
- `shimmering`
- `plucked`
- `soft-electronic`

A named family defines musical behavior; a palette biases instrument and texture
selection. For example, `sunlit + wooden` and `sunlit + airy` should share the
same positive rhythmic identity but sound materially different.

Improve the catalog report with clusters based on existing audio descriptors and
instrument metadata. Use clusters to:

- Detect overrepresented near-duplicate timbres.
- Prefer underused coherent banks during candidate generation.
- Measure palette coverage in a batch.
- Avoid selecting several piano, marimba, or bell variations that listeners
  perceive as the same idea.

This need not begin with a learned embedding model. Standardized spectral,
transient, pitch, width, duration, and source/category features can establish a
transparent baseline. Introduce embeddings only if listening tests show a clear
benefit.

## Workstream 4: Use more of the downloaded library

First produce a gap and utilization report for the safe catalog:

- Assets and coherent banks by collection, family, articulation, register,
  velocity layer, duration, and timbre cluster.
- Assets used by production candidate pools and selected results.
- Assets rejected, with reasons such as license, mapping quality, excessive
  transient, noise, silence, or missing pitch metadata.
- Locally promoted assets that are no longer referenced by the production core.

Then prioritize currently downloaded but underused collections that fill a
specific need, especially:

- Natural piano and keyboard registers.
- Classical guitar and gentle plucked instruments.
- Natural bass articulations.
- Mbira, psaltery, subdued mallets, ocarina, harmonica, pizzicato strings, and
  soft winds.
- Organic, found-object, and world percussion with controlled transients.
- Low-level textures that remain unobtrusive under speech.

Additional downloads should be proposed only after the report identifies a
missing timbre, register, articulation, or rhythmic role. Continue to exclude
aggressive club loops, unclear licenses, unsuitable long recordings, duplicate
format conversions, and sources that cannot be redistributed under the chosen
production-bundle policy.

## Production asset bundle

Treat the promoted sample set as a versioned companion artifact rather than
ordinary Git source. The repository should contain:

- A lock manifest of logical IDs, source revisions, SHA-256 checksums, sizes,
  licenses, attribution, and expected relative paths.
- A script/command to verify an installed bundle.
- An explicit command to install or promote the bundle; normal generation never
  invokes it.
- A complete `NOTICE.md` and machine-readable license report.

Publish the binary bundle through Git LFS. A conventional Git object history is
not a good home for approximately 1.8 GB of binary samples. Preserve one logical
bundle version and checksums even if host limits later require splitting it.

Prefer CC0 assets for the redistributable production core. Attributed compatible
assets require preserved attribution. Collections that restrict sample
repackaging are not supported or cataloged.

## Novelty policy

Support three distinct selection contexts:

- **Single production result:** generate a small pool and choose randomly from
  the accepted top tier.
- **Generate another:** optionally avoid the last few fingerprints supplied by
  the caller.
- **Bake-off or option set:** maximize pairwise audio, motif, grammar, family,
  instrument, and palette distance with explicit coverage constraints.

Do not condition ordinary generation on hidden global history. All information
that influences selection must come from the request, profile, catalog version,
or optional fingerprints and must be recorded in the result.

## Evaluation loop

For each meaningful grammar, palette, or sample-mapping expansion:

1. Generate a larger candidate pool from fixed evaluation seeds.
2. Apply automated hard validation and quality scoring.
3. Select a 12-16 clip listening set with coverage and distance constraints.
4. Reuse identical speech across every candidate.
5. Record ratings and comments in the existing CSV format.
6. Convert recurring feedback into profile or mapping rules, not per-seed edits.
7. Retain accepted examples as named regression fixtures.

Track at least:

- Family and palette coverage.
- Instrument-bank and articulation coverage.
- Percussion energy and transient outliers.
- Downbeat clarity and swing amount.
- Motif/grammar distance.
- Speech masking and music loudness.
- Rejection reasons and selection frequency per source.

## Verification

Add tests for:

- Deterministic round-robin and velocity-layer resolution.
- Backward-compatible sample references.
- Meter-safe new motif, bass, and percussion grammars.
- Instrument-range and articulation validity.
- Palette bias without hard-coded fixed tracks.
- Timbre clustering and coverage selection.
- Catalog utilization and rejection reports.
- Bundle lock verification, license manifests, and checksum failure.
- Operation from the promoted bundle with the external SSD disconnected.
- Absence of implicit downloads.

Run the complete unit suite plus synth-only, every newly enabled major sampled
family, and mixed-speech integration renders. Final listening candidates must
remain finite 44.1 kHz stereo, within the established duration and peak limits,
and maintain exact speech downbeats and bar boundaries.

## Completion criteria

- Repeated sampled notes sound more natural without losing reproducibility.
- Production batches show measurably broader motif, articulation, and timbre
  coverage while maintaining the current acceptance quality.
- More of the existing catalog is used for defensible musical reasons.
- The core production assets can be installed and verified independently of the
  original external repositories.
- The API and explorer can select, explain, reproduce, and compare every new
  variation.
