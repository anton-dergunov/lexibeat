# Step 2: Advanced Web Explorer

## Outcome

Build a local browser interface for learning and experimentation on top of the
versioned API from Step 1. The explorer must not contain a second implementation
of music generation: every randomization, validation, preview, and full render
goes through the public API and produces an ordinary `BedSpec`.

The interface has two complementary modes:

- **Simple** offers the same safe high-level controls suitable for a future
  product.
- **Lab** exposes most resolved musical parameters for detailed exploration.

It is a local development tool, not the later personal-vocabulary application.
User accounts, deployment, vocabulary management, and production authentication
are out of scope.

## Simple mode

On opening the page, create a reasonable random `MusicRequest` and resolve it
through `production-v1`. Present only:

- Style: `Auto` plus the friendly production families.
- Energy: `Calm`, `Balanced`, or `Bright`.
- Rhythm: `Sparse`, `Steady`, or `Groovy`, defaulting to `Steady`.
- Palette: `Acoustic`, `Hybrid`, or `Electronic`, defaulting to `Hybrid`.
- `Generate another`, `Play`, `Stop`, and `Open in Lab` actions.

Do not expose raw volume or swing here. The established music/speech balance and
metrical clarity are part of the production profile rather than ordinary user
preferences.

`Generate another` should create a fresh seed and use the small candidate pool.
The browser may optionally pass fingerprints from its last three results for
stronger novelty, but clearing or reloading the page must not affect correctness.

## Lab mode

Display the resolved `BedSpec` as grouped, typed controls:

1. Identity: engine/profile versions, family, seed, duration, and meter.
2. Harmony: root, scale, progression, chord extensions, voicings, texture, and
   register.
3. Bass: instrument, pattern grammar, register, articulation, density, velocity,
   and stereo placement.
4. Lead/motif: instrument, contour, notes, onset grid, call-and-response choices,
   register, velocity, and articulation.
5. Percussion: lanes, sample references, patterns, accents, velocity, density,
   and bus level.
6. Texture/effects: drones or loops, ambience, delay, filtering, reverb, stereo
   width, and fade behavior.
7. Samples: logical ID, collection, checksum, category, source license, local or
   external availability, and promotion status.
8. Quality: hard validation status, score, metrical clarity, percussion share,
   density, peak, loudness, and novelty fingerprint.

Use musical names alongside numeric values. For example, show both a sixteenth
step and its bar/beat position, and both a MIDI pitch and note name.

## Editing workflow

- Allow a field, section, or sample choice to be locked.
- `Randomize unlocked` creates a new valid resolved specification while
  preserving locked values.
- `Validate` reports errors and warnings without mutating the specification.
- `Render preview` produces a short music-only loop quickly.
- `Render lesson preview` may use cached/example speech to demonstrate ducking;
  it must not invoke a hosted voice service without a separate explicit action.
- Keep an in-memory undo/redo history and allow named snapshots for A/B
  comparison.
- Load and save complete `.bed.json` files.
- Provide a read-only JSON view initially; a raw editor may be enabled in an
  explicitly advanced section with schema validation before use.
- Offer `Copy Python`, `Copy CLI`, and `Copy JSON` representations so discoveries
  can move into tests or another application.

Editing may intentionally move outside production constraints. The page should
show one of three states:

- Production-safe.
- Experimental but renderable, with specific warnings.
- Invalid, with exact fields that must be corrected.

It should never silently clamp a user's Lab value. A separate `Return to safe
range` action can make that change explicitly.

## Service boundary

Use a small local HTTP service with a static browser client. Define typed JSON
endpoints around Step 1 rather than exposing internal renderer functions:

- `GET /api/schema`: versions, control schema, ranges, enums, and family names.
- `POST /api/resolve`: `MusicRequest` to `MusicGenerationResult` metadata.
- `POST /api/randomize`: locked fields plus a base `BedSpec` to a new spec.
- `POST /api/validate`: validation and quality report.
- `POST /api/render-preview`: preview audio for a resolved spec.
- `POST /api/render`: complete bed audio for a resolved spec.
- `GET /api/samples/{logical_id}`: metadata only.
- `POST /api/samples/{logical_id}/promote`: explicit local promotion.

Choose the smallest maintainable Python server during implementation and keep it
as an optional project dependency. The browser should use ordinary HTML, CSS,
and TypeScript or JavaScript unless a richer framework clearly reduces total
complexity. Generated audio is written beneath a managed `out/explorer/`
directory or streamed directly; never write to arbitrary client-provided paths.

## Responsiveness and safety

- Show progress for candidate analysis and full rendering.
- Support cancellation between candidates and render stages.
- Cache previews by `BedSpec` checksum, renderer version, and duration.
- Prevent duplicate simultaneous renders of the same cache key.
- Limit accepted duration, candidate count, and request body size.
- Resolve all filesystem access through managed output, cache, and library roots.
- Do not permit arbitrary paths in JSON, sample references, or download requests.
- Do not download libraries implicitly. Promotion is explicit and only copies a
  cataloged, checksum-verified asset.
- Display a clear external-library status and explain which selected assets are
  already available offline.

## Verification

Add tests for:

- JSON request/response schemas and version reporting.
- Simple controls producing valid `MusicRequest` values.
- `BedSpec` round trips through the service without loss.
- Lock-aware randomization and deterministic behavior with a fixed seed.
- Validation messages for unsafe versus invalid experimental values.
- Preview cache keys and invalidation across renderer versions.
- Cancellation and error cleanup.
- Rejection of arbitrary file paths and oversized requests.
- No implicit network calls.
- Browser smoke flows: initial generation, playback, Lab editing, undo, A/B,
  save, and reload.

Complete manual checks with the external SSD attached and detached. Promoted
specs must remain playable offline; unpromoted specs must show an actionable
message rather than substitute another sound.

## Completion criteria

- A user can open the local page, hear a strong random result, and generate
  another without understanding music parameters.
- The same result can be opened in Lab, modified, validated, rendered, saved,
  and reproduced from its JSON.
- Simple mode cannot accidentally leave the `production-v1` safe range.
- Lab mode exposes the resolved configuration and provenance without duplicating
  generator rules in the frontend.
- The service is documented with one local launch command and is optional for
  command-line/library users.

