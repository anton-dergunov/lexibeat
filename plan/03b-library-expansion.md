# Step 3 follow-up: staged natural sample-library expansion

## Decision from the Step 3B listening test

The grammar and palette treatment was more varied, but it was less suitable as
background music for language-learning speech. High-register foreground sounds
were especially distracting in clips 6, 7, 10 and 11; unusual bottle-like and
electronic sounds also pulled attention away from the words. Production should
therefore retain the control composition policy.

Future variety should come primarily from better natural recordings inside the
same restrained musical role. Real piano was the clearest prior quality gain, so
piano depth is the first target. Storage is not the limiting factor; mapping,
level calibration and speech suitability are.

## Stage 1 — audit the already downloaded library

Build a report comparing the complete external catalog with the production
bundle. Do not copy files yet. Rank complete, redistributable banks in this
order:

1. Piano: wider useful register, two or more velocity layers, coherent room or
   microphone perspective, and same-articulation round robins.
2. Sustained and lightly articulated strings that remain smooth under speech.
3. Classical/acoustic guitar and natural bass with controlled attacks.
4. Soft organic percussion with a clear low pulse and restrained mid/high
   transients.
5. Other natural instruments only after they pass the register and attention
   checks below.

Prefer complete banks over isolated notes. Keep articulation, microphone and
round-robin groups coherent, and quantify transposition gaps before promotion.

## Stage 2 — speech-safety gates

Every candidate bank must be tested in the register and role where production
would use it. Reject or remap a bank when any of these conditions holds:

- High notes have piercing, metallic, whistle-like or resonant partials that
  become foreground events at normal listening level.
- A note or transient is noticeably louder than adjacent mapped notes after
  loudness normalization.
- The instrument masks consonants or makes the spoken words harder to follow.
- The sound creates a novel effect—bottle-like, cosmic or otherwise attention
  grabbing—that is interesting alone but distracting as a repeated background.
- The attack is so soft that the rhythmic pulse becomes difficult to hear, or
  so sharp that it competes with speech.

Calibrate gain by register and velocity zone where necessary. Start with a
conservative lead ceiling and compare perceived level, not peak level alone.

## Stage 3 — candidate bundle and listening test

Promote a first candidate expansion from the existing library. A practical
starting size is 8–12 GB, but quality determines the size. Keep the candidate
bundle separate from the current production core.

Use paired manifests with identical families, seeds and speech:

1. Control: current production bundle and control composition policy.
2. Treatment: candidate natural banks with the same composition policy.
3. Listen to each set independently before making direct A/B comparisons.
4. Rate speech clarity, rhythmic usefulness, background suitability, perceived
   foreground level, naturalness and preference.
5. Remove or remap weak banks, then repeat with the surviving set.

The treatment should improve naturalness or preference without reducing speech
clarity or rhythmic usefulness. Mere distinctiveness is not an acceptance goal.

## Stage 4 — versioned production bundle v2

- Generate a lock manifest with logical IDs, source revisions, checksums, sizes,
  licenses, attribution and expected paths.
- Require explicit note, velocity, articulation, microphone and round-robin
  mappings where the source provides them.
- Reject demos, noise, silence, clipped files, unclear licenses and duplicate
  conversions before promotion.
- Verify the bundle offline and preserve the no-implicit-download contract.
- Publish only after the paired speech test shows no material regressions.

## Acceptance targets

- Meaningfully broader coherent piano coverage than the current bundle.
- At least two useful velocity layers for priority foreground banks where source
  material permits.
- More same-articulation, same-microphone round-robin depth without audible
  timbre switching.
- Finite 44.1-kHz stereo output, peak no higher than 0.97, deterministic replay
  and unchanged speech downbeats.
- No accepted clip with the piercing high-register problem heard in Step 3B.
- Listener preference or naturalness improves while speech clarity and pulse do
  not decline.

Only after this audit should new sample sources be researched. Prefer CC0;
compatible attributed sources require complete NOTICE entries.

## Implementation status

Stages 1 and 2 are implemented as a non-destructive gate:

```bash
uv run sample_library.py audit-expansion --refresh-index \
  --workspace out/library-expansion --target-gb 10
uv run sample_library.py audition-expansion \
  --workspace out/library-expansion
uv run sample_library.py audition-expansion \
  --workspace out/library-expansion --wave secondary
```

The first command deep-indexes the attached SSD into a metadata-only workspace,
compares it with the immutable production catalog, ranks coherent banks and
writes `audit.json`, `candidate-manifest.json` and `report.md`. It does not copy
or promote audio. The second command reads shortlisted samples directly from the
SSD and creates isolated, level-preserving probes plus a rating sheet.

The initial audit found 8,399 indexed assets (9.92 GB of audio), 6,617 payloads
not present in production, and 256 coherent or percussion groups to evaluate.
After natural-instrument and speech-safety screening, 16 banks containing 586
new assets (0.40 GB) entered the audition shortlist. The result is deliberately
far below the 10 GB ceiling: library size is not a quality target.

The current external catalog has no immediately production-ready new piano bank.
Two additional pianos are useful review candidates but have sparse note spacing;
one also has a bright high-register warning. The shortlist therefore also tests
section strings, natural contrabass, cajon, conga, bongos, frame drum and subdued
shakers. Snare, marching, metallic, miscellaneous, tenor-drum, electronic/FM,
glock/bell, bottle/cosmic/effect and unsupported foreground groups are excluded.

Stage 3 is intentionally waiting at the human listening gate. Rate the complete
isolated set in `out/library-expansion/auditions/ratings.csv` before any audio is
promoted or integrated into the production candidate path. Only retained banks
will be rendered with identical speech for the final paired A/B test.

A separate secondary wave covers the remaining non-rejected banks without
changing or overwriting the primary audition. Exact audio aliases are removed by
SHA-256 before rendering. Secondary clips use `S01`, `S02`, and so on, and are
written under `out/library-expansion/secondary-auditions/`.
