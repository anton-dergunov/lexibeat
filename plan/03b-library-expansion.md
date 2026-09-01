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

## Listener decision and candidate-v2 integration

All 16 primary banks and all 23 checksum-distinct secondary banks were accepted.
Primary clip 04 is retained with 2 dB additional attenuation. Secondary clips
S17 and S20 are retained with 4 dB attenuation; bright high-percussion clips S21
and S22 are retained with 5 dB attenuation. These constraints are applied when
the bank is selected and serialized into the resolved BedSpec.

The accepted set is integrated in the separate checksum-locked bundle at
`out/library-expansion/candidate-v2/`. It combines production v1 with 976 new
catalog assets, for 2,088 catalog assets and 2,183 total assets (2.58 GB). Bundle
verification reports no missing files, checksum mismatches or catalog-count
mismatches. Production v1 remains unchanged.

Candidate generation can now select all 39 accepted banks across lead, pad,
bass and percussion roles. Pitched banks are restricted to their audited safe
registers and use the audit's bank/high-register gain recommendations. Sustained
strings can serve as restrained pads; accepted contrabasses can replace synth
bass; cajon, frame drum, conga, darbuka, bongo, clave, shaker and tambourine
banks enter their compatible rhythmic roles.

The paired treatment in `out/library-expansion/paired-treatment/` replays the 14
family/seed pairs from `out/music-bakeoff-3/` with byte-identical speech and
identical composition events. It exercises 16 distinct accepted banks across 23
placements.

The listener accepted all 14 treatments as appropriate background music and
preferred the greater natural-instrument variety. Treatment clips 7 and 14 also
confirm that the existing electronic colors remain useful when occasional:
clip 7 resolves the VCSL TX81Z/FM Piano bank and clip 14 uses the built-in synth
lead. Neither sound is removed by the natural-library expansion. Candidate-v2
is therefore the approved foundation for Wave 3, while production-v1 remains
unchanged until the broader wave has completed its own listening gate.

## Wave 3 — broader natural instruments

This audit is more inclusive while retaining role-specific speech safety. The
attached catalog has mapped banks across accordion, classical guitar, flute,
recorder, ocarina, clarinet, oboe, bassoon, harmonica, harp, strumstick, quiet
organ, harpsichord, kalimba, mbira and marimba. They are screened in low/middle
registers and auditioned in actual musical roles rather than rejected merely
for being outside the first-wave family list.

The policy is intentionally more aggressive than Waves 1 and 2. Technically
usable coherent natural banks are auditioned by default; brightness, expressive
vibrato, unusual timbre and attack become role/register/gain notes rather than
automatic aesthetic rejections. A bank is rejected only for a concrete problem
such as silence, release/noise/demo material, unusable note coverage, excessive
mapping gaps, duplicate payload, or inability to remain usable under speech.

The implemented metadata-only audit evaluates 66 mapped banks against
candidate-v2. Ten contain no new payload because they are already in the
candidate bundle. All 56 checksum-distinct additions proceed to audition:
1,136 assets totaling 1.51 GB across accordion, recorder, ocarina, flute,
clarinet, oboe, bassoon, harmonica, harp, plucked string, organ, harpsichord,
lamellophone and marimba. No audio is copied or promoted by the audit.

```bash
uv run sample_library.py audit-wave3 \
  --workspace out/library-expansion \
  --baseline-bundle out/library-expansion/candidate-v2
uv run sample_library.py audition-expansion \
  --workspace out/library-expansion --wave wave3
```

The auditions and rating sheet are in
`out/library-expansion/wave3/auditions/`. The generated probes are finite and
peak at 0.42. The new FreePats Button Accordion HN bank is a 17-note CC0 SFZ
bank explicitly downloaded to the external tier; its separate release-trigger
recordings are excluded from the playable foreground mapping. Its source is
https://freepats.zenvoid.org/Organ/accordion.html.

No coherent bandoneon multisample bank with a sufficiently clear redistributable
license was found in this source pass. Do not substitute a single demonstration
recording or a bank with sample-repackaging restrictions; bandoneon remains an
explicit acquisition gap for a later source search.

## Wave 3 listener decision and candidate-v3 integration

All 56 Wave 3 banks are accepted. Harpsichord is also accepted, but as an
occasional baroque color: all six harpsichord banks receive 8 dB listener
attenuation and a 0.35 selection weight. This is approximately 40% of the
uncautioned amplitude before existing bank/high-register calibration. The
resolved zone gains remain serialized in each BedSpec.

Candidate-v3 is built separately at
`out/library-expansion/candidate-v3/` on top of candidate-v2. It contains all 95
accepted banks, 3,224 catalog assets, 3,319 total assets and 4.09 GB of audio.
Verification reports no missing files, checksum mismatches, LFS pointers or
catalog-count mismatches. Candidate-v2 and production-v1 remain unchanged.

The eight-clip speech-context check in
`out/library-expansion/wave3-integrated-listening/` contains two harpsichords
(English Normal and the brighter Flemish High) followed by accordion, flute,
vibrato recorder, harmonica, Renaissance organ and oboe. All clips reuse the
same bilingual speech, start speech on downbeats, remain finite 44.1-kHz stereo
audio and stay below the 0.97 peak ceiling.

Listener feedback showed that generic short-note lead grammar did not reveal
the character of several Wave 3 banks. The final role experiment therefore
keeps candidate-v3 unchanged and applies explicit, replayable treatments only
to the listening BedSpecs: sparse long notes for winds and harmonica, a
cantabile accordion phrase, held organ chords, and short-note patterns reserved
for genuinely plucked or struck instruments. Featured levels, bank gains,
events and reduced reverb are all serialized; no renderer-only mix exception is
used.

The exact 14-family matrix is tracked in
`plan/03c-wave3-instrument-role-experiment.json`. Its rendered speech-context
set is in `out/library-expansion/wave3-final-role-experiment/`, with one clip
each for accordion, bassoon, clarinet, flute, harmonica, harp, harpsichord,
lamellophone, marimba, oboe, ocarina, organ, recorder and strumstick. The
listening gate decides which role treatments, if any, should later become
production policy; creating this experiment does not alter production-v1 or
candidate-v3 selection behavior.

## Final listener decision and closure

The final role-aware listening gate accepts six Wave 3 families: harp,
lamellophone, marimba, ocarina, organ and plucked-string/strumstick. Accordion,
bassoon, clarinet, flute, harmonica, harpsichord, oboe and recorder are rejected.
Ocarina is treated as accepted because both the detailed clip note and the
listener's accepted list approve clip 11; its duplicate appearance in the
rejected list is recorded as a transcription slip.

The accepted production treatments are resolved into each BedSpec: harp uses a
restrained arpeggio and low selection weight; lamellophone uses a strongly
ducked percussive-backing pattern; marimba uses a louder soft-mallet accent;
ocarina uses sparse sustained notes; organ replaces the lead with held pad
chords; and strumstick uses a rhythmic plucked pattern. Rejected families are
absent from both the accepted-bank policy and the final catalog, including 76
stray inherited catalog assets that predated the Wave 3 proposal.

The final checksum-locked bundle is
`out/library-expansion/final-v3/`. It contains 57 accepted banks, including 18
Wave 3 banks, with 2,345 catalog assets and 2,440 total assets. Its logical audio
size is 3,061,928,345 bytes (3.062 GB / 2.852 GiB). Bundle verification reports
no missing files, checksum mismatches, LFS pointers or catalog-count mismatch.
All six accepted roles pass normal candidate generation and deterministic
sampled rendering. This closes Steps 3 and 3B; further library expansion is
deferred unless a later product requirement identifies a specific gap.
