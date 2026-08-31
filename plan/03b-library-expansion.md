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
