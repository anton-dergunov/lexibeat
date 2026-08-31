# Step 3B follow-up: staged sample-library expansion

## Starting point

The production bundle currently contains 1,112 catalog assets (1.96 GB), 27
coherent pitched banks and only five explicit round-robin groups, each with two
takes. Storage is not the limiting factor; mapping quality and audible use are.
The existing external library should be mined before adding new sources.

## Stage 1 — expand from already downloaded material

Build a candidate report from the complete external catalog and compare it with
the production bundle. Promote assets only when they fill one of these gaps:

- At least three coherent banks for each new palette: airy, wooden, warm,
  shimmering and plucked.
- Explicit same-articulation, same-microphone round robins for gentle piano,
  classical guitar, natural bass, subdued mallets, mbira and soft percussion.
- Two or more useful velocity layers per foreground bank, with restrained
  attacks suitable under speech.
- Register coverage from bass through upper melody without large transposition
  gaps.
- Organic low, mid and high percussion roles with controlled transients.

Target the first bundle expansion at roughly 8–12 GB, but accept less if the
quality gates are met. Size is a ceiling, not a success metric.

## Stage 2 — versioned production bundle v2

- Generate a lock manifest containing logical IDs, source revisions, checksums,
  sizes, licenses, attribution and expected paths.
- Keep one complete bank together; do not promote isolated notes merely to save
  space.
- Require explicit articulation, velocity, microphone and round-robin mappings.
- Reject demos, noise, silence, clipped files, unclear licenses and duplicate
  conversions before promotion.
- Verify the bundle offline and preserve the no-implicit-download contract.
- Keep the larger server catalog separate from the smaller production core so
  future experiments do not enlarge every deployment automatically.

## Stage 3 — only then evaluate new sources

After Stage 1 listening tests, produce a remaining-gap table. Research and add
new redistributable sources only for gaps that the existing 31 GB library cannot
fill. Prefer CC0; attributed compatible sources require complete NOTICE entries.
Do not admit a source based on library size or instrument count alone.

## Acceptance targets

- At least 60 coherent production banks.
- At least 50 usable explicit round-robin groups, preferably three or more takes.
- Every natural palette has three recognizably different foreground-bank options.
- A 16-clip batch uses at least four palettes, five bass grammars, five motif
  grammars and eight foreground instrument families.
- Expanded-library candidates maintain speech downbeats, finite 44.1-kHz stereo,
  the 0.97 peak ceiling and the existing percussion/masking quality gates.
- Listener ratings show a clear preference or distinctiveness gain before the
  bundle becomes the default.

## Evaluation sequence

1. Report complete-catalog gaps and proposed promotions without copying files.
2. Render music-only palette/grammar batches from the candidate bundle.
3. Remove or remap weak banks based on listening notes.
4. Render the surviving 12–16 options with identical bilingual speech.
5. Lock, verify and publish production bundle v2 only after acceptance.
