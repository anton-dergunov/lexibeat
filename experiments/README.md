# Experiments

Small, self-contained trials that answer one question each before anything is built into LexiBeat.
Each folder has its own environment and README and is not part of the package, the service or its
tests. An experiment imports nothing from `lexibeat`, unless it says so and runs in the repository's
own environment. Outputs go to the folder's `out/`, which is not tracked.

- [`sfx_stable_audio/`](sfx_stable_audio/) — can a local text-to-audio model make the short word
  sounds that `docs/plans/programme-loops.md` (P4) wants, instead of a noisy CC0 catalogue?
