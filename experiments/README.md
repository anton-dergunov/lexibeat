# Experiments

Small, self-contained trials that answer one question each before anything is built into LexiBeat.
Each folder has its own environment and README and is not part of the package, the service or its
tests. An experiment imports nothing from `lexibeat`, unless it says so and runs in the repository's
own environment. Outputs go to the folder's `out/`.

**Results are committed; what can be regenerated is not.** An experiment's raw results — labels,
scores, reports, model replies — are the work itself and are tracked, provided they hold no personal
data, credentials or private paths and come to a few MB of text. Audio, pictures and caches can be
rebuilt from them and stay ignored. The split is written in `.gitignore`, which ignores `out/` and
lets the result files back in by name. CI's check for the prohibited word skips `out/`, so a result
file is checked for it by hand before it is first committed.

- [`sfx_stable_audio/`](sfx_stable_audio/) — can a local text-to-audio model make the short word
  sounds that `docs/plans/programme-loops.md` (P4) wants, instead of a noisy CC0 catalogue?
- [`programme_blocks/`](programme_blocks/) — the blocks `docs/plans/programme-loops.md` would build
  programmes from (drill structures over music, syllables, mixed-language voices, examples,
  stories, callbacks, commentary, framing), each tried alone and scored 1–5 on a tablet; and a
  follow-up asking whether a dictionary pronunciation, rather than the spelling, makes a voice say
  a hard word right.
