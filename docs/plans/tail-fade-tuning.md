# Tuning the tail fade

`fade_tail` (`lexibeat/dsp.py`) is what happens when a take runs past its slot — the next
downbeat: it stays audible rather than being cut mid-word, ducking down and then fading out
under the voice that starts next. The goal is the real-life feel of two people talking: the
next speaker is heard clearly, but the one who was already talking hasn't been switched off.
This can only be judged by ear, not from a spectrogram, so it's a separate task from this
plan — the plan here is what to render and compare, not a decision on values yet.

## The parameters

| Constant | Current | Controls |
|---|---|---|
| `TAIL_DUCK_DB` | -4.0 | The level, in dB, the instant ducking finishes (at `TAIL_DUCK_SECONDS`) |
| `TAIL_DUCK_SECONDS` | 0.25 | How long the initial duck takes |
| `TAIL_FLOOR_DB` | -10.0 | The level the tail settles into by `TAIL_MAX_SECONDS`, where it stops |
| `TAIL_MAX_SECONDS` | 1.2 | Total length of the tail after the slot; the point it reaches silence |
| `TAIL_END_FADE_SECONDS` | 0.08 | A short cosine at the very end so the cut doesn't click |

The shape is: full level up to the slot, a fast duck to `TAIL_DUCK_DB` over `TAIL_DUCK_SECONDS`,
then a straight-line (in dB) fade from there down to `TAIL_FLOOR_DB` by `TAIL_MAX_SECONDS`.

## Where this came from

Originally `TAIL_DUCK_DB = -8.0`, `TAIL_FLOOR_DB = -30.0` — a 22 dB fade, and -30 dB reads as
near-silent, so the overrunning voice all but disappeared under the next one. `f9b4176` loosened
both to -4.0 / -10.0 (a 6 dB fade) to keep it present longer, closer to how an overlapping
conversation actually sounds. `test_the_tail_ducks_then_fades_lazily_and_ends_silent`
(`tests/test_lexibeat.py`) had the old 22 dB range's shape baked into it as a hardcoded 6 dB
drop, which is why it broke; it now derives its expectation from the constants directly, so
retuning these values no longer requires touching the test.

## Reasonable ranges to try

- `TAIL_DUCK_DB`: -2 to -6 dB. Past -2 it barely reads as ducked against the new voice; past -6
  it's back towards the old, more-hidden feel.
- `TAIL_FLOOR_DB`: -6 to -12 dB. Above -6 it risks competing with the new voice for the rest of
  the tail; below -12 it's headed back towards inaudible.
- `TAIL_DUCK_SECONDS` / `TAIL_MAX_SECONDS`: secondary knobs — how long the initial duck takes,
  and how long the whole tail lasts. Worth a listen if the dB range alone doesn't land right,
  but start with `TAIL_DUCK_DB` / `TAIL_FLOOR_DB`.

## Listening

Following the existing round pattern (`scripts/listening/defects.py`,
`scripts/listening/serve.py`): render a handful of loops with two consecutive words close enough
together that the first overruns its slot, once per candidate (`TAIL_DUCK_DB`, `TAIL_FLOOR_DB`)
pair, same seed, so only the tail changes between clips. Serve the round, judge by ear on a
tablet, adopt the pair that sounds most like an overlapping conversation without either voice
getting lost.
