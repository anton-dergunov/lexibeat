# How LexiBeat generates music

LexiBeat does not ask a generative model to produce an audio file. It composes a
short symbolic phrase with explicit rules, saves every decision in a `BedSpec`,
and then renders that phrase with synthesizers and recorded samples. This gives
the lesson arranger something a free-running audio model cannot guarantee: the
exact time of every bar and downbeat.

This document describes the active `production-v1` path exposed by
`MusicRequest`, `resolve_music()`, and `generate_music()`. Every current BedSpec
contains a fully resolved phrase; the superseded phrase-less renderers and their
four prototype styles have been removed.

The main implementation is split across
[`api.py`](../lexibeat/api.py),
[`generator.py`](../lexibeat/generator.py),
[`bedspec.py`](../lexibeat/bedspec.py),
[`music.py`](../lexibeat/music.py), and
[`mix.py`](../lexibeat/mix.py).

## A small music-theory vocabulary

You do not need advanced theory to follow the generator. The following terms are
enough:

- A **beat** is the pulse you might count as “one, two, three, four.” A
  **measure** or **bar** groups a fixed number of beats. In 3/4 there are three
  quarter-note beats per bar; in 4/4 there are four; in 5/4 there are five.
- LexiBeat divides a bar into **sixteenth-note steps**. A 4/4 bar has 16 steps,
  3/4 has 12, and 5/4 has 20. Events store a step number rather than an
  approximate timestamp.
- A **MIDI note** is a number for a piano key: middle C is 60, A4 is 69, and
  adding 12 moves up one octave.
- A **scale** selects notes around a tonal home. A scale **degree** is an index
  into that selection. LexiBeat numbers degrees from zero, so degree `0` is the
  tonic, degree `1` is the second scale note, and so on.
- A **chord** sounds several notes together. An **inversion** moves one or more
  chord notes to another octave. **Voice leading** means choosing inversions so
  the individual notes move short distances between consecutive chords.
- A **bass line** supplies low notes beneath the chords. A **motif** is a short,
  recognizable melodic idea that can repeat. A **percussion lane** is one drum
  role—low, middle, or high—with its own onset pattern.
- **Timbre** is sound colour: the difference between the same pitch played by a
  piano, marimba, or synthesizer. An **envelope** controls how a sound grows and
  fades. A low-pass **filter** softens high frequencies. **Reverb** adds a sense
  of acoustic space.

## The complete production pipeline

```mermaid
flowchart LR
    A[MusicRequest] --> B[Choose profile and family]
    B --> C[Generate deterministic candidates]
    C --> D[Resolve 4/8-bar phrases]
    D --> E[Assign synths or catalog samples]
    E --> F[Render and validate previews]
    F --> G[Rank and choose from top tier]
    G --> H[Save complete BedSpec]
    H --> I[Render pad, bass, drums, lead]
    I --> J[Align and duck under speech]
    J --> K[Normalize and limit final track]
```

There are two intentionally separate stages:

1. **Resolution composes the music.** It samples a family, harmony, rhythms,
   motif, instruments, and performance values. The result is a complete
   `BedSpec` containing explicit chord, bass, lead, and percussion events.
2. **Rendering performs the saved music.** The renderer repeats those events for
   the requested duration. It does not invent a new melody or progression while
   rendering.

This distinction explains why saving the resolved `BedSpec` is sufficient to
replay a composition. It also means that changing a generation-time field such
as `progression` after a phrase has been resolved does not rewrite the already
saved `phrase.chords`.

## The public controls

`MusicRequest` is the small, safe interface intended for applications:

| Field | Values | Effect |
|---|---|---|
| `family` | `auto` or a production family | Chooses the ranges and grammars from which the phrase is composed. `auto` makes one seeded family choice before candidate generation. |
| `energy` | `calm`, `balanced`, `bright` | `calm` lowers tempo, percussion, lead, and pad brightness; `bright` raises tempo, lead level, and brightness within production limits. |
| `rhythm` | `sparse`, `steady`, `groovy` | `sparse` keeps at most two percussion lanes and lowers the drum bus; `groovy` raises the drum bus slightly; `steady` leaves the family result unchanged. |
| `palette` | `electronic`, `hybrid`, `acoustic` | Controls whether catalog recordings may replace synthesized parts and how likely those replacements are. Electronic always remains sample-free. |
| `seed` | unsigned 64-bit integer or `None` | Reproduces resolution when supplied. `None` creates a random 64-bit request seed. |
| `profile` | normally `production-v1` | Selects versioned family, quality, swing, and candidate-selection policy. |

The energy adjustments are deliberately small. `calm` multiplies BPM by 0.94
(with a floor of 56), while `bright` multiplies it by 1.04 (with a ceiling of
104). Calm also multiplies drum level by 0.90, lead level by 0.95, and the pad
cutoff by 0.92. Bright multiplies lead level by 1.04 and pad cutoff by 1.10.
Sparse rhythm retains the low and mid lanes and multiplies drum level by 0.88;
groovy multiplies it by 1.06 but caps it at 0.62. Production swing is capped at
`0.025`, even if a family's raw range is wider.

### Production families

Families are not finished songs or genres. Each is a collection of allowed
tempo, scale, texture, density, and instrumentation ranges. The same family and
different seeds therefore sound related without resolving to the same phrase.

| Family | BPM | Scales | Harmony textures | Bass choices |
|---|---:|---|---|---|
| `meditative` | 58–76 | natural minor, Dorian, major | sustain, drone, open | sustain, drone, root–fifth |
| `organic` | 68–88 | Dorian, natural minor, major | pulse, open, arpeggio | root–fifth, syncopated, sustain |
| `acoustic` | 64–84 | major, Dorian, natural minor | arpeggio, sustain, pulse | sustain, root–fifth, passing |
| `sunlit` | 78–98 | major, Lydian, Dorian | pulse, arpeggio, open | root–fifth, syncopated, passing |
| `radiant` | 86–102 | major, Lydian | open, pulse, arpeggio | root–fifth, sustain |
| `acoustic-flow` | 68–88 | major, Dorian | sustain, open, arpeggio | sustain, root–fifth |
| `playful-minimal` | 76–94 | major, Lydian, Dorian | pulse, open | root–fifth, sustain |
| `warm-motion` | 72–88 | major, Dorian | pulse, open | root–fifth, sustain |
| `bright-organic` | 76–92 | major, Dorian, Lydian | open, arpeggio, pulse | root–fifth, passing |
| `gentle-game` | 82–98 | major, Lydian | arpeggio, pulse | root–fifth, passing |
| `sunlit-acoustic` | 76–92 | major, Lydian | open, sustain, arpeggio | root–fifth, sustain |
| `gentle-movement` | 70–86 | major, Dorian | pulse, open, sustain | root–fifth, passing |
| `playful-plucked` | 78–94 | major, Lydian | open, pulse | root–fifth, sustain |
| `bright-pastoral` | 72–88 | major, Lydian, Dorian | sustain, open, arpeggio | sustain, root–fifth |

The code also contains broader exploration families, but they are not members of
the production profile and are not selected by the public production API.

The remaining family ranges control timbre and activity. “Density” is the
probability used when considering lead onsets and also influences percussion
levels. Brightness is the initial pad-filter cutoff range. The raw family swing
is shown for completeness; `production-v1` caps the resolved value at `0.025`.

| Family | Root MIDI choices | Pad timbres | Lead roles | Density | Brightness (Hz) | Raw swing |
|---|---|---|---|---:|---:|---:|
| `meditative` | 40, 43, 45, 48 | sine, triangle, strings | synth, piano, glockenspiel | 0.12–0.28 | 550–1,050 | 0–0.030 |
| `organic` | 43, 45, 48, 50 | triangle, strings, soft saw | marimba, piano, synth | 0.28–0.52 | 700–1,350 | 0–0.050 |
| `acoustic` | 45, 48, 50, 53 | strings, triangle | piano, marimba, glockenspiel | 0.22–0.46 | 850–1,500 | 0–0.040 |
| `sunlit` | 48, 50, 53, 55 | triangle, soft saw, strings | piano, marimba, glockenspiel | 0.30–0.55 | 1,050–1,800 | 0–0.040 |
| `radiant` | 48, 50, 53, 55 | strings, triangle, soft saw | piano, marimba, glockenspiel | 0.30–0.50 | 1,150–1,850 | 0–0.025 |
| `acoustic-flow` | 45, 48, 50, 53 | strings, triangle | piano, marimba | 0.20–0.40 | 800–1,450 | 0–0.025 |
| `playful-minimal` | 48, 50, 53, 55 | sine, triangle | marimba, piano, glockenspiel | 0.16–0.34 | 900–1,650 | 0–0.020 |
| `warm-motion` | 45, 48, 50, 53 | triangle, strings, soft saw | piano, marimba, synth | 0.26–0.44 | 750–1,350 | 0–0.030 |
| `bright-organic` | 48, 50, 53, 55 | strings, triangle | marimba, piano, synth | 0.24–0.46 | 950–1,600 | 0–0.035 |
| `gentle-game` | 48, 50, 53, 55 | triangle, sine | marimba, piano, glockenspiel | 0.26–0.44 | 1,000–1,700 | 0–0.025 |
| `sunlit-acoustic` | 48, 50, 53, 55 | strings, triangle | piano, marimba, synth | 0.20–0.38 | 1,050–1,750 | 0–0.015 |
| `gentle-movement` | 45, 48, 50, 53 | triangle, strings | piano, marimba, synth | 0.22–0.40 | 800–1,400 | 0–0.012 |
| `playful-plucked` | 48, 50, 53, 55 | sine, triangle | marimba, glockenspiel, synth | 0.18–0.36 | 1,000–1,700 | 0–0.010 |
| `bright-pastoral` | 48, 50, 53, 55 | strings, triangle | piano, marimba, synth | 0.18–0.36 | 950–1,550 | 0–0.012 |

## How one phrase is composed

### 1. Meter and tempo

The family proposes a discrete list of BPM values. Meter is drawn from
`[4, 4, 4, 4, 3, 5]`, so 4/4 has probability 4/6 and 3/4 and 5/4 each have
probability 1/6. The active families always use quarter-note beats
(`beat_unit = 4`). BPM values that would make the selected bar shorter than 2.2
seconds are filtered out, leaving enough room for a spoken phrase. If none are
compatible, the meter falls back to 4/4 and restores the family's full BPM
list.

`bpm` always means quarter notes per minute. For beat unit `u`, the duration is:

```text
beat seconds = (60 / bpm) × (4 / u)
bar seconds  = beat seconds × beats_per_bar
steps/bar    = beats_per_bar × (16 / u)
```

Events land on this grid. Swing delays steps `2, 6, 10, ...`—the off-eighth
sixteenths—by `swing × one_step`. Production keeps that delay subtle so spoken
downbeats remain unambiguous.

### 2. Scale and chord progression

LexiBeat supports natural minor, Dorian, harmonic minor, major, and Lydian as
lists of semitone offsets from a MIDI root. For example, A Dorian is:

```text
A  B  C  D  E  F# G
0  2  3  5  7  9  10 semitones above A
```

A phrase contains four bars most often, or eight bars less often. It begins on
degree `0`, then follows this transition table. Repeated entries are weights,
so degree `0` moves to degree `5` twice as often as to each other listed choice.

| Current degree | Possible next degree(s) |
|---:|---|
| 0 | 2, 3, 4, 5, 5, 6 |
| 1 | 4, 6, 0 |
| 2 | 4, 5, 6 |
| 3 | 0, 4, 5 |
| 4 | 0, 5, 6; harmonic minor instead uses 0, 0, 5, 6 |
| 5 | 0, 2, 3, 4 |
| 6 | 0, 3, 4 |

These are hand-tuned probabilistic transitions, not a learned Markov model and
not a full system of classical functional harmony. The selected scale degree's
pitch is octave-folded to remain within six semitones of the tonic before the
chord is built.

For a degree with root `r`, the basic open voicing is:

```text
[r, r + 7, r + 12, r + 12 + diatonic_third, r + 19]
```

This deliberately repeats the root and fifth across octaves. A seventh, added
ninth, or both may be added. The third, seventh, and ninth are derived from the
selected scale, but the fifth is always a perfect fifth (`+7`). Consequently,
these are stable custom voicings rather than strictly diatonic triads; an
otherwise scale-based chord can contain a chromatic perfect fifth.

For each chord, `_smooth_voicing()` tries up to four inversions and octave shifts
of -12, 0, and +12 semitones, retaining notes between MIDI 35 and 84. The first
chord is placed near MIDI 57. Later chords minimize the sum of absolute movement
from the preceding chord's ordered voices. This is a compact voice-leading
heuristic: it reduces audible jumping without attempting full counterpoint.
Duplicate pitches are removed and the search retains at most the five lowest
voices before trying inversions.

### 3. Harmonic texture

The selected texture converts each bar's voicing into explicit `ChordEvent`s:

| Texture | Events produced |
|---|---|
| `sustain` | One chord at the bar start, lasting `pad.overlap × steps_per_bar`. |
| `drone` | The same long duration, using tonic, fifth, octave, and the current chord's top two notes. |
| `open` | The same long duration, using the lowest two and highest two voicing notes. |
| `pulse` | Two chords at step 0 and the integer half-bar, each 0.34 bar long; the second uses a 0.72 accent multiplier. |
| `arpeggio` | One 1.7-step note every two sixteenth steps, travelling up and partway back down the voicing. |

Once these events are saved, `harmony_texture` is descriptive metadata; replay
uses the events themselves.

### 4. Bass line

The bass mode is selected from the family's allowed set and converted to
`NoteEvent`s:

| Grammar | Behavior |
|---|---|
| `drone` | Holds the tonic an octave down from step 0 for 0.82 bar, regardless of chord. |
| `sustain` | Holds the current chord root from step 0 for 0.82 bar. |
| `root_fifth` | Plays the root at step 0 for 0.42 bar, then the fifth at half-bar for 0.35 bar. |
| `passing` | Holds the root for 0.55 bar, then at three-quarter-bar plays a 0.20-bar approach to the next root, clamped to two semitones. |
| `syncopated` | Places root/fifth/root at step 0, approximately three-eighths, and approximately three-quarters, each for the larger of 1.5 steps or one-fifth bar. |

### 5. Lead motif

The lead uses a five-note subset of the chosen scale across its configured
register. The subset is pentatonic-like: it omits scale notes that would create
more frequent close clashes over changing chords.

The generator walks through that note list in steps chosen from
`[-2, -1, 0, 0, +1, +2]`, considering even sixteenth steps across up to four
bars. Density determines whether a considered onset becomes a note. The motif
then repeats through the phrase with deterministic light omissions and velocity
variation. Piano motifs receive extra notes and a wider enforced pitch span;
other leads receive at least two notes and at least a small pitch span when the
available register permits it.

This is why the result sounds like a repeated idea rather than unrelated random
notes, while still remaining sparse enough not to compete with speech.

### 6. Percussion

Production first chooses one of five restrained grammars: straight, half-time,
gentle syncopation, sparse, or another weighted straight choice. It then writes
three phrase-length binary strings:

- The **low** lane supplies the metrical anchor and always strikes step zero of
  every bar. A second low hit may appear near the middle of the bar.
- The **mid** lane supplies rim, wood, or brush accents on meter-derived
  positions, with an optional gentle syncopation.
- The **high** lane supplies shaker or soft-hat motion, normally on alternating
  eighth-note subdivisions and thinned for sparse/half-time patterns.

In a pattern such as `x...x.......`, `x` means “trigger a hit” and `.` means a
rest. Optional hits may be omitted deterministically. Each lane also has a
per-hit probability and at most 2 ms of deterministic timing variation. The
renderer compensates the drum bus when many short bright hits accumulate.

## Candidate generation and quality selection

One request does not simply return the first random phrase:

1. `auto` chooses one production family using a request-seeded random generator.
2. The engine tries at most `6 × 4 = 24` candidates from that family. Candidate
   `i` uses `(request_seed + i × 104729) mod 2^64` as its BedSpec seed.
3. Each candidate is fully composed, assigned instruments, and rendered as a
   four/eight-bar preview. Resolution targets six accepted candidates and stops
   as soon as it reaches that target. If all 24 attempts are consumed, it can
   still select from a smaller accepted pool, but it fails if none passed.
4. A preview is rejected if it is silent or non-finite, exceeds the swing cap,
   lets percussion exceed 60% of preview RMS, has weak downbeat clarity, or
   falls below the production preference threshold. The last two cutoffs are
   `0.72` for metrical clarity and `0.62` for preference.
5. The preference score rewards nearly straight timing, a low hit on each
   downbeat, restrained drum level, major/Lydian colour, and metrical clarity.
   These are explicit priors distilled from the project's listening tests.
6. Accepted candidates are ranked by quality plus optional distance from recent
   `BedFingerprint`s. The engine takes the top 34% (normally two of six) and
   makes a final seeded choice within that tier. This retains controlled variety
   instead of always returning one maximally scored pattern.

The request seed controls this whole process, but the chosen candidate can have
a later candidate seed. The chosen `BedSpec.seed`, not merely the original
request seed, identifies the actual composition that was rendered.

The preference score is exactly `0.28 × straight + 0.23 × downbeats + 0.20 ×
restrained_drums + 0.12 × positive_scale + 0.17 × clarity`. Here straight falls
linearly from 1 as swing approaches 0.08; restrained drums rewards a level near
0.55; and major/Lydian score 1 while the other scales score 0.55. Ranking adds
`0.20 × novelty`, where novelty is the minimum fingerprint distance from the
optional recent beds.

## Active BedSpec fields

The following fields take part in production resolution or replay. “Compose”
means a value helps create explicit phrase events; “replay” means the renderer or
final mixer reads it after those events exist.

| Group | Active fields | Stage and meaning |
|---|---|---|
| Identity | `seed`, `schema_version`, `engine_version`, `profile_version` | The seed controls deterministic choices and performance variation; versions identify the contract and policy. |
| Timing | `bpm`, `beats_per_bar`, `beat_unit`, `swing` | Compose and replay: define the metrical grid and event timestamps. |
| Harmony | `root`, `scale`, `progression`, `chord_extension` | Compose: create chord, bass, and lead pitches. The progression remains useful to role-specific instrument treatment, but normal replay follows explicit events. |
| Pad | `enabled`, `instrument`, `level`, `cutoff_base`, `cutoff_motion`, `cutoff_curve`, `cutoff_period_bars`, `overlap`, `duck_db` | `overlap` helps compose long chord durations; the other fields select/render/filter the pad and control its speech ducking. |
| Bass | `enabled`, `level`, `attack`, `duck_db` | Replay: enable, shape, balance, and duck bass events. |
| Drums | `enabled`, `level`, `duck_db` | Replay: enable and balance the resolved percussion lanes. |
| Lead | `enabled`, `instrument`, `level`, `register`, `velocity`, `duck_db` | Register and velocity guide motif composition; instrument and level guide selection/rendering; ducking protects speech. |
| Space | `reverb_seconds`, `reverb_mix` | Replay: control convolution-reverb length and wet/dry balance. |
| Phrase identity | `family`, `loop_bars`, `harmony_texture`, `pad_timbre`, `bass_timbre`, `bass_grammar`, `motif_grammar`, `palette` | Family can influence catalog routing; texture, grammars, and palette record how the phrase was resolved. Replay uses loop length and synth timbres. |
| Chord events | `step`, `duration_steps`, `midi_notes`, `velocity`, `articulation` | Replay: exact chord onsets, lengths, pitches, and strength; articulation records the resolved instrument role. |
| Bass/lead events | `step`, `duration_steps`, `midi_note`, `velocity`, `articulation` | Replay: exact monophonic event data; articulation records the resolved instrument role. |
| Percussion lanes | `sound`, `pattern`, `level`, `probability`, `humanize`, `role`, `sample` | Sample assignment uses the low/mid/high role; replay uses the phrase-length pattern, probability, timing variation, gain, and resolved synth or sample. |
| Instruments | `pad_instrument`, `bass_instrument`, `lead_instrument`; zone `sample`, `root_note`, `lo_note`, `hi_note`, `lo_velocity`, `hi_velocity`, `gain_db`, `round_robin`, `articulation` | Replay: stable collection/asset/checksum references and complete pitch, velocity, gain, take, and articulation maps for multisample instruments. |

Schema 3 removed the phrase-less pad, bass, drum, and lead controls, along with
single-sample pitched replay, cyclic phrase-level sample variation, and unused
percussion pan. Event and multisample-zone articulation remain serialized
provenance.

## Worked example: `warm-motion`, request seed 42

This example uses the electronic palette so it requires no sample library:

```python
result = resolve_music(MusicRequest(
    family="warm-motion",
    energy="balanced",
    rhythm="steady",
    palette="electronic",
    seed=42,
))
```

With engine version 1.4.0 and profile `production-v1`, the request resolves to
candidate seed `523687`—candidate index 5, because
`42 + 5 × 104729 = 523687`. Its main choices are:

```text
84 BPM · 4/4 · root MIDI 53 (F3) · F major · seventh
8 bars · open chords · triangle bass synth · root–fifth bass
progression [0, 5, 0, 5, 2, 4, 0, 3] · random-walk motif
```

The F-major scale is F–G–A–B♭–C–D–E. Zero-based progression degrees
`[0, 5, 0, 5, 2, 4, 0, 3]` therefore start from F, D, F, D, A, C, F, and B♭.
The saved open voicings are:

| Bar | Degree | Saved voicing (duplicate notes are doubled voices) |
|---:|---:|---|
| 1 | 0 | F3–C4 |
| 2 | 5 | D3–F3–A3–D4 |
| 3 | 0 | C3–F3–A3–C4 |
| 4 | 5 | D3–A3 |
| 5 | 2 | A2–E3–C4–E4 |
| 6 | 4 | C3–E3–G3–C4 |
| 7 | 0 | C3–F3–A3–C4 |
| 8 | 3 | B♭2–F3–D4–F4 |

There are 16 steps in each 4/4 bar. Open texture places one overlapping chord
at each bar boundary. The root–fifth bass places two notes per bar, and the
random-walk lead stores 19 exact note events. Each percussion cell below repeats
for all eight bars, producing three 128-step saved lanes:

```text
low:  x.......x.......
mid:  ....x.......x...
high: ..x...x...x...x.
```

## Instruments and where they come from

### Procedural instruments

The electronic palette is entirely local mathematics:

- Pads use sine-plus-harmonic, triangle, or softened saw waves, a slow
  sine-shaped amplitude envelope, and moving low-pass filtering.
- Bass uses sine, round sine-plus-harmonic, triangle, or decaying pluck waves.
- The synth lead is an additive bell made from a fundamental and two quieter
  harmonics with a fast attack and exponential decay.
- Drums synthesize a falling-pitch kick, filtered-noise rim/brush/shaker/hat,
  and a short two-oscillator wood sound.

### Recorded instruments

The repository ships a checksum-locked production bundle. Normal generation
never downloads samples implicitly. The current bundle contains 1,112 catalog
assets that form 27 playable multisample banks, plus 95 files in four explicit
named packs:

| Source | Current material | License |
|---|---|---|
| Salamander Grand Piano | 60 Yamaha C5 files: 30 root notes in two selected velocity layers | CC BY 3.0 |
| VSCO 2 Community Edition named packs | 22 violin-section sustain files, 7 marimba files, 6 glockenspiel files | CC0 1.0 |
| Versilian Community Sample Library (VCSL) | Pianos, recorders, ocarina, organ, harp, harpsichord, FM pianos, kalimba/mbira, glockenspiel, marimba, vibraphone, xylophone, and percussion | CC0 1.0 |
| VSCO 2 catalog | Marimba, pizzicato solo violin, and percussion | CC0 1.0 |
| FreePats Spanish Classical Guitar | 48 pitched guitar recordings | CC0 1.0 |
| FreePats World Percussion | 30 world-percussion recordings | CC0 1.0 |
| Karoryfer Fashionbass | 55 natural electric-bass recordings | CC0 1.0 |
| Stargate Sample Pack | 29 production percussion recordings | CC0/public domain |

See [`NOTICE.md`](../NOTICE.md) and the
[bundle README](../assets/production-core/README.md) for attribution and source
details.

A multisample `InstrumentRef` stores logical sample identities rather than local
filesystem paths. Each zone declares its original MIDI note, playable note
range, velocity range, gain, articulation, and alternate-take index. At replay,
the renderer chooses the nearest matching zone and resamples only by the required
semitone distance. This is more convincing and reproducible than stretching one
recording across a whole keyboard.

For hybrid/acoustic candidates, sample assignment is itself seeded. Percussion
is chosen separately for low, mid, and high roles. Compatible foreground banks
are selected by the abstract lead role (piano, mallet, bell/harp, or a broader
organic set). Sustained strings may replace a pad and Fashionbass may replace
the synthesized bass. Acoustic requests make these substitutions more likely
than hybrid requests; electronic requests clear every catalog reference.

Every selected sample is also returned in `sample_manifest` with its collection,
logical ID, checksum, license, attribution, and original relative path.

## Rendering and mixing with speech

The phrase repeats until enough whole bars cover the requested duration. Four
mono stems are rendered at 44.1 kHz:

1. **Pad:** chords or arpeggios, amplitude envelope, moving low-pass filter,
   then reverb.
2. **Bass:** synthesized or sampled note events, attack/release shaping, then a
   low-pass filter.
3. **Drums:** synthesized or resolved one-shots with deterministic probability,
   timing, velocity, and density compensation.
4. **Lead:** synthesized or multisampled motif events, then a slightly shorter
   reverb.

Each stem is widened to stereo with a 12 ms right-channel Haas delay while
retaining 15% of the undelayed signal, so the downbeat itself remains clear. A
two-bar-or-shorter fade prevents clicks at the beginning and end. Music-only
`render_bed()` sums the stems and normalizes its peak to 0.7.

Lessons mix the unnormalized stems differently. The combined music is normalized
to -26 LUFS and speech to -16 LUFS by default. A speech envelope ducks each stem
by its configured depth, using roughly 100 ms attack and 400 ms release. Pads
and leads therefore move farther behind speech than bass or restrained drums
when their BedSpec requests it. The combined output is normalized to -16 LUFS
and passed through a look-ahead-style block limiter with a peak ceiling of 0.97.

## Relationship to research

These papers illuminate parts of the design, but they should not be read as a
claim that LexiBeat implements each complete theory. Repository history confirms
the code correspondences described below, but does not establish any of these
papers as direct implementation provenance; the research mappings are therefore
conceptual parallels.

### The Distance Geometry of Music

Demaine et al., [*The Distance Geometry of
Music*](https://arxiv.org/abs/0705.4085), represents cyclic rhythms as fixed
pulse lattices whose onsets are binary values. That maps directly to LexiBeat's
phrase-length `x`/`.` percussion strings and repeated metric grid.

The paper's central result is stronger: Euclidean rhythms distribute a chosen
number of onsets as evenly as possible on a circle and have additional
properties involving deepness and shelling. LexiBeat's current production
percussion does **not** implement those results. Listening work moved production
to explicit meter-aware anchors, and the unused Euclidean experiment helper was
removed. Production does not compute geometric evenness or deepness, shell
rhythms, derive scales with the Euclidean algorithm, or imitate the traditional
world timelines catalogued in the paper.

The accurate mapping is therefore: shared cyclic representation and historical
rhythm experiment, not an active implementation of the paper's full theory.

### Voice leading as distance

Tymoczko's [*The Geometry of Musical
Chords*](https://doi.org/10.1126/science.1126287) represents chords as points and
voice leadings as paths between them. [*Scale Theory, Serial Theory and Voice
Leading*](https://doi.org/10.1111/j.1468-2249.2008.00257.x) develops algorithms
for efficient voice leading more fully. LexiBeat shares the practical idea that
shorter per-voice motion generally connects chords more smoothly.

`_smooth_voicing()` is much narrower than Tymoczko's theory. It searches a
small, fixed set of inversions and octave shifts, preserves an ordered matching
between at most five voices, and minimizes an L1 sum of semitone distances. It
does not construct an orbifold, search all voice permutations, or model
independent contrapuntal lines.

### Algorithmic generation and structure

Herremans, Chuan, and Chew's [*A Functional Taxonomy of Music Generation
Systems*](https://arxiv.org/abs/1812.04186) provides useful vocabulary for
placing LexiBeat: it is a rule-based, stochastic, generate-and-test system built
for a specific functional role—controllable accompaniment beneath speech—not a
general autonomous composer.

Herremans and Chew's [*MorpheuS: generating structured music with constrained
patterns and tension*](https://arxiv.org/abs/1812.04832) addresses the broader
problem of long-term musical coherence with repeated patterns, tonal tension,
and optimization. LexiBeat takes a deliberately simpler approach suitable for
language lessons: it explicitly writes one coherent four/eight-bar phrase and
repeats it. The comparison is conceptual; the project does not implement the
MorpheuS optimization method or tension model.

## Limitations and design boundaries

- The system is explainable and deterministic because it is rule-based, but its
  rules cover a deliberately narrow musical world.
- The harmonic transition table and chord formula are product heuristics, not a
  complete account of classical, jazz, modal, or non-Western harmony.
- Repeating a short phrase creates coherence but not verse/chorus form,
  modulation, thematic development, or an evolving arrangement.
- Pentatonic-like lead notes reduce clashes; they do not guarantee that every
  melodic note has a functional relationship to the current chord.
- The quality gate catches silence, invalid audio, weak pulse, excessive swing,
  and percussion dominance. Its score encodes listening preferences; it cannot
  prove that a listener will find a bed beautiful or original.
- The central product constraint remains speech intelligibility. Musical ideas
  that are compelling in isolation can still be rejected when they compete with
  bilingual speech.
