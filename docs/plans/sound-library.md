# A curated sound library

The programme plan (`programme-loops.md`, P4) wants a short, real sound for a few iconic words: a
cat for "gato", a kettle for "la pava", thunder before "¡Qué susto!". The experiment in
`experiments/sfx_stable_audio/` tried the two ways of getting them, generating them and finding them
in open catalogues, and mixed some into a real loop. This plan turns what it found into a way of
building the sounds: a small, curated, checksum-locked library, approved by ear, from which a loop
draws an occasional treat. It is a plan for later, not a specification.

## What the experiment settled

- **In a loop, a sound works.** Starting on a beat and ending just before its word, it integrated
  well, and the moment was right. At first it was too loud. It belongs about 11–15 dB under the
  voice (each clip at −30 LUFS), where it is a nuance rather than an interruption.
- **Neither source is precise unattended.** The local model (Stable Audio 3 Small SFX) got roughly
  70–80% of words usable, and there is no telling in advance which will fail: its bee is a musical
  instrument, its nightingale a person whistling. Of 30 words, the catalogue's best clip beat the
  model's on 11, the model won on 9, and both were good on 5.
- **Catalogues sound better when they are right:** more dynamic range, more realism. Their typical
  failure is a person imitating the sound with their mouth. Next come ambience around the sound, a
  synth stand-in, and a label on the wrong thing.
- **Connotation matters as much as identity.** An air-raid siren is a siren, and it is still wrong.
- **Prompt steering barely works on the small model.** It changes *what* a sound is more than its
  character. Softening belongs in the mix, or awaits a larger model.
- **A spectrogram cannot say what a sound is.** Only listening can, so a person approves every clip.
- **Raw clips are uneven:** lengths up to 10 s, and silence at either end. Every clip is trimmed,
  levelled and faded before it is stored.

So: **high precision, low recall.** A sound is exactly its word, or it is absent. Most words have
none.

## The library

**Concepts, not words.** A concept is one sound: "cat meow", "punch", "kettle whistle". Many words
map to one ("golpe", "golpear" and "trompada" are all a punch; "hoguera" and "encender el fuego" are
both a fire). A concept records:

- an id and a short English label;
- a one-line description, which serves as both search query and generation prompt;
- its FSD50K / AudioSet labels, where it has them;
- a note on connotation to avoid ("a police siren, never an air-raid siren").

**Size.** In one real 919-entry vocabulary, about 9% of entries have an obvious sound, which comes to
about 65 distinct concepts. Soundable concepts grow far more slowly than vocabulary (AudioSet names
527 sound classes in all). **300–500 concepts with two or three approved clips each (1,000–1,500
clips) would cover nearly any learner's soundable words.** Start with the ~65 from the owner's own
vocabulary, then go on in order of how often concepts come up. A second and third clip per concept
buys more than breadth: the same word can then sound different in two loops.

**A clip** records its concept, its file, its source (dataset and id, or the model and seed), its
licence and author (for credits), its length and loudness, and who approved it and when.

## Building it

| Stage | What happens | Tooling |
|---|---|---|
| 1. Concepts | List the concepts and map words to them. The writer backend can propose a concept for a word; a person prunes. | A JSON concept list in the repository |
| 2. Candidates | About eight per concept: two from FSD50K (CC0, every rater calling the sound predominant), two from Freesound search (CC0, later CC BY), and two to four generated | `catalogue.py` and `compare.py` from the experiment, promoted to `scripts/sounds/` |
| 3. Normalize | Trim silence, cap at 4 s (a gap before a word is about 4.5 s at 78 BPM), fade in and out, level to −30 LUFS, 44.1 kHz stereo. Generate in whole seconds: fractional durations give noise. | The experiment's `demo_mix.py` trim and fade |
| 4. Pre-filter | Drop what a person would reject anyway, so the listener hears only plausible clips (below) | A local tagger and CLAP; optionally Gemini |
| 5. Approve | One page per concept on a tablet, tap to keep. About 100 decisions a day; two weeks covers the first 1,500 clips. | `scripts/listening/serve.py`, which already serves a round to a tablet on the home network and saves every verdict to `labels.json` |
| 6. Publish | A checksum-locked pack, like the sample bundle: Git LFS, sources and credits in `NOTICE.md` | `lexibeat/samples.py`, `lexibeat/bundle.py` |

The curation tools read their keys (Freesound, Gemini) from the owner's own config and live in
`scripts/`. None of it touches the service or the render, which hold no provider credential.

### The automatic pre-filter

It rejects, and never approves. Each step targets a failure the experiment actually heard:

- **Mechanical checks:** length, silence, clipping, finite samples. These are the checks that did
  work.
- **An AudioSet tagger** (PANNs, AST or BEATs, all runnable locally) vetoes a clip whose top labels
  include speech, human voice, whistling, music or a musical instrument. That catches the mouth
  imitations, the whistling nightingale and the instrument-like bee. It also checks that the
  concept's own label scores.
- **CLAP** (LAION-CLAP or Microsoft CLAP) ranks the survivors by how well they match the concept's
  description, with the other concepts as distractors. The listener hears the best first.
- **Optionally, Gemini:** "Is this a real X or an imitation? Does it suggest anything alarming?"
  That is the connotation check a classifier cannot make. It is cheap: audio counts as about 32
  tokens a second, so 500 concepts × 8 candidates × 4 s is roughly half a million audio tokens, about
  a dollar at Flash-tier prices at the time of writing. Check current prices before relying on this.

**Calibrate before trusting it.** The experiment's catalogue page has 231 clips over 30 words, most of
them now with a listener's verdict. Score them with the tagger and CLAP first: if the scores separate
good from bad, the filter earns its place; if not, it is dropped, and the listener hears everything.

## In the loop

- **Where:** in the gap after the previous word, starting on a beat, at most six beats early, and
  ending a quarter of a second before the word's first utterance.
- **How loud:** each clip at −30 LUFS, about 11–15 dB under the voice. The bed ducks under it by
  about 4 dB, as it does under speech.
- **How often:** at most about one word in ten. A sound is a treat, not a rule.
- **How it sounds:** dry or lightly "glued" (a band filter, light compression, a little room). The
  demo renders both; which one is not yet decided.
- **In the data:** the programme script's `sfx` field names a concept; the arrangement picks one of
  its approved clips, deterministically from the loop's seed; and the cue timeline for the host gains
  an `sfx` cue.

## Licences

A summary, not legal advice:

| Source | Ship in a paid product? |
|---|---|
| CC0 (most FSD50K and Freesound picks) | Yes, with no attribution needed |
| CC BY | Yes, with credits (author, licence, link) on a credits screen and in `NOTICE.md` |
| CC BY-NC (most of ESC-50) | No: a reference only |
| Stable Audio 3 (the local model) | Free for commercial use under a revenue threshold (US$1M a year at the time of writing); check the current Community License |
| Hosted generators | Usually commercial on paid plans only; free tiers are for evaluating |

- **Freesound's API terms reserve commercial use of the API itself** for agreement with its operator
  (the MTG at UPF). Confirm before building a shipped pack through it.
- **A pack uses original files, not previews:** the previews are lossy MP3s.

## Generating better sounds

- **Stable Audio 3 Medium** (1.4B parameters, against small's 0.4B) might steer, and might fail less.
  It is documented as CUDA-only; whether it runs on MPS in about 6 GB of fp32 weights on a 16 GB
  machine is untested, and is the first thing to try.
- **Hosted frontier generators**, for comparison. Each is worth one blind round against the local
  model and the catalogues on the same concepts:
  - **ElevenLabs Sound Effects:** the free tier gives 10,000 credits a month, and a sound costs 200,
    so about 50 sounds a month.
  - **Stable Audio 2.5 on stableaudio.com:** the free tier allows 20 generations a month and is
    non-commercial. Stable Audio 3 Large is available only through Stability's API.
  - **Model hosts such as fal.ai** serve several sound-effect models, paid per use.

## Risks

- **Curation time.** 1,500 approvals is two weeks of evenings. The pre-filter and a sensible order
  (the most frequent concepts first) keep it there, and the library is useful long before it is
  finished.
- **A wrong sound is worse than none.** It teaches the wrong association and breaks the spell. That
  is why a person approves every clip.
- **Connotation:** alarms, gunfire, screams and anything else distressing. Concepts carry an
  avoid-note, and the approver is asked about connotation, not only identity.
- **Repetition:** one clip heard in every loop. Keep two or three clips per concept, and choose
  among them by seed.

## Roadmap

| Phase | What | First concrete step |
|---|---|---|
| **S1** | Calibrate the pre-filter | Score the experiment's 231 judged clips with an AudioSet tagger and CLAP, and see whether the scores separate good from bad |
| **S2** | Try larger models | Run Stable Audio 3 Medium on MPS; one blind round with a hosted generator on its free tier |
| **S3** | The first 65 concepts | The owner's own vocabulary: candidates, pre-filter, a tablet round per batch |
| **S4** | The pack and its use in a loop | A checksum-locked pack published like the sample bundle; the `sfx` field, the placement rule and the `sfx` cue |
| **S5** | Grow to 300–500 concepts | By frequency, at about 100 approvals a day |
