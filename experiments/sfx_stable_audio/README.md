# Word sounds from Stable Audio 3 Small SFX

The programme plan (P4) wants a short, precise sound for a few iconic words, such as a cat for
"gato" or a siren for "sirena", and no sound for most words. The CC0 catalogues are noisy, so this
experiment asks whether [`stabilityai/stable-audio-3-small-sfx`](https://huggingface.co/stabilityai/stable-audio-3-small-sfx)
can generate those sounds on a laptop, how fast, and in how much memory.

## Running it

The model is gated: accept its licence on the model page once, and have a Hugging Face token
available (`hf auth login`). Then:

```bash
cd experiments/sfx_stable_audio
uv sync                                         # its own environment, ~0.5 GB, torch 2.7.1
uv run python generate.py                       # MPS, fp32 (the library's own choice off CUDA)
uv run python generate.py --half                # MPS, fp16
uv run python generate.py --device cpu
open out/mps/index.html                         # listen
```

The first run downloads 3.45 GB of weights into the Hugging Face cache (the DiT and autoencoder, and
the T5Gemma text encoder). Words, prompts, durations and seeds are in [`prompts.json`](prompts.json).
Each run writes `out/<device>/NN_word.wav` (44.1 kHz stereo, peak −1 dBFS), `report.json` and
`index.html`.

## Results

Apple M1, 16 GB, macOS 26.5, torch 2.7.1, 8 steps, the ten prompts in `prompts.json` (2–4 s each).

| | MPS fp32 | MPS fp16 | CPU fp32 |
|---|---|---|---|
| Load (weights already cached) | 9.2 s | 8.7 s | 7.8 s |
| First generation (warm-up) | 3.1 s | 2.7 s | 4.2 s |
| Per sound, mean | **2.03 s** | **1.56 s** | 3.97 s |
| Process peak RSS (during load) | 4.0 GB | 4.1 GB | 4.1 GB |
| Process RSS while generating | 0.3–0.5 GB | 0.4–0.6 GB | 2.9–3.0 GB |
| GPU memory (MPS driver, peak) | 3.1 GB | 2.1 GB | — |

- **Generation time barely depends on length.** A 2 s and a 4 s sound both take about 2 s on MPS,
  because the library adds 6 s of headroom to every latent. At these lengths the cost is per sound,
  not per second.
- **fp16 is safe here.** For the same seed it is about 23% faster and uses a third less GPU memory,
  and its spectra match fp32's closely (log-spectrum correlation ≥ 0.98 on every clip).
- **CPU timings are noisy.** An earlier CPU run of the same prompts averaged 7.0 s. MPS was steady
  across runs.
- In steady state the whole cost is about 3.5 GB (fp32) or 2.6 GB (fp16) of unified memory. Loading
  briefly needs about 4 GB, because the weights pass through the CPU first.

## Findings

- **Ask for whole seconds.** The same prompt and seed produces broadband noise at 2.4, 2.5 or
  3.5 s, and the sound at 2.0, 3.0 or 2.9999 s. That held for every prompt tried, and it is why the
  first cat, dog and keys clips were noise. `generate.py` now refuses a fractional duration. To get
  a shorter sound, generate whole seconds and trim.
- **The model ends a sound on its own.** A 4 s request for a train or a siren fills about 3.3 s and
  then goes silent, so the durations are ceilings rather than exact lengths.
- **Prompt adherence is good, and counts are loose.** "A dog barking twice" and "two loud barks"
  gave anywhere from two to six barks. Short prompts ("Dog bark.") were as affected by fractional
  durations as long ones. Descriptive prompts in the library's SFX style (source, action, recording
  character) worked.

## Steering and range

Two follow-up rounds, rendered side by side for listening with fp32 on MPS:

```bash
uv run python compare.py rounds/steer.json      # five sounds, original vs three directions
uv run python compare.py rounds/expand.json     # twenty less common words
open out/steer/index.html out/expand/index.html
```

A round is a JSON grid: words by prompt variants, each rendered in two takes. Every clip is matched
to −20 LUFS, so a comparison hears character rather than level. The page shows each clip's spectral
centroid, a rough measure of how bright it is.

**Steering** (cat, dog, door, train, siren). Each keeps its original prompt next to three versions:
soft and distant, warm lo-fi, and a character such as a sleepy cat, a puppy, a haunted door or a toy
siren.

- **By ear, steering hardly works on this model.** The measurements moved: at equal loudness the
  cat's spectral centroid fell from 2.2 kHz to 0.6–1.1 kHz. But the listener heard something else.
  Where the source is unmistakable (the siren, the train), the steered versions sound much like the
  original. Where it is not, the prompt changes *what* the sound is rather than its character: the
  cat variants sound like different cats, and the haunted door became a spacious, reverberant scene
  in a hall rather than a door. Steering towards "sits inside the music" needs a larger model, or
  processing after generation.
- **Put the source and its action first.** "An old wooden door gently creaking open … heard from the
  next room" came out as knocks and thumps. "Door hinge creaking slowly as an old wooden door opens,
  soft and gentle …" kept the creak.
- **"lo-fi" and "tape texture" can bring in crackle** like a vinyl record's (it did on the door).

**Range.** Twenty words from a real vocabulary. The listener's verdict:

| Verdict | Words |
|---|---|
| Good | woodpecker, frog, wolf, wild parrots, storm, uncorking, zipper, sweeping (surprisingly), kettle, yawn (realistic), snoring, braking |
| Passable | bonfire (a little like fire), snake (a snake, or a rope being dragged), frying pan (noise, but not the worst), padlock (a little strange) |
| Wrong | bee (a musical instrument), nightingale (a person whistling), cliff (not waves), stream |

That is 12 of 20 good and 16 of 20 usable: roughly 70–80%, with no way to tell in advance which
word will fail. The clips also sound limited in dynamic range. The two that fail most clearly were
the ones whose spectrograms looked most convincing (a harmonic buzz, a tonal song), so **a
spectrogram shows a sound's shape, not what it is: only listening decides.**

- **Takes vary little.** Two seeds 1000 apart usually give the same gesture: the same croak rhythm,
  the same broom strokes. For variety, change the prompt, not the seed.

## Against the catalogues

```bash
uv run python catalogue.py                      # needs a Freesound API key; see its docstring
open out/catalogue/index.html
```

All 30 words: each generated clip next to three clips from each of three open sources, taken by the
source's own ranking, not picked by ear.

| Source | Words covered | How the clips are chosen |
|---|---|---|
| ESC-50 | 16 of 30 | The first clips of the matching class, one per source recording. CC BY-NC (ESC-10: CC BY). |
| FSD50K | 21 of 30 | CC0 clips with the label, every rater calling it the predominant sound, fewest other labels first. |
| Freesound search | 30 of 30 | The first CC0 results of a plain query, at most 30 s long. |

The downloads stay small: ESC-50 one file at a time, FSD50K only its 7 MB of labels and metadata,
and Freesound's MP3 previews: about 110 MB in all, under `~/.cache/lexibeat/catalogue-sfx`.

- **FSD50K puts numbers on "noisy".** Of the clips people labelled "door", 85 of 191 are CC0 and only
  13 have a door as the predominant sound; for a siren 132 → 69 → 15, for a frog 110 → 37 → 7. A
  label says the sound is in the clip somewhere, not that the clip is that sound.

### What the listener heard

Per word, which was better: the generated clip or the best catalogue clip.

| Better | Words |
|---|---|
| Catalogue (11) | cat, dog (Freesound); train (Freesound: traditional, if a little intense; the others were a quiet high-speed train or wind); rain (ESC-50, FSD50K; Freesound was abstract, rain on an object); siren (Freesound; an ESC-50 clip is an air-raid siren, with a bad connotation); clapping (ESC-50, FSD50K; Freesound's sounded like TV, from one uploader); bee (ESC-50, FSD50K, far better than the model's); nightingale (every source; the model's is a person whistling); snake (weakly: one Freesound clip); uncorking (slightly); snoring (slightly more realistic) |
| Model (9) | wolf (Freesound's were synth notes and a siren); wild parrot (Freesound's were random birds); frying pan (no source was good); braking (the sources were bad); woodpecker (only one Freesound clip was right); yawn; storm, zipper, sweeping (slightly) |
| Both good (5) | clock, sneezing, frog (the Freesound frogs came with a whole pond around them), stream, bonfire |
| Neither (2) | cliff, padlock |
| Not compared (3) | keys (the catalogue clips were nice), door (usable, but the heavy doors sound funny if heard the wrong way), kettle (the model's was liked) |

The catalogues win the common, well-recorded sounds. The model wins where the catalogues are thin or
polluted.

- **Catalogue clips have better dynamic range** and more realism than the model's.
- **The catalogues' worst failure is human imitation.** People make the sound with their mouth: a
  bee, a snake, a cork. Next come ambience around the sound (a frog in a whole pond), synth or
  electronic stand-ins (the wolves), and a label on the wrong thing: FSD50K's "Hiss" returned a
  frying pan for "snake".
- **Connotation matters as well as identity.** The air-raid siren and the funny-sounding doors were
  right and still unusable. Curation has to judge what a sound suggests, not only what it is.
- **Lengths and edges vary.** Clips run up to 10 s and often start or end in silence. Before a sound
  can be laid over music it needs trimming, levelling and a fade (what the demo below does).
- **Freesound search vs FSD50K:** plain search was best for cat, dog, train and siren. It was worst
  where uploaders imitate or add ambience, which the FSD50K predominance filter mostly avoids.

## Sounds in a loop

```bash
uv run python experiments/sfx_stable_audio/demo_mix.py   # from the repository root
open experiments/sfx_stable_audio/out/demo/index.html
```

It needs the README demo's speech and timeline in `out/readme-demo/`, which
`python -m scripts.demos.generate_readme_demo` writes.

The README demo's 13 words over a freshly rendered "acoustic" bed, with a sound before 8 of them:
far more than the product would use, so there is plenty to judge. The sounds are clips the listener
rated well: thunder, wolf howl and yawn from the model; a clock, cork pop, applause, train and
nightingale from the catalogues. Each is trimmed of silence, starts on a beat and ends just before
its word, in the gap after the previous word. It sits 5–8 dB under the voice, and the bed ducks under
it. **Dry** is the sound as it is; **glued** adds a band filter, light compression and a little room,
to test whether processing gets the "inside the music" feel that prompt steering could not. This
script, unlike the rest, runs in the repository's own environment, because it mixes with LexiBeat
itself.

## What this means for the product

**Aim for high precision, low recall.** A sound should be exactly its word, or absent. Neither
source does that unattended. The model gets roughly 70–80% of words right, and the catalogues' first
results fail in the ways listed above. So sounds come from a **curated library**, not from live
generation or live search.

**How big the library needs to be.** In one real Spanish vocabulary of 919 entries, about 83 (9%)
have an obvious sound and another 40 or so a plausible one: a rough count by reading. Many share a
sound ("golpe", "golpear" and "trompada" are all a punch; "hoguera" and "encender el fuego" are both a
fire), leaving about 65 distinct sounds. Soundable concepts grow much more slowly than vocabulary:
AudioSet's ontology names 527 sound classes in total. **300–500 concepts with two or three approved
clips each (1,000–1,500 clips) would cover nearly every soundable word** a learner is likely to have.
Two weeks at about 100 approvals a day reaches that. Spend the effort on concepts in order of how
often they come up, and on a second or third clip per concept for variety, rather than on breadth.

**Make approval cheap:**

1. **Candidates.** For each concept, gather about six: two generated, two from FSD50K's
   predominant-and-CC0 filter, two from Freesound search.
2. **Automatic pre-filter,** so the listener only hears plausible ones:
   - **mechanical checks** (length, silence, clipping), which this experiment shows do work;
   - **an audio tagger** trained on AudioSet (PANNs, AST or BEATs; each runs locally) to veto clips
     whose top labels include speech, human voice, whistling, music or a musical instrument. That
     rule targets the actual failures heard here: the imitations, the whistling nightingale, the
     bee that sounds like an instrument;
   - **an audio-text model** (LAION-CLAP or Microsoft CLAP) to rank the rest by how well they match
     the concept's text, against the other concepts as distractors;
   - optionally, **an audio-understanding LLM** (Gemini takes audio input) to answer "is this a real
     X or an imitation, and does it suggest anything alarming?". That is the connotation check a
     classifier cannot do.
3. **Approve on a tablet:** one page per concept, tap to keep. Store the clip trimmed, levelled and
   faded, with its source and licence.

**Calibrate the pre-filter before trusting it.** The catalogue page's 231 clips, 30 words' worth,
now carry the listener's verdicts, many of them per clip. Scoring those clips with the tagger and CLAP shows at once whether
the scores separate good from bad, before anything is built on them. That is the obvious next
experiment, and it is cheap. Likelihood under a generative model (the "perplexity" idea) is harder
to compute for a diffusion model, and it measures typicality rather than correctness.

**Licences, for a paid product** (a summary, not legal advice):

| Licence | Ship it in a paid product? |
|---|---|
| CC0 | Yes. No attribution needed. |
| CC BY | Yes, with credit to each author, the licence and a link, for example on a credits screen. |
| CC BY-NC | No. "NonCommercial" rules out a paid product. That is most of ESC-50, so ESC-50 is a reference, not a source. |
| Sampling+ | Transformed use only. Avoid. |
| Stability AI Community License (the model's output) | Free for commercial use below a revenue threshold (US$1M a year at the time of writing); an enterprise licence above it. Check the current terms. |

Two caveats:

- **Freesound's API terms reserve commercial use of the API itself** for case-by-case agreement
  with its operator, separately from each sound's licence. A one-off, curated download for a shipped
  pack is worth confirming with them first.
- **Previews are lossy MP3s.** A shipped pack should use the original files.

Having an online model help curate raises no licence issue for CC0 or CC BY clips: that is use, not
redistribution.

## Next

- **Calibrate an automatic pre-filter** (an AudioSet tagger and CLAP) on the catalogue page's clips,
  which now have the listener's verdicts.
- **A larger model, for steering.** Stable Audio 3 Medium (1.4B parameters, against small's 0.4B)
  is released but documented as CUDA-only; whether it runs on MPS, in about 6 GB of fp32 weights on
  a 16 GB machine, is untested. Large is available only through Stability's API.

## Licence

Output is covered by the Stability AI Community License (commercial use above a revenue threshold
needs a Stability licence), and the text encoder by the Gemma Terms of Use. Unlike the CC0 sources in
the plan, these sounds are not candidates for the checksum-locked bundle until the licence has been
reviewed for that use.
