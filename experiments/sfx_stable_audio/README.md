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
- **First listening (cat only):** of the three clips, two were good and one was several cats at once.
- **The listener's view so far:** availability will be spotty, and community uploads vary in what
  they contain, so a catalogue is hard to keep consistent. A model promises more consistency. Not
  every word needs a sound, so gaps in coverage are acceptable either way.

## Next

- **A larger model, for steering.** Stable Audio 3 Medium (1.4B parameters, against small's 0.4B)
  is released but documented as CUDA-only; whether it runs on MPS, in about 6 GB of fp32 weights on
  a 16 GB machine, is untested. Large is available only through Stability's API.

## Licence

Output is covered by the Stability AI Community License (commercial use above a revenue threshold
needs a Stability licence), and the text encoder by the Gemma Terms of Use. Unlike the CC0 sources in
the plan, these sounds are not candidates for the checksum-locked bundle until the licence has been
reviewed for that use.
