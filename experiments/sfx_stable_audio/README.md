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

## Licence

Output is covered by the Stability AI Community License (commercial use above a revenue threshold
needs a Stability licence), and the text encoder by the Gemma Terms of Use. Unlike the CC0 sources in
the plan, these sounds are not candidates for the checksum-locked bundle until the licence has been
reviewed for that use.
