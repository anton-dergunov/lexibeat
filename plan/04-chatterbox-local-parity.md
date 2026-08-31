# Step 4: Chatterbox Local Parity and Runtime Evaluation

## Outcome

Determine whether LexiBeat can reproduce the hosted Chatterbox Multilingual V3
quality locally, and separate differences caused by model weights, inference
runtime, conditioning audio, and sampling controls. The result should identify a
preferred local backend for an Apple M1 MacBook Air with 16 GB of unified memory
without weakening the stable hosted ZeroGPU path.

This is an investigation plan, not an instruction to replace the existing MLX
backend before comparative renders have been reviewed.

## Current runtime differences

The hosted Space uses the official PyTorch implementation:

- `ChatterboxMultilingualTTS.from_pretrained("cuda", t3_model="v3")`;
- the official V3 T3 checkpoint and acoustic decoder on CUDA;
- the checkpoint's built-in `conds.pt` voice conditioning;
- no reference upload or language-specific reference override.

The local backend uses the MLX conversion:

- `mlx-community/chatterbox-multilingual-v3` through MLX-Audio;
- converted V3 weights and the separately downloaded S3TokenizerV2;
- an explicit reference clip for every request;
- references currently synthesized by macOS `say`, using Paulina for Spanish
  and Daniel for English.

The model family is therefore substantially the same, but the conditioning is
not. Synthetic reference voices may transfer less-natural cadence or artifacts,
so CUDA versus MLX must not be assumed to be the primary quality difference.

Primary references:

- [Official Chatterbox repository](https://github.com/resemble-ai/chatterbox)
- [Official Chatterbox model card](https://huggingface.co/ResembleAI/chatterbox)
- [MLX Multilingual V3 model card](https://huggingface.co/mlx-community/chatterbox-multilingual-v3)
- [MLX-Audio Chatterbox documentation](https://github.com/Blaizzy/mlx-audio/blob/main/docs/models/tts/chatterbox.md)

## Experiment 1: establish controlled inputs

Create a small, versioned evaluation manifest containing:

- short Spanish and English words;
- multiword phrases of increasing length;
- questions and exclamations;
- difficult names and consonant clusters;
- three fixed voice seeds per text;
- the same CFG, temperature, and exaggeration values for every runtime.

Save raw model output before LexiBeat trimming, fitting, resampling, mixing, or
limiting. Also save the post-processed lesson clip so model behavior can be
distinguished from arrangement behavior.

Record for every take:

- checkpoint and runtime revision;
- conditioning source and checksum;
- seed and generation controls;
- model-load time, generation time, real-time factor, and output duration;
- peak process memory and MLX/MPS allocation where available;
- whether the model repeated, omitted, or added words;
- listening notes for pronunciation, speaker consistency, rhythm, and
  naturalness.

## Experiment 2: isolate conditioning quality in MLX

Compare the existing MLX runtime with at least three conditioning sets:

1. the current synthetic Paulina/Daniel references;
2. clean human recordings in the matching language and accent;
3. a converted equivalent of the official built-in conditions, if the current
   MLX conversion tooling can preserve them correctly.

Keep weights and sampling settings fixed. If human or official-equivalent
conditioning closes the quality gap, retain MLX as the efficient local runtime
and replace only the default references. Keep Spanish and English references
separate unless a shared voice is deliberately requested.

## Experiment 3: run the official model on Apple MPS

Use an isolated environment so the official PyTorch/NumPy dependency set does
not alter LexiBeat's existing MLX environment. Pin the same official source
revision used by the Space and load:

```python
ChatterboxMultilingualTTS.from_pretrained("mps", t3_model="v3")
```

First use the built-in conditions and exactly match the hosted controls. Then
repeat with the same human references used in the MLX comparison. If an MPS
operation is unsupported, record the failing operation and test CPU fallback
separately rather than silently mixing runtimes in the benchmark.

The M1/16 GB machine is likely capable of loading the roughly 0.5B-parameter
model, but PyTorch MPS may be slower and use more unified memory than MLX. Treat
successful loading, sustained memory pressure, and repeatable multi-utterance
generation as separate acceptance criteria.

## Experiment 4: compare another CUDA resource

Run the same official script and manifest on an ordinary NVIDIA CUDA machine,
not only ZeroGPU. This distinguishes model/runtime behavior from Space queueing,
cold starts, and GPU virtualization. Use the same checkpoint revision and built-
in conditions as the hosted callback.

## Decision criteria

Prefer the smallest runtime that meets all of these conditions:

- no material rate of added or repeated words across the evaluation manifest;
- Spanish and English pronunciation acceptable to native review;
- consistent voice identity within each language;
- natural variation across repeats without unstable exaggeration;
- reliable completion of a six-pair lesson on the M1/16 GB machine;
- acceptable cold-load time, real-time factor, and memory pressure;
- reproducible model, reference, and configuration provenance.

Possible outcomes are:

- keep MLX and upgrade its reference conditioning;
- add an optional official PyTorch-MPS backend for highest local parity;
- reserve the official model for CUDA and use MLX as the practical laptop
  approximation;
- offer both local backends behind stable logical model values after their
  differences are documented and tested.

## Implementation boundary

Do not add the official PyTorch package to the ordinary `local-tts` extra until
the isolated evaluation succeeds. If adopted, give it a separate optional extra
and cache namespace, retain the current language-specific MLX route, and add a
one-pair integration render for both backends.
