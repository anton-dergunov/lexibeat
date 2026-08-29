# Earworms Generator

Earworms Generator is a small experimental project for making language-learning
audio: a word or phrase is spoken in Spanish, followed by its English meaning,
over a calm procedural music bed. Speech lands on a known beat grid, repeats use
slightly different delivery, and the music ducks gently while either speaker is
talking.

The preferred voice backend is Chatterbox on Apple Silicon. It uses separate
native references by default—Paulina for Mexican Spanish and Daniel for British
English—so the two languages are easy to distinguish. Kokoro remains available
as a faster fallback.

## Setup

Install [`uv`](https://docs.astral.sh/uv/), then:

```bash
uv sync
# Optional research backends and benchmark resource monitoring
uv sync --extra experimental-tts
uv run generate.py --download-samples salamander  # piano
uv run generate.py --download-samples vsco        # strings, marimba, glockenspiel
```

Samples are downloaded once to `~/.cache/earworms/`; generation itself remains
local and offline. Kokoro additionally needs `brew install espeak-ng`.

## Generate a lesson

```bash
# Chatterbox is the default
uv run generate.py --words 12 --out out/lesson.wav

# Audition only the music
uv run generate.py --bed-only --bed-style nocturne --out out/nocturne.wav

# Choose a specific background, lead, meter and chord colour
uv run generate.py --pad-instrument strings --instrument piano \
  --meter 4/4 --chord-extension add9 --out out/strings-and-piano.wav

# Fast voice fallback
uv run generate.py --backend kokoro --words 6 --out out/quick.wav

# Experimental expressive backends (weights download to the Earworms cache)
uv run generate.py --backend indextts25 --words 1 --out out/index.wav
uv run generate.py --backend voxcpm2 --words 1 --out out/voxcpm.wav
uv run generate.py --backend qwen3 --words 1 --out out/qwen.wav
uv run generate.py --backend tada --words 1 --out out/tada.wav
uv run generate.py --backend fish-s2 --words 1 --out out/fish.wav
```

Vocabulary comes from Markdown files or directories passed with `--vocab`.
Every lesson also writes a timestamped `.txt` tracklist and the resolved
`.bed.json`, which can be replayed or hand-edited with `--bed-spec`.

## Experimental expressive voices

The experimental backends expose only controls their current MLX runtimes
actually implement:

| Backend | Emotion/variation | Timing | Voice |
|---|---|---|---|
| `indextts25` | exact eight-float vector | model-side duration factor, bounded fit fallback | Paulina/Daniel cloning |
| `voxcpm2` | natural-language emotion, pace and pitch | qualitative instruction | Paulina/Daniel cloning |
| `qwen3` | natural-language emotion and prosody | qualitative instruction | Serena/Ryan presets |
| `tada` | stochastic dynamic prosody | no explicit rate control | Paulina/Daniel cloning |
| `fish-s2` | inline emotion tags and style instruction | no native rate control | Paulina/Daniel cloning |

IndexTTS duration shaping is not sample-exact. When any backend still overruns a
bar, the existing bounded pitch-preserving fit is used and recorded in the JSON
statistics. Qwen CustomVoice has no native Spanish preset; Serena is used for
Spanish and Ryan for English.

Run the matched five-model review and resource benchmark with:

```bash
uv run benchmark_voices.py --words 1 --out-dir out/tts-bakeoff
```

It runs models sequentially, writes one complete lesson WAV and `.stats.json`
per backend, and produces `comparison.json`/`comparison.md`. Statistics include
cold load and synthesis time, RTF, process-tree peak RSS, MLX peak allocation,
macOS memory pressure, swap state, output validation and any timing fallback.
The five model caches require roughly 23 GB of disk space.

Structural success is deliberately separate from listening quality. Very short
phrases can expose reference-boundary artifacts, wrong stress, degraded later
repetitions, or hallucinated trailing speech even when a WAV is finite and
correctly timed. Treat the generated comparison as a listening bake-off, not a
model-quality certification.

IndexTTS weights use the bilibili Model Use License, TADA uses the Llama 3.2
Community License, and Fish S2 Pro is research-only. See [NOTICE](NOTICE) before
using these backends outside local research.

See [DESIGN.md](DESIGN.md) for the research, musical design, measured behaviour,
and alternatives considered.
