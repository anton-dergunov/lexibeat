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
```

Vocabulary comes from Markdown files or directories passed with `--vocab`.
Every lesson also writes a timestamped `.txt` tracklist and the resolved
`.bed.json`, which can be replayed or hand-edited with `--bed-spec`.

See [DESIGN.md](DESIGN.md) for the research, musical design, measured behaviour,
and alternatives considered.
