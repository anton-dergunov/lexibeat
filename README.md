---
title: LexiBeat Language Lesson Generator
emoji: 🎵
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.26.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# LexiBeat

LexiBeat is a small experimental project for making language-learning
audio: a word or phrase is spoken in Spanish, followed by its English meaning,
over a calm procedural music bed. Speech lands on a known beat grid, repeats use
slightly different delivery, and the music ducks gently while either speaker is
talking.

The preferred local voice backend is Chatterbox on Apple Silicon. It uses separate
native references by default—Paulina for Mexican Spanish and Daniel for British
English—so the two languages are easy to distinguish. Kokoro remains available
as a faster fallback. The public Hugging Face Space uses the official CUDA
Chatterbox Multilingual runtime on ZeroGPU.

## Demo

A short example using **Spanish 🇦🇷 → English 🇬🇧**:
12 words and phrases set to a generated rhythmic backing track.

> [!TIP]
> 🔊 Turn on your sound to hear the generated speech and music.

https://github.com/user-attachments/assets/46a03899-5088-402c-afdc-73bd68ce1b27

<details>
<summary><strong>🎵 More music styles — extended 20-pair demos</strong></summary>

<br>

The same set of 20 Spanish–English word and phrase pairs, rendered with
different procedurally generated music styles.

### ☀️ Sunlit

Bright, relaxed, and lightly rhythmic.

https://github.com/user-attachments/assets/16cc6764-1d06-4a05-8fd3-048f46eb22b6

### 🌊 Warm Motion

Warm, flowing, and gently propulsive.

https://github.com/user-attachments/assets/8f97331a-703a-415f-8b18-cbf02fb5be49

### 🌙 Nocturnal

Slower, darker, and more atmospheric.

https://github.com/user-attachments/assets/9a83a5b9-1427-4216-b78c-aa6aed5d8a0f

</details>

### Try it yourself

Generate your own tracks with any language pair, vocabulary, and music style:

**[▶ Open LexiBeat on Hugging Face Spaces](https://huggingface.co/spaces/AntonDergunov/LexiBeat)**

## Setup

Install [`uv`](https://docs.astral.sh/uv/), then:

```bash
uv sync
# Local Chatterbox, Kokoro, and other MLX voice backends
uv sync --extra local-tts
# Optional research backends and benchmark resource monitoring
uv sync --extra local-tts --extra experimental-tts
# Optional hosted Gemini and Cloudflare speech backends
uv sync --extra hosted-tts
uv run generate.py --download-samples salamander  # piano
uv run generate.py --download-samples vsco        # strings, marimba, glockenspiel
```

The repository includes a checksum-locked production sample bundle through Git
LFS. Run `git lfs pull` after cloning. Additional samples can be downloaded
explicitly into `~/.cache/lexibeat/`. The default Chatterbox and other local
backends run offline after their weights are cached; explicitly selected Gemini
and Cloudflare backends send transcript text to their provider. Kokoro
additionally needs `brew install espeak-ng`.

## Web interface

Run the optional browser explorer locally with:

```bash
./scripts/run_explorer.sh
```

It installs the explorer and local-TTS extras and opens a Gradio interface.
**Lesson** accepts one to six Spanish/English pairs, synthesizes the established
retrieval-practice sequence with Chatterbox, mixes it over the current music
bed, and displays timed text in the audio player. A caption begins with its
utterance and remains visible until the next utterance, so the English answer is
never exposed before it is spoken. Lesson progress distinguishes validation,
model allocation/loading, individual utterances, music rendering, and mixing.
The first local request may download model weights and create the language-
specific reference cache.

**Music** exposes the safe product controls. **Lab** edits the resolved
`BedSpec`, validates production safety, randomizes unlocked fields, renders WAV
previews, and loads or saves complete `.bed.json` files. A lesson reuses the
last applied Music/Lab bed, or creates a safe automatic bed when none exists.
Local music-only renders may be up to 180 seconds; the public
[Hugging Face Space](https://huggingface.co/spaces/AntonDergunov/LexiBeat)
caps music-only renders at 30 seconds and lessons at six vocabulary pairs.

The local launcher retains FastAPI routes beneath `/api/`, with interactive API
documentation at `/api/docs`. The public Space launches Gradio directly so
ZeroGPU can discover its GPU callback and does not expose these custom routes.
The service never accepts client filesystem paths and writes only beneath
`out/explorer/` unless `LEXIBEAT_EXPLORER_OUT` selects another managed root.
Hosted sample promotion remains disabled.

Deployment from `main` is handled by `.github/workflows/deploy-huggingface.yml`.
Add a write-capable Hugging Face token as the GitHub Actions secret `HF_TOKEN`.
The workflow verifies the Linux music/web installation and synchronizes a small
staging directory to `AntonDergunov/LexiBeat`. It includes a revision-pinned
official Chatterbox CUDA runtime with the multilingual v3 model and hosted-only
dependency pins, but excludes
the checkout's `.venv` and the Git-LFS production audio, so the Space repository
stays well below its Git storage limit.

The hosted sample library lives separately in the private
`AntonDergunov/LexiBeatSamples` Storage Bucket. Upload the checksum-locked local
bundle once (review the plan before the real upload):

```bash
hf auth login
./scripts/sync_hf_samples.sh --dry-run
./scripts/sync_hf_samples.sh
```

In the Space settings, attach that bucket at `/data`, preferably read-only. The
application automatically discovers
`/data/lexibeat-production-core/v1/catalog.sqlite3`. A custom mount can be set
with `LEXIBEAT_BUCKET_MOUNT`, or the complete bundle path can be selected with
`LEXIBEAT_BUNDLE_ROOT`. If the bundle is not mounted, the hosted UI remains
usable but offers only the sample-free electronic palette. Rendering and
validation never perform implicit downloads.

The production bundle catalog contains only checksum-verified files copied by
the bundle builder. Music generation therefore trusts and caches that immutable
inventory instead of issuing per-sample existence checks against the mounted
bucket. Mutable local catalogs retain availability checks. If a cataloged bundle
file is missing when selected, generation reports that the bundle must be
repaired or reattached.

The production bundle is currently about 1.8 GB. A private bucket is appropriate
for the public Space: its visibility does not control the Space's visibility,
and visitors use samples through the application rather than receiving direct
bucket access. The bucket can be made public later if direct redistribution is
desired and every included source remains license-compatible.

The Space runs Chatterbox Multilingual v3 on ZeroGPU. Its module-level model is
placed on the CUDA-emulated device during startup, and the exact Gradio lesson
handler is decorated with `@spaces.GPU`; its reservation grows from 70 to 120
seconds with the number of vocabulary pairs. Only speech synthesis occupies the
GPU allocation. Music rendering and final mixing run in the chained CPU stage.
The Space launches Gradio directly with server-side rendering disabled, which
also lets ZeroGPU complete its startup scan before serving requests.
The first process start downloads the model checkpoint, and a cold ZeroGPU
allocation can therefore take longer than later requests. Generated speech and
finished lessons are cached beneath the managed explorer output directory.

## Versioned music API

Applications can request safe variety without understanding the complete music
schema:

```python
from lexibeat import MusicRequest, generate_music

audio, result = generate_music(
    MusicRequest(
        family="auto",
        energy="balanced",
        rhythm="steady",
        palette="hybrid",
        seed=42,
    ),
    duration_seconds=75,
)

# Persist this complete contract to reproduce the composition later.
result.bed_spec.to_json()
```

`MusicRequest` resolves several candidates through the versioned
`production-v1` profile, rejects unsafe previews, and selects randomly within
the highest-quality tier. Supplying recent `BedFingerprint` values is optional
and increases novelty without introducing hidden engine state. Resolution and
rendering never download assets.

The CLI exposes the same production path while preserving the legacy style
flags:

```bash
uv run generate.py --bed-only --music-family auto \
  --music-energy bright --music-rhythm steady --music-palette acoustic \
  --out out/production-bed.wav
```

## Large sample library and varied procedural beds

The expanded music engine has six additional procedural families:
`meditative`, `organic`, `acoustic`, `nocturnal`, `sunlit`, and `lofi-wide`.
Unlike the four legacy styles, these resolve a complete repeating phrase—chord
voicings, bass rhythm, motif and multi-lane percussion—into the saved
`.bed.json`. A fixed seed therefore reproduces both the composition and audio.

Bulk sample collections are managed explicitly and never downloaded by normal
generation. On this workstation the defaults are a 500 GB external library and
a 50 GB local working cache:

```bash
uv run sample_library.py status
uv run sample_library.py download library-core   # CC0 sources
uv run sample_library.py index --deep
uv run sample_library.py report
```

To audit the complete attached library against the shipped production bundle
without copying audio, then render isolated speech-safety probes:

```bash
uv run sample_library.py audit-expansion --refresh-index \
  --workspace out/library-expansion --target-gb 10
uv run sample_library.py audition-expansion \
  --workspace out/library-expansion
uv run sample_library.py audition-expansion \
  --workspace out/library-expansion --wave secondary
```

The audit prioritizes natural piano, restrained strings, acoustic guitar,
natural bass and soft organic percussion. It records safe-register coverage,
velocity and round-robin depth, normalized perceived-level spread, spectral
brightness and transient risk. Its candidate manifest is a listening shortlist,
not permission to promote the files.

The secondary wave is written separately under `secondary-auditions/`. It
contains the remaining non-rejected banks after checksum-identical aliases are
removed, and never overwrites an existing ratings sheet.

After recording an all-accepted listening decision, build and verify the
separate candidate-v2 bundle with explicit attenuation for retained caution
clips:

```bash
uv run sample_library.py integrate-expansion --accept-all \
  --caution 04:-2 --caution S17:-4 --caution S20:-4 \
  --caution S21:-5 --caution S22:-5
LEXIBEAT_BUNDLE_ROOT=out/library-expansion/candidate-v2 \
  uv run sample_bundle.py verify
```

Selecting this bundle through `LEXIBEAT_BUNDLE_ROOT` enables its audited safe
registers, per-bank gains, sustained-string pads, natural contrabasses and
expanded organic percussion. Every resolved choice and gain remains serialized
in the saved BedSpec. The existing production-v1 bundle is not modified.

The library contains VCSL, VSCO 2 CE, FreePats World Percussion, FreePats
Spanish Classical Guitar, Karoryfer Fashionbass and the Stargate public-domain
pack. Downloads use staging directories, the index stores SHA-256 identities,
and samples selected for a saved bed can be promoted into the local cache so it
remains playable when the external SSD is disconnected.
Old extensionless promotions can be migrated or removed only after a matching
catalog checksum is verified:

```bash
uv run sample_library.py migrate-promotions
```

When the external volume is offline, saved beds and bake-off candidates can use
locally promoted catalog assets without attempting to recreate the missing
mount. Download and index operations still require the configured volume.

The bake-off uses only the redistributable collection set. Promoted catalog
files retain their WAV/FLAC/AIFF extensions. To make a Foobar2000-compatible
audition list without copying the bulk library:

```bash
uv run sample_library.py playlist vcsl --category pitched --out out/vcsl.m3u8
```

The catalog also groups pitch-labelled directories into resolved multisample
instruments with note, velocity, articulation, microphone and round-robin zones.
The renderer keeps those dimensions coherent. Production uses the control's
first-take policy; saved Step 3 experiments can still request deterministic
cyclic alternates. This keeps the mapping infrastructure available without
changing the accepted production sound. Every active choice is represented in
the saved BedSpec, so a fixed seed remains replayable.

Resolved phrases also label the selected control bass pattern, the existing
random-walk motif, and the acoustic/hybrid/electronic palette. These labels make
saved beds easier to inspect without changing candidate selection or the music.
The wider Step 3B grammar and palette experiment remains loadable from its saved
BedSpec files, but is not used by production generation.

Generate a controlled listening set with:

```bash
uv run --extra hosted-tts --env-file .env compare_beds.py \
  --count 30 --sample-policy safe --voice-backend gemini-vertex \
  --out-dir out/music-bakeoff
```

The command builds a larger candidate pool, balances the selected families and
uses audio plus melodic-interval distance to choose the requested count. Every final clip uses the
same two bilingual items, so timbre, rhythm and speech masking can be compared
directly. It writes WAV/BedSpec pairs, a complete manifest, a listening guide
and a rating CSV.

An earlier manifest can be replayed after an engine change to render the same
family/seed pairs for a controlled A/B comparison:

```bash
uv run compare_beds.py --count 5 --voice-backend none \
  --replay-manifest out/music-before/manifest.json \
  --out-dir out/music-after
```

To compare the three production sample policies across an option set:

```bash
uv run compare_beds.py --count 12 --family-profile positive \
  --voice-backend none --palettes acoustic hybrid electronic \
  --out-dir out/palette-bakeoff
```

The feedback-focused profile adds `radiant`, `acoustic-flow`,
`playful-minimal`, `warm-motion`, `bright-organic`, `gentle-game`,
`sunlit-acoustic`, `gentle-movement`, `playful-plucked`, and `bright-pastoral`.
It keeps swing subtle, uses meter-aware low anchors, density-compensates the
percussion bus, broadens piano writing and excludes metallic ornaments from
ordinary drum selection:

```bash
uv run --extra hosted-tts compare_beds.py --count 14 \
  --family-profile positive --sample-policy safe \
  --speech-cache-from out/music-bakeoff-2 --out-dir out/music-bakeoff-3
```

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

# Experimental expressive backends (weights download to the LexiBeat cache)
uv run generate.py --backend indextts25 --words 1 --out out/index.wav
uv run generate.py --backend voxcpm2 --words 1 --out out/voxcpm.wav
uv run generate.py --backend qwen3 --words 1 --out out/qwen.wav
uv run generate.py --backend tada --words 1 --out out/tada.wav
uv run generate.py --backend fish-s2 --words 1 --out out/fish.wav

# Hosted backends read credentials from the environment
uv run --extra hosted-tts --env-file .env generate.py \
  --backend gemini --words 1 --out out/gemini.wav
# Paid Vertex AI via Application Default Credentials (no API key)
GOOGLE_CLOUD_PROJECT=your-project-id \
GOOGLE_CLOUD_LOCATION=global \
uv run --extra hosted-tts generate.py --backend gemini-vertex \
  --words 1 --out out/gemini-vertex.wav
uv run --extra hosted-tts --env-file .env generate.py \
  --backend cloudflare-aura2 --words 1 --out out/aura2.wav
uv run --extra hosted-tts --env-file .env generate.py \
  --backend cloudflare-melotts --words 1 --out out/melotts.wav
```

Vocabulary comes from Markdown files or directories passed with `--vocab`.
Every lesson also writes a timestamped `.txt` tracklist and the resolved
`.bed.json`, which can be replayed or hand-edited with `--bed-spec`.

## Experimental expressive voices

The experimental backends expose only controls their current local runtimes or
hosted APIs actually implement:

| Backend | Emotion/variation | Timing | Voice |
|---|---|---|---|
| `indextts25` | exact eight-float vector | model-side duration factor, bounded fit fallback | Paulina/Daniel cloning |
| `voxcpm2` | natural-language emotion, pace and pitch | qualitative instruction | Paulina/Daniel cloning |
| `qwen3` | natural-language emotion and prosody | qualitative instruction | Serena/Ryan presets |
| `tada` | stochastic dynamic prosody | no explicit rate control | Paulina/Daniel cloning |
| `fish-s2` | inline emotion tags and style instruction | no native rate control | Paulina/Daniel cloning |
| `gemini` | natural-language emotion, pace and pitch | qualitative instruction | Sulafat/Achird presets |
| `gemini-vertex` | same controls via paid Vertex AI/ADC | qualitative instruction | Sulafat/Achird presets |
| `cloudflare-aura2` | gentle local pitch/speed variation | local post-process | Aquila/Luna presets |
| `cloudflare-melotts` | gentle local pitch/speed variation | local post-process | provider default; currently English-only |

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

Hosted credentials are never loaded from tracked configuration. Copy
`.env.example` to the ignored `.env`, fill in the three values, and pass
`--env-file .env` to `uv run`. Gemini and Cloudflare do not honor `voice_seed`;
their output is not reproducible even though placement on the beat grid remains
deterministic. Gemini requests retry transient failures and adaptively pace
themselves for the preview endpoint's 3-RPM free-tier limit. The current project
also has a 10-request daily limit, while ten bilingual items repeated three times
need 60 requests; use increased quota or collect that comparison across multiple
quota days. A daily-quota error fails immediately because retrying cannot resolve
it. Cloudflare's current MeloTTS deployment rejects its documented
Spanish language code, so it remains available as an English-only diagnostic
until [cloudflare/ai#221](https://github.com/cloudflare/ai/issues/221) is fixed.
A hosted listening comparison can be attempted with the command below. MeloTTS
contributes English only, and its statistics record that language restriction.

```bash
uv run --extra hosted-tts --env-file .env compare_voices.py \
  --words 10 --reps 3 \
  --configs gemini cloudflare-aura2 cloudflare-melotts \
  --out-dir out/hosted-tts
```

For projects with billing and Vertex AI enabled, authenticate once with
`gcloud auth application-default login`, then export `GOOGLE_CLOUD_PROJECT` and
`GOOGLE_CLOUD_LOCATION=global`. The comparison presets
`gemini-vertex-31`, `gemini-vertex-25-flash`, `gemini-vertex-25-lite`, and
`gemini-vertex-25-pro` cover all currently documented Gemini-TTS models. The
Vertex path uses paid standard quota and does not read `GEMINI_API_KEY`.

Gemini 3.1 also has a separate six-call batching experiment. It groups all ten
items by language and repetition style, inserts the documented `[long pause]`
tag, splits the returned PCM at the nine strongest silent gaps, and refuses any
batch without ten credible segments. This reduces 60 provider requests to six,
but it does not provide authoritative word timestamps and the silent output is
still billed. Run it with:

```bash
GOOGLE_CLOUD_PROJECT=your-project-id \
GOOGLE_CLOUD_LOCATION=global \
uv run --extra hosted-tts compare_gemini_batched.py \
  --words 10 --reps 3 --out-dir out/hosted-tts-batched
```

The acceptance run split every batch cleanly, but cost `$0.099465` versus
`$0.059309` for 60 separate Gemini 3.1 calls because the long pauses consume
output audio tokens. Treat batching as a request-quota optimization, not a cost
optimization, and listen to the saved raw files before trusting the split.

IndexTTS weights use the bilibili Model Use License, TADA uses the Llama 3.2
Community License, and Fish S2 Pro is research-only. See [NOTICE](NOTICE.md) before
using these backends outside local research.

See [DESIGN.md](DESIGN.md) for the research, musical design, measured behaviour,
and alternatives considered.
