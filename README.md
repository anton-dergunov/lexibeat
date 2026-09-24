# LexiBeat

LexiBeat makes **loops**: a handful of words, each spoken in the language you are learning,
followed by its translation, over a calm procedural music bed. Speech lands on a known beat grid,
repeats are delivered slightly differently, and the music ducks gently while either speaker is
talking. The `retrieval` pattern leaves a silent bar between the word and its answer to recall it in.

The music engine is procedural, not a model: numpy and `scipy.signal` over CC0 samples, no GPU, no
network, and byte-identical from a seed. That is what lets a loop be built on a small always-on
machine.

There are two ways in.

- **A library and a versioned HTTP service.** A host posts words and a direction for each, injects
  its own speech backend, and follows an operation until the finished MP3 and its timeline are
  ready. LexiBeat holds no provider credential of its own. The contract is
  [`docs/service.md`](docs/service.md).
- **A local command and a Gradio Lab**, for working on the music and trying a voice this machine
  can run.

The preferred local voice backend is Chatterbox on Apple Silicon. It uses separate native
references by default — Paulina for Mexican Spanish and Daniel for British English — so the two
languages are easy to distinguish. Kokoro remains available as a faster fallback.

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

Generate your own tracks with any language pair, vocabulary, and music style — see **Setup** below,
then either the local command or the Lab.

## Setup

Install [`uv`](https://docs.astral.sh/uv/), then:

```bash
uv sync                                    # the slim runtime: music, mixing, MP3
uv sync --extra service                    # the versioned HTTP surface a host integrates against
uv sync --extra explorer                   # the Gradio Lab
uv sync --extra local-tts                  # Chatterbox, Kokoro and other MLX voice backends
uv sync --extra local-tts --extra experimental-tts   # research backends and resource monitoring
uv sync --extra hosted-tts                 # Gemini and Cloudflare speech backends
uv run lexibeat --download-samples salamander  # piano
uv run lexibeat --download-samples vsco        # strings, marimba, glockenspiel
```

The slim runtime is numpy, scipy, soundfile, pyloudnorm, soxr and pedalboard, and nothing else.
`librosa` is an extra of `local-tts`: the music path used it for resampling alone, and through it
numba and llvmlite rode into an image whose job is to render a bed and mix speech it did not
synthesise.

The repository includes a checksum-locked production sample bundle through Git LFS. Run
`git lfs pull` after cloning, or fetch a published one with `lexibeat-bundle fetch`. Without it the
engine still works and offers the sample-free `electronic` palette. Additional samples can be
downloaded explicitly into `~/.cache/lexibeat/`. The default Chatterbox and other local backends run
offline after their weights are cached; explicitly selected Gemini and Cloudflare backends send
transcript text to their provider. Kokoro additionally needs `brew install espeak-ng`.

### The sample bundle

```bash
lexibeat-bundle verify                                   # every asset against its own digest
lexibeat-bundle fetch --into /data/lexibeat --from URL.001 --from URL.002 --sha256 DIGEST
lexibeat-bundle publish --out dist/bundle                # archive parts plus SHA256SUMS, for a release
export LEXIBEAT_BUNDLE_ROOT=/data/lexibeat/lexibeat-production-core-v3
```

It is the listener-approved Wave 3 library — 2,440 assets, 3.1 GB — so it never rides a wheel or a
release of source: it is fetched once into a volume and named by `LEXIBEAT_BUNDLE_ROOT`. Its
manifest carries the expansion policy that switches on the sustained strings, natural contrabasses,
Wave 3 leads and their audited gains; a bundle without one renders the plainer control behaviour.
`/api/v1/schema` reports which bundle is mounted and whether every file is present. It is published
as numbered parts under 1.9 GiB, because GitHub refuses a release asset of 2 GiB or more; `fetch`
joins them, checks the whole archive's digest, and replaces any older bundle in the directory. Licensing is not the reason — the one attribution-bearing source is
CC-BY 3.0 and its credit already travels inside the bundle — size is.

## The loop service

```python
from pathlib import Path
from lexibeat.service import ServiceConfig, create_service

app = create_service(
    config=ServiceConfig(output_root=Path("/var/lib/lexibeat")),
    backend_factory=lambda context: MyBackend(context.credentials),
)
```

`POST /api/v1/loops` takes the words, a free-text direction for each and the two languages, and
answers an operation to follow; `GET /api/v1/loops/{id}/audio` is the finished MP3 at 128 kbps.
The full contract, including the `Backend` protocol a host implements, is
[`docs/service.md`](docs/service.md), with `docs/openapi-v1.json` as the machine copy.

There is no `lexibeat-service serve`: a process with no speech backend has nothing to run, and a
default one would be exactly the provider credential this package refuses to hold.

## Web interface

Run the optional browser explorer locally with:

```bash
./scripts/run_explorer.sh
```

It installs the explorer and local-TTS extras and opens a Gradio interface. **Loop** takes one row
a word — the word, its translation, and free text saying how it should be said — with the two
languages given as a code and a name. It synthesizes the retrieval-practice sequence with whatever
voice this machine can run, mixes it over the current music bed, and writes an MP3. There is no row
limit. The first local request may download model weights and create the language-specific
reference cache.

**Music** exposes the safe product controls. **Lab** edits the resolved `BedSpec`, validates
production safety, randomizes unlocked fields, renders WAV previews, and loads or saves complete
`.bed.json` files. A loop reuses the last applied Music/Lab bed, or creates a safe automatic bed
when none exists. Music-only renders may be up to 180 seconds.

The Lab launcher retains FastAPI routes beneath `/api/`, with interactive API documentation at
`/api/docs`. These are the *music* routes and are separate from the versioned loop service at
`/api/v1`; the Lab never accepts client filesystem paths and writes only beneath `out/explorer/`
unless `LEXIBEAT_EXPLORER_OUT` selects another managed root.

The production bundle catalog contains only checksum-verified files. Music generation therefore
trusts and caches that immutable inventory instead of issuing per-sample existence checks against
the mount. Mutable local catalogs retain availability checks. If a cataloged bundle file is missing
when selected, generation reports that the bundle must be repaired or reattached. Rendering and
validation never perform implicit downloads.

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

See [How LexiBeat generates music](docs/music-generation.md) for a
beginner-friendly explanation of the composition rules, parameters,
instruments, rendering pipeline, and related music-theory research.

The CLI uses the same production path by default:

```bash
uv run lexibeat --bed-only --music-family auto \
  --music-energy bright --music-rhythm steady --music-palette acoustic \
  --out out/production-bed.wav
```

## Tuning the music by ear

[`docs/plans/music-listening.md`](docs/plans/music-listening.md) is the method: blind A/B pairs for
the known defects, then rounds of clips rated part by part on a tablet, summarised into rules. The
tools are `scripts/listening/`.

What a loop could become beyond the drill — examples, commentary, a final review, sound effects,
formats to choose from — is planned in [`docs/plans/programme-loops.md`](docs/plans/programme-loops.md).

## Large sample library and varied procedural beds

The exploration profile retains six procedural families used by the listening
benchmarks and README demo:
`meditative`, `organic`, `acoustic`, `nocturnal`, `sunlit`, and `lofi-wide`.
They resolve a complete repeating phrase—chord voicings, bass rhythm, motif and
multi-lane percussion—into the saved
`.bed.json`. A fixed seed therefore reproduces both the composition and audio.

Bulk sample collections are managed explicitly and never downloaded by normal
generation. On this workstation the defaults are a 500 GB external library and
a 50 GB local working cache:

```bash
uv run python -m scripts.samples.sample_library status
uv run python -m scripts.samples.sample_library download library-core   # CC0 sources
uv run python -m scripts.samples.sample_library index --deep
uv run python -m scripts.samples.sample_library report
```

To audit the complete attached library against the shipped production bundle
without copying audio, then render isolated speech-safety probes:

```bash
uv run python -m scripts.samples.sample_library audit-expansion --refresh-index \
  --workspace out/library-expansion --target-gb 10
uv run python -m scripts.samples.sample_library audition-expansion \
  --workspace out/library-expansion
uv run python -m scripts.samples.sample_library audition-expansion \
  --workspace out/library-expansion --wave secondary
uv run python -m scripts.samples.sample_library audit-wave3 \
  --workspace out/library-expansion \
  --baseline-bundle out/library-expansion/candidate-v2
uv run python -m scripts.samples.sample_library audition-expansion \
  --workspace out/library-expansion --wave wave3
```

The audit prioritizes natural piano, restrained strings, acoustic guitar,
natural bass and soft organic percussion. It records safe-register coverage,
velocity and round-robin depth, normalized perceived-level spread, spectral
brightness and transient risk. Its candidate manifest is a listening shortlist,
not permission to promote the files.

The secondary wave is written separately under `secondary-auditions/`. It
contains the remaining non-rejected banks after checksum-identical aliases are
removed. Audition regeneration keeps existing ratings matched by bank while
updating clip identifiers and filenames.

Wave 3 is a deliberately broader natural-instrument pass. It compares the
complete attached catalog with candidate-v2, retains brightness and vibrato as
role/register/gain warnings instead of aesthetic rejection reasons, and writes
its isolated probes under `wave3/auditions/`. The explicitly downloadable CC0
sources now include FreePats Button Accordion HN; normal generation still never
downloads samples.

After the Wave 3 listening gate, build its separate candidate on top of
candidate-v2. Family cautions are serialized into every affected instrument;
this accepted configuration keeps harpsichords occasional and 8 dB lower:

```bash
uv run python -m scripts.samples.sample_library integrate-wave3 --accept-all \
  --caution-family harpsichord:-8
LEXIBEAT_BUNDLE_ROOT=out/library-expansion/candidate-v3 \
  uv run python -m scripts.samples.sample_bundle verify
```

The completed listener gate retains six families and removes every rejected
family from the final catalog. Rebuild and verify the finalized bundle with:

```bash
uv run python -m scripts.samples.sample_library integrate-wave3 \
  --base-bundle out/library-expansion/candidate-v2 \
  --out out/library-expansion/final-v3 \
  --keep-family harp --keep-family lamellophone --keep-family marimba \
  --keep-family ocarina --keep-family organ --keep-family plucked-string \
  --reject-family accordion --reject-family bassoon \
  --reject-family clarinet --reject-family flute \
  --reject-family harmonica --reject-family harpsichord \
  --reject-family oboe --reject-family recorder
LEXIBEAT_BUNDLE_ROOT=out/library-expansion/final-v3 \
  uv run python -m scripts.samples.sample_bundle verify
```

The final bundle is the production bundle, committed as `assets/production-core/v3`. Its policy automatically
applies the approved role treatment whenever one of these banks is selected;
rejected source collections may remain on the external archive, but none of
their assets or catalog rows are shipped in this bundle.

After recording an all-accepted listening decision, build and verify the
separate candidate-v2 bundle with explicit attenuation for retained caution
clips:

```bash
uv run python -m scripts.samples.sample_library integrate-expansion --accept-all \
  --caution 04:-2 --caution S17:-4 --caution S20:-4 \
  --caution S21:-5 --caution S22:-5
LEXIBEAT_BUNDLE_ROOT=out/library-expansion/candidate-v2 \
  uv run python -m scripts.samples.sample_bundle verify
```

Selecting this bundle through `LEXIBEAT_BUNDLE_ROOT` enables its audited safe
registers, per-bank gains, sustained-string pads, natural contrabasses and
expanded organic percussion. Every resolved choice and gain remains serialized
in the saved BedSpec.

The library contains VCSL, VSCO 2 CE, FreePats World Percussion, FreePats
Spanish Classical Guitar, Karoryfer Fashionbass and the Stargate public-domain
pack. Downloads use staging directories, the index stores SHA-256 identities,
and samples selected for a saved bed can be promoted into the local cache so it
remains playable when the external SSD is disconnected.
Old extensionless promotions can be migrated or removed only after a matching
catalog checksum is verified:

```bash
uv run python -m scripts.samples.sample_library migrate-promotions
```

When the external volume is offline, saved beds and bake-off candidates can use
locally promoted catalog assets without attempting to recreate the missing
mount. Download and index operations still require the configured volume.

The bake-off uses only the redistributable collection set. Promoted catalog
files retain their WAV/FLAC/AIFF extensions. To make a Foobar2000-compatible
audition list without copying the bulk library:

```bash
uv run python -m scripts.samples.sample_library playlist vcsl --category pitched --out out/vcsl.m3u8
```

The catalog also groups pitch-labelled directories into resolved multisample
instruments with note, velocity, articulation, microphone and round-robin zones.
The production renderer deterministically uses the first coherent take, while
the sample-library audit tools can still audition alternate zones. Every active
production choice is represented in the saved BedSpec, so a fixed seed remains
replayable.

Resolved phrases also label the selected production bass pattern, motif, and
acoustic/hybrid/electronic palette. The abandoned Step 3B grammar and palette
values are no longer accepted by the current BedSpec schema.

Generate a controlled listening set with:

```bash
uv run --extra hosted-tts --env-file .env python -m scripts.benchmarks.compare_beds \
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
uv run python -m scripts.benchmarks.compare_beds --count 5 --voice-backend none \
  --replay-manifest out/music-before/manifest.json \
  --out-dir out/music-after
```

To compare the three production sample policies across an option set:

```bash
uv run python -m scripts.benchmarks.compare_beds --count 12 --family-profile positive \
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
uv run --extra hosted-tts python -m scripts.benchmarks.compare_beds --count 14 \
  --family-profile positive --sample-policy safe \
  --speech-cache-from out/music-bakeoff-2 --out-dir out/music-bakeoff-3
```

## Generate a loop from local notes

```bash
# Chatterbox is the default
uv run lexibeat --vocab ~/notes/spanish --words 12 --out out/loop.wav

# Any pair of languages: the code picks the voice, the name goes into the director note
uv run lexibeat --vocab ~/notes/mandarin --source-language zh-Hans \
  --source-language-name "Mandarin Chinese" --target-language pt-BR \
  --target-language-name "Brazilian Portuguese" --out out/mandarin.wav

# Audition only synth-based production music
uv run lexibeat --bed-only --music-family meditative \
  --music-palette electronic --out out/meditative.wav

# Choose product-level music controls
uv run lexibeat --music-family sunlit-acoustic \
  --music-energy bright --music-rhythm steady --out out/sunlit.wav

# Fast voice fallback
uv run lexibeat --backend kokoro --vocab ~/notes/spanish --words 6 --out out/quick.wav

# Experimental expressive backends (weights download to the LexiBeat cache)
uv run lexibeat --backend indextts25 --vocab ~/notes/spanish --words 1 --out out/index.wav
uv run lexibeat --backend voxcpm2 --vocab ~/notes/spanish --words 1 --out out/voxcpm.wav
uv run lexibeat --backend qwen3 --vocab ~/notes/spanish --words 1 --out out/qwen.wav
uv run lexibeat --backend tada --vocab ~/notes/spanish --words 1 --out out/tada.wav
uv run lexibeat --backend fish-s2 --vocab ~/notes/spanish --words 1 --out out/fish.wav

# Hosted backends read credentials from the environment
uv run --extra hosted-tts --env-file .env python -m lexibeat.cli \
  --backend gemini --vocab ~/notes/spanish --words 1 --out out/gemini.wav
# Paid Vertex AI via Application Default Credentials (no API key)
GOOGLE_CLOUD_PROJECT=your-project-id \
GOOGLE_CLOUD_LOCATION=global \
uv run --extra hosted-tts python -m lexibeat.cli --backend gemini-vertex \
  --vocab ~/notes/spanish --words 1 --out out/gemini-vertex.wav
uv run --extra hosted-tts --env-file .env python -m lexibeat.cli \
  --backend cloudflare-aura2 --vocab ~/notes/spanish --words 1 --out out/aura2.wav
uv run --extra hosted-tts --env-file .env python -m lexibeat.cli \
  --backend cloudflare-melotts --vocab ~/notes/spanish --words 1 --out out/melotts.wav
```

Vocabulary comes from Markdown files or directories passed with `--vocab`, which is required and
has no default — those are your own notes, in your own place. Every run also writes a timestamped
`.txt` tracklist and the resolved `.bed.json`, which can be replayed or hand-edited with
`--bed-spec`.

## Experimental expressive voices

The experimental backends expose only controls their current local runtimes or
hosted APIs actually implement:

| Backend | Direction/variation | Timing | Voice |
|---|---|---|---|
| `indextts25` | neutral eight-float vector | model-side duration factor, bounded fit fallback | Paulina/Daniel cloning |
| `voxcpm2` | the caller's direction, plus pace and pitch | qualitative instruction | Paulina/Daniel cloning |
| `qwen3` | the caller's direction, plus prosody | qualitative instruction | Serena/Ryan presets |
| `tada` | stochastic dynamic prosody | no explicit rate control | Paulina/Daniel cloning |
| `fish-s2` | the caller's direction as a style instruction | no native rate control | Paulina/Daniel cloning |
| `gemini` | the caller's direction, plus pace and pitch | qualitative instruction | Sulafat/Achird presets |
| `gemini-vertex` | same controls via paid Vertex AI/ADC | qualitative instruction | Sulafat/Achird presets |
| `cloudflare-aura2` | gentle local pitch/speed variation | local post-process | Aquila/Luna presets |
| `cloudflare-melotts` | gentle local pitch/speed variation | local post-process | provider default; currently English-only |

IndexTTS duration shaping is not sample-exact. When any backend still overruns a
bar, the existing bounded pitch-preserving fit is used and recorded in the JSON
statistics. Qwen CustomVoice has no native Spanish preset; Serena is used for
Spanish and Ryan for English.

Run the matched five-model review and resource benchmark with:

```bash
uv run python -m scripts.benchmarks.benchmark_voices --vocab ~/notes/spanish --words 1 --out-dir out/tts-bakeoff
```

It runs models sequentially, writes one complete loop WAV and `.stats.json`
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
uv run --extra hosted-tts --env-file .env python -m scripts.benchmarks.compare_voices \
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
uv run --extra hosted-tts python -m scripts.benchmarks.compare_gemini_batched \
  --words 10 --reps 3 --out-dir out/hosted-tts-batched
```

The acceptance run split every batch cleanly, but cost `$0.099465` versus
`$0.059309` for 60 separate Gemini 3.1 calls because the long pauses consume
output audio tokens. Treat batching as a request-quota optimization, not a cost
optimization, and listen to the saved raw files before trusting the split.

IndexTTS weights use the bilibili Model Use License, TADA uses the Llama 3.2
Community License, and Fish S2 Pro is research-only. See [NOTICE](NOTICE.md) before
using these backends outside local research.

See [the design notes](docs/design.md) for the research, musical design, measured behaviour,
and alternatives considered.
