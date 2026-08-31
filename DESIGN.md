# LexiBeat — Design Notes & Options

Research and design decisions behind LexiBeat: generating language-learning
audio (vocabulary spoken over a rhythmic ambient bed) from a word list, fully
locally on an Apple Silicon Mac.

---

## 0. What was built

```
generate.py            CLI entry point
compare_voices.py      renders the same words through several voice setups
lexibeat/vocab.py      parses the Obsidian notes into (Spanish, English, emoji) triples
lexibeat/bedspec.py    every music parameter, as styles and as JSON
lexibeat/music.py      renders a BedSpec into audio on a known beat grid
lexibeat/instruments.py  synth and sampled instruments behind one interface
lexibeat/samples.py    sample-pack registry and downloader
lexibeat/emotion.py    emoji -> delivery mapping
lexibeat/voice.py      Kokoro and Chatterbox backends, per-repeat variation
lexibeat/arrange.py    places utterances on downbeats according to a pattern
lexibeat/mix.py        LUFS normalisation, sidechain ducking, block limiter
```

### Running it

```bash
brew install espeak-ng          # required by Kokoro for Spanish G2P
uv sync

uv run generate.py --words 12 --out out/lesson.wav
uv run generate.py --words 6 --dry-run              # show the plan only
uv run generate.py --bed-only --bed-style lofi      # audition a bed in ~1 s
uv run generate.py --download-samples               # fetch the piano (~88 MB)
uv run generate.py --words 6                        # Chatterbox, expressive default
uv run generate.py --words 6 --backend kokoro       # faster fallback
uv run compare_voices.py --words 3                  # A/B the voice setups
```

Flags worth knowing: `--bed-style yoga|nocturne|lofi|warm`, `--bed-seed`,
`--bed-spec <file>`, `--instrument`, `--pad-instrument`, `--meter`,
`--chord-extension`, `--mode words|phrases|mixed`, `--pattern retrieval|alternating`,
`--backend chatterbox|kokoro`, `--ref-audio-es`, `--ref-audio-en`,
`--prosody-strength`, `--no-emotion`, `--duck-db`.

Every run writes a `.txt` tracklist and a `.bed.json` with the resolved music
parameters, so a bed that sounds good can be replayed or hand-edited.

### Measured behaviour

| Check | Result |
|---|---|
| Utterance onset vs downbeat | median **15 ms** |
| Integrated loudness / true peak | −17.4 LUFS / 0.97 |
| Recovered tempo of the finished mix | 80.75 BPM (requested 80) |
| Bed render | 0.3–1.4 s for ~70 s of audio, all styles |
| Kokoro | ~0.4 s per utterance; 4.2 min track in 36 s |
| Chatterbox (MLX) | ~5.5–6.1 s per utterance — about 14× Kokoro |
| Emoji coverage in the vocabulary notes | **922 / 939 entries (98%)** |

### Known limitations

- Kokoro's Spanish voices are European-leaning. Chatterbox defaults to macOS
  Paulina (`es_MX`) for Latin American Spanish and Daniel (`en_GB`) for English.
- Chatterbox needs Apple Silicon; `--backend kokoro` remains the portable path.
- `mlx-community/chatterbox-multilingual-v3` ships without built-in voice
  conditioning, so language-specific reference clips are generated and cached
  automatically in `~/.cache/lexibeat/refs/`.
- Only 40% of entries carry an example sentence, so `--mode phrases` falls back
  to the headword for the rest.
- Multi-sense glosses are truncated to the first sense.
- Repeats of a word are blocked together rather than spaced across the track.


---

## 1. Early timing analysis

The initial prototype used a private, untracked 184.6-second reference recording
to establish timing. It was measured with `librosa.beat.beat_track` plus a crude
speech-band energy VAD (300–3400 Hz, 70th-percentile threshold):

| Property | Measured value |
|---|---|
| Tempo | **117.45 BPM** |
| Beat interval | 0.511 s |
| Bar length (4/4) | **~2.05 s** |
| Speech-ish segments | 67 |
| Median utterance duration | **0.33 s** |
| Median gap between utterances | **1.57 s** |
| Implied utterance cycle | ~1.90 s |

**Key finding:** the utterance cycle (~1.90 s) fits inside one 4/4 bar (~2.05 s).
The structure is *one short utterance per bar, onset on the downbeat, remainder of
the bar silent*. "Finding insertion points" is therefore not a semantic problem —
it is a beat grid problem.

The private reference recording is intentionally not redistributed.

---

## 2. Does the method actually work?

**Split verdict.**

Claims made by commercial predecessors were marketing copy. No peer-reviewed
validation of a particular commercial product was found; the relevant evidence
supports individual learning mechanisms rather than a branded implementation.

The *components* underneath do have support:

- **Spaced repetition** — robust, well-replicated.
- **Song superiority effect** — melody and rhythm aid verbatim recall of text.
  Important caveat: the effect is strongest when text is **sung**, not spoken over
  music. Spoken-over-a-beat gets rhythmic chunking and pleasant repeatability, but
  not the full melodic mnemonic.
- **Passive background listening** — weak for *acquiring* new vocabulary, but
  reasonable for *consolidating* items already encountered, and for pronunciation.
  This fits the intended use case (words sourced from the learner's own notes).

### Evidence-based improvements over the original

1. **Retrieval gap** (implemented). Play the Spanish, leave a silent bar for the
   learner to recall the English *before* it is spoken. Converts passive listening
   into retrieval practice — the strongest single effect in the learning
   literature. Costs one bar.
2. **Expanding-interval spacing** (future work). Rather than blocking all repeats
   of a word together, interleave repeats at expanding intervals across the track
   (the Pimsleur "graduated interval recall" idea).
3. **Phrases over bare words** (supported). The source notes contain example
   sentences, which are often better learning material than isolated headwords.

### Prior art

No project was found that does this. Nearest neighbours:

- [langlearnai](https://pypi.org/project/langlearnai/0.0.3/) — builds Anki decks from
  word lists with TTS audio and images. No music, no beat alignment.
- [vocabulary-to-speech-api](https://github.com/elhalili/vocabulary-to-speech-api) —
  generates an audio file from a vocabulary list. No music.
- **Glossika** (commercial) — spaced audio repetition, no music.
- **Pimsleur** (commercial) — graduated interval recall, no music.
- Commercial musical-vocabulary courses — human-recorded, not generated.

The niche is open.

---

## 3. Music source — options

All downstream stages consume the same interface: **`(audio, beat_grid)`**. Only
the provider differs, so these are swappable.

### Option A — Procedural synthesis (chosen for v1)

Synthesise the bed directly in NumPy at a chosen BPM.

- **Pro:** the grid is known *by construction* — zero estimation error. Instant,
  deterministic (seeded), arbitrary length, loops perfectly, no model downloads,
  no GPU.
- **Con:** ambient wallpaper rather than an authored track.
- **Note:** the target bed is deliberately unobtrusive, so the quality gap is
  smaller than it first appears.

### Option B — Neural music generation

| Model | Route on Apple Silicon | Notes |
|---|---|---|
| MusicGen (Meta) | [musicgen-mlx](https://github.com/andrade0/musicgen-mlx), [mlx-audiocraft](https://github.com/theashishmaurya/mlx-audiocraft) | `audiocraft` 1.3.0 on PyPI is pinned to 2024-era torch — painful on Python 3.12. Reported: musicgen-medium ≈ 2–3 min for 15 s on an M1 **Max**. |
| Stable Audio Open 1.0 / small | [stable-audio-mlx](https://github.com/sandst1/stable-audio-mlx), [HF](https://huggingface.co/stabilityai/stable-audio-open-small) | `stable-audio-tools` 0.0.20 on PyPI requires **Python <3.11**. The MLX port sidesteps this. BPM is expressed in the prompt ("90 BPM ambient loop"), not as a structured parameter. |

- **Pro:** richer, more "real" timbre.
- **Con:** slow on M1/16 GB; output capped at ~30–47 s so a multi-minute track
  requires stitching with crossfades; **BPM is only a prompt hint**, so beat
  tracking is still required to recover the true grid.

### Option C — User-supplied clips + beat tracking

| Library | Notes |
|---|---|
| [beat_this](https://github.com/CPJKU/beat_this) (CPJKU, ISMIR) | Current state of the art for beats **and downbeats**; drops DBN post-processing. `pip install beat-this`, then `beat_this.inference.File2Beats`. |
| [BeatNet](https://github.com/mjhydri/BeatNet) | Joint beat/downbeat/tempo/meter, real-time capable. |
| madmom | Long-standing baseline, older. |
| `librosa.beat.beat_track` | Beats only, no downbeats. Fine for tempo estimation (used above). |

- **Pro:** the most musically satisfying beds.
- **Con:** curation burden falls on the user.

### Option D — Hybrid (recommended next step)

Generate a single 8-bar loop neurally, snap it to a grid with `beat_this`, then
loop it for the full track. Neural timbre with an exact grid.

### Iteration 4 — resolved procedural phrases and a tiered sample catalog

The later music expansion stayed with Option A but removed its main source of
sameness. The old styles all used the same topology: one sustained chord and
one bass root per bar, the same synthetic three-piece kit, and unrelated random
pentatonic lead notes. Range changes alone could not make those outputs sound
structurally different.

New styles instead resolve a four- or eight-bar phrase before rendering. The
saved BedSpec contains exact chord voicings, chord rhythm, bass events, a
repeating motif, Euclidean or syncopated percussion lanes, timbres and stable
sample references. The renderer repeats that coherent phrase and does not make
new compositional decisions. Legacy BedSpec JSON follows the old path unchanged.

Complete source libraries live outside the repository. A checksum-locked
production subset is shipped with Git LFS. A SQLite catalog maps logical
`collection:asset` references to SHA-256-verified files in the bundled, external,
or local promoted tier. Normal generation performs no network access, and
missing/offline assets fail explicitly. The bake-off generator renders a larger
candidate pool, extracts audio descriptors and greedily chooses a family-balanced,
maximally separated listening set rather than hard-coding tracks.

Listener feedback from the first 30-bed comparison established a clearer
production prior: nearly straight subdivision, audible first-beat hierarchy,
restrained percussion and coherent acoustic foreground instruments. The second
comparison profile therefore caps ordinary swing, forces the low percussion
lane onto every bar downbeat, applies transient/RMS-aware one-shot gain, and
keeps bells and other metallic ornaments out of normal drum roles.

Individual catalog samples remain useful for percussion. Pitched foreground
parts now use a resolved `InstrumentRef`: a directory-level set of logical
sample zones with pitch and velocity bounds saved into BedSpec. This restores
the dedicated Salamander piano path and unlocks coherent VCSL/VSCO pianos,
harps, mallets, organs and winds. The production catalog is restricted to
redistributable sources.

### Iteration 5 — pulse clarity and foreground-instrument coverage

The third listening round replaced phrase-wide Euclidean low drums with
meter-aware anchor patterns. Sample roles are explicit in BedSpec; kicks and
bass drums can provide the low anchor, while darbuka, bongo and conga remain
mid accents. Dense multi-lane percussion receives automatic gain compensation,
and positive families keep only subtle grid swing.

Piano motifs now span four bars and a wider register, with instrument-specific
gain and shallower speech ducking. Candidate distance includes normalized pitch
intervals and onset gaps so a new timbre cannot disguise a repeated melody.
The safe catalog exposes CC0 mbiras, psaltery, ocarina, harmonica, pizzicato and
spiccato strings, classical guitar and natural electric bass. Coverage-aware
selection ensures the targeted guitar, bass and expanded foreground families
appear in comparison sets while retaining one result per procedural family.

### Iteration 6 — deterministic natural sample variation

The catalog previously collapsed same-note alternatives to one file. A resolved
multisample bank now preserves coherent note, velocity, articulation and
microphone groups, then labels explicit alternate takes with stable round-robin
indexes. Conservative filename inference avoids combining ambiguous siblings or
switching microphone perspectives during a phrase.

BedSpec 1.1 serializes event articulation, sample-variation indexes, percussion
take groups and the cyclic selection strategy. Pitched parts offset that saved
choice on each phrase repetition, while percussion advances through its resolved
take group per hit. Replaying the same BedSpec is byte-deterministic; older JSON
receives compatible natural-articulation and first-take defaults.

Catalog reports now expose register, articulation and timbre-cluster coverage,
round-robin depth, instrument-bank utilization and explicit rejection reasons.
The bake-off tool can replay an earlier manifest's exact family/seed pairs, so
intentional engine-level PCM changes can be evaluated with controlled before and
after renders.

### Iteration 7 — audible grammar and palette variety

Round robins improve repetition at the sample level but are intentionally hard
to notice. Audible option-set variety now comes from two separately seeded,
serialized dimensions: six bass grammars and six melodic motif grammars. The
motifs include directed contours and call-and-response rather than relying only
on unrelated random walks. Independent grammar seeds preserve the historical
style RNG sequence and prevent one new choice from reshuffling tempo, harmony,
percussion or timbre.

The public palette control now adds airy, wooden, warm, shimmering, plucked and
soft-electronic identities. Natural palettes bias coherent catalog banks and
also apply restrained synthesis/mix differences; soft-electronic remains fully
sample-free. Fingerprints include the resolved grammar, palette and harmony
identity. Bake-off selection first covers those structures, then improves the
minimum and mean pairwise distance while retaining one result per family.

---

## 4. Voice — options

| Engine | Spanish | Expressive control | Speed on M1 | License | Verdict |
|---|---|---|---|---|---|
| **Kokoro-82M** (chosen) | 3 voices: `ef_dora`, `em_alex`, `em_santa` — European-leaning | Speed only; vary prosody post-hoc | Faster than realtime on CPU | Apache-2.0 | Best speed/quality trade-off for v1 |
| Chatterbox Multilingual (Resemble AI) | 23 languages, better Spanish | Real `exaggeration` / `cfg_weight` knobs; zero-shot cloning | ~0.5 B params, uneven MPS support — slow | MIT | Best quality if speed is acceptable |
| XTTS-v2 (Coqui) | Yes, 16 languages | Cloning | Moderate | **CPML — non-commercial** | License blocks commercial use |
| F5-TTS | Yes | Cloning | Moderate | MIT | Viable alternative |
| macOS `say` | Paulina (Mexican), Mónica (Spain) | Rate only | Instant | System | Zero-dependency fallback |

**Measured:** Kokoro `ef_dora` renders "la cuenta, por favor" in **1.57 s** of
audio. At 117 BPM (bar = 2.05 s) that barely fits; phrases need a slower tempo or
a two-bar slot. This is why the prototype defaults to a lower BPM.

**Requirement:** Kokoro needs `espeak-ng` for Spanish G2P — `brew install espeak-ng`.

### Intonation variation — three tiers

1. **Post-hoc prosody** (implemented): re-synthesise each repeat with ±1–2
   semitone pitch shift, 0.95–1.05 rate, and a slight gain contour. Cheap and
   effectively indistinguishable from "changed emphasis" at background-listening
   attention.
2. **Native expressive control**: swap in Chatterbox and drive `exaggeration`.
3. **Seed variation**: regenerate with different random seeds. Weakest, since
   Kokoro is largely deterministic per voice.

### What was actually built and measured (iteration 2)

`mlx-audio` turned out to be the practical route on Apple Silicon. It ships MLX
ports of **Chatterbox, IndexTTS, Zonos2, Higgs Audio, Qwen3-TTS, MOSS-TTS,
Dia, Sesame** and more, and its dependencies are light — `mlx`, `numpy`,
`scipy`, `transformers>=5.14`, `huggingface_hub`, `miniaudio`, `sounddevice`
— with **no torch pin**, so it installs alongside Kokoro without conflict.

This matters: the PyPI `chatterbox-tts` package pins `torch==2.6.0` and
`numpy<2.0.0`, which can never share an environment with Kokoro's torch 2.13
and numpy 2.x. The MLX route sidesteps that entirely.

Verified against the MLX source and by running it: `generate()` takes
`lang_code`, `ref_audio`, `exaggeration`, `cfg_weight`, `temperature`, `speed`,
`repetition_penalty`, `min_p`, `top_p`. Note it is `lang_code`, not
`language_id` as the class docstring suggests.

**Why the later repeats sounded mechanical.** The v1 pipeline pitch-shifted
each repeat with `librosa.effects.pitch_shift`, which uses librosa's own phase
vocoder — documented as a reference implementation that "makes no attempt to
handle transients, producing many audible artifacts". Three changes:

1. Chatterbox varies delivery natively, so no post-processing is applied at all.
2. Where a shift is still used (Kokoro), it goes through Rubber Band via
   `pedalboard.PitchShift`.
3. The variation table was rescaled from ±0.9 to ±0.4 semitones, leaning on
   speed — which is native to the engine and artifact-free — instead of pitch.
   `--prosody-strength` scales the whole table.

**Emotion from emoji.** `vocab.py` already captured the emoji trailing each
headword and discarded it. 922 of 939 entries (98%) carry one, so the notes are
effectively pre-annotated. Faces and gestures map to a named delivery; object
emoji fall through to a warm neutral. Resulting distribution: warm 76%,
emphatic 7.5%, thoughtful 3.8%, sad 3.0%, angry 2.2%, the rest below 2%.

### Bilingual references (iteration 3)

Chatterbox now conditions Spanish and English independently. The defaults clone
macOS Paulina (`es_MX`) for Spanish and Daniel (`en_GB`) for English, using a
natural reference script in the matching language. This fixes the Spanish accent
that resulted when the old Paulina clip was reused for English. Kokoro keeps its
separate `ef_dora` and `af_heart` voices as the fast fallback.

### Music parameterisation

`BedSpec` (`lexibeat/bedspec.py`) holds metre, harmony, four layers and space.
Chords are derived from `root` + `scale` + scale degrees rather than a hard-coded
table; the voicing formula `[r, r+7, r+12, r+12+third, r+19]` reproduces the v1
progression **exactly**, which is how the refactor was verified as
sound-preserving. Drum patterns use one string step per sixteenth note, with
their length derived from the selected meter, so rhythm remains data.

Four styles — `yoga` (the original), `nocturne`, `lofi`, `warm` — define
*ranges* that a seeded rng samples within, so the same style gives related but
distinct beds.

### Sampled instruments

Salamander Grand Piano (Yamaha C5, CC-BY 3.0, Alexander Holm): 30 root notes at
**minor-third** spacing across 16 velocity layers. Two layers are fetched (~88 MB
of the 748 MB library) into `~/.cache/lexibeat/samples/`. Minor-third spacing
means resampling never exceeds 1.5 semitones, so the timbre holds. Pure Python
plus `soundfile`, so it works on Linux as well as macOS.

Iteration 3 replaces filename inference with explicit sample manifests and adds
curated CC0 VSCO 2 CE strings, marimba and glockenspiel. Strings can replace the
synth pad without replacing the piano lead; marimba and glockenspiel are sparse
lead alternatives. Downloads remain explicit and are grouped by
`--download-samples vsco`.

Beds can now be 3/4, 4/4 or 5/4; use unextended, seventh, add-nine or ninth
voicings; and move the pad filter along seeded sine, triangle or smoothed-random
curves. The renderer exposes pad, bass, drum and lead stems so speech can duck
each layer by a different amount.


---

## 5. Timing and mixing

Two details that make or break the result:

- **Onset alignment, not file alignment.** TTS output carries leading silence.
  Trim to the first onset and align *that* to the downbeat.
- **Overrun handling.** If an utterance is longer than its slot, time-stretch it
  slightly (pitch-preserving) rather than letting it drift off the grid.

**Loudness balance** — fixed gains are the wrong tool. Use:

- **Sidechain ducking:** music down ~4–6 dB while speech is present, ~100 ms
  attack, ~400 ms release.
- **LUFS normalisation:** speech ≈ −16 LUFS, music bed ≈ −26 LUFS
  (`pyloudnorm`).

This is what "clear but not shouting" means numerically, and it adapts
automatically to whatever bed is used.

---

## 6. Environment constraints

Python version is unusually tight:

| Package | Requires |
|---|---|
| `kokoro` 0.9.4 | `>=3.10, <3.13` |
| `misaki` 0.9.4 | `>=3.8, <3.13` |
| `librosa` 1.0.0 | `>=3.12` |
| `stable-audio-tools` 0.0.20 | `>=3.10, <3.11` |
| `audiocraft` 1.3.0 | `>=3.8` (but old torch pins) |
| `beat-this` 1.1.0 | `>=3` |

**Python 3.12 is the only version satisfying Kokoro + librosa together.** The
system default here is 3.14.7, which has wheels for almost none of this. Stable
Audio Open cannot share the environment at all (needs 3.10) — use the MLX port or
a separate venv.

---

## 7. Things to try next

Done in iterations 2–3: Chatterbox for genuine intonation control; separate
native Paulina/Daniel references; parameterised and randomisable music beds;
recorded piano, strings, marimba and glockenspiel; chord extensions; alternative
meters; filter automation; and per-layer sidechain ducking.

Done in the expressive-voice iteration:

- **IndexTTS 2.5 via an isolated native-MLX worker** — consumes the explicit
  `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]` vector
  from `Emotion.vector()` and applies the model's `duration_factor`. A first
  overlong render is regenerated once with a calibrated factor. The published
  control is not sample-exact, so the ordinary bounded fit remains a measured
  fallback rather than being hidden.
- **VoxCPM2, Qwen3-TTS, TADA and Fish Audio S2 Pro** — experimental comparison
  backends spanning instruction-controlled emotion/prosody, dynamic stochastic
  duration, and word-level emotion tags. Each declares its real capabilities,
  licence and fallback behavior in the generated statistics.
- **Zonos2 deferred** — current ZONOS2 in mlx-audio does not expose the emotion
  and pitch-variance controls described by the earlier design note. Those
  controls belong to Zonos v0.1, whose released language list excludes Spanish.

The first one-item listening bake-off also made clear that successful generation
is not linguistic correctness. IndexTTS 2.5 followed Spanish stress best among
the new models, but exposed short-reference boundary artifacts. Fish rendered
Spanish with incorrect stress and a pronounced final `h`; Qwen's Spanish retained
an unsuitable accent and hallucinated trailing material; and VoxCPM2 degraded on
later Spanish repetitions despite strong English variation. Chatterbox remained
the subjective quality leader, although it too has occasionally inserted or
mispronounced a short initial syllable. These observations are artifact-specific,
so the benchmark records them separately from finite/stereo/timing validation.

Still open:

- Hybrid music: neural 8-bar loop, grid-snapped with `beat_this`, then looped.
- Expanding-interval spacing across the track rather than blocked repeats.
- Sung rather than spoken delivery, to capture the actual song superiority effect.
- A "test mode" track: Spanish only, with silence where the English would be.

### Hosted expressive TTS snapshot (2026-08-29)

Hosted speech is opt-in and does not replace Chatterbox. Gemini 3.1 Flash TTS is
the most promising current fit because its exact-text TTS endpoint accepts
director instructions for tone, accent, pace and style in both Spanish and
English. Cloudflare Aura 2 supplies separate Spanish and English models but only
exposes speaker and encoding controls; MeloTTS exposes only text and language.
The two Cloudflare paths therefore use the same restrained local speed/pitch
variation as Kokoro, and record that post-processing rather than describing it
as native model control. A live REST/schema check on 2026-08-29 also confirmed
that Cloudflare still rejects MeloTTS `lang: "es"` with AiError 8002, matching
[cloudflare/ai#221](https://github.com/cloudflare/ai/issues/221); that backend is
therefore an English-only diagnostic rather than a viable lesson renderer.

| Service/model | Direct expressive control | Free allocation | Published paid rate | Five minutes/day (150 min/month) |
|---|---|---:|---:|---:|
| Gemini 3.1 Flash TTS Preview | prompt controls for style, accent, pace and tone | free tier, rate-limited | ~$0.03/audio min | ~$4.50 if billed |
| Gemini 2.5 Flash / Flash-Lite TTS | same prompt controls; Flash-Lite is preview and single-speaker | no Vertex free allocation | ~$0.015/audio min | ~$2.25 on paid Vertex |
| Gemini 2.5 Pro TTS | higher-control structured speech workflows | no Vertex free allocation | ~$0.03/audio min | ~$4.50 on paid Vertex |
| Cloudflare Aura 2 ES/EN | none beyond contextual delivery and speaker | 10,000 Workers AI neurons/day shared | $0.03/1k characters | about 4,500 chars/day; exceeds free allocation slightly and requires the $5 Workers Paid plan |
| Cloudflare MeloTTS | language only; deployed Spanish path currently broken | same 10,000 neurons/day | $0.0002/audio min | ~$0.03; well inside free allocation |
| Hume Octave 1 | acting instructions and speed | ~10 min/month | $0.15/1k characters above Creator quota | roughly $15/month at current Creator allowance |
| Cartesia Sonic 3 | explicit emotion, speed and volume | ~27 min/month | $5 plan includes ~133 min | slightly above the $5 plan allowance |
| Eleven v3 | emotional audio tags and stability modes | ~10 min/month | ~$0.10/audio min | ~$15 usage-equivalent; current Creator plan includes 220 min |
| Google Cloud Chirp 3 HD | pace, pause and pronunciation; no comparable emotion prompt | first 1M chars/month, billing required | $30/M characters | normally inside the character allowance |
| OpenAI GPT-4o Mini TTS | natural-language voice instructions | no API free tier | ~$0.015/audio min | ~$2.25, but the model is deprecated |

For the smaller target of 50 vocabulary pairs per day, three Spanish/English
repetitions are roughly 3,000 billed characters when each short utterance
averages ten characters. That is about 8,182 Aura neurons and remains inside the
daily Cloudflare allocation. Five continuous minutes is a deliberately more
conservative projection and depends on language, word length and delivery rate.
The Gemini free project observed during the 2026-08-29 acceptance run is limited
to 3 RPM and 10 RPD for each preview TTS model. Consequently the requested ten
items x three repetitions x two languages (60 requests) cannot complete in one
quota day without batching or a quota upgrade; pacing fixes RPM errors but cannot
fix an exhausted daily allowance.

The paid Vertex experiment uses Application Default Credentials and the project
and `global` location already established by the sibling `vim-mastery` project.
Google Cloud currently exposes four Gemini-TTS endpoints there:
`gemini-3.1-flash-tts-preview`, `gemini-2.5-flash-tts`,
`gemini-2.5-flash-lite-preview-tts`, and `gemini-2.5-pro-tts`. Vertex returns
headerless 24 kHz mono PCM, which the backend decodes directly. The separate
Cloud Text-to-Speech API is required for Chirp 3 HD and was not enabled when the
Vertex setup was inspected.

#### Batched Gemini pause experiment

Gemini 3.1 documents `[short pause]` and `[long pause]` inline audio tags, but
the Gemini-TTS API does not document word timestamps, SSML marks, or distinct
audio parts for multiple phrases. Cloud TTS streaming chunks are transport
chunks and are not promised to align with input phrase boundaries. The batching
experiment therefore makes six calls (two languages by three repeat styles),
puts ten phrases behind `[long pause]` separators in each call, then splits at
the nine longest local silence regions. It uses one neutral emotional direction
per batch so every phrase in a batch truly shares the same delivery settings.

The 2026-08-29 live run found exactly ten segments in all six responses. The
shortest chosen gap was 0.85 seconds and the weakest selected/unselected gap
confidence was 5.16x. The reconstructed 60-utterance listening file was finite
with a 0.95 peak. The experiment used six calls instead of 60 but cost
`$0.099465`, compared with `$0.059309` for separate Gemini 3.1 requests, because
the generated silence is billable output audio. This route is useful when
request quota is the constraint; it still requires listening for omitted words,
spoken tags, or bad segment boundaries.

Sources: [Gemini TTS](https://ai.google.dev/gemini-api/docs/speech-generation),
[Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing),
[Gemini-TTS on Google Cloud](https://docs.cloud.google.com/text-to-speech/docs/gemini-tts),
[Gemini 3.1 audio-tag prompting](https://cloud.google.com/blog/products/ai-machine-learning/gemini-3-1-flash-tts-on-google-cloud/),
[Cloudflare Aura 2 ES](https://developers.cloudflare.com/workers-ai/models/aura-2-es/),
[Aura 2 EN](https://developers.cloudflare.com/workers-ai/models/aura-2-en/),
[MeloTTS](https://developers.cloudflare.com/workers-ai/models/melotts/),
[Workers AI pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/),
[Hume acting instructions](https://dev.hume.ai/docs/text-to-speech-tts/acting-instructions),
[Hume pricing](https://www.hume.ai/pricing),
[Cartesia controls](https://docs.cartesia.ai/build-with-cartesia/capability-guides/volume-speed-emotion),
[Cartesia pricing](https://www.cartesia.ai/pricing),
[ElevenLabs TTS](https://elevenlabs.io/docs/overview/capabilities/text-to-speech),
[ElevenLabs pricing](https://elevenlabs.io/pricing/api),
[Chirp 3 HD](https://docs.cloud.google.com/text-to-speech/docs/chirp3-hd),
[Google Cloud TTS pricing](https://cloud.google.com/text-to-speech/pricing), and
[OpenAI GPT-4o Mini TTS](https://developers.openai.com/api/docs/models/gpt-4o-mini-tts).

---

## 8. Sources

- [Beat This! Accurate beat tracking without DBN postprocessing](https://arxiv.org/pdf/2407.21658)
- [beat_this — GitHub](https://github.com/CPJKU/beat_this)
- [BeatNet — GitHub](https://github.com/mjhydri/BeatNet)
- [Beat Transformer: Demixed Beat and Downbeat Tracking](https://arxiv.org/pdf/2209.07140)
- [BeatFM: Improving Beat Tracking with Pre-trained Music Foundation Model](https://arxiv.org/pdf/2508.09790)
- [musicgen-mlx — GitHub](https://github.com/andrade0/musicgen-mlx)
- [mlx-audiocraft — GitHub](https://github.com/theashishmaurya/mlx-audiocraft)
- [stable-audio-mlx — GitHub](https://github.com/sandst1/stable-audio-mlx)
- [stable-audio-open-small — Hugging Face](https://huggingface.co/stabilityai/stable-audio-open-small)
- [stable-audio-open-1.0 — Hugging Face](https://huggingface.co/stabilityai/stable-audio-open-1.0)
- [Kokoro Spanish voice bug report](https://github.com/hexgrad/kokoro/issues/301)
- [Kokoro-82M voices & language codes](https://soniqo.audio/guides/kokoro)
- [Kokoro vs XTTS vs Chatterbox comparison](https://localaimaster.com/blog/kokoro-vs-xtts-vs-chatterbox)
- [Best Local TTS Models 2026](https://localaimaster.com/blog/best-local-tts-models)
- [MOSS-TTS — GitHub](https://github.com/OpenMOSS/MOSS-TTS)
- [Coqui TTS — GitHub](https://github.com/coqui-ai/TTS)
- [Amphion toolkit — GitHub](https://github.com/open-mmlab/Amphion)
- [langlearnai — PyPI](https://pypi.org/project/langlearnai/0.0.3/)
- [vocabulary-to-speech-api — GitHub](https://github.com/elhalili/vocabulary-to-speech-api)
- [Music Training Program based on language development and neuroscience principles](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3846262/)
- [mlx-audio — GitHub](https://github.com/Blaizzy/mlx-audio)
- [Chatterbox — Resemble AI](https://github.com/resemble-ai/chatterbox)
- [IndexTTS — GitHub](https://github.com/index-tts/index-tts)
- [IndexTTS2 paper](https://arxiv.org/pdf/2506.21619)
- [Salamander Grand Piano (CC-BY 3.0)](https://archive.org/details/SalamanderGrandPianoV3)
- [SalamanderGrandPiano SFZ mirror](https://github.com/sfzinstruments/SalamanderGrandPiano)
- [VSCO 2 Community Edition (CC0)](https://github.com/sgossner/VSCO-2-CE)
- [librosa phase vocoder caveat](https://librosa.org/doc/main/generated/librosa.phase_vocoder.html)
- [pedalboard — Spotify](https://github.com/spotify/pedalboard)
