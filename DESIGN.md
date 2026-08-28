# Earworms Generator — Design Notes & Options

Research and design decisions behind this prototype: generating Earworms-style
language-learning audio (vocabulary spoken over a rhythmic ambient bed) from a
word list, fully locally on an Apple Silicon Mac.

---

## 0. What was built

```
generate.py          CLI entry point
earworms/vocab.py    parses the Obsidian markdown notes into (Spanish, English) pairs
earworms/music.py    procedural ambient bed synthesised at a known BPM
earworms/voice.py    Kokoro TTS with per-repetition prosody variation
earworms/arrange.py  places utterances on downbeats according to a pattern
earworms/mix.py      LUFS normalisation, sidechain ducking, block limiter
DESIGN.md            this document
```

### Running it

```bash
brew install espeak-ng          # required by Kokoro for Spanish G2P
uv sync

uv run generate.py --words 12 --out out/lesson.wav
uv run generate.py --words 6 --dry-run          # show the plan, generate nothing
```

Useful flags: `--mode words|phrases|mixed`, `--pattern retrieval|earworms`,
`--bpm`, `--seed`, `--voice-es`, `--voice-en`, `--duck-db`, `--speech-lufs`,
`--music-lufs`. A sidecar `.txt` tracklist with timestamps is written next to
the audio.

### Measured behaviour of the output

| Check | Result |
|---|---|
| Render time | ~40 s for a 4.2 min / 10-word track (M1, 16 GB) |
| Music bed render | 0.5 s for 49 s of audio |
| Utterance onset vs downbeat | **median 12 ms** offset |
| Utterance durations (single words) | 0.62–1.07 s, against a 2.76 s slot at 80 BPM |
| Integrated loudness | −17.5 LUFS, true peak 0.97 |
| Recovered tempo of the finished mix | 80.75 BPM (requested 80) |

Speech synthesis dominates the runtime at roughly 1.6 s per utterance; repeated
strings are cached.

### Known limitations

- Kokoro's Spanish voices are European-leaning; `ef_dora` is not a Latin
  American accent. See §4 for alternatives.
- Only 40% of the 939 vocabulary entries carry an example sentence, so
  `--mode phrases` silently falls back to the headword for the other 60%.
- The `Item` translation takes only the first semicolon-separated sense, so
  "the check; the bill" is taught as "the check".
- Repeats of a word are blocked together rather than spaced across the track
  (see §2, improvement 2 — not yet implemented).
- Single fixed chord progression and one arrangement; the bed varies only by
  `--seed`.

---

## 1. Analysis of the reference track

Measured from `Rapid Spanish (Latin American), Vols. 1–3 - Earworms Learning.mp3`
(184.6 s) using `librosa.beat.beat_track` plus a crude speech-band energy VAD
(300–3400 Hz, 70th-percentile threshold):

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

Reproduce with:

```bash
uv run --with librosa python -c "
import librosa, numpy as np
y, sr = librosa.load('Rapid Spanish (Latin American), Vols. 1-3 - Earworms Learning.mp3', sr=22050, mono=True)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
print(tempo, np.median(np.diff(librosa.frames_to_time(beats, sr=sr))))
"
```

---

## 2. Does the method actually work?

**Split verdict.**

Earworms' own claims ("engages both hemispheres of the brain", "proven to be
highly effective") are marketing copy. No peer-reviewed validation of the product
itself was found — search results for the method returned only vendor material and
app reviews.

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
3. **Phrases over bare words** (supported). Earworms teaches phrases, not isolated
   vocabulary. The Obsidian notes contain example sentences, which are arguably
   better material than the headwords.

### Prior art

No project was found that does this. Nearest neighbours:

- [langlearnai](https://pypi.org/project/langlearnai/0.0.3/) — builds Anki decks from
  word lists with TTS audio and images. No music, no beat alignment.
- [vocabulary-to-speech-api](https://github.com/elhalili/vocabulary-to-speech-api) —
  generates an audio file from a vocabulary list. No music.
- **Glossika** (commercial) — spaced audio repetition, no music.
- **Pimsleur** (commercial) — graduated interval recall, no music.
- **Earworms / Berlitz MBT** (commercial) — human-recorded, not generated.

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
- **Note:** the Earworms bed is deliberately unremarkable, so the quality gap is
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

- Hybrid music: neural 8-bar loop, grid-snapped with `beat_this`, then looped.
- Chatterbox for genuine intonation control.
- Expanding-interval spacing across the track rather than blocked repeats.
- Sung rather than spoken delivery, to capture the actual song superiority effect.
- Latin American Spanish voice (Kokoro's are European-leaning) — Chatterbox
  cloning from a short reference sample, or macOS Paulina.
- A "test mode" track: Spanish only, with silence where the English would be.

---

## 8. Sources

- [Earworms — the Musical Memorisation Method](https://www.earwormslearning.com/about)
- [Earworms MBT review (Mezzofanti Guild)](https://www.mezzoguild.com/review-berlitz-earworms-music-brain-trainer/)
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
- [The song that never ends: repeated exposure and earworm development](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10585939/)
- [Music Training Program based on language development and neuroscience principles](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3846262/)
