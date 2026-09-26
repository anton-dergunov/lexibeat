# Programme blocks

**Question.** `docs/plans/programme-loops.md` lists the blocks a loop could be built from besides
today's drill: other drill structures, examples, stories, callbacks, commentary and framing.
Before any of them goes into `arrange.PATTERNS` or a writer backend, each one is tried **on its
own**. What does it sound like, what does a model write for it, and which versions are worth
building? Each candidate is scored 1–5 on a tablet, and the scores decide what P1–P3 build and
which prompts are iterated. Sound effects are out of scope; they have their own experiment and
plan.

## Running it

This runs in the repository's own environment, because the drills are rendered with LexiBeat's
beds, stretch, fit and mix. What you hear is what a loop would do.

```bash
uv run --extra hosted-tts --env-file .env python experiments/programme_blocks/run.py all --dry-run
uv run --extra hosted-tts --env-file .env python experiments/programme_blocks/run.py all
uv run python experiments/programme_blocks/run.py serve        # then open the printed address
```

One stage at a time: `drills`, `syllables`, `mixed`, `context`, `commentary`, `framing` or
`pronounce`. `page`
rebuilds `out/index.html` from the reports without calling anything.

- **Speech** goes to Cloud Text-to-Speech with Application Default Credentials. This is the path
  Acervo's production voices use: `gemini-3.1-flash-tts-preview` for directed lines, WaveNet for
  clear ones. The quota project comes from `GOOGLE_CLOUD_PROJECT` or from the ADC file.
- **Text** goes to the Gemini free tier through `GEMINI_API_KEY`. The strongest free model answers
  first (`gemini-3.8-flash`, then 3.7, then 3.5), and the lite models are the fallback.
- **Cached.** Every master and every reply is cached under `out/cache/`. Re-running a stage after
  editing one prompt buys only what that prompt changed.
- **Dry run.** `--dry-run` buys no speech and prints how many calls would be bought. A full run is
  about 330 speech calls, well under a dollar.

Scores are saved to `out/labels.json`, with a draft kept in the browser. `out/labels.json` and each
stage's `out/<stage>/report.json` are tracked, since they are the result; the audio, the cache and
`out/index.html`, which `page` rebuilds from the reports, are not.

**Open the page through `serve`, never as a file.** Opened from disk, the page cannot reach the
server, and its scores stay only in that browser's storage for that file's address, where a page
served on a port cannot see them. **Download scores**, at the top of the page, saves whatever the
page holds as a file to merge into `out/labels.json`: that is how G's scores were recovered. A page
that finds scores the server lacks, such as ones kept while the server was down, sends them as soon
as it loads.

## The example vocabulary

`words.json` is a sample of the owner's own vocabulary, taken from an Acervo export: 226 words,
100 random Spanish and 100 random English, plus the words the stages name. It is tracked on
purpose, so the experiment runs from a clean clone. Spanish is glossed in English and English in
Russian, as in the vocabulary. `syllable_words.json` is the hand-picked syllable set.

## Sections

**A. Drills over music.** One pair, *el atasco — traffic jam*, in every structure, over one bed
(bright-organic, 80 BPM). The native voice is Kore, production's default, and the guide is Charon.
- Structures:
  - classic (today's `retrieval`, the reference);
  - five two-or-four rhythms;
  - the same five reversed, plus translation, a silent bar, then the word;
  - four echo variants;
  - slow–normal–fast by stretching, by asking, or both;
  - a speed round over a quicker bed (gentle-game at 100 BPM) at 1.0–1.4×, asked to be fast,
    or two pairs a bar;
  - a chant, with syllables on eighth notes: one take cut and snapped to the grid, each
    syllable recorded alone, or pause tags.
- Each card shows the grid of what starts on each eighth note.
- Production's director note (`delivery_instruction`, framed by Acervo's take prompt) is used
  throughout. The speed and chant lines are the exception and use a bare "Speak …" note, because
  the production note's pace words would contradict them.

**B. Syllables.** Splitting is only worth it for hard words, so the words are chosen twice:
- a detector prompt (`prompts/hard_words.md`) runs over all of `words.json`;
- a hand-curated set, `syllable_words.json`, covers Spanish, English, Russian, Italian and French.

`prompts/syllabify.md` splits every item, marking stress in capitals because the lite models got
an index wrong about one word in four. Each item is then spoken four ways:
- a Gemini voice asked to split the word;
- Gemini `[short pause]` tags;
- WaveNet SSML `<break>`;
- one natural take cut at its loudness dips (`cut.py`).

**C. Mixed-language voices.** Ten presenter lines, each mixing two or three of English, Spanish,
Russian, French, German, Italian and Mandarin, spoken by ten configurations:
- Gemini 3.1 with a prompt naming each part's language, in two voices;
- Gemini 3.1 with no prompt;
- Gemini 3.1 with the quoted language's `languageCode`;
- Gemini 2.5 with no prompt;
- the Gemini API's 3.8 Flash TTS, which detects the language;
- WaveNet: plain, with SSML `<lang>`, and with SSML `<voice>`;
- Aura-1.

Every card shows the exact request.

**D. Context** (text only):
- punchy examples from two prompts compared on the same 20 words;
- joint examples;
- mini-stories in two styles, shown as they would play between the drills;
- callbacks of five kinds (reuse, contrast, quiz, chain, twist) instead of "remember X?".

**E. Commentary** (text only). One call per lesson of about twelve words, about fifty words in all.
- The prompt lists seven kinds of remark: register, contrast, false friend, pronunciation,
  mnemonic, joke and culture.
- The model writes candidates only for the kinds that fit a word, flags factual claims, and
  picks one, or two at most.
- The candidates are kept, as material for a later judge.

**F. Framing** (text only):
- for three lessons, the teaching order with a reason for each word, section headers, a title,
  an intro and an outro;
- concrete examples of further framing blocks, and three the model proposes itself.

**G. Pronunciation hints**, the follow-up to B. B cut words into syllables by their **spelling**
and got several wrong, and a pronunciation clip that is wrong teaches the wrong word. Here the
words B got wrong, plus one it got right as a control, are given the dictionary's pronunciation
instead. `pronunciation_words.json` holds each word's IPA from Wiktionary (two written by hand,
where Wiktionary has none, and flagged), its syllables split from that IPA, and a human recording
from Wikimedia Commons where one exists.
- **A probe first.** Each mechanism is given a decoy word with another word's pronunciation, so
  it can be heard whether the pronunciation is honoured at all or the voice just said the word.
- **Three conditions per word:** the plain word asked for slowly; the IPA, whole, slowly; the
  IPA syllables, then the word.
- **Four voices:**
  - Gemini 3.1 on Cloud TTS, production's voice, which documents no pronunciation input: IPA in
    slashes in the text, or named in the prompt;
  - Gemini 3.8 Flash TTS through the Gemini API's Interactions endpoint, whose guide says IPA in
    slashes is followed; pace goes in a `speech_metadata` style;
  - Chirp 3 HD, the same speakers, with `customPronunciations`, `speakingRate` and SSML
    `<phoneme>`;
  - WaveNet with SSML `<phoneme>`.
- **Beside them,** the human recording (rated, to check against) and B's directed clip (not
  rated, from the cache).
- **Scoring is stricter than elsewhere:** 1 if any sound is wrong, 2–5 only for a correct clip.

## What the documentation says about mixed-language speech

- **Must the text be in the voice's language?**
  - **Gemini voices:** no. The voices are multilingual, so the same voice speaks any supported
    language and may switch within a line.
    - Cloud TTS still *requires* a `languageCode`. It sets the output language and accent: the
      language the presenter is "from".
    - The Gemini API's own TTS takes no language code and "detects the input language
      automatically".
  - **WaveNet voices:** yes. Each voice belongs to one locale, and other languages are read with
    that locale's phonetics unless SSML says otherwise.
- **Coverage.**
  - Gemini-TTS on Cloud TTS lists about 25 GA languages, including Spanish, Russian, French,
    German, Italian and US English, and 75 or more in preview, including Mandarin.
  - The Gemini API's 3.8 Flash TTS claims more than 130 languages, and its Lite model more than 100.
  - No page quantifies how quality falls off for rarer languages.
- **Marking which part is in which language.**
  - The only standard is SSML `<lang xml:lang="…">`. Cloud TTS supports it "on a best effort
    basis", in the same voice unless `<voice>` switches speaker. Semitic languages come out as
    silence, and Japanese Kanji is read as Chinese.
  - Gemini voices take plain text and a prompt, with no SSML and no per-span language field. The
    prompt is the only tool, and nothing is unified across providers.
  - Aura-1 is English only.

Sources: [Cloud TTS: Gemini-TTS](https://docs.cloud.google.com/text-to-speech/docs/gemini-tts),
[Cloud TTS: SSML](https://docs.cloud.google.com/text-to-speech/docs/ssml),
[Gemini API: speech generation](https://ai.google.dev/gemini-api/docs/speech-generation),
[Cloudflare: Aura-1](https://developers.cloudflare.com/workers-ai/models/aura-1/).

## Measured so far

**Which voices honour a written pronunciation** (G's probe, checked by transcribing the decoy
clips with a Gemini model, then confirmed by ear; the scored findings are in
[report.md](report.md), section G):
- Chirp 3 HD `customPronunciations`, Chirp 3 HD `<phoneme>` and WaveNet `<phoneme>` said the
  target, not the decoy, in English, Spanish and French.
- Gemini 3.1 said the target when the IPA was named in the prompt and the decoy was in the text, in
  English, Spanish and Russian, but not in French.
- **Russian has no working route on Google's voices.** `<phoneme>` is silently ignored by both
  Chirp and WaveNet, whatever the IPA is written with, and a Russian custom pronunciation is refused
  as IPA. As X-SAMPA it is accepted only without soft consonants, which Russian cannot do without.
  So G renders no Chirp or WaveNet IPA clip for Russian.
- Chirp refuses some IPA symbols as invalid rather than approximating them: the /ʝ/ of Wiktionary's
  first transcription of *ayuntamiento*, so the control uses its /j/ variant.
- On *lethargy*, Gemini 3.8 read the IPA syllables as "le · ther · gy" and Gemini 3.1 garbled them.
- **Gemini 3.8's free key allows three calls a minute and ten a day.** Its 45 clips take five days
  of runs on that key; each run fills in from the cache onward.


**Asking for speed** (section A's table: twelve lines, trimmed length relative to "naturally",
no stretching):
- "fast" gives 1.14–1.68× (about 1.3× typical);
- the `[extremely fast]` tag gives 0.74–1.73× (about 1.5× typical, but once *slower*);
- "very fast and rhythmic, like a rapper on the beat" gives 0.87–1.45×, and is as often slower
  as faster.

So the plan's advice holds: stretch for speed, and let a direction only nudge. A short plain word
("fast") beats an elaborate one.

**Asking for slowness** overshoots:
- "slowly and deliberately, stretching every syllable" spread *el atasco* over 6.6 s, which no bar
  holds;
- "slowly, like a patient teacher, without pausing between syllables" gives about 3 s, roughly
  3.4× the natural 0.87 s;
- a chanted take is slow too: *la berenjena* came back at 4.9 s.

So the chant stretches its take towards its slots before cutting it, capped at 2×.

**Text models.** The strong free models were overloaded ("high demand", 503) for the whole first
run. Apart from the syllabifier (3.8-flash) and one example prompt (3.5-flash), every reply came
from `gemini-3.5-flash-lite`, and the model that answered is on every card. The lite model follows
the commentary prompt loosely: one candidate per word, mostly `contrast`, and many factual claims.
That is a finding about the model as much as about the prompt. A reply from a lite model is kept
but not final: re-running a stage asks the strong models once more before settling for it.

**Aura-1.** Cloudflare's daily free allocation was already spent on the first run, so the Aura
cards say so instead. A later run fills them in from where it stopped.

## Verdicts

The first run has been scored; what the listening found, section by section, is in
[report.md](report.md).
