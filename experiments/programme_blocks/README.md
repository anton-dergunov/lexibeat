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

One stage at a time: `drills`, `syllables`, `mixed`, `context`, `commentary` or `framing`. `page`
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

Scores are saved to `out/labels.json`, with a draft kept in the browser. `out/` is not tracked.

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

To be written from `out/labels.json` once the page has been scored.
