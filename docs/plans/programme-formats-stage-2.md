# Programme formats, stage 2: rendering the richer formats

Stage 1 is built. A loop is described by a format ([`docs/programme-format.md`](../programme-format.md),
`lexibeat/formats/`); `format` and `switches` replaced `pattern`; and `classic` and `alternating`
render byte-identically to the drills they replaced. But this version renders only one `words`
section of `say`, `gap` and `rest`. Everything else in the grammar parses and is refused by name.

This plan makes the rest render, in two parts that each ship on their own:

- **2a: formats that need no model.** `echo` and `review`; the `quiz`, `review`, `intro` and
  `outro` sections; cues, pace and stretch. The timeline becomes a cue list here.
- **2b: formats a writer model fills in.** `radio-lesson` and `story`, through an injected writer.

Stage 3, the host integration in Acervo, comes after both
([`programme-loops.md`](programme-loops.md), "What changes for the host").

**Rules this plan keeps.** They come from the programme-blocks listening
(`experiments/programme_blocks/report.md`) and from this repository's invariants:
- speed comes from stretching, to at most 1.2×, and slowness from asking the voice;
- pronunciation help is the whole word said slowly;
- a remark carries no grammar, spelling or linguistic term;
- a remark or mnemonic is followed by the word, and a story by a review;
- a mixed-language line goes only to a voice that can mix, and otherwise the format falls back to a
  plain one;
- no constrained decoding;
- no credential on the integration path.

## 2a — template formats and the cue timeline

### The voice: a role, a pace, one capability (`lexibeat/voice.py`)

- **`SpeechRequest.role`:** `"native"` for a source-language line (the word, an example, a story
  line) and `"guide"` for a learner-language line (the translation, a cue, a remark, an intro).
  - It is a hint. A host backend with a second speaker uses it. LexiBeat's own backends already
    pick a voice per language, so for them the guide is simply their learner-language voice.
  - Because of this, **`guide_voice` leaves the grammar's `requires`**: a format degrades rather than
    being refused. The spec and its examples change with it.
- **`Delivery.pace`:** empty by default, or `"slow"`, `"natural"` or `"fast"`. When it is set,
  `delivery_instruction` uses it in place of the prosody band, so `pace: "slow"` reaches the
  director note.
- **`BackendCapabilities.mixes_languages`:** false by default, true for the Gemini backends. It is
  what `requires: ["multilingual_voice"]` is checked against.
- **`stretch`:** applied after `Speaker.say` with `dsp.time_stretch`, capped at 1.2. It never slows a
  line down.

### Phrases in the learner's language (`lexibeat/formats/phrases/{en,ru,es}.json`)

- **What they hold:** 10–20 `your_turn` cues, and `intro` and `outro` templates with `{count}`.
- **Choosing a line:** a cue is picked deterministically from the seed and the word's index, so
  neighbouring words hear different cues and a re-render hears the same ones.
- **A learner language with no phrase file** is refused by name when a format needs phrases. In 2b a
  writer, when there is one, translates the English phrases once for that render.

### The planner (`lexibeat/programme.py`)

- `plan(format, items, languages, phrases, seed) -> Programme`. A programme is an ordered list of
  segments: `kind`, `section`, `item` (or none), `role`, `language`, `text`, `bars` (a fixed count,
  or none for as long as the line takes), `take`, `pace`, `stretch`, `direction`.
- **Order:**
  - sections run in the format's order;
  - a `quiz` with `at: "middle"` goes after the first half of the words and covers that half;
  - `at: "end"`, the default, follows all the words and covers them all;
  - `review` covers every word;
  - an `intro` or `outro` with `text: "template"` uses the phrases.
- **Take indices** count each word's lines per side across sections, so repetitions keep varying.

### The arranger and the timeline (`lexibeat/arrange.py`, `lexibeat/loop.py`)

- **The arranger:**
  - `arrange()` walks a programme. A fixed segment is fitted to `bar × 0.92`, as now. A variable one
    is spoken naturally and takes `ceil((length + margin) / bar)` bars.
  - The long-take retry keeps working per word and side.
- **The result's `timeline` becomes two fields:**
  - `items`: one row per word — `index`, `source`, `target`, `direction`, plus `start`, `end`,
    `source_reveal` and `target_reveal` within the words section. This is what a retrieval display
    turns on.
  - `cues`: every line — `kind`, `section`, `item`, `role`, `language`, `text`, `start`, `end`,
    `take`. It is ordered by start, and a word may appear in it more than once.
- **The wire version:** if `v0.7.0` has not been tagged when this starts, it stays API 2.0.0 and
  package 0.7.0; otherwise it becomes 3.0.0 and 0.8.0.

### Formats (`lexibeat/formats/`)

- **What renders** grows to:
  - sections `words`, `quiz`, `review`, and `intro`/`outro` with template text;
  - steps `say` (with `pace` and `stretch`), `gap`, `rest` and `cue`;
  - the section parameters `at` and `stretch`.
- **Still refused by name until 2b:** writer steps, any `when` but `always`, `chunk`, any `order` but
  `as_given`, `group_headers`, and `bed`.
- **The requirement check moves to render time**, because it needs the backend and the writer:
  `renderable(format, switches, capabilities, has_writer)`. A missing requirement with a `fallback`
  renders the fallback, and the result reports `format: "classic", fallback_from: "radio-lesson"`.
  Without a fallback it is refused, naming the requirement.
- **New built-ins:**
  - `echo.json`: the word, a cue, a silent bar, the word, the translation, a rest.
  - `review.json`: the meaning, a gap, the word; then every word again at 1.2×. It has no `bed`
    change: changing the bed's tempo mid-loop is a music-engine change of its own.
- **`/schema`** lists every built-in this version renders, with its `requires`, so a host can show
  or hide the writer formats.

### Everything else in 2a

- `service.py`: the result fields, and a refreshed OpenAPI snapshot.
- The CLI's `--format` gains the new ids.
- The Lab lists the formats that need no writer.
- `docs/service.md`: the result shape.
- The spec: its "What renders today" table, and `guide_voice` removed.

### Tests (2a)

- **Planner:** the section order and a middle and an end quiz cover the right words. Take indices
  continue across sections. The cue choice is deterministic and varies from word to word.
- **Speech requests:** each line carries the right role and language; `pace` reaches the director
  note; `stretch` shortens a take and is capped.
- **Timeline:** the `items` reveals for `classic` are unchanged; the `cues` for `review` hold each
  word twice.
- **Requirements:** a fallback is taken and reported; a missing requirement is refused by name.
- **Byte-identical guard:** `classic` still renders byte-identically to the previous version, checked
  as stage 1 was, by rendering both from a clean checkout with the same words and seed.

## 2b — writer formats

### The writer interface (`lexibeat/writer.py`)

- **The `Writer` protocol:** `write(request: WriteRequest) -> str`, where a `WriteRequest` is a
  prompt and a purpose. The host owns the model call and returns text. LexiBeat owns the prompt, the
  parser and what a good line is.
- **`create_service(writer_factory=None)`:** the factory is given the `RenderContext`, as
  `backend_factory` is. A format requiring `writer` falls back or is refused when there is none.
- **`GeminiWriter`, for the CLI and tests only:**
  - It reads `GEMINI_API_KEY` and uses `google-genai` (the `hosted-tts` extra), as the CLI's Gemini
    voice does.
  - It tries `gemini-3.8-flash`, then 3.7, then 3.5-flash, falling through on 429 and 5xx only.
  - It is never on the service's integration path. The CLI selects it with `--writer gemini`.

### The prompt and the script (`lexibeat/prompts/programme.md`, `lexibeat/script.py`)

- **One writer call per render.** Its prompt is assembled from what the resolved format needs:
  - per-word examples;
  - remarks of the allowed kinds, with "no remark" an explicit option;
  - `hard_to_say` flags;
  - callbacks, with "none" allowed;
  - story beats per chunk;
  - the intro and outro;
  - a word order or topic groups with titles;
  - phrase translations, when the learner language has no phrase file.
- **What seeds the prompt:** the experiment's winning prompts
  (`experiments/programme_blocks/prompts/example_punchy_v1.md`, `story.md`, `callback.md`,
  `commentary.md`, `framing.md`), with the listening's rules written in:
  - no grammar, spelling or linguistic terms;
  - a remark is short;
  - `hard_to_say` only for a word commonly said wrong, never merely a long one;
  - a callback only where the link reads naturally;
  - an example is short, funny and memorable;
  - every line names its language.
- **The reply is JSON, read by `script.parse`** with no constrained decoding:
  - it takes the object out of a fenced or bare reply;
  - it checks indices, kinds, lengths (a line at most 500 characters, a direction at most 200) and
    that every required part is present;
  - a missing optional part means it did not happen.
- **A reply that cannot be read fails the render**, and the parser's sentence survives every
  hand-off. Retrying belongs to the host's model chain, not to LexiBeat.

### The planner, extended

- **The `when` checks:**
  - `writer_decides`: a remark or callback is present;
  - `hard_to_say`: the flag is set;
  - `natural_link`: a callback is present;
  - `then` runs only when its step happened.
- **Text written by the writer:**
  - `pronounce` is the word said with `pace: "slow"`, native role;
  - an example is a native line, then its translation in the guide voice when `translate` is set;
  - a remark and the writer's intro and outro are guide lines;
  - a story beat is native lines, each followed by its translation.
- **Ordering:** `order: "writer"` and `group_by_topic` use the writer's order and groups;
  `group_headers` speaks each group's title; `chunk` and `after_chunk` place story beats.
- **New built-ins, taken from the spec's examples:**
  - `radio-lesson.json`: requires `writer` and `multilingual_voice`, with fallback `classic`, because
    its remarks quote the word inside a learner-language sentence;
  - `story.json`: requires `writer`.

### Tests (2b)

- **Offline, with a fake writer returning canned replies:**
  - each of the parser's refusals names the fault;
  - `when` and `then` behave as described;
  - story chunks and topic groups come out right;
  - with no writer, `radio-lesson` falls back to `classic`;
  - a render with the fake writer produces the expected cues.
- **Live, gated by `RUN_LIVE_WRITER_TESTS=true`, on `GEMINI_API_KEY`:**
  - scripts for `radio-lesson` and `story` over two word sets, Spanish from English and English from
    Russian;
  - each asserts that the reply parses, every word has an example, the remark kinds are allowed, and
    no remark uses a grammar term (a small word list).
  - The scripts are saved to a scratch file to be read by a person, and each records which model
    answered.

## Verification

- The full suite passes after 2a and again after 2b, with the OpenAPI snapshot refreshed each time.
- **2a:** `classic` is still byte-identical to the previous version. `echo` and `review` render
  through the service with a fake backend: the cues are ordered, the peak stays at or below 0.97,
  and the review lines are shorter by the stretch.
- **2b:** the live writer test passes, and the generated scripts are read before the part is called
  done.
- **Optional:** one real render each of `radio-lesson` and `story` through the CLI
  (`--backend gemini-vertex --writer gemini`), billed to the Cloud project, for a listener to judge.
- The prohibited-word grep is clean.

## Model calls, estimated

Every text call goes to the Gemini API on `GEMINI_API_KEY`. A render makes **one** writer call,
however long it is. The offline tests and CI use a fake writer and make none.

| What | Text calls (`GEMINI_API_KEY`) | Speech calls |
|---|---|---|
| 2a, all of it | 0 | 0: a fake voice |
| 2b prompt development: the assembled prompt on the two word sets, read, adjusted | ~10–20 | 0 |
| 2b live test: 2 formats × 2 word sets = 4 calls a run, ~3–5 runs while the prompt settles | ~12–20 | 0 |
| **Total for implementing and testing** | **~25–40** | 0 |
| Optional real renders, 8 words each (a radio lesson is ~100 lines, a story ~60) | 2 | ~160 on Vertex (the Cloud project, not the key), roughly $0.10–0.30 |

- **Quota:** 25–40 calls over the work is well inside a free key's daily text allowance. The known
  risk is overload, not quota: in the experiment the strong models answered 503 for a whole run, and
  the replies came from `gemini-3.5-flash-lite`.
- **The fallback:** the CLI writer falls through to the next model on 503. Each saved script records
  which model answered, so a lite model's script is not mistaken for the prompt's best.
- **No speech on the key:** the Gemini API's own TTS allows 10 calls a day on the free tier, far too
  few for a render. Real audio goes through Vertex.
