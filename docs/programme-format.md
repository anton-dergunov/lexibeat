# Programme formats

A loop is made from a **format**: a small JSON document that says what a loop consists of — a
classic drill, a short radio lesson, a story told between the words, a quick review. Formats live in
LexiBeat, one file each under `lexibeat/formats/`. A host chooses one by name and may set the few
switches it declares; it never builds one. `lexibeat/formats/__init__.py` parses the whole grammar
described here, and a test parses every example on this page.

## Three layers

| Layer | What it is | Made by | Carries text? |
|---|---|---|---|
| **Format** | The recipe: sections, and the steps each word goes through | A file in LexiBeat | Never |
| **Programme script** | One render's segments, with the sentences a writer model wrote for these words | The planner, from the format, the words and the writer | Yes |
| **Timeline** | What starts when in the finished track | The arranger | Yes |

A format never contains a sentence, a word or a language. The same format teaches Mandarin from
Portuguese as readily as Spanish from English.

## Choosing a format

```json
{
  "items": [{"source": "el atasco", "target": "traffic jam", "direction": ""}],
  "source_language": {"code": "es", "name": "Spanish"},
  "target_language": {"code": "en", "name": "English"},
  "format": "radio-lesson",
  "switches": {"remarks": false, "review": "fast"}
}
```

- **`format`** is a built-in format's id, as `/schema` lists them. It may instead be a whole format
  inline, which is for experiments and the command line (`--format-file`), not for a host.
- **`switches`** sets the switches that format declares, by name, and nothing else. A switch left out
  takes its default. An unknown switch, or a value the switch does not offer, is refused.
- **`/schema`** lists every built-in format this version can render, each as
  `{id, label, description, switches, requires, bars_per_item, utterances_per_item,
  has_recall_gap}`. A host draws its dropdown from `label` and `description`, and its checkboxes and
  choices from `switches`, so a format added to LexiBeat appears in the host with nothing changing
  there.

Why a name and not the structure: the host's interface is a dropdown and a few switches, and the
grammar below is LexiBeat's to change. Were the structure the API, every change to it would be a
versioned contract, and its mistakes would surface as validation errors in someone else's app.

## The grammar

### A format

| Key | Required | Meaning |
|---|---|---|
| `id` | yes | Lower-case letters, digits and hyphens. A built-in format's file is named after it |
| `label` | yes | What a listener chooses it by, at most 60 characters |
| `description` | yes | One sentence saying what it is |
| `sections` | yes | The programme, in order. At least one is a `words` section |
| `order` | no | How the words are ordered: `as_given` (the default), `group_by_topic`, or `writer` (the writer orders them, e.g. to fit a plot) |
| `requires` | no | What the render needs: `writer` (a writer model), `multilingual_voice` (a voice that can say words of two languages in one line) |
| `fallback` | no | The format rendered instead when a requirement is missing |
| `switches` | no | The choices a listener may make, by name |

### Sections

| `kind` | What it is | Takes |
|---|---|---|
| `intro` | The opening line | `text`: `template` (the default) or `writer` |
| `words` | Each word in turn, through its `block` | `block`; `repetitions` (how many times the block's repeated run plays, 1–6); `group_headers` (a line before each topic group); `chunk` (words taken so many at a time) with `after_chunk` (steps after each chunk); `choice` |
| `quiz` | A recall pass over the words | `block`; `at`: `middle` or `end`; `announce` |
| `review` | Every word again | `block`; `stretch`; `bed`: `same` or `quicker`; `announce`; `choice` |
| `outro` | The closing line | `text` |

Any section may carry `switch`, naming a switch that turns it off.

A `quiz` and a `review` **announce themselves**: the guide says what is coming ("Now, all the words
once more") before the first word, in one of several wordings from the phrase files. `announce:
false` leaves it out. A word heard again with no warning sounds like a mistake, or like the drill
starting over.

### Steps

A `block` is a list of steps, run once per word. Each step names exactly one kind:

| Step | What is heard | Parameters |
|---|---|---|
| `{"say": "word"}` / `{"say": "translation"}` | The word, or its translation | `pace`: `slow`, `natural` or `fast`, asked of the voice; `stretch`: 1.0–1.2 |
| `{"gap": n}` | *n* silent bars to recall in | |
| `{"rest": n}` | *n* bars of music alone | |
| `{"cue": "your_turn"}` | A short prompt to speak, from a bank of lines in the learner's language | |
| `{"example": "writer"}` | An example sentence the writer wrote for this word | `translate`: true or false |
| `{"remark": {"kinds": [...]}}` | One short remark about the word | `kinds` from `contrast`, `register`, `false_friend`, `mnemonic`, `culture`, `joke` |
| `{"pronounce": "slow_whole"}` | The word said whole and slowly, as pronunciation help | |
| `{"story_beat": "writer"}` | The next piece of the story | |
| `{"callback": {"kinds": [...]}}` | A line that links back to an earlier word | `kinds` from `reuse`, `contrast`, `chain`, `quiz` |

A `words` section takes every step. A `quiz` takes `say`, `gap`, `rest` and `cue`. A `review`
takes `say`, `gap` and `rest`. `after_chunk` takes `story_beat`, `callback`, `remark` and `rest`.

Every step may also carry:
- **`when`**: whether it happens. One of a closed list of named checks, each implemented by the
  planner: `always` (the default), `writer_decides` (the writer judges it worthwhile),
  `hard_to_say` (the word is one a learner is likely to say wrong), `natural_link` (the writer found
  a link that reads naturally).
- **`then`**: steps that follow it only when it actually happened, such as repeating the word after
  a remark. A step with `then` needs a `when` other than `always`, and a `then` step takes no `when`,
  `switch` or `then` of its own.
- **`switch`**: a switch that turns the step off.
- **`repeat`**: marks the step as part of the block's **repeated run**, which plays as many times as
  the words section's `repetitions` says (once by default). The steps marked `repeat` stand
  together, and there is one run per block. It is how a listener chooses how many times each word
  is said, through a switch whose `choice` sets `repetitions`.

**Bars.** A step takes one bar, and each line starts on its bar's downbeat. `gap` and `rest` take
the number of bars they name. A line whose length depends on its text — a remark, a story beat —
takes as many whole bars as it needs.

### Switches

```json
{
  "switches": {
    "remarks":     {"label": "Remarks about words", "default": true},
    "repetitions": {"label": "Times each word is said", "default": "3", "choices": ["2", "3", "4"]}
  }
}
```

A switch is either **on or off** (`default` true or false), or a **choice** among named values
(`choices`, with `default` one of them). A step or section naming a switch is removed when that
switch is `false` or `"off"`. A section's `choice` maps a choice to overrides of its own parameters,
applied when that choice is made.

## Rules the listening fixed

These come from the programme-blocks experiment (`experiments/programme_blocks/report.md`) and are
built into the grammar rather than left to each format:

- **Pronunciation help is the whole word, said slowly** (`pronounce: "slow_whole"`). Split into
  syllables it was wrong on about half the words tried, on every voice, with or without a written
  pronunciation.
- **Slowness is asked for, never made.** `pace: "slow"` goes to the voice, because a slowed recording
  sounds metallic. `stretch` can speed a line up, to at most 1.2, but **no built-in format uses
  it**: in a real loop even 1.1× on the Gemini voice was heard as metallic against the rest.
- **Remarks are about hearing, meaning and use.** There is no grammar kind, and no remark about
  spelling: the programme is listened to.
- **A remark comes before the example**, as something about the word itself, and the word is not
  said again after it: in a real loop that repetition was heard as a recap starting.
- **A pair belongs together.** Where a quiz or a review says both sides, it says them back to back
  and then rests, so the two are heard as one unit; a gap between them splits it.
- **A word's block ends in a rest**, and so does a story beat, so each is heard as one block rather
  than running straight into the next.
- **A quiz and a review announce themselves**, and **a story is followed by a review**, so its words
  are heard plainly once more.

## Examples

### Classic drill

Each word three times with its translation, with a silent bar to recall the meaning the first time.
This is the built-in `classic`.

```json
{
  "id": "classic",
  "label": "Classic drill",
  "description": "Each word three times with its translation, with a pause to recall the meaning the first time.",
  "sections": [
    {"kind": "words", "block": [
      {"say": "word"}, {"gap": 1}, {"say": "translation"},
      {"say": "word"}, {"say": "translation"},
      {"say": "word"}, {"say": "translation"},
      {"rest": 1}
    ]}
  ]
}
```

### Say it back

```json
{
  "id": "echo",
  "label": "Say it back",
  "description": "Hear the word, say it yourself in the pause, then hear it again.",
  "sections": [
    {"kind": "words", "block": [
      {"say": "word"}, {"cue": "your_turn"}, {"gap": 1},
      {"say": "word"}, {"say": "translation"}, {"rest": 1}
    ]}
  ]
}
```

### Radio lesson

An intro, then each word said three times with a recall gap after the first (the listener can
choose two or four), a remark where the writer finds one worth it, an example, and the word said
slowly if it is one learners say wrong; a quiz halfway; every word again at the end.

```json
{
  "id": "radio-lesson",
  "label": "Radio lesson",
  "description": "An intro, each word with an example and sometimes a remark, a quiz halfway, then every word again.",
  "requires": ["writer", "multilingual_voice"],
  "fallback": "classic",
  "order": "group_by_topic",
  "switches": {
    "repetitions": {"label": "Times each word is said", "default": "3", "choices": ["2", "3", "4"]},
    "remarks": {"label": "Remarks about words", "default": true},
    "quiz": {"label": "Quiz halfway", "default": true},
    "review": {"label": "Final review", "default": true}
  },
  "sections": [
    {"kind": "intro", "text": "writer"},
    {"kind": "words", "group_headers": true, "switch": "repetitions", "repetitions": 2,
     "choice": {"2": {"repetitions": 1}, "3": {"repetitions": 2}, "4": {"repetitions": 3}},
     "block": [
      {"say": "word"}, {"gap": 1}, {"say": "translation"},
      {"say": "word", "repeat": true}, {"say": "translation", "repeat": true},
      {"remark": {"kinds": ["contrast", "register", "false_friend", "mnemonic", "culture"]},
       "when": "writer_decides", "switch": "remarks"},
      {"example": "writer", "translate": true},
      {"pronounce": "slow_whole", "when": "hard_to_say", "then": [{"say": "word"}]},
      {"rest": 1}
    ]},
    {"kind": "quiz", "at": "middle", "switch": "quiz",
     "block": [{"say": "translation"}, {"say": "word"}, {"rest": 1}]},
    {"kind": "review", "switch": "review",
     "block": [{"say": "word"}, {"say": "translation"}, {"rest": 1}]},
    {"kind": "outro", "text": "writer"}
  ]
}
```

### Story

```json
{
  "id": "story",
  "label": "Story",
  "description": "A short story told in pieces between the words, then every word again.",
  "requires": ["writer"],
  "order": "writer",
  "sections": [
    {"kind": "intro", "text": "writer"},
    {"kind": "words", "chunk": 2,
     "block": [{"say": "word"}, {"say": "translation"}, {"say": "word"}, {"rest": 1}],
     "after_chunk": [{"story_beat": "writer"}, {"rest": 1}]},
    {"kind": "review", "block": [{"say": "word"}, {"say": "translation"}, {"rest": 1}]},
    {"kind": "outro", "text": "writer"}
  ]
}
```

Words come two at a time, each block ending in a rest, and a piece of the story follows each pair,
with a rest of its own. `order: "writer"` lets the
writer arrange the words to fit its plot.

### Review, for words already known

```json
{
  "id": "review",
  "label": "Review",
  "description": "For words you know: the meaning first, a pause to find the word, then all of them once more.",
  "sections": [
    {"kind": "words", "block": [{"say": "translation"}, {"gap": 1}, {"say": "word"}, {"rest": 1}]},
    {"kind": "review", "block": [{"say": "word"}, {"say": "translation"}, {"rest": 1}]}
  ]
}
```

### Degrading to a plainer voice

```json
{
  "id": "polyglot",
  "label": "Three languages",
  "description": "Each word with a remark that compares it with the same word in another language.",
  "requires": ["writer", "multilingual_voice"],
  "fallback": "classic",
  "sections": [
    {"kind": "words", "block": [
      {"say": "word"}, {"gap": 1}, {"say": "translation"},
      {"remark": {"kinds": ["contrast"]}, "when": "writer_decides",
       "then": [{"say": "word"}]},
      {"rest": 1}
    ]}
  ]
}
```

A voice that cannot mix languages gets `classic` instead, and the result says so. Directions on
individual lines are simply dropped for a plain voice; that is not a requirement.

## Where the grammar stops

The grammar is closed on purpose. It stays readable because it has fixed section kinds, a closed
list of named checks, `then`, and parameters one level deep. It becomes a programming language at the
first boolean combination, variable, counter or reference from one segment to another, and at that
point the wanted behaviour becomes a **named capability** instead: a new `when`, a new section
`kind`, or a decision the writer makes.

| Wanted | In the grammar? | How |
|---|---|---|
| A hard word gets two more bars | Yes | `when: "hard_to_say"` with `then` |
| Repeat the word after a remark | Yes | `then` |
| Say each word three times, or four, as the listener chooses | Yes, the one counter | A `repeat` run, `repetitions`, and a switch's `choice` |
| Speed up gradually over the loop | A parameter | A `ramp` on a section, if it is ever wanted |
| A B A B across two words, not within one | A new section kind | `interleave`, not a construct |
| Every third word, a quiz on the last three | No | `repetitions` is the one counter the grammar has; a second is a language |
| Extra practice for a verb that is also hard to say | No | Needs `and` and word data from the host: an expression language |
| A callback to a related earlier word | No, and not needed | A judgement: the writer makes it (`natural_link`) |
| Under five minutes, dropping remarks first | No | A priority policy, and the planner's |

## Voices and roles

Every line has a **role**. A line in the language being learned — the word, an example, a story
line — is the **native** presenter's. A line in the learner's own language — the translation, a cue,
a remark, an intro — is the **guide's**. The role travels with the line to the voice as a hint: a
voice with a second speaker uses it, and one without says everything itself. So there is no
requirement for a second voice, and a format never has to be refused for want of one.

What a format *can* require is a writer, and a voice that can mix languages in one line
(`multilingual_voice`), which a remark quoting the word in a learner-language sentence needs. A
render missing a requirement renders the format's `fallback` instead, and says so in its result
(`format` is the fallback, `fallback_from` the one asked for); without a fallback it is refused,
naming the requirement.

**Learner-language phrases** — the cues, the quiz and review announcements, and the intro and outro
when their `text` is `template` — come from `lexibeat/formats/phrases/<language>.json`: English,
Russian and Spanish for now, with several wordings of each, varied from line to line by the seed. For
a learner language with no phrase file, a writer, when there is one, writes them for that render;
without a writer the format is refused, naming the language.

## The timeline

A render's result carries two views of what was said:

- **`items`**, one row per word, from its block in the words section: `index`, `source`, `target`,
  `direction`, `start`, `end`, and `source_reveal` and `target_reveal` — when each side is first
  heard, which is what a retrieval display turns on. A side the block never says has no reveal.
- **`cues`**, every line in the order it is heard: `kind` (`say`, `cue`, `announce`, `intro`,
  `outro`, and the written kinds),
  `section`, `item` (none for a line that belongs to no word), `side` (`source` or `target`, for a
  word's own line), `role`, `language`, `text`, `take`, `start`, `end`. A word appears in it as
  often as it is said — twice, in a format with a review.
- **`group`**, on every cue, says which lines are **shown together**: a word's own lines for one
  word in one section (its drill, a quiz or review pair), a line and its `translation`. Every other
  line — an example, a remark, a story line, a header, an announcement — is a group of its own.
  Groups rise in the order heard, and a player shows a group from its first line's start until the
  next group starts, revealing each line as it is first heard. One function assigns them
  (`programme.group_lines`), so a player never has to guess which line translates which.

## Written lines

A format that takes text from a writer — examples, remarks, `pronounce` and every other step gated
by a check the writer answers, a story, a written intro or outro, topic groups — gets it from **one
writer call per render** (`lexibeat/script.py`). The prompt is assembled from `lexibeat/prompts/`,
one part for each thing the format will use, so the writer is never asked for a remark the format
has switched off. It carries the rules above, plus two the listening added:

- `hard_to_say` is true only for a word learners commonly say wrong — a misleading spelling, an
  unexpected stress — never for one that is merely long;
- a story line is at most ten words, so a listener can follow it by ear.

The reply is plain text, read by `script.parse`: no JSON mode, no schema. The parser checks every
part it relies on and refuses the reply naming the first thing wrong — "beats: needs one beat per
group of 2 words: 4" — and a missing optional part means it did not happen. A reply that cannot be
used fails the render; retrying belongs to the host's model chain.

How the named checks read the script:

- **`writer_decides`**: the step happens when the writer wrote it (an example, a remark, a
  callback);
- **`hard_to_say`**: the writer flagged the word;
- **`natural_link`**: the writer wrote a callback;
- **`then`**: runs only when its step happened.

`pronounce` is the word said with `pace: "slow"` by the native voice, taking as many bars as it
needs. A written line — an example and its translation, a remark, a story line, a header — likewise
takes the bars it needs; it is never squeezed into one.

## What renders today

The whole grammar parses. This version renders all of it except:

| Not yet | Why |
|---|---|
| `bed` on a `review` | Changing the bed's tempo mid-loop is a music-engine change of its own |
| `remark` and `callback` after a chunk | A chunk is followed by a story beat or a rest |
| More than one `words` section | The quiz and the review are defined over the one words section |

A format using any of these is refused by name — "format 'x' uses 'bed' on the 'review' section,
which this version cannot render yet" — never rendered in part. `/schema` lists the built-in
formats: `classic`, `alternating`, `echo`, `review`, `radio-lesson` and `story`, with what each
requires, so a host can show the writer formats only where it has a writer.
