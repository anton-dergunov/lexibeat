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
| `requires` | no | What the render needs: `writer`, `guide_voice`, `multilingual_voice` |
| `fallback` | no | The format rendered instead when a requirement is missing |
| `switches` | no | The choices a listener may make, by name |

### Sections

| `kind` | What it is | Takes |
|---|---|---|
| `intro` | The opening line | `text`: `template` (the default) or `writer` |
| `words` | Each word in turn, through its `block` | `block`; `group_headers` (a line before each topic group); `chunk` (words taken so many at a time) with `after_chunk` (steps after each chunk) |
| `quiz` | A recall pass over the words | `block`; `at`: `middle` or `end` |
| `review` | Every word again | `block`; `stretch`; `bed`: `same` or `quicker`; `choice` |
| `outro` | The closing line | `text` |

Any section may carry `switch`, naming a switch that turns it off.

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

**Bars.** A step takes one bar, and each line starts on its bar's downbeat. `gap` and `rest` take
the number of bars they name. A line whose length depends on its text — a remark, a story beat —
takes as many whole bars as it needs.

### Switches

```json
{
  "switches": {
    "remarks": {"label": "Remarks about words", "default": true},
    "review":  {"label": "Final review", "default": "normal", "choices": ["off", "normal", "fast"]}
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
- **Fast comes from stretching, slow from asking.** `stretch` goes no higher than 1.2: past that, a
  recording sounds processed. Slowness is asked of the voice (`pace: "slow"`), because a slowed
  recording sounds metallic.
- **Remarks are about hearing, meaning and use.** There is no grammar kind, and no remark about
  spelling: the programme is listened to.
- **A remark or a mnemonic is followed by the word**, with `then`.
- **A story is followed by a review**, so its words are heard plainly once more.

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
  "requires": ["guide_voice"],
  "sections": [
    {"kind": "words", "block": [
      {"say": "word"}, {"cue": "your_turn"}, {"gap": 1},
      {"say": "word"}, {"say": "translation"}, {"rest": 1}
    ]}
  ]
}
```

### Radio lesson

An intro, each word with an example and, where the writer finds one worth it, a remark; a quiz
halfway; every word again at the end.

```json
{
  "id": "radio-lesson",
  "label": "Radio lesson",
  "description": "An intro, each word with an example and sometimes a remark, a quiz halfway, then every word again.",
  "requires": ["writer", "guide_voice"],
  "order": "group_by_topic",
  "switches": {
    "remarks": {"label": "Remarks about words", "default": true},
    "quiz":    {"label": "Quiz halfway", "default": true},
    "review":  {"label": "Final review", "default": "normal", "choices": ["off", "normal", "fast"]}
  },
  "sections": [
    {"kind": "intro", "text": "writer"},
    {"kind": "words", "group_headers": true, "block": [
      {"say": "word"}, {"gap": 1}, {"say": "translation"},
      {"say": "word"}, {"say": "translation"},
      {"example": "writer", "translate": true},
      {"remark": {"kinds": ["contrast", "register", "false_friend", "mnemonic", "culture"]},
       "when": "writer_decides", "switch": "remarks",
       "then": [{"say": "word"}, {"say": "word"}]},
      {"pronounce": "slow_whole", "when": "hard_to_say",
       "then": [{"say": "word"}]},
      {"rest": 1}
    ]},
    {"kind": "quiz", "at": "middle", "switch": "quiz",
     "block": [{"say": "translation"}, {"gap": 1}, {"say": "word"}]},
    {"kind": "review", "switch": "review",
     "block": [{"say": "word"}, {"say": "translation"}],
     "choice": {"fast": {"stretch": 1.2, "bed": "quicker"}}},
    {"kind": "outro", "text": "writer"}
  ]
}
```

### Story

```json
{
  "id": "story",
  "label": "Story",
  "description": "A short story told in pieces between the words, then every word again, briskly.",
  "requires": ["writer", "guide_voice"],
  "order": "writer",
  "sections": [
    {"kind": "intro", "text": "writer"},
    {"kind": "words", "chunk": 2,
     "block": [{"say": "word"}, {"say": "translation"}, {"say": "word"}],
     "after_chunk": [{"story_beat": "writer"}]},
    {"kind": "review", "block": [{"say": "word"}, {"say": "translation"}], "stretch": 1.1},
    {"kind": "outro", "text": "writer"}
  ]
}
```

Words come two at a time, and a piece of the story follows each pair. `order: "writer"` lets the
writer arrange the words to fit its plot.

### Review, for words already known

```json
{
  "id": "review",
  "label": "Review",
  "description": "The meaning first, a pause to find the word, then a quick run through all of them.",
  "sections": [
    {"kind": "words", "block": [{"say": "translation"}, {"gap": 1}, {"say": "word"}]},
    {"kind": "review", "block": [{"say": "word"}, {"say": "translation"}],
     "stretch": 1.2, "bed": "quicker"}
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
| Speed up gradually over the loop | A parameter | A `ramp` on a section, if it is ever wanted |
| A B A B across two words, not within one | A new section kind | `interleave`, not a construct |
| Every third word, a quiz on the last three | Borderline | One counter (`every: 3`) is tolerable; two is a language |
| Extra practice for a verb that is also hard to say | No | Needs `and` and word data from the host: an expression language |
| A callback to a related earlier word | No, and not needed | A judgement: the writer makes it (`natural_link`) |
| Under five minutes, dropping remarks first | No | A priority policy, and the planner's |

## What renders today

The whole grammar parses. This version renders:

| Part | Rendered |
|---|---|
| Sections | One `words` section |
| Steps | `say` (without `pace` or `stretch`), `gap`, `rest` |
| `when` | `always` |
| `order`, `requires`, `fallback` | `as_given`; no requirements |
| A `words` block | Must say both the word and its translation, because each timeline row is one word with both of its reveals |

A format using anything else is refused by name — "format 'story' uses order 'writer', requires
'writer', …, which this version cannot render yet" — never rendered in part. `/schema` lists only
the built-in formats this version renders: `classic` and `alternating`.
