# From drills to programmes

A loop today is one drill: each word three times, its translation three times, in varied voices,
over a bed. That works, and it stays. This plan is for what a loop could be besides — something
closer to a short radio lesson you put on and take part in — and for how to build it without
losing what already works. It is a research-and-design plan for the next session, not a
specification: it lists more options than will be built, groups them, and says roughly how each
would be generated.

## What it is for

The whole point is that words stick because the music carries them. Everything below is judged by
that: does it make a word more memorable, or come back more easily a day later? Three things follow.

- **Memorable beats complete.** A short, odd, funny example ("I didn't shave my moustache today")
  sticks; an accurate dictionary sentence does not. A joke, a sound, a callback to an earlier word is
  worth more than another repetition.
- **Participatory beats passive.** A silence you are invited to fill ("your turn") is recall, and
  recall is what the retrieval pattern already turns on.
- **Choice beats one method.** No single format suits every mood or every set of words. The owner
  picks a format when making a loop, or asks to be surprised.

Two constraints hold throughout:

- **It degrades gracefully.** Every format must still work with a plain voice that takes no
  direction: directions are dropped and nothing else changes. With an expressive voice, everything
  is directed and as emotional as the line allows.
- **The target language is heard a lot; commentary is in the learner's language.** A programme is
  not a lecture about Spanish in English. Commentary is short and gets out of the way.

## The building blocks

Each block is a kind of segment a programme can contain. For each: what it is, what it needs, and
what it costs.

**Cost tiers:**
- **T** — template only: no model call beyond the voice.
- **L** — one text-model call per loop, shared across many segments.
- **L+** — its own text-model call.
- **V** — voice calls, which the take cache makes cheap on a re-render.

### Drills — the words themselves

| Block | What you hear | Needs | Cost |
|---|---|---|---|
| Classic | Word, translation, three times each, in varied voices (today) | — | V |
| Two or four | The same pair, two or four times, e.g. four to fill four bars | — | V |
| Reverse | The translation first, a gap to recall the word, then the word | — | V |
| Echo | The word, then "your turn" and a silent bar, then the word again | a cue line | T, V |
| Slow, normal, fast | The word three times, each quicker; the last one on the beat | time-stretch | V |
| Speed round | Every pair once, back to back, at about 1.3×, as a closing sprint | time-stretch | V |
| Chant | The word's syllables placed on eighth notes, sung-spoken on the beat | syllabification | V |
| Syllables | A hard word split, then whole: "a-ta-sco … atasco" | syllabification | T, V |

### Context — the word in use

| Block | What you hear | Needs | Cost |
|---|---|---|---|
| Example | A short sentence using the word, then its translation. LexiBeat writes its own, short and punchy, for the radio format | writer | L, V |
| Joint example | One sentence tying two or three neighbouring words together: "buried in the desert up to my neck" | writer, ordering | L, V |
| Mini-story | A thread through the whole set, told in pieces between the drills | writer | L+, V |
| Callback | Refers back to an earlier word: "remember the desert? Here it is again …" | writer, ordering | L, V |

**Examples are LexiBeat's own.** Acervo holds examples, some of them generated, but they were
written for an article, not for the ear. Using them would also mean sending examples as well as
words through the host interface. LexiBeat writes its own first; if those fall short, Acervo's stored
examples are the fallback, and only then does the interface grow.

### Commentary — why the word is interesting

| Block | What you hear | Cost |
|---|---|---|
| Usage and register | "That's the blunt way to say it — here's the polite one", e.g. for "to pee oneself" | L, V |
| Contrast | How this word differs from the similar one you might confuse it with | L, V |
| False friend | "It looks like X in English, and it means something else entirely" | L, V |
| Pronunciation | "The double r rolls — listen again" | L, V |
| Mnemonic | A keyword or image that links sound and meaning | L, V |
| Joke or pun | Anything that makes you smile at the word | L, V |
| Cultural note | Where and when people actually say it | L, V |

Commentary is where invented facts are likeliest. See the risks section.

### Framing — the programme around the words

| Block | What you hear | Cost |
|---|---|---|
| Intro | "Today: twelve words about food and travel" — topics and count | T (templated) or L |
| Section header | "Now, three words for the kitchen" | T or L |
| Encouragement | "Nice — keep going", sparingly | T |
| Midpoint quiz | Translations only, a gap each, then the answers | T, V |
| Final review | Every pair once, in order, optionally as the speed round: "now all of them, faster" | T, V |
| Outro | A one-line recap and goodbye | T or L |

### Sound

| Block | What you hear | Cost |
|---|---|---|
| Sound effect | A short, real sound for an iconic word: a siren for "police", rain, a cat, a razor for "moustache" | tagging (L) plus a sound pack |
| Music cue | The bed responds: a swell into the intro, a drop for the speed round, stop-time under "your turn" | arrangement work |

### Voices

- **A native voice** says everything in the target language.
- **A guide voice** says commentary and framing in the learner's language.
- **Optionally, a two-person dialogue** for examples and jokes.
- Every line can carry its own direction: excited for the intro, conspiratorial for a mnemonic,
  playful for a repeated funny word. With a plain voice, those directions are dropped.

## Formats — presets built from blocks

| Format | Built from |
|---|---|
| Classic drill | Today's loop, unchanged |
| Quick fire | Two repetitions each at pace, then a speed-round review |
| Radio lesson | Intro; per word: classic plus an example plus one line of commentary; midpoint quiz; final review; outro |
| Story mode | Intro; the mini-story told between short drills; callbacks; final review |
| Review | Reverse drills and a speed round only — for words already known |
| Surprise me | A random, sensible mix, different every time |

In Acervo, the format is chosen in the make dialog next to the music, the same way: a short list,
each with a sentence saying what it is, and Surprise me first.

## Ordering the words

The words arrive as a random sample of the owner's scope (choosing words by hand is a separate,
later feature). Within a programme they can be reordered:

- **Keep near-synonyms and look-alikes apart.** Adjacent similar words interfere in memory.
- **Pair words that combine naturally**, so joint examples and callbacks have something to work
  with.
- **Group by topic** when the intro names topics.
- **Start with a heuristic** (topic, part of speech, similarity of spelling). Use a model only when a
  format wants joint examples, and then let the writer choose the pairs.

## How a programme is generated

**One structured artefact: the programme script.**
- A list of segments, each `{kind, voice, lang, text, direction, items, sfx, music_cue}`.
- Written as JSON and checked by LexiBeat's own parser, never by constrained decoding: the host's
  rule, measured there. A schema guarantees shape, not quality, and hides the model's drafting in
  the fields.

**Who writes it: an injected writer backend.**
- LexiBeat holds no provider credentials; its voice is injected by the host. The writer is injected
  the same way.
- **LexiBeat owns** the prompt templates, the script schema and the parser, and so owns what an
  example or a joke should be like.
- **The host owns** only the model call, a call home like `/pronunciations/take`, and supplies the
  words. The standalone CLI can inject a local writer.
- The host interface stays "words in, track out" for the first phases.

**The pipeline:**
1. **Order and group** the words (heuristic, or part of the writer's call).
2. **Write the script.** One call for the whole programme; optionally a second call that critiques
   and trims it (too long, not funny, wrong register). Templates fill the T-tier segments with no
   call at all.
3. **Speak each segment** through the injected voice. The take cache makes a re-render free.
4. **Arrange** the segments on the bar grid. A commentary line may span several bars; a sound effect
   sits under or just before its word; the bed ducks deeper under commentary than under a single
   word; music cues change the bed at section boundaries.
5. **Emit a cue timeline** for the host (below).

**Speed.** Asking a voice to "speak faster" is unreliable. Model providers say so themselves, and
the voice guesses. Instead:
- time-stretch the master, which the take cache holds losslessly, in one formant-preserving pass;
- let the direction ("briskly") only nudge;
- measure the usable bound by ear with the listening tools. The fit squeeze is now capped at 1.2×
  for exactly this reason, and a deliberate speed round may tolerate 1.3–1.4×.

**Overlap** is now possible and safe: a take that runs past its bar fades under the next voice. A
speed round can lean on that.

## What changes for the host

The loop timeline Acervo stores is four times per word plus a cadence (`repeats`, `repeat_seconds`).
The player assumes one row per word, starts that only increase, and strict source/target
alternation. A programme breaks all three: an intro has no word, an example interrupts the
alternation, and a final review says every word a second time.

**The replacement is a cue list:** `(kind, item?, text, lang, start, end, role)`. It touches:
- **Acervo:** `loops/client.py` (`_timeline`, `_cadence`), `services/loops.store`;
- **the player:** `loopMomentAt`, the lyric rows and their reveal rule, the seek ticks, `stepWord`,
  and the bar's current word.

A classic drill is expressible as a cue list too, so there is one timeline shape, not two.

Richer word data from Acervo — stored examples, notes, register, emoji — is a later option (P2's
fallback), not a starting point. Limits to design to: a take is at most 500 characters, and a
direction at most 200.

## Sound effects: where to get them

Each full URL is written out, so it can be copied.

**Can be bundled — CC0, no attribution needed:**
- Freesound, CC0 only — https://freesound.org/search/?f=license:%22Creative+Commons+0%22
  (API: https://freesound.org/docs/api/ — filter with `license:"Creative Commons 0"`)
- Pixabay sound effects — https://pixabay.com/sound-effects/
- OpenGameArt CC0 sound library — https://opengameart.org/content/cc0-sounds-library
  and https://opengameart.org/content/cc0-sound-effects
- Kenney audio packs — https://kenney.nl/assets/category:Audio

**Usable only with filtering:**
- FSD50K — https://zenodo.org/record/4060432
  (overview: https://annotator.freesound.org/fsd/release/FSD50K/).
  - About 51,000 Freesound clips in 200 AudioSet classes.
  - Each clip carries its own licence (CC0, CC-BY or CC-BY-NC), so filter to CC0, or CC-BY with
    attribution.
  - The AudioSet labels are exactly what matching a word to a sound needs.

**For reference only — not for a public bundle:**
- Sonniss GDC game audio bundles — https://sonniss.com/gameaudiogdc/
  (licence: https://sonniss.com/gdc-bundle-license/). Royalty-free for use in productions, but the
  sounds may not be redistributed on their own, and AI training is forbidden.
- BBC Sound Effects archive — https://sound-effects.bbcrewind.co.uk/
  (licence: https://sound-effects.bbcrewind.co.uk/licensing). The RemArc licence covers personal,
  educational and research use only.
- ESC-50 — https://github.com/karolpiczak/ESC-50 . CC BY-NC, except its ESC-10 subset, which is
  CC BY.

**How sounds would work:**
- The writer tags iconic words with a sound label from a fixed vocabulary (AudioSet-like: "siren",
  "rain", "cat meow"), or none.
- A small, curated, checksum-locked sound pack, built from the CC0 sources and published like the
  sample bundle, answers that label.
- Most words get no sound. A sound is a treat, not a rule.

## How to tell whether it works

- **Reuse the listening tools.**
  - Blind A/B between formats on the same words (the round-0 page).
  - The part-by-part page, extended with segment kinds, to say which blocks help and which grate.
- **Track, per format:** length, cost (text calls plus voice calls, against the free-tier limits the
  host lives within) and "would put it on again".
- **Later**, a recall check: words heard in a programme against words heard in a classic drill, a
  day apart.

## Risks

- **Length and cost.** A radio lesson is longer and calls a text model. Keep formats bounded by word
  count, and keep T-tier blocks the default.
- **Invented facts in commentary.** A model will happily make up an etymology. Prefer blocks that
  cannot be wrong (examples, jokes, mnemonics) over factual claims. Where a fact is wanted, ground it
  in Acervo's own notes (P2's fallback), and have the critique pass drop what it cannot support.
- **The wrong voice for a line.** Target-language text read by the guide voice, or the reverse. The
  script names the language of every segment; the arrangement refuses a mismatch.
- **Overload.** Too many blocks per word drowns the word. Formats cap commentary at one line per word.
- **Interference.** Similar words side by side blur together. That is the ordering rule above.

## Roadmap

| Phase | What | First concrete step |
|---|---|---|
| **P1** | New drill patterns (two or four, reverse, echo, slow-normal-fast), a final review, a speed round, a templated intro | Add patterns to `arrange.PATTERNS` beside `retrieval`, and a `final_review` section that says each pair once; the cue timeline for the host |
| **P2** | LexiBeat-written examples: one per word, then joint ones for neighbours | The injected writer backend and its first prompt; the script schema and parser; the host's writer call home |
| **P3** | The full programme script: commentary, callbacks, mini-story, critique pass | A `radio lesson` format end to end, judged blind against the classic drill |
| **P4** | Sound effects | The label vocabulary; a first CC0 pack of about 50 sounds from Freesound and Kenney; tagging in the writer's call |
| **P5** | Format picker and Surprise me in Acervo | The make dialog lists formats beside music; the cue-list player |
