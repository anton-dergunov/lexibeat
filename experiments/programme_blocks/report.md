# Programme blocks: what the listening found

The first full run of this experiment, scored on a tablet: 530 labels, 128 of them with a written
note, plus the listener's overall remarks; and a follow-up, section G, 157 labels and the
listener's remarks. The numbers below are read from `out/labels.json`, and
every label names a card in that stage's `out/<stage>/report.json`; both are tracked, so any number
here can be traced to what was scored. What each section plays and why is in the
[README](README.md); what the blocks are for is `docs/plans/programme-loops.md`.

**Read the text sections with one caveat.** The strong free text models were overloaded for the
whole run, so nearly every reply in D, E and F came from `gemini-3.5-flash-lite`. Those verdicts are
about what the lite model wrote. The audio sections do not depend on it.

## At a glance

| Section | Labels | Mean | In one line |
|---|---|---|---|
| A. Drills over music | 37 | 3.65 | The standard structures work; speed and chant only under conditions |
| B. Syllables | 173 | 2.68 | Directed wins; the detector picks the wrong words; splitting the spelling causes errors |
| C. Mixed-language voices | 90 | 4.04 | Gemini voices work; WaveNet does not |
| D. Context | 148 | 4.31 | Stories were the surprise; example prompt v1 beats v2 |
| E. Commentary | 49 | 4.59 | Good, apart from grammar and spelling remarks |
| F. Framing | 24 | 4.83 | Good; titles are the one weak spot |
| G. Pronunciation hints | 157 | — | A whole word said slowly is almost always right; syllables are wrong half the time, IPA or not |

G's labels are nearly all 1 or 5, so it is read as right or wrong rather than as a mean. Nine
labels are left out of these means because they do not score what they appear to (see
[The labels themselves](#the-labels-themselves)).

## A. Drills over music

One pair, *el atasco — traffic jam*, in every structure.

| Structure | Scores |
|---|---|
| Classic (today's `retrieval`) | 5 |
| Two or four | pair per bar 5, word–pause pairs 5, bar each 4, doubles 3, on the beat 3 |
| Reverse | pair per bar 5, word–pause pairs 5, bar each 5, recall gap 4, doubles 3, on the beat 3 |
| Echo | guide cue 5, native cue 5, reversed 5, no cue 3 |
| Slow, normal, fast | both 5, directed 4, on the beat 4, stretched 2 |
| Speed round | stretched 1.0× 5, 1.2× 5, 1.3× 4, 1.3× on the main bed 4, 1.4× 4, directed 5, directed and 1.2× 3, two pairs a bar 2 |
| Chant, *el atasco* | quarters, cut and snapped 4; markup 3; separate takes 2; eighths, cut and snapped 1 |
| Chant, *la berenjena* | markup 3; eighths 2; quarters 1; separate takes 1 |

- **Every structure has a place.** The ones scored 3 are less standard, not wrong: faster or
  single repetitions suit revision, and alternating words suits a recap. They become formats to
  choose from, not deletions.
- **The echo needs its cue**, and the cue needs variety: a bank of 10–20 short phrases, not one
  "your turn" every time.
- **Slow by asking, fast by stretching.** Slowing a recording in software sounds metallic; speeding
  one up holds well to about 1.3×. The voice can be asked to go as slow as needed, but asking it to
  go fast is not reliable. "Both", a directed take stretched a little for the fast one, is the
  compromise that won.
- **The speed round is a recap only.** Played at the start it is too fast to learn from. Two pairs
  a bar sounds like someone running for a bus. Directed takes carry rising intonation that the
  stretched ones lack, and stretching stays the fallback for a voice that takes no direction.
- **Asking for speed** (the twelve-line table): "fast" is the safe word. `[extremely fast]` is
  usually very fast but mispronounced *la berenjena*; "very fast and rhythmic" is as often slower.
- **The chant is not proven.** Its only good result, quarters cut and snapped on *el atasco*, got 1
  on *la berenjena*, where the cut was abrupt and wrong. Eighths stutter so badly the word is hard
  to understand; separate takes do not blend. Where it works, the listener heard it as useful for
  beginners and for anyone who wants a word fixed in their head, but it may be working only
  because *el atasco* is easy.

## B. Syllables

### The four methods, per recording

38 words, four recordings each.

| Method | Mean | Spanish (15) | Russian (8) | English (11) | Italian, French (4) |
|---|---|---|---|---|---|
| directed | **4.42** | 4.67 | 4.62 | 3.73 | 5.00 |
| markup (`[short pause]` tags) | 3.03 | 4.00 | 3.12 | 1.73 | 2.75 |
| SSML `<break>` on WaveNet | 2.34 | 2.73 | 3.38 | 1.09 | 2.25 |
| cut at loudness dips | **1.08** | 1.00 | 1.00 | 1.00 | 1.75 |

Counting ties, directed had the best recording for 35 of 38 words, markup for 17, SSML for 5 and
cut for 2. Cut scored 2 or less on 37 words.

- **Cut is unusable.** It cuts abruptly, and on most words the pieces do not add up to the word.
- **Markup often ignores the split.** About eight notes say it spoke the word once or twice at
  normal speed; once it said "short pause" aloud (*безнаказанность*), and once it said the word
  three times.
- **SSML is bound to its voice's locale**, so Russian and Italian words came out with an English
  accent.
- **Directed is the method**, and its rare failures are not the voice's fault (below).

### Why the split itself goes wrong

The syllables are cut from the spelling, and the voice is asked to say them. Where a word is not
said as it is written, the voice then says syllables that do not exist in speech:

- *imprescindible*: directed and markup both 2. Cutting the written word separates what is said
  as one, /impɾesθinˈdiβle/.
- *congeniality*: all four methods 1. The voice broke the word into different syllables again.
- *lethargy*: all four methods 1, each spelling out two letters instead of the last syllable.
- *perseverance*: directed 4, with the third syllable wrong.
- *оппортунистический*: the syllabifier split it wrongly, and markup said the correct split anyway.

So the split should come from how the word is **said**, the dictionary pronunciation, not from how
it is written. This matters most for English, where words are very often not said as written.

One word already tests that idea, and cautions against expecting too much from it:
*Worcestershire* was sent as a respelling of its sound, "wus-ter-sher", and still scored 2 (directed,
last syllable wrong) and 3 (markup, first two syllables run together).

### Choosing the words

| Picks | Mean | Scores |
|---|---|---|
| Spanish, 7 of 116 words | 1.29 | *el alforfón* 3; the other six 1 |
| English, 7 words | 2.57 | *hypocrite* 5; *chivalrous*, *mischievously*, *lethargy* 3; *unscrupulous* 2; *tantalizingly*, *unreciprocated* 1 |

- **Help is worth giving only where a learner would say the word wrong.** *Hypocrite* is the model
  case: the listener would have said it wrong themselves. A word that is merely long, or a
  tongue-twister that nobody actually mispronounces, deserves nothing: *tantalizingly*,
  *unreciprocated*, *escalofriante*, *desaprovechar*, *la predisposición*.
- **The detector prompt asks for the wrong thing.** Of its five reasons to pick a word, `long`,
  `cluster` and `sound` describe difficulty, not error; only `spelling`, and sometimes `stress`,
  describe a word a learner gets wrong. Spanish is said almost as it is written, so it should
  rarely qualify at all.
- **No linguistic terms, anywhere.** The *pésimo* and *súbitamente* picks explain themselves in terms
  only a linguist would use. That holds for the whole programme, not just this block.
- **When the detector does pick a word, its tip must be short**, and correctness is the whole
  point: a split that is said wrong teaches the wrong word. Misfiring on which words to pick is
  tolerable. Misspeaking a word it picked is not.
- **A hard word may deserve more bars** of the plain word said correctly, rather than a split.

### What the listener would try next

1. **Give the voice the pronunciation**, not the spelling: find which voices accept a written
   pronunciation, and how.
2. **Failing that, ask for the whole word said slowly**, unsplit. Slow is easy to get from a voice
   (section A). Even if it is not syllable by syllable, slow still serves the purpose.

Compare the two by ear on the words that failed above. There are two more reasons to prefer the
second:

- **The split is not heard on the beat.** Played over music, the syllable-by-syllable example does
  not audibly sit on the grid. A split suits writing a word part by part more than learning how it
  sounds.
- **It gets tiring.** Syllable by syllable on every word is too slow, so it is for the occasional
  word only.

The syllabifier itself (scored 4) mostly split correctly; the problem is what it is given to split.

## C. Mixed-language voices

Ten presenter lines, each mixing two or three languages.

| Configuration | Mean | Notes |
|---|---|---|
| Gemini API 3.8 Flash TTS, language detected | **4.90** | |
| Gemini 3.1, prompt names each part's language (Kore) | 4.80 | its Russian may not be natural (see labels) |
| Gemini 3.1, the same, Charon | 4.70 | speaks Russian well |
| Gemini 3.1, no prompt | 4.70 | |
| Gemini 3.1, the quoted language's `languageCode` | 4.60 | got French *bon appétit* right, but said "is" in German |
| Gemini 2.5, no prompt | 4.30 | once read the whole line in German; once nailed a hard Russian consonant |
| WaveNet with SSML `<voice>`, a second speaker | 3.90 | steady, but plainly inserted |
| WaveNet, plain | 2.30 | strong accent, often wrong |
| WaveNet with SSML `<lang>` | 2.20 | no better than plain |
| Aura-1 | — | not rendered: the day's free allocation was spent |

- **Mixed-language lines belong only on Gemini voices.** With WaveNet, a host should not attempt
  them; the programme degrades to a simple loop instead.
- The Gemini voices' slight accent on a foreign word was judged fine everywhere.

## D. Context

| Block | Items | Mean |
|---|---|---|
| Punchy examples, prompt v1 | 60 | **4.58** (Spanish 4.63, English 4.53) |
| Punchy examples, prompt v2 | 60 | 3.98 |
| Joint examples | 8 | 4.50 |
| Mini-stories, six, narrator and dialogue | 6 | **5.00** |
| Callbacks | 14 | 4.14 |

**Examples.**
- v1's sentences are funny and memorable; the onion sentence was the best of the whole test.
- v2 splits by style:

  | v2 style | Items | Mean |
  |---|---|---|
  | complaint | 17 | 4.71 |
  | confession | 13 | 4.15 |
  | boast | 9 | 4.11 |
  | reaction | 14 | 3.79 |
  | slip | 7 | **2.14** |

  - **Slip** ("Fui al ban… al banco equivocado") is bad style and should go.
  - **Reaction** appends an English retort after an arrow ("⟶ If you say so"), which does not
    belong in the audio. The arrow and its retort should go.
- One v1 English example had a wrong Russian translation.

**Mini-stories** were the surprise: all six scored 5, and the listener did not expect to like the
format. They should be followed by a quick repetition of their words, as a reminder.

**Callbacks** work when the link is natural and confuse when it is not.

| Kind | Items | Mean |
|---|---|---|
| reuse | 6 | 4.67 |
| chain | 1 | 5 |
| contrast | 1 | 5 |
| quiz | 5 | 3.60 |
| twist | 1 | 2 |

A reuse that merely pushes two words into one sentence ("I am going to dream of the border") reads
as strange. The rule: use a callback where the words combine naturally, and skip it otherwise.

## E. Commentary

49 distinct remarks for 50 words. The model offered about one candidate per word, mostly
`contrast`, which is the lite model following the prompt loosely.

| Kind | Remarks | Mean |
|---|---|---|
| culture | 7 | 4.71 |
| contrast | 24 | 4.67 |
| register | 6 | 4.67 |
| mnemonic | 6 | 4.67 |
| false friend | 5 | 4.20 |
| pronunciation | 1 | 3 |

- **Every 3 was a remark about grammar or spelling**, or a false friend put unclearly:
  - the commas around *sin embargo*;
  - the tenses of *acabar de*;
  - the future of *predecir*, filed as "pronunciation" although it is about verb forms;
  - an etymology for *empapado*.

  The programme is about hearing and saying a word. Commentary should say nothing about writing it
  and use no grammar terms.
- **Repeat the word after a mnemonic**, for a bar or two. The *la multa* and *hypocrite* mnemonics
  were liked, and both wanted the word again straight after.
- **Remarks flagged as factual scored no lower** (4.61 against 4.57). The listener did not penalise
  them, which says they read well, not that they were checked.
- The listener came to this block with low expectations and found it useful: commentary that gives
  a picture of the word, or tells it apart from a neighbour, earns its place.
- **A word with commentary needs fewer drill bars**, perhaps two instead of four.

## F. Framing

Three planned episodes, scored 4–5 throughout: order, section headers, intro and outro.

- The one 3 was the title of *lesson-es-1*. Titles are the weak spot.
- The intro should be written fresh each time, so that it is not the same every episode.
- One section header put *el bigote* where it did not quite fit.

## G. Pronunciation hints

The follow-up to B: the words B said wrong, plus one control, spoken with the dictionary's
pronunciation (Wiktionary's IPA) instead of their spelling. Scoring was strict: 1 if any sound is
wrong. In practice every label is right (4–5) or wrong (1), with a handful of 3s. The 28 Gemini 3.8
clips its free key has not yet allowed are missing, and nothing below waits for them.

### Is a written pronunciation honoured at all?

Each mechanism was given a decoy word with another word's IPA. 5 means the target was heard, 1
means the decoy was, so the pronunciation was ignored.

| Mechanism | English | Spanish | Russian | French |
|---|---|---|---|---|
| Chirp 3 HD `customPronunciations` | 4 | 5 | refused | 5 |
| Chirp 3 HD SSML `<phoneme>` | 4 | 5 | **1** | 5 |
| WaveNet SSML `<phoneme>` | 5 | 5 | **1** | 5 |
| Gemini 3.1, the IPA named in the prompt | 5 | 5 | **5** | **1** |

- **Google's phoneme route does not exist for Russian.** `<phoneme>` is ignored, and a custom
  pronunciation is refused.
- **Gemini 3.1 follows a pronunciation in its prompt over the words in its text**, in three
  languages of four: asked for *lethargy* with "banana" written, it said *lethargy*. That makes the
  prompt a lever for correcting one known word. It also means a wrong IPA would be spoken just as
  confidently.

### Right or wrong, per setup

| Setup | Right | Wrong | Wrong on | Length against the human recording |
|---|---|---|---|---|
| Gemini 3.1, plain word, slowly | 14 / 15 | 1 | *vituperative* | ×1.6 |
| Gemini 3.1, IPA in the text, slowly | 13 / 15 | 2 | *thoroughly*, *el ayuntamiento* | ×1.9 |
| Gemini 3.1, IPA in the prompt, slowly | 14 / 15 | 1 | *vituperative* | ×1.8 |
| Gemini 3.8, plain word, slowly | 5 / 5 | 0 | | ×1.6 |
| Gemini 3.8, IPA in the text, slowly | 5 / 5 | 0 | | ×2.0 |
| Chirp 3 HD, plain word, rate 0.7 | 14 / 15 | 1 | *несовершеннолетний* | ×1.3 |
| Chirp 3 HD, custom pronunciation, rate 0.7 | 13 / 13 | 0 | | ×1.3 |
| WaveNet, SSML `<phoneme>`, slow | 13 / 13 | 0 | | ×1.2 |
| **Syllables:** Gemini 3.1, IPA syllables | 7 / 15 | **8** | 8 of 15 words | |
| **Syllables:** Chirp 3 HD, `<phoneme>` per syllable | 6 / 13 | **7** | 7 of 13 words | |
| **Syllables:** Gemini 3.8, IPA syllables | 6 / 7 | 1 | *несовершеннолетний* | |
| A person: the Commons recording | 10 / 11 | 1 | *écureuil* | ×1.0 |

Chirp and WaveNet have no Russian rows: the probe showed the pronunciation would be ignored there,
so they cover 13 words, not 15. The length column is the median ratio of the trimmed clip to the
human recording of the same word.

### What it shows

- **The whole word was never the main problem; splitting it is.** Said whole and slowly, every
  voice got 13–15 of 15 right, with or without a pronunciation. Split into syllables, Gemini 3.1
  and Chirp were wrong on about half the words. That is *more* often than B's split by spelling on
  the same words, which scored 2 or less on 4 of 14, though B was not scored as strictly. The IPA
  did not rescue the split. Only Gemini 3.8 read IPA syllables well (6 of 7), on too few words to
  lean on.
- **For Gemini 3.1, a written pronunciation buys nothing.** In the text it fixed one word and broke
  two. In the prompt it failed on the same word as the plain one, *vituperative*, which has several
  accepted pronunciations, so the plain "wrong" may be a variant the listener doesn't use.
- **The error-free setups are the hard overrides, and they are the ones the listener liked least.**
  Chirp's custom pronunciation and WaveNet's `<phoneme>` made no mistakes on 13 words, which is
  what the phoneme-override practice in the literature predicts. But they barely slow down: at
  most ×1.3 the human length, even when asked for 0.7× or "slow", against ×1.6–2.0 for Gemini.
  The listener also heard their audio as clearly worse. They offer no route for Russian at all.
- **No setup is proven error-free.** Fifteen words cannot show it: 0 wrong of 13 is still
  consistent with about one word in four going wrong. The words were also chosen as B's failures
  in *splitting*, not as words voices mispronounce whole, so this is not a clean test of whole-word
  accuracy. The listener's impression was the same: fewer failures than before, but none of the
  voices was faultless.
- **A human recording is not the gold standard it looks like.** The recordings vary in register
  and delivery, as people do, and one (*écureuil*) was not quite right. A generated voice sounds
  consistent and is usually clearer.

### Decision

- **Pronunciation help is the whole word, said slowly, by the production Gemini voice, from the
  plain text.** It was right on 14 of 15 words, it is the slowest and best-sounding option, and it
  needs nothing looked up.
- **Syllable by syllable is dropped**, on every voice tried here.
- **No IPA by default.** A pronunciation named in the Gemini prompt is kept as a per-word
  correction for a word known to come out wrong, since the probe shows it overrides the text. It
  is not a default: it needs a trustworthy IPA, and a wrong one is spoken just as surely.
- **Chirp and WaveNet overrides are not used.** They are correct but fast and poorer in sound, and
  they have no Russian.
- **Still open: catching the wrong whole word automatically.** A transcription of the take,
  compared with the word, caught the garbled syllables in the smoke test. Whether it would catch a
  whole word said wrong is not measured.

## The labels themselves

- **Left out of the means above, nine labels that score something else:**
  - five Aura-1 recordings scored 1, which were never made;
  - four example-prompt cards scored 1, which record that the model was unavailable, not a
    judgement of the prompt.
- **Probably mis-scored:**
  - `C.line.1@g31-directed` is a 5, while its note says the voice's Russian "doesn't sound natural
    at all". It decides whether Kore should speak Russian, so it is worth re-listening.
  - The four method cards in B (cut 4, the others 3) were scored after the first few words and
    contradict the per-recording scores, where cut averaged 1.08. The per-recording scores are the
    ones to trust.
- **Never scored:** the ten configuration cards in C, the four prompt cards in E and the ten
  further framing ideas in F.
- **A defect in the page:** *blush* and *frantically* appear in two lessons each, and their cards
  share an id, so each pair shares one score. `page.py` should key commentary cards by lesson as
  well as word.

## What this means for the plan

For `docs/plans/programme-loops.md`:

- **P1 can be built from this:**
  - pair per bar, word–pause pairs and their reversed forms;
  - echo with a bank of cues;
  - slow, normal, fast as "both";
  - a speed round as a recap only: stretched to at most 1.2×, or directed "fast".

  Leave out two pairs a bar, the chant on eighths and the chant from separate takes.
- **Syllables become pronunciation help, and the help is the whole word said slowly** (section G):
  - only for words a learner is likely to say wrong;
  - picked by a detector rewritten around that rule, with no linguistic terms in its output;
  - spoken whole and slowly by the production Gemini voice, from the plain text, never split;
  - extra bars of the plain word, rather than a split;
  - a pronunciation in the prompt only as a correction for one known word.
- **P2 and P3:**
  - examples from prompt v1, without slips or retorts;
  - mini-stories promoted, each followed by a quick recap drill;
  - callbacks only where the link is natural;
  - commentary about sound and meaning only, followed by a bar or two of the word;
  - titles and intros written fresh each episode.

  Rerun D, E and F on a strong model before settling any prompt.
- **Mixed-language lines** only with a Gemini voice; a plain voice gets a simple loop.
