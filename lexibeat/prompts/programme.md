You write the spoken lines for one episode of an audio vocabulary programme. Music plays, and two
presenters take turns: a **native** presenter who speaks {{language}}, and a **guide** who speaks
{{learner_language}}, the learner's own language. The learner is learning {{language}}.

The words are the programme. Everything you write exists only to make a word stick, so it is short,
concrete and over before the listener notices it started. Nothing you write is chatter, filler or
praise.

Rules for every line:
- Never explain grammar or spelling, and never use a linguistic term (no "conjugation", "stress",
  "syllable", "infinitive", "palatal", "vowel" and the like). The programme is listened to, and a
  listener cares about what a word means, how it is used and how it sounds, not how it is written.
- A {{language}} line uses the word as the entry means it, in its register and in a natural form:
  conjugated, agreed, with the article where it belongs.
- No real people, brands or places that date; nothing cruel.
- A `direction` is a short English note for the voice actor ("deadpan, mock tragic").
- A guide line that quotes {{language}} — a word, a phrase, a whole expression — lists what it quotes
  in `quoted`, each exactly as it is written in `text`, so the guide can say it with a native
  {{language}} pronunciation. A guide line that quotes nothing gives `"quoted": []`.

The words, each with its number and what it means:

{{words}}

{{parts}}

Refer to a word by its number (`item`), never by repeating it as a key. Answer with one JSON object
and nothing else, shaped like this (leave out any key not asked for above):

{{shape}}
