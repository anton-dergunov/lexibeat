Split each item into its spoken syllables, the way a careful native speaker would say them slowly
to a learner, not the way a dictionary hyphenates for line breaks.

Rules:
- Syllables follow speech: Spanish "ayuntamiento" is `a-yun-ta-mien-to`, English "thoroughly" is
  `thor-ough-ly`, Russian "здравствуйте" is `здрав-ствуй-те`.
- Keep the original spelling, accents and letters. Joined together, the syllables must give back
  every letter of the item in order, with nothing added, dropped or changed. Leave out spaces and
  punctuation, but keep a short word such as an article as its own syllable ("el", "la").
- Write the syllable that carries the main stress in CAPITALS and every other syllable in
  lower case, e.g. `a`, `yun`, `ta`, `MIEN`, `to`. Exactly one syllable is in capitals.
- `note` is optional: one short phrase in English on what to listen for, e.g. "the rr is rolled",
  or "the w is silent". Leave it empty when there is nothing worth saying.

Answer with a JSON object and nothing else:

{"items": [{"id": "...", "syllables": ["..."], "note": ""}]}

The items:

{{items}}
