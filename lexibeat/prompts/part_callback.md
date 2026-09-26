**Callbacks.** After a word, a callback may bring back a word taught **earlier** in the order, so it
is recalled after a delay. Write one only where the two words genuinely combine in a natural line;
otherwise give `"callback": null`. Most words get none. Kinds: {{callback_kinds}}.
- `reuse`: one short, natural {{language}} sentence using this word and an earlier one together;
- `contrast`: this word set against an earlier one it could be confused with, in a line;
- `chain`: this word leads to an earlier one because one causes the other, in one line;
- `quiz`: the guide gives an earlier word's meaning in {{learner_language}}, then the native
  presenter answers with the word.
Give `kind`, `refers_to` (the earlier word's number) and `lines`: each with `speaker` (`native` or
`guide`), `text`, and `translation` (into {{learner_language}}, or "" for a guide line).
