You write for an audio vocabulary programme in {{language}} for a learner who speaks
{{learner_language}}. The programme drills the twelve words below in this order, over music, with
two presenters: a **native** one who speaks {{language}} and a **guide** who speaks
{{learner_language}}.

A **callback** is a moment that brings back a word taught earlier, after a later word's drill, so
the earlier word is recalled after a delay rather than only heard once. "Remember the desert? Here
it is again" is the weak version: it announces a callback instead of being one. A good callback
makes the listener *use* the earlier word. Write callbacks of these kinds:

- `reuse`: a new short sentence using the current word **and** an earlier one together;
- `contrast`: the current word set against an earlier one it could be confused with, or its
  opposite ("not *sweet*, *bitter*"), in a line or two;
- `quiz`: the guide gives an earlier word's meaning in {{learner_language}}, a gap follows (write
  `[gap]` as its own line), and then the native presenter answers with the word, perhaps in a
  short phrase;
- `chain`: the current word leads to an earlier one by sound, by a shared root, or because one
  causes the other, said in one line;
- `twist`: an example sentence said at an earlier word's drill, repeated with the current word
  swapped in so that it turns absurd. Write that earlier example as `earlier`, since it has not
  been written yet.

For each kind, write up to two callbacks where the words genuinely allow one. Skip a kind rather
than force it. Each callback:
- plays straight after the drill of the word named in `at`, and brings back words from earlier in
  the order (`refers_to`);
- is at most two short lines, plus `[gap]` for a quiz;
- keeps the target words in the sense given.

For each line give `speaker` (`native` or `guide`), `text`, and `translation` (into
{{learner_language}}, or "" when the line is already in it).

Answer with a JSON object and nothing else:

{"callbacks": [{"kind": "...", "at": "headword", "refers_to": ["headword"], "earlier": "", "lines": [{"speaker": "guide", "text": "...", "translation": ""}], "why": "..."}]}

The words, in the order they are taught:

{{words}}
