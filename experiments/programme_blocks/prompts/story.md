You write a mini-story for an audio vocabulary programme in {{language}}, for a learner who speaks
{{learner_language}}. The programme drills twelve words one at a time over music. Between the
drills, a very short story is told in pieces, each piece using the word or words that have just
been drilled. The story gives the words a place to live. It must never crowd them out: the drills
are the programme, and the story is the thread between them.

{{style}}

How it fits together:
- First choose the **order** in which to teach the twelve words, so that the story can use them
  as they arrive. Every word is used at least once, in its sense and register as given.
- Write between four and six **beats**. A beat comes straight after the drill of one word (`after`)
  and uses that word and any words drilled since the last beat (`uses`). The first beat may come
  after the second or third word.
- The whole story is at most 120 words of {{language}}, so each beat is one to three short lines.
- Keep it simple and concrete: plain words around the taught ones, one situation, and something
  that happens. A small twist or joke near the end is welcome.
- The target words appear exactly as they would in natural speech: conjugated, agreed, with the
  article where it belongs.

For each line give `speaker` (`native` or `guide`), `text`, `translation` (into
{{learner_language}}), and `direction` (a short English note for the voice actor).

Answer with a JSON object and nothing else:

{"title": "...", "order": ["headword", "..."], "beats": [{"after": "headword", "uses": ["headword"], "lines": [{"speaker": "native", "text": "...", "translation": "...", "direction": "..."}]}]}

The words:

{{words}}
