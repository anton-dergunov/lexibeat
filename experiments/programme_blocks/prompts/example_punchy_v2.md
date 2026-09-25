You write the example lines for a vocabulary programme that sounds like a radio show with two
presenters. The **native presenter** speaks {{language}}. The **guide** speaks
{{learner_language}}, the learner's language, and is a little slower on the uptake. Between the
drills, the native presenter lets slip a line that uses the word, as if something had just
happened to them, and sometimes the guide reacts in two or three words.

The words are the show; these lines are seasoning. Each one is over in a few seconds.

For each word below, write {{count}} candidates, and make them **different kinds** of moment:
- **confession**: owning up to something small ("I ate the whole cake. Again.");
- **complaint**: a mock-tragic grievance about daily life;
- **boast**: absurd pride in something tiny;
- **slip**: the presenter gets something slightly wrong and corrects it, which makes the word heard
  twice;
- **reaction**: the line is a reply to something the listener cannot see.

Rules:
- The native line is at most eight words in {{language}} and uses the word in the sense and
  register of the entry, in a natural form (conjugated, agreed).
- The guide's reaction is optional, at most four words in {{learner_language}}, and never explains
  the word. Most candidates have none.
- Everyday words around it; no real people or brands; nothing cruel.
- Never a dictionary sentence, never a question to the learner.

For each candidate give:
- `kind`: one of the five above;
- `text`: the native line in {{language}};
- `translation`: natural {{learner_language}};
- `direction`: a short English note for the voice actor;
- `reaction`: the guide's line in {{learner_language}}, or "".

Answer with a JSON object and nothing else:

{"words": [{"headword": "...", "candidates": [{"kind": "...", "text": "...", "translation": "...", "direction": "...", "reaction": ""}]}]}

The words:

{{words}}
