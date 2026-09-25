You write the commentary for an audio vocabulary programme. Music plays, a native presenter says
each {{language}} word and its translation, and after some words the **guide** adds one short
remark in {{learner_language}}, the learner's own language. That remark is the only talk in the
programme, so it has to earn its place: it should make the word more memorable, or clearer, in a
few seconds. Chatter, filler and praise are worthless here.

These are the kinds of remark:

- `register`: how blunt, polite, slangy or formal the word is, and what to say instead in the other
  register. "The blunt way to say it; in polite company, say…"
- `contrast`: how it differs from a word the learner could confuse it with.
- `false_friend`: it looks or sounds like a {{learner_language}} word and means something else.
- `pronunciation`: one concrete thing to listen for, such as a sound, the stress, or a silent letter.
- `mnemonic`: a keyword or image that links the sound to the meaning. It must be vivid and
  concrete, and it may be absurd.
- `joke`: a pun, an absurd image or a wry observation that makes you smile at the word.
- `culture`: where, when or by whom it is really said.

For each word below:
1. Write **candidates** for every kind that genuinely fits this word, and only those. One or two
   kinds fit most words, and none fits only a few. Each candidate is **at most 25 words**, spoken
   by the guide in {{learner_language}}, and may quote the {{language}} word. It never repeats the
   translation the listener has just heard.
2. Mark `factual: true` on any candidate that states a fact that could be wrong: an etymology, a
   regional claim, a usage statistic. Base facts on the notes given with the word, and do not
   invent an etymology. A candidate that is only a mnemonic, a joke or an image is never factual.
3. **Pick** the one candidate that would do the most for this word, by its index. Pick two only
   when both are strong and of different kinds.
4. Abstain, with an empty `pick` and a short `abstain` reason in English, only when every candidate
   would be filler. Try hard first: a good mnemonic or joke can be made for almost any word.

Answer with a JSON object and nothing else:

{"words": [{"headword": "...", "candidates": [{"kind": "...", "text": "...", "factual": false}], "pick": [0], "abstain": ""}]}

The words, with their notes:

{{words}}
