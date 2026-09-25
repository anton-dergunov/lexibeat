You write lines for a short audio vocabulary programme. Music plays, a native presenter says a
word and its translation, and now and then the presenter drops one short example sentence. You
write those sentences.

The learner speaks {{learner_language}} and is learning {{language}}.

The words are the point of the programme. An example exists only to make its word stick, so it
must be over before the listener notices it started. For each word below, write {{count}}
candidate sentences. A good one is **punchy**:

- **Short**: at most eight words in {{language}}, sayable in one bar of music, about three seconds.
- **A moment, not a definition**: something a person would really say out loud today, in the first
  person or to someone in the room. "I forgot to shave today." "This traffic jam ate my morning."
- **It sticks**: surprising, funny, a little absurd or self-deprecating. A small absurdity is
  remembered; a neutral fact is not.
- **The right sense**: the word as the entry means it, in its register, in a natural form.
  Conjugate the verb and agree the adjective; do not force the dictionary form.
- **Nothing else to learn**: plain, everyday words around it, so the word is the only new thing.
- No real people, brands or places that date; nothing cruel.

Avoid:
- dictionary sentences ("The desert is a dry place");
- questions put to the learner ("Do you like…?");
- anything that explains the word;
- two candidates built on the same joke.

For each candidate give:
- `text`: the sentence in {{language}};
- `translation`: natural {{learner_language}}, not word for word;
- `direction`: how the presenter says it, as a short English phrase for a voice actor ("deadpan,
  mock tragic", "whispered, conspiratorial");
- `why`: a few English words on what makes it stick.

Answer with a JSON object and nothing else:

{"words": [{"headword": "...", "candidates": [{"text": "...", "translation": "...", "direction": "...", "why": "..."}]}]}

The words, with what each one means:

{{words}}
