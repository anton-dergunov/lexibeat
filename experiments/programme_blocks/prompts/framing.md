You plan one episode of an audio vocabulary programme: twelve {{language}} words, drilled one at
a time over music, for a learner who speaks {{learner_language}}. Two presenters: a **native** one
who speaks {{language}}, and a **guide** who speaks {{learner_language}}. Framing is what the guide
says around the words. It must be short, warm and never chatty, because the words are the
programme.

Plan:

1. **Order.** Put the twelve words in teaching order:
   - keep look-alikes and near-synonyms apart, since similar words side by side blur together in
     memory;
   - put next to each other the words that combine naturally or share a topic;
   - end on a strong, memorable word.

   For each word, give a few English words saying why it is where it is.
2. **Sections.** Split the order into two to four sections of three to five words. Give each a
   spoken **header** of at most eight words in {{learner_language}}, e.g. "Now, three words for the
   kitchen." A header may name a theme the words share even if their topic labels differ; it must
   be true of every word in the section.
3. **Title.** A short episode title in {{learner_language}}, as a radio show would announce it.
4. **Intro.** At most 25 words, spoken by the guide: what today is about and how many words, with
   a hook.
5. **Outro.** At most 20 words: a one-line recap and goodbye, which may call back one word.

For the intro and the outro, add `direction`, a short English note for the voice actor.

Answer with a JSON object and nothing else:

{"title": "...", "order": [{"headword": "...", "why": "..."}], "sections": [{"header": "...", "headwords": ["..."]}], "intro": {"text": "...", "direction": "..."}, "outro": {"text": "...", "direction": "..."}}

The words:

{{words}}
