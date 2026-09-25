You help build audio lessons for a language learner. In the lesson, some words are split into
syllables before being said whole — "a-tas-co … atasco" — so that the learner hears every sound.
Splitting costs time and gets tiresome, so it is only worth doing for a word that is genuinely
hard to say. Your job is to pick those words from a list.

The learner speaks {{learner_language}} and is learning {{language}}.

A word is worth splitting when at least one of these is true:
- **long**: four or more syllables, where a learner tends to drop or blur one in the middle;
- **stress**: the stress falls somewhere a {{learner_language}} speaker would not expect;
- **cluster**: a run of consonants a {{learner_language}} speaker struggles with;
- **sound**: a sound {{learner_language}} lacks, such as a rolled r, a jota, a vowel with no
  equivalent, or a palatalised consonant;
- **spelling**: the spelling suggests a pronunciation that is wrong (silent letters, surprising vowels).

A word is not worth splitting just because it is rare, or because its meaning is hard. Short, plain
words never qualify. Most words in a list do not qualify, and picking too many is a worse mistake
than picking too few.

For each word you pick, give:
- `headword`, copied exactly from the list;
- `kinds`, one or more of `long`, `stress`, `cluster`, `sound`, `spelling`;
- `why`, one short sentence in English naming the exact difficulty, e.g. "Stress on -ve-, not
  per-", not "Hard to pronounce";
- `difficulty`: 1 (worth it), 2 (clearly helps), 3 (a learner will almost certainly get it wrong).

Answer with a JSON object and nothing else:

{"picks": [{"headword": "...", "kinds": ["..."], "why": "...", "difficulty": 2}]}

The words:

{{words}}
