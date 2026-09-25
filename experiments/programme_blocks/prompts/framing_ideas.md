You are helping design an audio vocabulary programme: twelve {{language}} words drilled over
music, for a learner who speaks {{learner_language}}, with a native presenter who speaks
{{language}} and a guide who speaks {{learner_language}}. The words are the programme; everything
else is framing, and framing must be brief.

Using this episode's twelve words, write a **concrete example** of each of the framing blocks below,
exactly as it would be spoken, and then propose **three more framing blocks of your own** that a
text model could write and that would make the words stick better without adding chatter.

Blocks:
- `episode_title`: the title, as announced;
- `cold_open`: two or three seconds before the intro that tease the most striking word without
  giving its meaning;
- `midpoint_quiz_leadin`: the one line before a quiz in which the guide gives meanings and the
  listener answers;
- `encouragement_bank`: ten different short lines of encouragement (at most five words each),
  written once and reused across episodes. They must not be generic cheerleading: make them wry
  or specific;
- `outro_recap`: the closing recap, calling back two words;
- `word_of_the_day`: the one word chosen to end on, and the line that sends the listener off with
  it.

For each block, give `name`, `what` (one English sentence on what it is for), `lines` (the spoken
lines, each with `speaker` (`native` or `guide`), `text` and, when it is not in
{{learner_language}}, `translation`) and `why` (why it helps, in a few English words).

Answer with a JSON object and nothing else:

{"blocks": [{"name": "...", "what": "...", "lines": [{"speaker": "guide", "text": "...", "translation": ""}], "why": "..."}]}

The words:

{{words}}
