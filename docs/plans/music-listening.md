# Tuning the music by ear

This is how the music gets better from here: listen to what the engine actually makes, say what is
wrong with each part, and turn those judgements into rules. It is written to be followed in order,
on a tablet, a round at a time.

## Why negative rules, and why there is room for them

It is usually unclear what makes a clip *good*, and almost always clear what makes one bad: a
note that is too high, an instrument that sounds electronic, a chord that sounds wrong. Those are
facts about one part of the clip, so each one can become a rule about that part.

Rules take things away, so the question is whether variety survives. It does, by a wide margin.
Every bed draws each of these independently from its seed:

- meter (3), tempo (5–6 per family), key (4 roots × 2–3 scales) and chord colour (4);
- one of 59 four-bar or 9,612 eight-bar progressions;
- a texture, a pad, a bass pattern and timbre, a lead instrument and register, and percussion
  lanes chosen from 28 / 296 / 97 samples;
- a melody that is a fresh random walk every time.

Ruling out an instrument, a register band or a chord removes one choice on one axis and leaves the
rest alone. What really limits how *different* two clips sound is timbre — each family reaches only
7–12 lead sources, 3–5 pads and 7–8 basses today — so `summarise --variety` reports exactly those
counts before a rule is adopted.

Variety comes back by adding: a newly approved instrument, progression or family. Comparing clip A
with clip B ("I prefer this one") is deliberately not the method. It says nothing about *which part*
made the difference, so it cannot become a rule.

## Round 0 — the three known defects (do this first)

Comparing the README demo, which sounds good, with what the service renders turned up three
concrete defects:

| Defect | What you hear | Where it comes from |
|---|---|---|
| **Unapproved lead instruments** | an electronic-sounding lead (an FM electric piano), a harsh glockenspiel, hard mallets | Catalogue banks the Wave 3 listening never approved can still become the lead, with no register limit at all |
| **Leads that go too high** | a thin, piercing top line | Lead notes reach MIDI 94–101; the demo never went above 88 |
| **Off-key chords** | a chord that sounds sour | Every chord gets a perfect fifth, which is outside the key on vii (major), #iv (lydian) and vi (dorian) — in about half of all beds |

Each has a switch in `ListenerPolicy` (`lexibeat/profiles.py`), off by default. A switch changes
*only* a bed that shows its defect: tested over 560 beds, every changed bed had the defect and no
other bed changed at all. So a pair of clips from the same seed, one with the switch on, differs in
that one part and nothing else.

**Steps**

1. On the laptop, render the pairs — 6 per defect, 18 in all, about 30 seconds each:

   ```bash
   uv run python -m scripts.listening.defects --out out/listening/round-00
   ```

2. Serve them, and open the printed address on the tablet (same Wi-Fi, or over Tailscale):

   ```bash
   uv run python -m scripts.listening.serve out/listening/round-00
   ```

3. For each pair: press play, switch between **A** and **B** while it plays (the position is kept,
   so you hear the same bar both ways), then choose **A is better**, **Same** or **B is better**.
   The card tells you which part to listen to, never which side is the fix. Add a note if something
   else stands out. Answers save as you go.

4. See what you decided:

   ```bash
   uv run python -m scripts.listening.summarise out/listening/round-00
   ```

   Each switch comes out as **ADOPT** (the fixed clip won more often), **keep as is**, or **no
   clear difference**.

5. Adopt the winners. On `PRODUCTION_V1` in `lexibeat/profiles.py`, set
   `listener=ListenerPolicy(...)` with the switches that won (the cap at 88 unless the notes said
   otherwise), and bump `ENGINE_VERSION` in `lexibeat/bedspec.py`, because a given seed now
   resolves to a different bed. Then release and re-pin the host. Every labelling round after this
   one hears the fixed engine.

### Round 0, as heard (September 2026)

| Switch | Fixed better | As is better | Same | Decision |
|---|---:|---:|---:|---|
| Approved lead instruments only | 4 | 2 | 0 | **Adopted** |
| Lead notes at or below MIDI 88 | 5 | 0 | 1 | **Adopted** |
| Diatonic chord fifths | 1 | 5 | 0 | Rejected — the perfect fifth is preferred by ear |

Both adopted switches are on `PRODUCTION_V1` from engine 1.5.0. The rejected one stays in the code,
off, because it is a finding too: the "off-key" fifth was the suspected defect, and by ear it is not
one. Running `defects` again now finds only that defect, because production no longer makes the
other two.

## Labelling rounds — rating each part

**Steps, for round N (starting at 1)**

1. Render a round, exactly the way a host's loops are made (the production profile, the hybrid
   palette, the bundled catalogue). From round 2 on, replay the previous round's seeds first:

   ```bash
   uv run python -m scripts.listening.make_round --out out/listening/round-01 --count 30
   uv run python -m scripts.listening.make_round --out out/listening/round-02 --count 20 \
       --replay out/listening/round-01
   ```

   Replayed clips are marked "heard before" on the page. They are the same seeds under the new
   rules, which is how you hear whether a change helped.

2. Serve and open it on the tablet:

   ```bash
   uv run python -m scripts.listening.serve out/listening/round-01
   ```

3. For each clip, play it and go down the parts:

   | Part | What the page tells you it is |
   |---|---|
   | Lead (melody) | the instrument and its range, e.g. "Concert Harp · C#5–A5" |
   | Pad and chords | the pad's source, texture and chord colour |
   | Bass | its source and pattern |
   | Percussion | each lane's sample or synth sound |
   | Harmony | the key and the progression as numerals, e.g. "I – IV – I – iii" |
   | Tempo and feel | the style, tempo and meter |

   Mark each part **Fine**, **Meh** or **Dislike**. On Meh or Dislike, tap the reasons that apply
   (too high, electronic, harsh attack, too loud, too quiet, too busy, sour / off-key, boring,
   other) and add a note if needed. Finish with an overall **Keep / OK / Reject**. You do not have
   to rate every part of every clip; rate what you noticed.

4. Summarise:

   ```bash
   uv run python -m scripts.listening.summarise out/listening/round-01 --variety
   ```

   It prints:
   - the share of clips where no part was disliked — the number that should rise from round to
     round;
   - every concrete choice (a lead bank, a top-note band, a pad, a percussion sample, a chord
     numeral, a family), ranked by how often its part was disliked, with the reasons;
   - the choices disliked in at least 60% of the clips they appeared in, as candidates to rule out;
   - with `--variety`, how many lead, pad and bass sources each family has left.

5. Turn the clear dislikes into rules. A rule is a field on `ListenerPolicy` applied to
   `PRODUCTION_V1`. The three that exist are the first; a new kind — forbidden banks, forbidden
   percussion samples, forbidden chord numerals per scale — is a new field, written so it changes
   only beds that contain the thing it forbids, with a test that says so. Bump `ENGINE_VERSION`.
   Before adopting a rule, check `--variety`: a family left with fewer than three leads needs an
   instrument added, not another one taken away.

6. Render the next round with `--replay` and repeat.

**When to stop:** when the share of clips with no disliked part stops rising between rounds. What
is left is then either taste worth writing down in a family's description, or a missing instrument
worth adding to the library — which is its own listening exercise, in the style of the Wave 3
auditions.

## Notes

- Clips are the music bed alone, loudness-matched to −20 LUFS, so "too loud" and "too quiet" are
  about a part inside a clip and never about which clip was mastered hotter. A loop adds speech on
  top and ducks the music under it, so something that is borderline alone may matter less there —
  and a piercing lead matters more.
- Everything a round produces stays under `out/listening/`, which is not tracked: `round.json` (what
  each clip is made of), the clips, `labels.json` (your answers) and, for round 0, `key.json` (which
  side was fixed — the server never serves it).
- The tools are `scripts/listening/`: `defects`, `make_round`, `serve`, `summarise`, and the one
  page they serve.
