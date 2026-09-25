"""Section A: drill structures, each one a short clip over real music.

Every structure is written as lines on a grid of bars and beats, rather than as the one-line-a-bar
slots `arrange.PATTERNS` holds. Several of these structures put two lines in a bar or a syllable on
every eighth note, which that shape cannot say. Each line goes through the production voice and
LexiBeat's own trim, stretch and fit. It is placed on the grid, laid over a production bed with
`mix_stems` at production loudness, and written as MP3 the way a loop is.

All the drills share one pair and one bed, so that only the structure differs between them. The
speed round needs several pairs and a quicker bed; it also measures how much faster a voice
actually goes when it is asked to.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
from dataclasses import dataclass, replace

import numpy as np

from lexibeat import dsp
from lexibeat.api import MusicRequest, resolve_music
from lexibeat.loop import write_mp3
from lexibeat.mix import mix_stems
from lexibeat.music import SR, Grid, render_stems

import cut
import tts
import voice
from common import OUT, parallel, relative, word, write_report

SECTION = "drills"
DIR = OUT / SECTION
INTRO_BARS, OUTRO_BARS = 1, 1

# The pair every drill is heard with, and the six the speed round runs through.
PAIR = ("el atasco", "es", "traffic jam", "en")
FOOD = ["la berenjena", "la canela", "la cebolla", "el perejil", "la mostaza", "la avellana"]
# Syllables for the chant, by hand: the syllabifier in section B is itself under test.
CHANT = {"el atasco": (["el", "a", "tas", "co"], 2),
         "la berenjena": (["la", "be", "ren", "je", "na"], 3)}


@functools.lru_cache(maxsize=None)
def _resolved(family: str, seed: int):
    return resolve_music(MusicRequest(family=family, seed=seed)).bed_spec


def bed(family: str, seed: int, bpm: float | None = None):
    spec = copy.deepcopy(_resolved(family, seed))
    if bpm:
        spec.bpm = bpm
    return spec


BEDS = {"main": ("bright-organic", 7, None), "fast": ("gentle-game", 7, 100.0)}


@dataclass
class Line:
    """One utterance: where it starts, who says it, and how it is asked for and treated."""

    bar: int
    beat: float
    role: str  # "W" (the word), "T" (its translation) or "cue"
    take: int = 0
    stretch: float = 1.0
    direction: str = ""
    raw: bool = False
    beats: float | None = None  # its slot; by default, until the next line starts
    text: str | None = None
    language: str | None = None
    audio: np.ndarray | None = None  # already prepared, e.g. one syllable of a cut word
    gain_db: float = 0.0
    label: str | None = None


@dataclass
class Drill:
    group: str
    id: str
    title: str
    why: str
    lines: list[Line]
    bars: int
    bed: str = "main"
    pair: tuple[str, str, str, str] = PAIR
    notes: str = ""


def _speak(line: Line, pair: tuple[str, str, str, str]) -> tts.Take:
    if line.role == "cue":
        role = "guide" if line.language != pair[1] else "native"
        return voice.say(line.text, line.language, role=role, direction=line.direction,
                         take=line.take, raw=line.raw)
    text, language = (pair[0], pair[1]) if line.role == "W" else (pair[2], pair[3])
    return voice.say(line.text or text, language, role="native" if line.role == "W" else "guide",
                     direction=line.direction, take=line.take, raw=line.raw)


def render(drill: Drill) -> dict:
    spec = bed(*BEDS[drill.bed])
    grid = Grid.from_spec(spec)
    total_bars = INTRO_BARS + drill.bars + OUTRO_BARS
    speech = np.zeros(grid.samples(total_bars * grid.bar) + grid.sr, dtype=np.float32)
    starts = sorted(grid.bar_start(INTRO_BARS + l.bar) + l.beat * grid.beat for l in drill.lines)
    pattern_end = grid.bar_start(INTRO_BARS + drill.bars)
    requests, placed, lengths = [], [], []
    for line in drill.lines:
        pair = drill.pair
        start = grid.bar_start(INTRO_BARS + line.bar) + line.beat * grid.beat
        if line.beats is not None:
            slot = line.beats * grid.beat
        else:
            later = [s for s in starts if s > start + 1e-6]
            slot = (later[0] if later else pattern_end + grid.bar) - start
        if line.audio is not None:
            audio = line.audio
        else:
            take = _speak(line, pair)
            requests.append(take.request)
            audio = voice.prepared(take, stretch=line.stretch, max_seconds=slot * 0.92,
                                   slot_seconds=slot, gain_db=line.gain_db)
            lengths.append(f"{_label(line, pair)}: {voice.trimmed_seconds(take) / line.stretch:.2f} s")
        at = grid.samples(start)
        end = min(len(speech), at + len(audio))
        speech[at:end] += audio[: end - at]
        placed.append({"bar": line.bar + 1, "beat": line.beat + 1,
                       "label": line.label or _label(line, pair),
                       "start": round(start, 3), "seconds": round(len(audio) / SR, 3)})
    stems = render_stems(spec, total_bars)
    depths = {name: getattr(spec, name).duck_db for name in stems}
    track = mix_stems(stems, speech, depths)
    peak = float(np.abs(track).max())
    assert track.ndim == 2 and track.shape[1] == 2 and np.isfinite(track).all(), drill.id
    assert peak <= 0.97 + 1e-6, (drill.id, peak)
    path = DIR / f"{drill.group}--{drill.id}.mp3"
    write_mp3(path, track)
    notes = drill.notes
    if drill.group == "slow-normal-fast":
        notes = "Take lengths before fitting: " + "; ".join(dict.fromkeys(lengths)) + ". " + notes
    return {"id": f"A.{drill.group}.{drill.id}", "title": drill.title, "why": drill.why,
            "notes": notes,
            "audio": [{"src": relative(path), "label": f"{len(track) / SR:.0f} s"}],
            "grid": _grid(placed, drill.bars),
            "facts": f"{spec.bpm:g} BPM · bar {grid.bar:.2f} s · {drill.bars} bars "
                     f"+ {INTRO_BARS} in + {OUTRO_BARS} out · bed {BEDS[drill.bed][0]}",
            "details": [{"summary": "Requests sent", "body": _unique(requests)}]}


def _label(line: Line, pair) -> str:
    if line.role == "cue":
        return f"cue “{line.text}”"
    text = line.text or (pair[0] if line.role == "W" else pair[2])
    extra = []
    if line.stretch != 1.0:
        extra.append(f"×{line.stretch:g}")
    if line.direction:
        extra.append(line.direction.split(",")[0])
    return f"{line.role} {text}" + (f" ({'; '.join(extra)})" if extra else "")


def _grid(placed: list[dict], bars: int) -> list[list[str]]:
    """Eight cells a bar, one per eighth note: what starts there."""
    rows = [["·"] * 8 for _ in range(bars)]
    for item in placed:
        steps = int(round((item["beat"] - 1) * 2))
        bar, cell = item["bar"] - 1 + steps // 8, steps % 8
        if 0 <= bar < bars and 0 <= cell < 8:
            rows[bar][cell] = item["label"]
    return rows


def _unique(requests: list[dict]) -> str:
    seen, out = set(), []
    for request in requests:
        key = json.dumps(request, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            out.append(request)
    return json.dumps(out, ensure_ascii=False, indent=1)


# --- the structures ---------------------------------------------------------------------------

def _mirror(lines: list[Line]) -> list[Line]:
    return [replace(l, role={"W": "T", "T": "W"}.get(l.role, l.role)) for l in lines]


def two_or_four() -> list[Drill]:
    W, T = "W", "T"
    shapes = [
        ("pair-per-bar", "Pair per bar",
         "The word on beat 1, its translation on beat 3, four bars running: no gaps, one steady "
         "pulse. The densest version of today's drill that still leaves each line room.",
         [Line(b, 0, W, take=b) for b in range(4)] + [Line(b, 2, T, take=b) for b in range(4)], 4),
        ("word-pause-pairs", "Word, a bar of silence, then pairs",
         "The word alone, then a whole empty bar for the meaning to surface before it is given, "
         "then two tight pairs. Retrieval first, reinforcement after.",
         [Line(0, 0, W), Line(2, 0, W, take=1), Line(2, 2, T), Line(3, 0, W, take=2),
          Line(3, 2, T, take=1)], 4),
        ("doubles", "Doubles",
         "Word twice, translation twice, twice over. Repetition in pairs is how a chorus is built; "
         "it gives the ear a second chance at each line before it changes.",
         [Line(0, 0, W), Line(0, 2, W, take=1), Line(1, 0, T), Line(1, 2, T, take=1),
          Line(2, 0, W, take=2), Line(2, 2, W, take=3), Line(3, 0, T, take=2),
          Line(3, 2, T, take=3)], 4),
        ("bar-each", "A bar each",
         "Today's rhythm, one line to a bar, without the retrieval gap: four bars, two pairs. "
         "Spacious, the easiest to follow while doing something else.",
         [Line(0, 0, W), Line(1, 0, T), Line(2, 0, W, take=1), Line(3, 0, T, take=1)], 4),
        ("on-the-beat", "Four on the beat",
         "Word, translation, word, translation on the four beats of one bar, twice. As rhythmic "
         "as it gets: every line squeezed into one beat, the overflow fading under the next.",
         [Line(b, beat, W if beat % 2 == 0 else T, take=b * 2 + beat // 2)
          for b in range(2) for beat in range(4)], 2),
    ]
    drills = [Drill("two-or-four", sid, title, why, lines, bars)
              for sid, title, why, lines, bars in shapes]
    reverse = [Drill("reverse", d.id, d.title + ", reversed",
                     "The same structure with the translation first, so each word is recalled "
                     "from its meaning instead of recognised from its sound. " + d.why,
                     _mirror(d.lines), d.bars) for d in drills]
    reverse.append(Drill(
        "reverse", "recall-gap", "Translation, a silent bar, the word",
        "The plan's reverse drill: hear the meaning, produce the word yourself in the empty bar, "
        "then hear it said, and once more with its meaning.",
        [Line(0, 0, "T"), Line(2, 0, "W"), Line(3, 0, "T", take=1), Line(3, 2, "W", take=1)], 4))
    return drills + reverse


def echo() -> list[Drill]:
    cue_en = dict(role="cue", text="Your turn.", language="en",
                  direction="inviting and encouraging, like a teacher handing over the microphone")
    cue_es = dict(role="cue", text="¡Tu turno!", language="es",
                  direction="inviting and encouraging, like a teacher handing over the microphone")
    return [
        Drill("echo", "guide-cue", "Echo, cued by the guide in English",
              "Repeat after me. The word, the guide's “your turn”, a bar of silence to say it in, "
              "then the word again to check against. Saying it aloud is production, and the "
              "second hearing corrects it.",
              [Line(0, 0, "W"), Line(1, 0, **cue_en), Line(3, 0, "W", take=1)], 4),
        Drill("echo", "native-cue", "Echo, cued in the target language",
              "The same, but the native voice gives the cue in Spanish. It keeps the language "
              "switch out of the drill, and teaches the cue itself.",
              [Line(0, 0, "W"), Line(1, 0, **cue_es), Line(3, 0, "W", take=1)], 4),
        Drill("echo", "no-cue", "Echo with no cue, only the music",
              "No spoken cue: the word, two bars of music, the word. Cheapest and least "
              "talkative, and it relies on the listener learning the convention.",
              [Line(0, 0, "W"), Line(3, 0, "W", take=1)], 4),
        Drill("echo", "reverse", "Reverse echo: meaning, your turn, the word",
              "The guide gives the meaning and hands over; you say the Spanish word in the gap, "
              "then hear it. Recall and production together, the hardest of the four.",
              [Line(0, 0, "T"), Line(1, 0, **cue_en), Line(3, 0, "W")], 4),
    ]


# Not "slowly and deliberately, stretching every syllable", which is what this was: the voice took
# it literally and spread *el atasco* over 6.6 seconds, which no bar holds. This one comes back at
# about 1.6× the natural length.
SLOW = "slowly, like a patient teacher, without pausing between syllables"
NATURAL = "naturally, at an ordinary conversational pace"
FAST = "quickly and crisply, as one short snap"


def slow_normal_fast() -> list[Drill]:
    why = ("The word three times, slow enough to hear every sound, then natural, then quick, "
           "and the translation after. Slow is for the ear; fast is how it is really said. ")
    return [
        Drill("slow-normal-fast", "stretch", "Speeds by stretching one recording",
              why + "Here one recording is stretched to 0.8×, 1.0× and 1.3× (formant-preserving): "
              "identical delivery, only the speed changes.",
              [Line(0, 0, "W", stretch=0.8), Line(1, 0, "W"), Line(2, 0, "W", stretch=1.3),
               Line(3, 0, "T")], 4),
        Drill("slow-normal-fast", "directed", "Speeds by asking the voice",
              why + "Here the voice is asked for each speed in words, with nothing stretched. "
              "A bare note is used, since production's pace words would contradict it.",
              [Line(0, 0, "W", direction=SLOW, raw=True), Line(1, 0, "W", direction=NATURAL, raw=True),
               Line(2, 0, "W", direction=FAST, raw=True), Line(3, 0, "T")], 4),
        Drill("slow-normal-fast", "both", "Asked, then nudged by stretching",
              why + "The voice is asked for each speed, and the stretch finishes the fast one "
              "(1.15×). The slow one is left alone: asked for, it is already 3.4× the natural "
              "length.",
              [Line(0, 0, "W", direction=SLOW, raw=True),
               Line(1, 0, "W", direction=NATURAL, raw=True),
               Line(2, 0, "W", direction=FAST, raw=True, stretch=1.15), Line(3, 0, "T")], 4),
        Drill("slow-normal-fast", "on-the-beat", "Slow, natural, fast, then on the beat",
              why + "Ends with the fast take on three beats in a row, which makes the last "
              "repetition part of the rhythm, before the meaning.",
              [Line(0, 0, "W", direction=SLOW, raw=True),
               Line(1, 0, "W", direction=NATURAL, raw=True),
               Line(2, 0, "W", direction=FAST, raw=True, stretch=1.15),
               *[Line(3, b, "W", direction=FAST, raw=True, stretch=1.15, beats=1) for b in range(3)],
               Line(4, 0, "T")], 5),
    ]


SPEED_DIRECTION = "fast, rhythmic and punchy, landing hard on the beat"


def _food() -> list[tuple[str, str, str, str]]:
    return [(h, "es", word(h, "es")["primaryGloss"], "en") for h in FOOD]


def speed_round() -> list[Drill]:
    pairs = _food()
    why = ("Every pair once, back to back, as a closing sprint over a quicker bed "
           f"({BEDS['fast'][0]}, {BEDS['fast'][2]:g} BPM): a review that tests whether the word "
           "comes before the meaning does. ")

    def lines(stretch: float, direction: str = "", per_bar: int = 1) -> list[Line]:
        out = []
        for index, pair in enumerate(pairs):
            bar, half = divmod(index, per_bar)
            beat = half * (4 / per_bar)
            span = 2 / per_bar
            out += [Line(bar, beat, "W", text=pair[0], stretch=stretch, direction=direction,
                         raw=bool(direction), beats=span, label=f"W {pair[0]}"),
                    Line(bar, beat + span, "T", text=pair[2], stretch=stretch, direction=direction,
                         raw=bool(direction), beats=span, label=f"T {pair[2]}")]
        return out

    drills = [
        Drill("speed-round", f"stretch-{s:g}", f"Stretched to {s:g}×",
              why + f"Production takes, stretched to {s:g}× (formant-preserving).",
              lines(s), 6, bed="fast") for s in (1.0, 1.2, 1.3, 1.4)]
    drills += [
        Drill("speed-round", "directed", "Asked to be fast, not stretched",
              why + f"Every line asked for “{SPEED_DIRECTION}”, nothing stretched.",
              lines(1.0, SPEED_DIRECTION), 6, bed="fast"),
        Drill("speed-round", "directed-1.2", "Asked to be fast, and stretched to 1.2×",
              why + "Both: the voice asked for pace, the stretch taking it the rest of the way.",
              lines(1.2, SPEED_DIRECTION), 6, bed="fast"),
        Drill("speed-round", "two-per-bar", "Two pairs a bar",
              why + "Twice as dense, a line on every beat; asked for pace and stretched to "
              "1.3×. Where the plan's “overlap is safe” is tested: long lines fade under the next.",
              lines(1.3, SPEED_DIRECTION, per_bar=2), 3, bed="fast"),
        Drill("speed-round", "stretch-1.3-main-bed", "Stretched to 1.3×, over the usual bed",
              why + "The 1.3× version over the ordinary drill bed, to hear what the quicker "
              "music adds.", lines(1.3), 6, bed="main"),
    ]
    return drills


MEASURE = [("naturally, at an ordinary pace", ""), ("fast", ""),
           ("very fast and rhythmic, like a rapper on the beat", ""),
           ("naturally, at an ordinary pace", "[extremely fast] ")]


def measure() -> dict:
    """How much faster the voice goes when asked: trimmed seconds, per direction."""
    rows = []
    texts = [(p[0], "es") for p in _food()] + [(p[2], "en") for p in _food()]
    header = ["line"] + [(tag + d).strip() for d, tag in MEASURE]
    for text, language in texts:
        cells: list = [{"text": f"{text} ({language})"}]
        base = None
        for direction, tag in MEASURE:
            take = voice.say(tag + text, language, role="native" if language == "es" else "guide",
                             direction=direction, raw=True)
            seconds = voice.trimmed_seconds(take)
            base = base or seconds
            name = hashlib.sha256(f"{text}|{direction}|{tag}".encode()).hexdigest()[:12]
            path = tts.write_clip(DIR / "measure" / f"{name}.mp3",
                                  take.audio, take.sr)
            cells.append({"text": f"{seconds:.2f} s · {base / seconds if seconds else 0:.2f}×",
                          "src": relative(path)})
        rows.append(cells)
    return {"id": "A.speed-round.measure", "title": "How fast does the voice go when asked?",
            "why": "Each line recorded under four directions (a bare note, no stretching). The "
                   "figure is the trimmed length, and the speed relative to the first column. "
                   "Tap a cell to hear it.",
            "table": {"columns": header, "rows": rows}}


def chant() -> list[Drill]:
    drills = []
    for headword, (syllables, stress) in CHANT.items():
        gloss = word(headword, "es")["primaryGloss"]
        pair = (headword, "es", gloss, "en")
        spoken = " ".join(syllables)
        drills.append(_chant_snapped(pair, syllables, stress, 0.5))
        drills.append(_chant_snapped(pair, syllables, stress, 1.0))
        drills.append(_chant_separate(pair, syllables, stress))
        drills.append(Drill(
            "chant", f"markup--{_slug(headword)}", f"Chant with pause tags: {spoken}",
            "The syllables written out with [short pause] tags between them and one request for "
            "the chant, placed on the downbeat, not snapped to the grid. The simplest to build, and "
            "at the voice's mercy for the rhythm.",
            [Line(0, 0, "W", text=" [short pause] ".join(syllables),
                  direction="chanting it rhythmically, every syllable even", raw=True, beats=4,
                  label=f"chant {spoken}"),
             Line(1, 0, "W"), Line(1, 2, "T"),
             Line(2, 0, "W", text=" [short pause] ".join(syllables),
                  direction="chanting it rhythmically, every syllable even", raw=True, beats=4,
                  take=1, label=f"chant {spoken}"),
             Line(3, 0, "W", take=1), Line(3, 2, "T", take=1)], 4, pair=pair))
    return drills


def _slug(text: str) -> str:
    return text.replace(" ", "-")


CHANT_DIRECTION = ("chanting it syllable by syllable, evenly and rhythmically like a drum beat, "
                   "every syllable the same length")


def _on_steps(pieces: list[np.ndarray], bar: int, stress: int, step: float) -> list[Line]:
    """One syllable per step (in beats), each fitted into its step, the stressed one louder."""
    seconds = Grid.from_spec(bed(*BEDS["main"])).beat * step
    lines = []
    for index, piece in enumerate(pieces):
        piece = dsp.fit(piece, seconds * 0.95, SR, slot_seconds=seconds)
        lines.append(Line(bar, index * step, "W", audio=piece * (1.4 if index == stress else 1.0),
                          label=f"♪{index + 1}"))
    return lines


def _chant_snapped(pair, syllables: list[str], stress: int, step: float) -> Drill:
    """One chanted take, stretched as a whole towards its slots, then cut and snapped.

    A chanted take is slow: *la berenjena* came back at almost five seconds. Squeezing each
    syllable into an eighth note on its own would ask the stretch for 2.5×, so the whole take is
    first brought to the length of its slots (capped at 2×), and only then cut.
    """
    take = voice.say(pair[0], "es", direction=CHANT_DIRECTION, raw=True)
    raw = voice.prepared(take)
    target = Grid.from_spec(bed(*BEDS["main"])).beat * step * len(syllables) * 0.9
    rate = min(max(len(raw) / SR / target, 0.8), 2.0)
    audio = voice.prepared(take, stretch=rate)
    pieces = cut.split(audio, SR, len(syllables))
    notes = (f"The chanted take was {len(raw) / SR:.1f} s, stretched ×{rate:.2f} before cutting."
             + ("" if pieces else " The cut failed (fewer loudness peaks than syllables), so the "
                                  "whole take is placed."))
    name = "eighths" if step == 0.5 else "quarters"
    if step == 0.5:
        chant = lambda bar: _on_steps(pieces, bar, stress, step) if pieces else \
            [Line(bar, 0, "W", audio=audio, label="chant (uncut)")]
        lines = chant(0) + chant(2) + [Line(1, 0, "W"), Line(1, 2, "T"), Line(3, 0, "W", take=1),
                                       Line(3, 2, "T", take=1)]
        bars = 4
    else:
        lines = (_on_steps(pieces, 0, stress, step) if pieces else
                 [Line(0, 0, "W", audio=audio, label="chant (uncut)")])
        lines += [Line(2, 0, "W"), Line(2, 2, "T")]
        bars = 3
    return Drill(
        "chant", f"snapped-{name}--{_slug(pair[0])}",
        f"Chant on {name}, one take cut and snapped: {'-'.join(syllables)}",
        f"The voice chants the word once. The take is cut at its loudness dips into "
        f"{len(syllables)} syllables, and each is placed on {'an eighth note' if step == 0.5 else 'a beat'}, "
        "with the stressed one louder, then the whole word and its meaning. The rhythm is the "
        "grid's, not the voice's.", lines, bars, pair=pair, notes=notes)


def _chant_separate(pair, syllables: list[str], stress: int) -> Drill:
    pieces = []
    for index, syllable in enumerate(syllables):
        take = voice.say(syllable, "es", direction="just this one syllable, crisply, as one beat "
                         "of a chant" + (", stressed" if index == stress else ""), raw=True)
        pieces.append(voice.prepared(take))
    return Drill(
        "chant", f"separate--{_slug(pair[0])}", f"Chant, each syllable recorded alone: {'-'.join(syllables)}",
        "Each syllable is its own request, placed on an eighth note. Exact timing, but a syllable "
        "said alone is not the syllable said inside the word; this is the test of how much that "
        "matters.",
        _on_steps(pieces, 0, stress, 0.5) + _on_steps(pieces, 2, stress, 0.5)
        + [Line(1, 0, "W"), Line(1, 2, "T"), Line(3, 0, "W", take=1), Line(3, 2, "T", take=1)],
        4, pair=pair)


GROUPS = [
    ("classic", "Classic (today's loop)", "The reference: today's `retrieval` pattern, unchanged."),
    ("two-or-four", "Two or four", "The pair two or four times, in different rhythms, filling four bars."),
    ("reverse", "Reverse", "The same structures with the translation first: recall the word from its meaning."),
    ("echo", "Echo", "Repeat after me: a cue and a silent bar to say the word in."),
    ("slow-normal-fast", "Slow, normal, fast", "Three speeds of one word, by stretching, by asking, or both."),
    ("speed-round", "Speed round", "Every pair once, quickly, as a closing sprint."),
    ("chant", "Chant", "The word's syllables on eighth notes, then the whole word."),
]


def classic() -> list[Drill]:
    return [Drill("classic", "retrieval", "Classic retrieval drill",
                  "Word, a silent bar to recall the meaning, the meaning; then the pair twice "
                  "more; then a rest. Production takes 0–2 per line, as a loop renders them.",
                  [Line(0, 0, "W", take=0), Line(2, 0, "T", take=0), Line(3, 0, "W", take=1),
                   Line(4, 0, "T", take=1), Line(5, 0, "W", take=2), Line(6, 0, "T", take=2)], 8)]


def drills() -> list[Drill]:
    return classic() + two_or_four() + echo() + slow_normal_fast() + speed_round() + chant()


def run() -> dict:
    DIR.mkdir(parents=True, exist_ok=True)
    for old in DIR.rglob("*.mp3"):
        old.unlink()
    everything = drills()
    # Buy every take first, a few at a time; rendering then reads the cache.
    speakable = [(line, d.pair) for d in everything for line in d.lines if line.audio is None]
    parallel(lambda item: _speak(*item), speakable)
    cards = {group: [] for group, _, _ in GROUPS}
    for drill in everything:
        cards[drill.group].append(render(drill))
        print(f"  A {drill.group}/{drill.id}")
    cards["speed-round"].append(measure())
    report = {"section": "A", "title": "Drills over music",
              "intro": f"One pair, “{PAIR[0]}” — “{PAIR[2]}”, in every structure, over one bed "
                       f"({BEDS['main'][0]}), so only the structure changes. The native voice "
                       f"({voice.NATIVE}) says Spanish; the guide ({voice.GUIDE}) says English. "
                       "Each clip opens and closes with a bar of music. The grid under each clip "
                       "shows what starts on each eighth note.",
              "groups": [{"id": g, "title": t, "intro": i, "cards": cards[g]}
                         for g, t, i in GROUPS]}
    write_report(SECTION, report)
    return report
