"""Section B: split a hard word into its syllables, then say it whole.

Splitting helps only where a word is hard to say, so the section first chooses its words, in two
ways shown side by side. A detector prompt looks through the whole example vocabulary, and a hand
curated list covers the cases most worth hearing. A second prompt syllabifies every chosen item,
and its answer is checked: joined together, the syllables must give back every letter of the item.

Every item is then spoken four ways, over no music:
1. **directed**: the Gemini voice is given the hyphenated word and asked to split it, then say it;
2. **markup**: the syllables with `[short pause]` tags between them, then the word;
3. **SSML**: the WaveNet (clear) voice with a `<break>` between syllables, then the word;
4. **cut**: one natural take of the word, cut at its loudness dips (`cut.py`) and spaced out, then
   the same take whole.
"""

from __future__ import annotations

import json
import re
import unicodedata
from html import escape

import numpy as np

import cut
import llm
import tts
import voice
from common import HERE, NAMES, OUT, parallel, prompt, relative, words, write_report

SECTION = "syllables"
DIR = OUT / SECTION
EXTRA_FROM_DETECTOR = 6

METHODS = [
    ("directed", "Directed: asked to split, then say it whole",
     "The hyphenated word and a bare director note in one Gemini request. The voice decides the "
     "rhythm, and may add a word or lose the split."),
    ("markup", "Markup: [short pause] between syllables",
     "The syllables joined with Gemini's pause tags, then [medium pause] and the word. The pause "
     "length is the voice's idea of short."),
    ("ssml", "SSML breaks on the clear voice",
     "WaveNet, production's clear voice, reading SSML with an exact 300 ms `<break>` between "
     "syllables and 700 ms before the word. The most predictable of the four, and the flattest."),
    ("cut", "One natural take, cut and spaced",
     "The word said once, naturally, then cut at its loudness dips into syllables that are played "
     "with 250 ms between them, followed by the same take whole. The sounds are the real word's, "
     "and the cut can land in the wrong place."),
]


def _letters(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    return "".join(ch for ch in text if ch.isalpha())


def detect() -> dict:
    """The detector, one call per language over the whole example vocabulary."""
    results = {}
    for language in ("es", "en"):
        rows = [row for row in words() if row["language"] == language]
        heads = {row["headword"] for row in rows}
        listing = "\n".join(f"- {row['headword']} ({row.get('pos')}: {row.get('primaryGloss')})"
                            for row in rows)

        def check(reply, heads=heads):
            picks = reply["picks"]
            llm.need(isinstance(picks, list), "picks is not a list")
            kept = [p for p in picks if p.get("headword") in heads]
            llm.need(len(kept) >= len(picks) * 0.8, "too many picks name words not in the list")
            return kept

        other = "Russian" if language == "en" else "English"
        results[language] = llm.ask(prompt("hard_words", language=NAMES[language],
                                           learner_language=other, words=listing), check=check)
    return results


def syllabify(items: list[dict]) -> dict:
    listing = [{"id": str(i), "text": item["text"], "language": NAMES[item["lang"]]}
               for i, item in enumerate(items)]

    def check(reply):
        out = {}
        for entry in reply["items"]:
            item = items[int(entry["id"])]
            marked = [str(s) for s in entry["syllables"]]
            llm.need(bool(marked), f"no syllables for {item['text']}")
            # The stress is written in capitals, which models get right far more often than an
            # index; the syllables are then put back into the item's own case.
            upper = [i for i, s in enumerate(marked) if s.isupper() and s.lower() != s]
            stress = upper[0] if len(upper) == 1 else 0
            syllables = _recased(item["text"], marked)
            out[entry["id"]] = {"syllables": syllables, "stress": stress,
                                "marked": len(upper) == 1,
                                "note": entry.get("note") or "",
                                "joins": _letters("".join(syllables)) == _letters(item["text"])}
        llm.need(len(out) >= len(items) * 0.9, "items missing from the reply")
        return out

    return llm.ask(prompt("syllabify", items=listing), check=check)


def _recased(text: str, marked: list[str]) -> list[str]:
    """The syllables with the item's own letter case, where they join back to it."""
    letters = [ch for ch in unicodedata.normalize("NFC", text) if ch.isalpha()]
    if _letters("".join(marked)) != _letters(text):
        return [s.lower() for s in marked]
    out, at = [], 0
    for syllable in marked:
        count = len([ch for ch in syllable if ch.isalpha()])
        out.append("".join(letters[at:at + count]))
        at += count
    return out


def _shown(syllables: list[str], stress: int) -> str:
    return "·".join(s.upper() if i == stress else s for i, s in enumerate(syllables))


def speak(item: dict, split: dict, slug: str) -> list[dict]:
    text, language = item["text"], item["lang"]
    syllables = split["syllables"]
    role = "guide" if language in ("en",) else "native"
    out = []

    take = voice.say(f"{'-'.join(syllables)}... {text}", language, role=role, raw=True,
                     direction="slowly, syllable by syllable, with an even beat between the "
                               "syllables, then say the whole word once, naturally")
    out.append(("directed", take.audio, take.sr, take.request, ""))

    tagged = " [short pause] ".join(syllables) + " [medium pause] " + text
    take = voice.say(tagged, language, role=role, raw=True, direction="clearly, like a teacher")
    out.append(("markup", take.audio, take.sr, take.request, ""))

    ssml = ("<speak>" + '<break time="300ms"/>'.join(escape(s) for s in syllables)
            + f'<break time="700ms"/>{escape(text)}</speak>')
    take = tts.google(ssml=ssml, locale=tts.LOCALES[language], voice=tts.WAVENET[language])
    out.append(("ssml", take.audio, take.sr, take.request, ""))

    take = voice.say(text, language, role=role)
    whole = voice.trimmed(take)
    pieces = cut.split(whole, take.sr, len(syllables))
    gap = np.zeros(int(0.25 * take.sr), dtype=np.float32)
    if pieces:
        parts = [p for piece in pieces for p in (piece, gap)]
        audio = np.concatenate([*parts, np.zeros(int(0.35 * take.sr), np.float32), whole])
        note = ""
    else:
        audio, note = whole, "The cut failed: fewer loudness peaks than syllables."
    out.append(("cut", audio, take.sr, take.request, note))

    clips = []
    for method, audio, sr, request, note in out:
        path = tts.write_clip(DIR / f"{slug}--{method}.mp3", audio, sr)
        clips.append({"id": method, "src": relative(path), "label": method, "note": note,
                      "request": request})
    return clips


def run() -> dict:
    DIR.mkdir(parents=True, exist_ok=True)
    for old in DIR.glob("*.mp3"):
        old.unlink()
    found = detect()
    curated = json.loads((HERE / "syllable_words.json").read_text(encoding="utf-8"))
    chosen = [dict(item, source="curated") for item in curated]
    have = {_letters(item["text"]) for item in chosen}
    picks = sorted(((p, lang) for lang, result in found.items() for p in (result["reply"] or [])),
                   key=lambda pair: -int(pair[0].get("difficulty", 1)))
    for pick, language in picks:
        if len([c for c in chosen if c["source"] == "detector"]) >= EXTRA_FROM_DETECTOR:
            break
        if _letters(pick["headword"]) not in have:
            chosen.append({"text": pick["headword"], "lang": language, "why": pick["why"],
                           "source": "detector"})
            have.add(_letters(pick["headword"]))

    split = syllabify(chosen)
    splits = split["reply"] or {}

    def one(pair):
        index, item = pair
        entry = splits.get(str(index))
        if not entry:
            return None
        slug = f"{index:02d}-{re.sub(r'[^a-z0-9]+', '-', _letters(item['text']) or str(index))}"
        return speak(item, entry, slug)

    spoken = parallel(one, list(enumerate(chosen)), workers=4)

    cards = []
    for index, (item, clips) in enumerate(zip(chosen, spoken)):
        entry = splits.get(str(index))
        if not entry or clips is None:
            cards.append({"id": f"B.word.{index}", "title": f"{item['text']} ({item['lang']})",
                          "why": item["why"], "notes": "The syllabifier left this item out."})
            continue
        joins = "" if entry["joins"] else " ⚠ The syllables do not join back to the word."
        if not entry["marked"]:
            joins += " ⚠ No single stressed syllable was marked."
        cards.append({
            "id": f"B.word.{_letters(item['text'])}",
            "title": f"{item['text']} · {NAMES[item['lang']]}",
            "why": f"{'Chosen by hand' if item['source'] == 'curated' else 'Picked by the detector'}: "
                   f"{item['why']}",
            "facts": f"{_shown(entry['syllables'], entry['stress'])}"
                     + (f" — {entry['note']}" if entry["note"] else "") + joins,
            "audio": [{"id": c["id"], "src": c["src"], "label": c["label"], "note": c["note"],
                       "rate": True} for c in clips],
            "details": [{"summary": "Requests sent",
                         "body": json.dumps([c["request"] for c in clips], ensure_ascii=False,
                                            indent=1)}],
            "rate": False})

    detector_cards = []
    for language, result in found.items():
        detector_cards.append({
            "id": f"B.detector.{language}",
            "title": f"What the detector picked from the {NAMES[language]} words",
            "why": f"{len(result['reply'] or [])} of "
                   f"{len([r for r in words() if r['language'] == language])} words. Rate each "
                   "pick: was it worth splitting?",
            "items": [{"id": _letters(p["headword"]),
                       "text": f"{p['headword']}  ·  difficulty {p.get('difficulty')}  ·  "
                               f"{', '.join(p.get('kinds', []))}",
                       "sub": p.get("why", "")} for p in (result["reply"] or [])],
            "details": [{"summary": "Prompt", "body": result["prompt"]},
                        {"summary": f"Answered by {result['model']} in {result['seconds']} s",
                         "body": json.dumps(result["passed_over"], ensure_ascii=False, indent=1)
                         if result["passed_over"] else result["text"]}],
            "rate": True})

    method_cards = [{"id": f"B.method.{mid}", "title": title, "why": why,
                     "notes": "Rate the method overall, after listening to a few words."}
                    for mid, title, why in METHODS]
    report = {"section": "B", "title": "Syllables",
              "intro": "Hard words split into syllables, then said whole. The stressed syllable "
                       "is in capitals. Rate the four methods overall at the top, and each "
                       "recording under its word.",
              "groups": [
                  {"id": "methods", "title": "The four methods", "cards": method_cards},
                  {"id": "detector", "title": "Choosing the words", "cards": detector_cards},
                  {"id": "words", "title": f"{len(cards)} words, four ways each",
                   "intro": "The syllabifier's prompt is at the foot of this group.",
                   "cards": cards + [{"id": "B.syllabifier", "title": "The syllabifier",
                                      "why": f"Answered by {split['model']} in {split['seconds']} s.",
                                      "details": [{"summary": "Prompt", "body": split["prompt"]}],
                                      "rate": True}]},
              ]}
    write_report(SECTION, report)
    return report
