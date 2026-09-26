"""Section G: give the voice the word's pronunciation, not its spelling.

Section B cut words into syllables by their spelling and handed the pieces to a voice. On words
that are not said as written, the voice then said syllables the word does not have, and a
pronunciation clip that is wrong is worse than none. Here the words B got wrong are spoken again
with the dictionary's pronunciation (IPA from Wiktionary, `pronunciation_words.json`) in three
conditions:

1. **plain, slowly**: the word as written, asked for slowly and whole, with no pronunciation;
2. **IPA, slowly**: the same, with the written pronunciation given;
3. **IPA, syllable by syllable**: syllables taken from the pronunciation, then the word whole.

Each condition goes to the voices that have a way to take a pronunciation:
- **Gemini 3.1** on Cloud TTS, production's voice, which documents none: IPA in slashes in the
  text, or named in the prompt;
- **Gemini 3.8** through the Gemini API, whose guide says IPA in slashes is followed;
- **Chirp 3 HD**, the same speakers, with `customPronunciations` and the API's own speaking rate,
  and SSML `<phoneme>` per syllable;
- **WaveNet** with SSML `<phoneme>`, production's plain voice.

Two references sit beside them: a human recording from Wikimedia Commons where one exists, and B's
directed clip, which is the same request as before and so comes from the cache.

**A probe comes first**, because a clip of the right word proves nothing if the voice would have
said it right anyway. Each mechanism is given a decoy word with another word's pronunciation: hear
the decoy and the pronunciation was ignored. It found that Google's voices ignore `<phoneme>` in
Russian and refuse a Russian custom pronunciation, so those setups are not rendered for Russian.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from html import escape

import httpx

import tts
import voice
from common import HERE, NAMES, OUT, parallel, relative, write_report

SECTION = "pronounce"
DIR = OUT / SECTION
REFS = OUT / "cache" / "refs"

SLOW = "slowly and clearly, the whole word, without splitting it"
BY_SYLLABLE = ("slowly, syllable by syllable, with an even beat between the syllables, then say the "
               "whole word once, naturally")
CHIRP_RATE = 0.7

SETUPS = [
    ("human", "A person: the Wikimedia Commons recording"),
    ("g31-plain-slow", "Gemini 3.1 · plain word, slowly"),
    ("g31-ipa-text", "Gemini 3.1 · IPA in the text, slowly"),
    ("g31-ipa-prompt", "Gemini 3.1 · IPA in the prompt, slowly"),
    ("g31-ipa-syll", "Gemini 3.1 · IPA syllables, then the word"),
    ("g38-plain-slow", "Gemini 3.8 · plain word, slowly"),
    ("g38-ipa", "Gemini 3.8 · IPA in the text, slowly"),
    ("g38-ipa-syll", "Gemini 3.8 · IPA syllables, then the word"),
    ("chirp-plain-slow", f"Chirp 3 HD · plain word at {CHIRP_RATE}×"),
    ("chirp-ipa", f"Chirp 3 HD · custom pronunciation at {CHIRP_RATE}×"),
    ("chirp-ipa-syll", f"Chirp 3 HD · SSML phoneme syllables, then the word, at {CHIRP_RATE}×"),
    ("wavenet-ipa", "WaveNet · SSML phoneme, slow"),
    ("b-directed", "For comparison, not rated: B's directed clip, split by spelling"),
]

# Measured by the probe: Russian `<phoneme>` is ignored and a Russian custom pronunciation refused.
IGNORED = {"ru": {"chirp-ipa", "chirp-ipa-syll", "wavenet-ipa"}}
IGNORED_NOTE = ("Not rendered: Google's voices ignore a written pronunciation in Russian, so this "
                "would be the plain word (see the probe).")

# A decoy word per language, and the entry whose pronunciation it is given.
PROBES = {"en": ("banana", "lethargy"), "es": ("manzana", "imprescindible"),
          "ru": ("яблоко", "достопримечательность"), "fr": ("pomme", "écureuil")}
PROBE_SETUPS = [
    ("chirp-custom", "Chirp 3 HD · customPronunciations"),
    ("chirp-ssml", "Chirp 3 HD · SSML <phoneme>"),
    ("wavenet-ssml", "WaveNet · SSML <phoneme>"),
    ("g31-prompt", "Gemini 3.1 · the pronunciation named in the prompt"),
]


def entries() -> list[dict]:
    return json.loads((HERE / "pronunciation_words.json").read_text(encoding="utf-8"))


def _letters(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalpha())


def _role(entry: dict) -> str:
    # As in B: the guide (Charon) speaks English, the native presenter (Kore) everything else.
    return "guide" if entry["lang"] == "en" else "native"


def _speaker(entry: dict) -> str:
    return voice.voice_for(entry["lang"], _role(entry))


def _syllable_text(entry: dict) -> str:
    return " ".join(f"/{s}/" for s in entry["syllables"]) + f"... /{entry['ipa']}/"


def _phonemes(text: str, ipa: str) -> str:
    """SSML for a word or short phrase, one `<phoneme>` per word, since a tag directs one word."""
    words, sounds = text.split(), ipa.split()
    if len(words) != len(sounds):
        words, sounds = [text], [ipa.replace(" ", "")]
    return " ".join(f'<phoneme alphabet="ipa" ph="{escape(s)}">{escape(w)}</phoneme>'
                    for w, s in zip(words, sounds))


def _gemini31(entry: dict, text: str, direction: str) -> tts.Take:
    return voice.say(text, entry["lang"], role=_role(entry), raw=True, direction=direction)


def _gemini38(entry: dict, text: str, style: str) -> tts.Take:
    return tts.gemini_styled(text, style=f"speaking {style}", voice=_speaker(entry))


def _chirp(entry: dict, **kwargs) -> tts.Take:
    return tts.google(locale=tts.LOCALES[entry["lang"]], voice=tts.chirp(entry["lang"],
                      _speaker(entry)), speaking_rate=CHIRP_RATE, **kwargs)


def _human(entry: dict) -> tts.Take | None:
    recording = entry.get("recording")
    if not recording:
        return None
    REFS.mkdir(parents=True, exist_ok=True)
    path = REFS / (hashlib.sha256(recording["mp3"].encode()).hexdigest()[:24] + ".mp3")
    if not path.exists():
        if tts.DRY:
            return None
        for attempt in range(4):  # Wikimedia answers a burst of downloads with 429
            answer = httpx.get(recording["mp3"], timeout=60, follow_redirects=True,
                               headers={"User-Agent": "lexibeat-experiment/0.1 (listening test)"})
            if answer.status_code != 429 or attempt == 3:
                break
            time.sleep(10 * (attempt + 1))
        answer.raise_for_status()
        path.write_bytes(answer.content)
    audio, sr = tts._decode(path.read_bytes())
    return tts.Take(audio, sr, {"GET": recording["page"]}, True)


def _b_directed(entry: dict) -> tts.Take | None:
    """B's directed request, sent again unchanged, so the cache answers it for nothing."""
    report = OUT / "syllables" / "report.json"
    if not report.exists():
        return None
    groups = json.loads(report.read_text(encoding="utf-8"))["groups"]
    card = next((c for g in groups for c in g["cards"]
                 if c["id"] == f"B.word.{_letters(entry['text'])}"), None)
    if not card:
        return None
    sent = json.loads(card["details"][0]["body"])[0]["json"]
    return tts.google(sent["input"]["text"], locale=sent["voice"]["languageCode"],
                      voice=sent["voice"]["name"], model=sent["voice"].get("modelName"),
                      prompt=sent["input"].get("prompt"))


def speak(entry: dict, setup: str) -> tts.Take | None:
    text, ipa, lang = entry["text"], entry["ipa"], entry["lang"]
    if setup == "human":
        return _human(entry)
    if setup == "b-directed":
        return _b_directed(entry)
    if setup == "g31-plain-slow":
        return _gemini31(entry, text, SLOW)
    if setup == "g31-ipa-text":
        return _gemini31(entry, f"/{ipa}/", SLOW)
    if setup == "g31-ipa-prompt":
        return _gemini31(entry, text, f"{SLOW}, pronouncing it exactly as /{ipa}/")
    if setup == "g31-ipa-syll":
        return _gemini31(entry, _syllable_text(entry), BY_SYLLABLE)
    if setup == "g38-plain-slow":
        return _gemini38(entry, text, SLOW)
    if setup == "g38-ipa":
        return _gemini38(entry, f"/{ipa}/", SLOW)
    if setup == "g38-ipa-syll":
        return _gemini38(entry, _syllable_text(entry), BY_SYLLABLE)
    if setup == "chirp-plain-slow":
        return _chirp(entry, text=text)
    if setup == "chirp-ipa":
        return _chirp(entry, text=text, pronunciations={text: ipa})
    if setup == "chirp-ipa-syll":
        pieces = '<break time="300ms"/>'.join(_phonemes(re.sub(r"[ˈˌ]", "", s) or s, s)
                                              for s in entry["syllables"])
        return _chirp(entry, ssml=f'<speak>{pieces}<break time="700ms"/>'
                                  f'{_phonemes(text, ipa)}</speak>')
    if setup == "wavenet-ipa":
        return tts.google(ssml=f'<speak><prosody rate="slow">{_phonemes(text, ipa)}</prosody>'
                               '</speak>', locale=tts.LOCALES[lang], voice=tts.WAVENET[lang])
    raise ValueError(setup)


def probe(language: str, setup: str, decoy: str, ipa: str) -> tts.Take:
    locale, tag = tts.LOCALES[language], (f'<speak><phoneme alphabet="ipa" ph="{escape(ipa)}">'
                                          f"{escape(decoy)}</phoneme></speak>")
    if setup == "chirp-custom":
        return tts.google(decoy, locale=locale, voice=tts.chirp(language),
                          pronunciations={decoy: ipa})
    if setup == "chirp-ssml":
        return tts.google(ssml=tag, locale=locale, voice=tts.chirp(language))
    if setup == "wavenet-ssml":
        return tts.google(ssml=tag, locale=locale, voice=tts.WAVENET[language])
    if setup == "g31-prompt":
        return voice.say(decoy, language, raw=True, direction=f"clearly, pronouncing it exactly "
                                                              f"as /{ipa}/")
    raise ValueError(setup)


def _probe_cards() -> list[dict]:
    by_text = {_letters(e["text"]): e for e in entries()}
    jobs = [(lang, setup) for lang in PROBES for setup, _ in PROBE_SETUPS]

    def one(job):
        language, setup = job
        decoy, target = PROBES[language]
        try:
            take = probe(language, setup, decoy, by_text[_letters(target)]["ipa"])
        except Exception as failure:
            return {"id": setup, "error": str(failure)[:300]}
        path = tts.write_clip(DIR / f"probe-{language}--{setup}.mp3", take.audio, take.sr)
        return {"id": setup, "src": relative(path), "request": take.request}

    results = parallel(one, jobs, workers=4)
    labels, cards = dict(PROBE_SETUPS), []
    for language, (decoy, target) in PROBES.items():
        clips = [r for (lang, _), r in zip(jobs, results) if lang == language]
        cards.append({
            "id": f"G.probe.{language}",
            "title": f"{NAMES[language]}: “{decoy}” written, “{target}” asked for",
            "why": f"The text says “{decoy}”; the pronunciation given is {target}'s. Score 5 if "
                   f"you hear {target}, 1 if you hear {decoy}, which means the pronunciation was "
                   "ignored.",
            "audio": [{"id": c["id"], "src": c.get("src"), "label": labels[c["id"]],
                       "note": c.get("error", ""), "rate": bool(c.get("src"))} for c in clips],
            "details": [{"summary": "Requests sent",
                         "body": json.dumps([{c["id"]: c.get("request") or c.get("error")}
                                             for c in clips], ensure_ascii=False, indent=1)}],
            "rate": False})
    return cards


def _facts(entry: dict) -> str:
    parts = [f"/{entry['ipa']}/", "syllables " + " · ".join(entry["syllables"])]
    if entry["ipa_source"] == "hand":
        parts.append("⚠ IPA written by hand: Wiktionary has none")
    if entry.get("split") == "hand":
        parts.append("syllables split by hand")
    if entry.get("ipa_note"):
        parts.append(entry["ipa_note"])
    return " — ".join(parts)


def run(only: set[str] | None = None) -> dict:
    DIR.mkdir(parents=True, exist_ok=True)
    words = [e for e in entries() if not only or _letters(e["text"]) in only]
    jobs = [(entry, setup) for entry in words for setup, _ in SETUPS]

    def one(job):
        entry, setup = job
        if setup in IGNORED.get(entry["lang"], set()):
            return {"id": setup, "error": IGNORED_NOTE}
        slug = re.sub(r"[^a-z0-9]+", "-", _letters(entry["text"]).encode("ascii", "ignore")
                      .decode() or hashlib.sha1(entry["text"].encode()).hexdigest()[:8])
        try:
            take = speak(entry, setup)
        except tts.Quota as failure:
            return {"id": setup, "error": f"Not rendered yet: {failure}"[:300]}
        except Exception as failure:  # a refusal is a finding, not a crash
            return {"id": setup, "error": str(failure)[:400]}
        if take is None:
            return None
        path = tts.write_clip(DIR / f"{slug}--{setup}.mp3", take.audio, take.sr)
        return {"id": setup, "src": relative(path), "request": take.request,
                "seconds": round(len(voice.trimmed(take)) / take.sr, 2)}

    results = parallel(one, jobs, workers=4)
    labels = dict(SETUPS)
    cards = []
    for entry in words:
        clips = [r for (e, _), r in zip(jobs, results) if e is entry and r is not None]
        rows = []
        for clip in clips:
            note = clip.get("error", "")
            if clip["id"] == "human" and not note:
                rec = entry["recording"]
                note = f"{rec['accent']} · {rec['author']} · {rec['licence']} · {rec['page']}"
            rows.append({"id": clip["id"], "src": clip.get("src"), "label": labels[clip["id"]],
                         "note": note if clip["id"] == "human" or not clip.get("src") else "",
                         "rate": bool(clip.get("src")) and clip["id"] != "b-directed"})
        cards.append({
            "id": f"G.word.{_letters(entry['text'])}",
            "title": f"{entry['text']} · {NAMES[entry['lang']]}"
                     + (" · control" if entry.get("control") else ""),
            "why": entry["why"],
            "facts": _facts(entry),
            "audio": rows,
            "details": [{"summary": "Requests sent",
                         "body": json.dumps([{c["id"]: c.get("request") or c.get("error")}
                                             for c in clips], ensure_ascii=False, indent=1)},
                        {"summary": "Trimmed lengths",
                         "body": "\n".join(f"{c['id']}: {c['seconds']} s" for c in clips
                                           if "seconds" in c)}],
            "rate": False})
    report = {"section": "G", "title": "Pronunciation hints",
              "intro": "The words section B said wrong, spoken with the dictionary's pronunciation "
                       "instead of their spelling. Score 1 if any sound is wrong, since a wrong "
                       "clip teaches the wrong word; 2–5 only when it is right, for how useful it "
                       "is as a pronunciation clip: pace, clarity, not tiresome. The human "
                       "recording is there to check against, and is rated too.",
              "groups": [{"id": "probe", "title": "Is the pronunciation honoured at all?",
                          "cards": _probe_cards()},
                         {"id": "words", "title": f"{len(cards)} words", "cards": cards}]}
    write_report(SECTION, report)
    return report
