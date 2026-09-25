"""Section C: one line, several languages, and what each voice does with it.

A programme's presenters will sometimes quote another language inside their own: "Today's word is
*el atasco*". So before examples or stories are written, it is worth hearing how each voice copes.
Ten lines, each mixing two or three of the owner's languages, are each spoken by every
configuration below. Every card carries the exact request, so what was asked can be read beside
what came back.

The findings from the documentation are in `RESEARCH`, shown at the top of the section and copied
into README.md.
"""

from __future__ import annotations

import json
from html import escape

import tts
import voice
from common import NAMES, OUT, parallel, relative, write_report

SECTION = "mixed"
DIR = OUT / SECTION

# (main language, spans): the main language is the presenter's own.
LINES = [
    ("en", [("en", "Today's word is "), ("es", "el atasco"), ("en", ", a traffic jam.")]),
    ("ru", [("ru", "Сегодня учим слово "), ("es", "el desierto"), ("ru", " — пустыня.")]),
    ("es", [("es", "¿Cómo se dice "), ("en", "traffic jam"), ("es", "? ¡El atasco!")]),
    ("en", [("en", "The French say "), ("fr", "bon appétit"), ("en", ", and so do we.")]),
    ("en", [("en", "In German, a traffic jam is "), ("de", "der Stau"),
            ("en", ": short and grumpy.")]),
    ("ru", [("ru", "Английское слово "), ("en", "doppelganger"), ("ru", " — это двойник.")]),
    ("es", [("es", "En italiano se dice "), ("it", "buongiorno"),
            ("es", "; en español, buenos días.")]),
    ("en", [("en", "In Mandarin, thank you is "), ("zh", "谢谢"), ("en", ".")]),
    ("en", [("es", "La cebolla"), ("en", ": onion. In Russian, "), ("ru", "лук"),
            ("en", ". Three languages, one tear.")]),
    ("es", [("es", "Mi amiga siempre dice "), ("fr", "c'est la vie"),
            ("es", " cuando pierde el autobús.")]),
]


def _text(spans) -> str:
    return "".join(text for _, text in spans)


def _foreign(main, spans) -> list[tuple[str, str]]:
    return [(lang, text.strip(" ,.:;—")) for lang, text in spans if lang != main]


def _prompt(main, spans) -> str:
    quoted = "; ".join(f"say “{text}” in {NAMES[lang]}, with a native {NAMES[lang]} pronunciation"
                       for lang, text in _foreign(main, spans))
    return (f"Read this line aloud as a bilingual radio presenter whose own language is "
            f"{NAMES[main]}. {quoted[0].upper() + quoted[1:]}, and everything else in "
            f"{NAMES[main]}. Do not add, drop, translate or explain any words.")


def _ssml_lang(main, spans) -> str:
    return "<speak>" + "".join(
        escape(text) if lang == main else
        f'<lang xml:lang="{tts.LOCALES[lang]}">{escape(text)}</lang>' for lang, text in spans
    ) + "</speak>"


def _ssml_voice(main, spans) -> str:
    return "<speak>" + "".join(
        escape(text) if lang == main else
        f'<voice name="{tts.WAVENET[lang]}">{escape(text)}</voice>' for lang, text in spans
    ) + "</speak>"


CONFIGS = [
    ("g31-directed", "Gemini 3.1, the prompt names each part's language",
     "Production's directed voice (Kore), with `languageCode` set to the presenter's language and "
     "a prompt saying which words are in which language. The only way to label a span for a "
     "Gemini voice is to say it in words.",
     lambda main, spans: tts.google(_text(spans), locale=tts.LOCALES[main], voice=voice.NATIVE,
                                    model=tts.GEMINI_31, prompt=_prompt(main, spans))),
    ("g31-directed-charon", "The same, a second Gemini voice",
     "As above, in Charon (the guide voice here), to hear whether it is the model or the voice.",
     lambda main, spans: tts.google(_text(spans), locale=tts.LOCALES[main], voice=voice.GUIDE,
                                    model=tts.GEMINI_31, prompt=_prompt(main, spans))),
    ("g31-plain", "Gemini 3.1, no prompt",
     "The same voice with no prompt at all: only `languageCode` says what the line is. What the "
     "model does with a foreign phrase on its own.",
     lambda main, spans: tts.google(_text(spans), locale=tts.LOCALES[main], voice=voice.NATIVE,
                                    model=tts.GEMINI_31)),
    ("g31-other-locale", "Gemini 3.1, the quoted language's languageCode",
     "No prompt, and `languageCode` set to the *quoted* language instead of the presenter's. If "
     "the code sets the accent, the presenter's own words should now sound foreign.",
     lambda main, spans: tts.google(_text(spans), locale=tts.LOCALES[_foreign(main, spans)[0][0]],
                                    voice=voice.NATIVE, model=tts.GEMINI_31)),
    ("g25-plain", "Gemini 2.5 Flash TTS, no prompt",
     "Production's second directed model, no prompt: the older model's handling of the switch.",
     lambda main, spans: tts.google(_text(spans), locale=tts.LOCALES[main], voice=voice.NATIVE,
                                    model=tts.GEMINI_25)),
    ("api-38", "Gemini API 3.8 Flash TTS, language detected",
     "The Gemini API's newest voice on the free key: no language code exists in the request, so "
     "the model decides everything. Not a production path today.",
     lambda main, spans: tts.gemini_api(_text(spans))),
    ("wavenet-plain", "WaveNet, the presenter's voice",
     "Production's clear voice for the presenter's language, plain text. A WaveNet voice speaks "
     "one language, so the quoted words are read with its phonetics.",
     lambda main, spans: tts.google(_text(spans), locale=tts.LOCALES[main],
                                    voice=tts.WAVENET[main])),
    ("wavenet-lang", "WaveNet with SSML <lang>",
     "The same voice, each quoted span wrapped in `<lang xml:lang=…>`, SSML's standard way of "
     "labelling a span's language. Cloud TTS supports it “on a best effort basis”.",
     lambda main, spans: tts.google(ssml=_ssml_lang(main, spans), locale=tts.LOCALES[main],
                                    voice=tts.WAVENET[main])),
    ("wavenet-voice", "WaveNet with SSML <voice>: a second speaker",
     "Each quoted span handed to that language's own WaveNet voice with `<voice name=…>`, in one "
     "request: two people, each in their own language.",
     lambda main, spans: tts.google(ssml=_ssml_voice(main, spans), locale=tts.LOCALES[main],
                                    voice=tts.WAVENET[main])),
    ("aura", "Cloudflare Aura-1 (English lines only)",
     "Deepgram's Aura-1 on Cloudflare: English only, no language field, no direction. Only the "
     "English-led lines are sent.",
     lambda main, spans: tts.aura(_text(spans)) if main == "en" else None),
]

RESEARCH = """
<p><b>Must the text be in the voice's language?</b> For a Gemini voice, no. The voices are
multilingual: the same voice speaks any supported language, and a line may switch language
midway. Cloud TTS still <i>requires</i> a <code>languageCode</code>, which sets the output language
and accent. It is the language the presenter is "from", which the <i>other-locale</i> configuration
tests. The Gemini API's own TTS takes no language code at all and "detects the input language
automatically". For a WaveNet voice, yes: each voice belongs to one locale (its name says which),
and other languages are read with that language's phonetics unless SSML says otherwise.</p>
<p><b>Coverage.</b> On Cloud TTS, Gemini-TTS lists about 25 GA languages, including Spanish,
Russian, French, German, Italian and US English, and 75 or more in preview, including Mandarin
and British English. The Gemini API's 3.8 Flash TTS claims more than 130 languages, and its Lite
model more than 100. Quality falls off for rarer languages, which the docs do not quantify;
"best effort" is the most any page promises.</p>
<p><b>Marking which part is in which language.</b> The only standard is SSML
<code>&lt;lang xml:lang="es-ES"&gt;</code>. Cloud TTS supports it "on a best effort basis. Not all
language combinations produce the same quality", all in one voice unless
<code>&lt;voice&gt;</code> switches the speaker. Semitic languages are read as silence, and
Japanese Kanji as Chinese. Gemini voices take plain text and a prompt, with no SSML and no
per-span language field. The only tool is to say it in the prompt, and nothing in the API is
unified across providers. Aura-1 is English only and ignores language entirely.</p>
<p><b>Sources:</b>
<a href="https://docs.cloud.google.com/text-to-speech/docs/gemini-tts">Cloud TTS: Gemini-TTS</a> ·
<a href="https://docs.cloud.google.com/text-to-speech/docs/ssml">Cloud TTS: SSML (&lt;lang&gt;, &lt;voice&gt;)</a> ·
<a href="https://ai.google.dev/gemini-api/docs/speech-generation">Gemini API: speech generation</a> ·
<a href="https://developers.cloudflare.com/workers-ai/models/aura-1/">Cloudflare: Aura-1</a></p>
"""


def run() -> dict:
    DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(i, cid, fn) for i, _ in enumerate(LINES) for cid, _, _, fn in CONFIGS]

    def one(job):
        index, cid, fn = job
        main, spans = LINES[index]
        try:
            take = fn(main, spans)
        except Exception as failure:  # a refusal is a finding, not a crash
            return {"id": cid, "error": str(failure)[:400]}
        if take is None:
            return None
        path = tts.write_clip(DIR / f"{index:02d}--{cid}.mp3", take.audio, take.sr)
        return {"id": cid, "src": relative(path), "request": take.request}

    results = parallel(one, jobs, workers=5)
    by_line: dict[int, list] = {}
    for (index, cid, _), result in zip(jobs, results):
        if result is not None:
            by_line.setdefault(index, []).append(result)

    titles = {cid: title for cid, title, _, _ in CONFIGS}
    line_cards = []
    for index, (main, spans) in enumerate(LINES):
        clips = by_line.get(index, [])
        languages = " + ".join(dict.fromkeys(NAMES[lang] for lang, _ in spans))
        line_cards.append({
            "id": f"C.line.{index}",
            "title": _text(spans),
            "why": f"{languages}. The presenter speaks {NAMES[main]}.",
            "audio": [{"id": c["id"], "src": c.get("src"), "label": titles[c["id"]],
                       "note": c.get("error", ""), "rate": True} for c in clips],
            "details": [{"summary": "Requests sent",
                         "body": json.dumps([{c["id"]: c.get("request") or c.get("error")}
                                             for c in clips], ensure_ascii=False, indent=1)}],
            "rate": False})
    config_cards = [{"id": f"C.config.{cid}", "title": title, "why": why,
                     "notes": "Rate the configuration overall, after listening to the lines."}
                    for cid, title, why, _ in CONFIGS]
    report = {"section": "C", "title": "Mixed-language voices", "intro_html": RESEARCH,
              "groups": [{"id": "configs", "title": "The configurations", "cards": config_cards},
                         {"id": "lines", "title": "Ten lines", "cards": line_cards}]}
    write_report(SECTION, report)
    return report
