"""The two presenters, and what a take is told: production's director note, or a bare one.

**Production's note.** A loop's take reaches Acervo as LexiBeat's `delivery_instruction`, which is
the caller's direction followed by the repetition's pace and pitch words. Acervo frames that with
`prompts/acervo_pronounce_take.md` and sends it to Cloud TTS's `gemini-3.1-flash-tts-preview` in
voice `Kore`. `say()` reproduces that exactly, so a classic drill here is what the owner already
hears.

**A bare note.** The speed and chant experiments ask the voice for something the prosody words
would contradict: "slowly" followed by "at a natural pace". `say(raw=True)` therefore sends only
"Speak {direction}." inside the same frame, and says so on the card.

**Two presenters.** A native voice for the target language and a guide voice for the learner's
language, so a drill sounds like two people rather than one person switching languages.
"""

from __future__ import annotations

import numpy as np

from lexibeat import dsp
from lexibeat.music import SR
from lexibeat.voice import Delivery, delivery_instruction

import tts
from common import NAMES

# Acervo's prompts/acervo_pronounce_take.md, verbatim.
TAKE_PROMPT = ("Read this {language} line aloud as a native speaker would say it. {direction} "
               "Do not add, drop, translate or explain any words.")

NATIVE = "Kore"   # production's default Gemini voice
GUIDE = "Charon"  # a second presenter for the learner's language


def voice_for(language: str, role: str) -> str:
    return GUIDE if role == "guide" else NATIVE


def note(direction: str = "", take: int = 0, *, raw: bool = False) -> str:
    if raw:
        return f"Speak {direction.strip().rstrip('.')}." if direction else ""
    return delivery_instruction(Delivery.for_take(take, direction))


def say(text: str, language: str, *, role: str = "native", direction: str = "", take: int = 0,
        raw: bool = False, model: str = tts.GEMINI_31, voice: str | None = None) -> tts.Take:
    """One directed take, framed exactly as production frames it."""
    prompt = TAKE_PROMPT.format(language=NAMES[language],
                                direction=note(direction, take, raw=raw)).replace("  ", " ")
    return tts.google(text, locale=tts.LOCALES[language], voice=voice or voice_for(language, role),
                      model=model, prompt=prompt, take=take)


def prepared(take: tts.Take, *, stretch: float = 1.0, max_seconds: float | None = None,
             slot_seconds: float | None = None, gain_db: float = 0.0) -> np.ndarray:
    """A take ready to place on the grid: trimmed, optionally stretched, at the music's rate, fitted.

    The same chain `Speaker.say` runs, with the stretch as an explicit argument rather than a
    property of the take, because stretching is what several of these experiments are about.
    """
    audio = dsp.trim(take.audio)
    if abs(stretch - 1.0) > 1e-3 and len(audio) > take.sr // 10:
        audio = dsp.time_stretch(audio, take.sr, stretch)
    audio = dsp.resample(audio, take.sr, SR)
    if max_seconds:
        audio = dsp.fit(audio, max_seconds, SR, slot_seconds=slot_seconds)
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if peak > 0:
        audio = audio / peak * 0.9
    return (audio * 10 ** (gain_db / 20)).astype(np.float32)


def trimmed(take: tts.Take) -> np.ndarray:
    return dsp.trim(take.audio)


def trimmed_seconds(take: tts.Take) -> float:
    return len(trimmed(take)) / take.sr
