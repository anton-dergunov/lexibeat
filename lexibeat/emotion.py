"""Colour each word's delivery using the emoji already in the vocabulary notes.

98% of the entries carry one, so the notes are effectively pre-annotated. An
emoji that names a feeling picks that feeling; an emoji that names an object
falls through to a neutral, engaged reading.
"""

from __future__ import annotations

from dataclasses import dataclass

# IndexTTS orders its vector this way; kept here so that backend is cheap to add.
VECTOR_ORDER = ("happy", "angry", "sad", "afraid", "disgusted",
                "melancholic", "surprised", "calm")


@dataclass(frozen=True)
class Emotion:
    name: str
    exaggeration: float = 0.45  # Chatterbox: 0 flat, 1 theatrical
    cfg_weight: float = 0.5  # lower pacing is more deliberate
    pitch_bias: float = 0.0  # semitones, for the Kokoro path
    speed_bias: float = 1.0  # multiplier, for the Kokoro path

    def vector(self) -> list[float]:
        """An 8-float emotion vector, for backends that take one."""
        weights = dict.fromkeys(VECTOR_ORDER, 0.0)
        vector_name = (self.name if self.name in VECTOR_ORDER
                       else _VECTOR_NAME.get(self.name, "calm"))
        weights[vector_name] = self.exaggeration
        return [weights[k] for k in VECTOR_ORDER]


NEUTRAL = Emotion("neutral", exaggeration=0.40, cfg_weight=0.5)

EMOTIONS = {
    "neutral": NEUTRAL,
    "warm": Emotion("warm", 0.50, 0.45, pitch_bias=0.2, speed_bias=0.99),
    "happy": Emotion("happy", 0.62, 0.45, pitch_bias=0.4, speed_bias=1.02),
    "delighted": Emotion("delighted", 0.66, 0.42, pitch_bias=0.45, speed_bias=1.01),
    "emphatic": Emotion("emphatic", 0.70, 0.40, pitch_bias=0.3, speed_bias=1.03),
    "surprised": Emotion("surprised", 0.72, 0.40, pitch_bias=0.5, speed_bias=1.04),
    "angry": Emotion("angry", 0.68, 0.38, pitch_bias=-0.2, speed_bias=1.03),
    "sad": Emotion("sad", 0.45, 0.60, pitch_bias=-0.45, speed_bias=0.95),
    "afraid": Emotion("afraid", 0.60, 0.45, pitch_bias=0.35, speed_bias=1.02),
    "disgusted": Emotion("disgusted", 0.58, 0.45, pitch_bias=-0.25, speed_bias=0.98),
    "thoughtful": Emotion("thoughtful", 0.38, 0.60, pitch_bias=-0.15, speed_bias=0.96),
    "calm": Emotion("calm", 0.32, 0.62, pitch_bias=-0.1, speed_bias=0.95),
}

_VECTOR_NAME = {
    "warm": "happy", "delighted": "happy", "emphatic": "surprised",
    "thoughtful": "calm", "neutral": "calm",
}

# Emoji that name a feeling. Everything else falls through to neutral.
_EMOJI = {
    "happy": "😀😃😄😁😆😊🙂😌🤩🥳🎉😸👏🌞🌈",
    "delighted": "🤤😋😍🥰😘💖❤️🧡💛💚💙💜🍰🍫",
    "emphatic": "💪🔥⚡‼️❗💥🚀🏆✅👊🗣",
    "surprised": "😮😲😯🤯😱🙀❓⁉️😳",
    "angry": "😠😡🤬👿💢🤯🖕",
    "sad": "😢😭😥😞😔😟🙁☹️💔😿🥺",
    "afraid": "😨😰😧🥶👻⚠️🆘",
    "disgusted": "🤢🤮😖😣🤧🦠🚫",
    "thoughtful": "🤔🧐💭📚🎓🧠💡❔",
    "calm": "😐😑🧘🌙💤😴🕊️🌿🍃",
}

EMOJI_TO_EMOTION = {ch: name for name, chars in _EMOJI.items() for ch in chars}


def for_item(source: str, emoji: str = "", *, enabled: bool = True) -> Emotion:
    """Pick the delivery for one vocabulary item."""
    if not enabled:
        return NEUTRAL

    for ch in emoji:
        if ch in EMOJI_TO_EMOTION:
            return EMOTIONS[EMOJI_TO_EMOTION[ch]]

    # Spanish marks exclamations and questions at both ends, so the punctuation
    # is a reliable hint even when no emoji says anything.
    if "¡" in source or source.rstrip().endswith("!"):
        return EMOTIONS["emphatic"]
    if "¿" in source or source.rstrip().endswith("?"):
        return EMOTIONS["thoughtful"]
    return EMOTIONS["warm"]
