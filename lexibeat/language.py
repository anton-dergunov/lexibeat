"""A language, as far as LexiBeat needs to know one.

LexiBeat used to be Spanish-and-English shaped: pattern slots were literally ``("es", 0)`` and
``("en", 0)``, and the Gemini director note said *"native Spanish"* or *"native English"* on a
two-way branch. Nothing about the engine needed that, and a host that teaches Mandarin from
Portuguese could not ask for it.

So a language is a pair. ``code`` is what voices, references and locales are keyed on; ``name`` is
what goes into a director note a model reads. The host supplies both, because the host is the one
holding a vocabulary record that says what the language is called. When it supplies only a code,
the code stands in for the name — worse director notes, never a crash.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str
    name: str = ""

    def __post_init__(self) -> None:
        code = str(self.code or "").strip()
        if not code:
            raise ValueError("A language needs a code.")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "name", str(self.name or "").strip() or code)

    def __str__(self) -> str:
        return self.name

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "name": self.name}

    @classmethod
    def from_value(cls, value: object) -> "Language":
        """Accept a bare code, a mapping, or a Language."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value)
        if isinstance(value, dict):
            return cls(str(value.get("code") or ""), str(value.get("name") or ""))
        raise ValueError(f"Cannot read a language from {type(value).__name__}.")


SPANISH = Language("es", "Spanish")
ENGLISH = Language("en", "English")
