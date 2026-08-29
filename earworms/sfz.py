"""Small, dependency-free SFZ reader for indexing sample libraries.

This is deliberately not a complete SFZ synthesizer.  It resolves the opcodes
needed to understand the common instrument libraries used by Earworms and
reports everything else so an imported bank is never treated as fully
supported by accident.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from .samples import midi


SUPPORTED_OPCODES = {
    "sample", "key", "pitch_keycenter", "lokey", "hikey", "lovel", "hivel",
    "seq_length", "seq_position", "lorand", "hirand", "volume", "pan",
    "offset", "end", "loop_mode", "loop_start", "loop_end", "transpose",
    "tune", "group", "off_by", "ampeg_attack", "ampeg_decay",
    "ampeg_sustain", "ampeg_release",
}


@dataclass(frozen=True)
class SFZZone:
    sample: str
    key_center: int | None = None
    lo_key: int | None = None
    hi_key: int | None = None
    lo_vel: int = 0
    hi_vel: int = 127
    seq_length: int = 1
    seq_position: int = 1
    volume_db: float = 0.0
    pan: float = 0.0
    loop_mode: str = "no_loop"
    opcodes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SFZDocument:
    path: Path
    zones: tuple[SFZZone, ...]
    unsupported_opcodes: tuple[str, ...]


_HEADER = re.compile(r"<(global|group|region)>\s*", re.IGNORECASE)
_OPCODE = re.compile(r"([A-Za-z0-9_]+)=")


def _note(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return midi(value.upper())


def _tokens(text: str) -> list[tuple[str, dict[str, str]]]:
    """Return ``(header, opcodes)`` sections while tolerating quoted paths."""
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    matches = list(_HEADER.finditer(text))
    sections: list[tuple[str, dict[str, str]]] = []
    for index, match in enumerate(matches):
        body = text[match.end():matches[index + 1].start()
                    if index + 1 < len(matches) else len(text)]
        starts = list(_OPCODE.finditer(body))
        values: dict[str, str] = {}
        for op_index, op in enumerate(starts):
            raw = body[op.end():starts[op_index + 1].start()
                       if op_index + 1 < len(starts) else len(body)].strip()
            if raw:
                try:
                    parsed = shlex.split(raw, posix=True)
                    raw = " ".join(parsed) if parsed else raw
                except ValueError:
                    pass
            values[op.group(1).lower()] = raw
        sections.append((match.group(1).lower(), values))
    return sections


def parse(path: Path) -> SFZDocument:
    """Parse common SFZ region inheritance and return resolved zones."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    global_ops: dict[str, str] = {}
    group_ops: dict[str, str] = {}
    zones: list[SFZZone] = []
    unsupported: set[str] = set()
    for header, values in _tokens(text):
        unsupported.update(set(values) - SUPPORTED_OPCODES)
        if header == "global":
            global_ops.update(values)
            continue
        if header == "group":
            group_ops = {**global_ops, **values}
            continue
        ops = {**global_ops, **group_ops, **values}
        sample = ops.get("sample")
        if not sample:
            continue
        key = _note(ops.get("key"))
        center = _note(ops.get("pitch_keycenter"))
        zones.append(SFZZone(
            sample=sample.replace("\\", "/"),
            key_center=center if center is not None else key,
            lo_key=_note(ops.get("lokey")) if "lokey" in ops else key,
            hi_key=_note(ops.get("hikey")) if "hikey" in ops else key,
            lo_vel=int(ops.get("lovel", 0)), hi_vel=int(ops.get("hivel", 127)),
            seq_length=int(ops.get("seq_length", 1)),
            seq_position=int(ops.get("seq_position", 1)),
            volume_db=float(ops.get("volume", 0.0)),
            pan=float(ops.get("pan", 0.0)),
            loop_mode=ops.get("loop_mode", "no_loop"), opcodes=ops,
        ))
    return SFZDocument(path, tuple(zones), tuple(sorted(unsupported)))
