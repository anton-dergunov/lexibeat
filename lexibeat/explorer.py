"""Application services for the local and hosted LexiBeat music explorer.

This module deliberately has no dependency on FastAPI or Gradio.  It is the
typed boundary between those optional interfaces and the stable music API.
All paths accepted here are JSON Pointers into a BedSpec; filesystem paths are
never accepted from clients.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from .api import MusicGenerationResult, MusicRequest, resolve_music
from .bedspec import BedSpec, SCALES, STYLES
from .generator import ENGINE_VERSION
from .library import (BUNDLED_ROOT, COLLECTIONS, SampleAsset, SampleLibrary,
                      SampleRef, infer_articulation, infer_round_robin)
from .music import Grid, SR, render_bed, render_stems
from .profiles import PROFILES, get_profile
from .quality import evaluate_preview

EXPLORER_API_VERSION = "explorer-v1"
HOSTED_MAX_DURATION_SECONDS = 30.0
LOCAL_MAX_DURATION_SECONDS = 180.0
PREVIEW_MAX_DURATION_SECONDS = 15.0
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_LOCKS = 128

ValidationState = Literal["production-safe", "experimental", "invalid"]
ProgressCallback = Callable[[float, str], None]
CancelCheck = Callable[[], bool]


@dataclass(frozen=True)
class ControlField:
    path: str
    group: str
    label: str
    kind: Literal["number", "integer", "boolean", "enum", "text", "table"]
    minimum: float | None = None
    maximum: float | None = None
    safe_minimum: float | None = None
    safe_maximum: float | None = None
    step: float | None = None
    choices: tuple[str | int, ...] = ()
    unit: str = ""
    read_only: bool = False


@dataclass(frozen=True)
class ValidationIssue:
    severity: Literal["error", "warning"]
    path: str
    code: str
    message: str
    safe_value: object | None = None
    has_safe_value: bool = False

    def to_dict(self) -> dict:
        result = asdict(self)
        if not self.has_safe_value:
            result.pop("safe_value", None)
        result.pop("has_safe_value", None)
        return result


@dataclass(frozen=True)
class ExplorerValidationReport:
    state: ValidationState
    issues: tuple[ValidationIssue, ...]
    measurements: dict[str, float]
    fingerprint: dict | None = None

    @property
    def renderable(self) -> bool:
        return self.state != "invalid"

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "renderable": self.renderable,
            "issues": [issue.to_dict() for issue in self.issues],
            "measurements": self.measurements,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class RandomizationResult:
    bed_spec: BedSpec
    validation: ExplorerValidationReport
    seed: int

    def to_dict(self) -> dict:
        return {
            "bed_spec": asdict(self.bed_spec),
            "validation": self.validation.to_dict(),
            "seed": self.seed,
        }


@dataclass(frozen=True)
class RenderArtifact:
    artifact_id: str
    audio_url: str
    duration_seconds: float
    sample_rate: int
    sha256: str
    cache_hit: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExplorerConfig:
    hosted: bool = False
    output_root: Path = Path("out/explorer")
    max_cache_bytes: int = 512 * 1024 * 1024
    max_pending_renders: int = 8

    @property
    def max_duration_seconds(self) -> float:
        return (HOSTED_MAX_DURATION_SECONDS if self.hosted
                else LOCAL_MAX_DURATION_SECONDS)

    @classmethod
    def from_environment(cls) -> "ExplorerConfig":
        hosted = bool(os.environ.get("SPACE_ID") or
                      os.environ.get("LEXIBEAT_EXPLORER_HOSTED"))
        root = Path(os.environ.get("LEXIBEAT_EXPLORER_OUT", "out/explorer"))
        cache_mb = int(os.environ.get("LEXIBEAT_EXPLORER_CACHE_MB", "512"))
        return cls(hosted=hosted, output_root=root,
                   max_cache_bytes=max(cache_mb, 32) * 1024 * 1024)


CONTROL_FIELDS: tuple[ControlField, ...] = (
    ControlField("/engine_version", "Identity", "Engine version", "text",
                 read_only=True),
    ControlField("/profile_version", "Identity", "Profile version", "text",
                 read_only=True),
    ControlField("/schema_version", "Identity", "Schema version", "integer",
                 read_only=True),
    ControlField("/seed", "Identity", "Resolved seed", "integer", read_only=True),
    ControlField("/bpm", "Identity", "Tempo", "number", 30, 240, 56, 104,
                 1, unit="BPM"),
    ControlField("/beats_per_bar", "Identity", "Beats per bar", "integer", 2, 12,
                 3, 5, 1),
    ControlField("/beat_unit", "Identity", "Beat unit", "enum",
                 choices=(4, 8, 16)),
    ControlField("/swing", "Identity", "Swing", "number", 0, 0.5, 0, 0.025,
                 0.005),
    ControlField("/root", "Harmony", "Root MIDI note", "integer", 0, 127,
                 36, 60, 1, unit="MIDI"),
    ControlField("/scale", "Harmony", "Scale", "enum",
                 choices=tuple(SCALES)),
    ControlField("/progression", "Harmony", "Progression", "text"),
    ControlField("/chord_extension", "Harmony", "Chord colour", "enum",
                 choices=("none", "seventh", "add9", "ninth")),
    ControlField("/pad/enabled", "Pad", "Enabled", "boolean"),
    ControlField("/pad/instrument", "Pad", "Instrument", "enum",
                 choices=("synth", "strings")),
    ControlField("/pad/level", "Pad", "Level", "number", 0, 2, 0.15, 0.8, 0.01),
    ControlField("/pad/detune", "Pad", "Detune", "number", 0, 1, 0, 0.2,
                 0.01, unit="semitones"),
    ControlField("/pad/cutoff_base", "Pad", "Filter base", "number", 80, 18000,
                 400, 4000, 10, unit="Hz"),
    ControlField("/pad/cutoff_motion", "Pad", "Filter motion", "number", 0, 10000,
                 0, 1800, 10, unit="Hz"),
    ControlField("/pad/cutoff_curve", "Pad", "Filter curve", "enum",
                 choices=("sine", "triangle", "random_walk")),
    ControlField("/pad/cutoff_period_bars", "Pad", "Filter period", "number",
                 0.25, 64, 2, 20, 0.25, unit="bars"),
    ControlField("/pad/overlap", "Pad", "Chord overlap", "number", 0.1, 4,
                 0.5, 2.2, 0.05, unit="bars"),
    ControlField("/pad/duck_db", "Pad", "Speech duck", "number", 0, 24,
                 2, 10, 0.25, unit="dB"),
    ControlField("/bass/enabled", "Bass", "Enabled", "boolean"),
    ControlField("/bass/level", "Bass", "Level", "number", 0, 2, 0.12, 0.75, 0.01),
    ControlField("/bass/octave", "Bass", "Octave", "integer", -4, 3, -2, 0, 1),
    ControlField("/bass/attack", "Bass", "Attack", "number", 0, 2, 0.003, 0.5,
                 0.005, unit="seconds"),
    ControlField("/bass/decay_bars", "Bass", "Decay", "number", 0.02, 4,
                 0.1, 1.5, 0.05, unit="bars"),
    ControlField("/bass/duck_db", "Bass", "Speech duck", "number", 0, 24,
                 0, 8, 0.25, unit="dB"),
    ControlField("/drums/enabled", "Percussion", "Enabled", "boolean"),
    ControlField("/drums/kick", "Percussion", "Legacy kick pattern", "text"),
    ControlField("/drums/rim", "Percussion", "Legacy rim pattern", "text"),
    ControlField("/drums/level", "Percussion", "Bus level", "number", 0, 2,
                 0.15, 0.65, 0.01),
    ControlField("/drums/kick_level", "Percussion", "Legacy kick level", "number",
                 0, 2, 0.1, 0.8, 0.01),
    ControlField("/drums/rim_level", "Percussion", "Legacy rim level", "number",
                 0, 2, 0, 0.4, 0.01),
    ControlField("/drums/shaker_level", "Percussion", "Legacy shaker level", "number",
                 0, 2, 0, 0.2, 0.01),
    ControlField("/drums/shaker_density", "Percussion", "Shaker density", "number",
                 0, 1, 0, 1, 0.05),
    ControlField("/drums/duck_db", "Percussion", "Speech duck", "number", 0, 24,
                 0, 8, 0.25, unit="dB"),
    ControlField("/lead/enabled", "Lead", "Enabled", "boolean"),
    ControlField("/lead/instrument", "Lead", "Instrument", "enum",
                 choices=("synth", "piano", "marimba", "glockenspiel")),
    ControlField("/lead/level", "Lead", "Level", "number", 0, 2, 0.1, 1.2, 0.01),
    ControlField("/lead/bar_probability", "Lead", "Bar probability", "number", 0, 1,
                 0.05, 0.75, 0.05),
    ControlField("/lead/max_notes", "Lead", "Maximum notes", "integer", 1, 16,
                 1, 4, 1),
    ControlField("/lead/register", "Lead", "Register range", "text", unit="semitones"),
    ControlField("/lead/velocity", "Lead", "Velocity range", "text"),
    ControlField("/lead/humanize", "Lead", "Timing humanize", "number", 0, 0.25,
                 0, 0.04, 0.005, unit="seconds"),
    ControlField("/lead/duck_db", "Lead", "Speech duck", "number", 0, 24,
                 2, 10, 0.25, unit="dB"),
    ControlField("/space/reverb_seconds", "Space", "Reverb length", "number", 0,
                 12, 0.2, 6, 0.1, unit="seconds"),
    ControlField("/space/reverb_mix", "Space", "Reverb mix", "number", 0, 1,
                 0.05, 0.68, 0.01),
    ControlField("/phrase/family", "Resolved phrase", "Family", "enum",
                 choices=tuple(STYLES)),
    ControlField("/phrase/loop_bars", "Resolved phrase", "Loop length", "integer",
                 1, 32, 4, 8, 1, unit="bars"),
    ControlField("/phrase/harmony_texture", "Resolved phrase", "Harmony texture",
                 "enum", choices=("sustain", "drone", "open", "pulse", "arpeggio")),
    ControlField("/phrase/pad_timbre", "Resolved phrase", "Pad timbre", "enum",
                 choices=("sine", "triangle", "strings", "soft_saw")),
    ControlField("/phrase/bass_timbre", "Resolved phrase", "Bass timbre", "enum",
                 choices=("sine", "round", "triangle", "pluck")),
    ControlField("/phrase/round_robin_strategy", "Resolved phrase",
                 "Sample variation", "enum", choices=("cyclic",), read_only=True),
    ControlField("/phrase/chords", "Resolved phrase", "Chord events", "table"),
    ControlField("/phrase/bass", "Resolved phrase", "Bass events", "table"),
    ControlField("/phrase/lead", "Resolved phrase", "Lead events", "table"),
    ControlField("/phrase/percussion", "Resolved phrase", "Percussion lanes", "table"),
)

CONTROL_BY_PATH = {field.path: field for field in CONTROL_FIELDS}
TABLE_LOCK_PATHS = {
    "/phrase/chords", "/phrase/bass", "/phrase/lead", "/phrase/percussion",
    "/phrase/lead_sample", "/phrase/pad_sample", "/phrase/lead_instrument",
    "/phrase/pad_instrument", "/phrase/bass_instrument",
}
LOCKABLE_PATHS = {
    field.path for field in CONTROL_FIELDS if not field.read_only
} | TABLE_LOCK_PATHS


def explorer_schema(config: ExplorerConfig | None = None) -> dict:
    config = config or ExplorerConfig.from_environment()
    production_bundle = BUNDLED_ROOT.joinpath("catalog.sqlite3").is_file()
    return {
        "api_version": EXPLORER_API_VERSION,
        "engine_version": ENGINE_VERSION,
        "profiles": {
            name: {"version": profile.version, "families": list(profile.families)}
            for name, profile in PROFILES.items()
        },
        "simple": {
            "families": ["auto", *get_profile("production-v1").families],
            "energy": ["calm", "balanced", "bright"],
            "rhythm": ["sparse", "steady", "groovy"],
            "palette": (["acoustic", "hybrid", "electronic"]
                        if production_bundle else ["electronic"]),
        },
        "controls": [asdict(field) for field in CONTROL_FIELDS],
        "lockable_paths": sorted(LOCKABLE_PATHS),
        "limits": {
            "request_bytes": MAX_REQUEST_BYTES,
            "max_locks": MAX_LOCKS,
            "preview_seconds": PREVIEW_MAX_DURATION_SECONDS,
            "render_seconds": config.max_duration_seconds,
        },
        "capabilities": {
            "hosted": config.hosted,
            "sample_promotion": not config.hosted,
            "production_bundle": production_bundle,
            "voice": False,
            "persistent_storage": False,
        },
    }


_ROOT_KEYS = {
    "bpm", "beats_per_bar", "beat_unit", "swing", "root", "scale",
    "progression", "chord_extension", "seed", "pad", "bass", "drums", "lead",
    "space", "phrase", "schema_version", "engine_version", "profile_version",
}
_NESTED_KEYS = {
    "pad": {"instrument", "level", "detune", "cutoff_base", "cutoff_motion",
            "cutoff_curve", "cutoff_period_bars", "overlap", "duck_db", "enabled"},
    "bass": {"level", "octave", "attack", "decay_bars", "duck_db", "enabled"},
    "drums": {"kick", "rim", "kick_level", "rim_level", "shaker_level",
              "shaker_density", "level", "duck_db", "enabled"},
    "lead": {"instrument", "level", "bar_probability", "max_notes", "register",
             "velocity", "humanize", "duck_db", "enabled"},
    "space": {"reverb_seconds", "reverb_mix"},
}
_PHRASE_KEYS = {
    "family", "loop_bars", "harmony_texture", "pad_timbre", "bass_timbre",
    "chords", "bass", "lead", "percussion", "lead_sample", "pad_sample",
    "lead_instrument", "pad_instrument", "bass_instrument",
    "round_robin_strategy",
}
_NOTE_KEYS = {"step", "duration_steps", "midi_note", "velocity",
              "articulation", "sample_variation"}
_CHORD_KEYS = {"step", "duration_steps", "midi_notes", "velocity",
               "articulation", "sample_variation"}
_LANE_KEYS = {"sound", "pattern", "level", "probability", "humanize", "pan",
              "sample", "role", "articulation", "round_robin_samples"}
_SAMPLE_KEYS = {"collection", "asset_id", "sha256"}
_INSTRUMENT_KEYS = {"name", "zones"}
_ZONE_KEYS = {"sample", "root_note", "lo_note", "hi_note", "lo_velocity",
              "hi_velocity", "gain_db", "round_robin", "articulation"}
_LOGICAL_PART = re.compile(r"^[A-Za-z0-9._-]+$")


def _issue(issues: list[ValidationIssue], severity: Literal["error", "warning"],
           path: str, code: str, message: str, *, safe_value: object = None,
           has_safe_value: bool = False) -> None:
    issues.append(ValidationIssue(severity, path, code, message,
                                  safe_value, has_safe_value))


def _unknown_keys(value: object, allowed: set[str], path: str,
                  issues: list[ValidationIssue]) -> None:
    if not isinstance(value, dict):
        _issue(issues, "error", path or "/", "type", "Expected an object.")
        return
    for key in sorted(set(value) - allowed):
        _issue(issues, "error", f"{path}/{key}", "unknown_field",
               "Unknown field is not allowed at the web API boundary.")


def _validate_sample(value: object, path: str, issues: list[ValidationIssue]) -> None:
    if value is None:
        return
    _unknown_keys(value, _SAMPLE_KEYS, path, issues)
    if not isinstance(value, dict):
        return
    for key in ("collection", "asset_id"):
        part = value.get(key)
        if not isinstance(part, str) or not _LOGICAL_PART.fullmatch(part):
            _issue(issues, "error", f"{path}/{key}", "logical_id",
                   "Sample identifiers may contain only letters, numbers, dot, dash, and underscore.")
    digest = value.get("sha256", "")
    if digest and (not isinstance(digest, str) or
                   not re.fullmatch(r"[0-9a-fA-F]{64}", digest)):
        _issue(issues, "error", f"{path}/sha256", "checksum",
               "SHA-256 must contain exactly 64 hexadecimal characters.")


def _validate_instrument(value: object, path: str,
                         issues: list[ValidationIssue]) -> None:
    if value is None:
        return
    _unknown_keys(value, _INSTRUMENT_KEYS, path, issues)
    if not isinstance(value, dict):
        return
    zones = value.get("zones", [])
    if not isinstance(zones, (list, tuple)) or len(zones) > 1024:
        _issue(issues, "error", f"{path}/zones", "size",
               "Instrument zones must be a list containing at most 1024 rows.")
        return
    for index, zone in enumerate(zones):
        zone_path = f"{path}/zones/{index}"
        _unknown_keys(zone, _ZONE_KEYS, zone_path, issues)
        if isinstance(zone, dict):
            _validate_sample(zone.get("sample"), f"{zone_path}/sample", issues)
            for low_name, high_name in (("lo_note", "hi_note"),
                                        ("lo_velocity", "hi_velocity")):
                low = zone.get(low_name, 0)
                high = zone.get(high_name, 127)
                if (not isinstance(low, int) or isinstance(low, bool) or
                        not isinstance(high, int) or isinstance(high, bool) or
                        not 0 <= low <= high <= 127):
                    _issue(issues, "error", zone_path, "zone_range",
                           f"{low_name}/{high_name} must be ordered MIDI values from 0 to 127.")
            round_robin = zone.get("round_robin", 0)
            if (not isinstance(round_robin, int) or isinstance(round_robin, bool)
                    or not 0 <= round_robin <= 1024):
                _issue(issues, "error", f"{zone_path}/round_robin",
                       "round_robin", "Round-robin index must be an integer from 0 to 1024.")


def _validate_shape(data: object) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    _unknown_keys(data, _ROOT_KEYS, "", issues)
    if not isinstance(data, dict):
        return issues
    for name, allowed in _NESTED_KEYS.items():
        if name in data:
            _unknown_keys(data[name], allowed, f"/{name}", issues)
    phrase = data.get("phrase")
    if phrase is None:
        return issues
    _unknown_keys(phrase, _PHRASE_KEYS, "/phrase", issues)
    if not isinstance(phrase, dict):
        return issues
    for collection, allowed in (("chords", _CHORD_KEYS), ("bass", _NOTE_KEYS),
                                ("lead", _NOTE_KEYS), ("percussion", _LANE_KEYS)):
        rows = phrase.get(collection, [])
        if not isinstance(rows, list) or len(rows) > 2048:
            _issue(issues, "error", f"/phrase/{collection}", "size",
                   "Resolved event collections must contain at most 2048 rows.")
            continue
        for index, row in enumerate(rows):
            row_path = f"/phrase/{collection}/{index}"
            _unknown_keys(row, allowed, row_path, issues)
            if collection == "percussion" and isinstance(row, dict):
                _validate_sample(row.get("sample"), f"{row_path}/sample", issues)
                variations = row.get("round_robin_samples", [])
                if not isinstance(variations, (list, tuple)) or len(variations) > 32:
                    _issue(issues, "error", f"{row_path}/round_robin_samples",
                           "size", "Round-robin sample groups may contain at most 32 takes.")
                else:
                    for variation, sample in enumerate(variations):
                        _validate_sample(
                            sample, f"{row_path}/round_robin_samples/{variation}", issues)
    for name in ("lead_sample", "pad_sample"):
        _validate_sample(phrase.get(name), f"/phrase/{name}", issues)
    for name in ("lead_instrument", "pad_instrument", "bass_instrument"):
        _validate_instrument(phrase.get(name), f"/phrase/{name}", issues)
    return issues


def _pointer_tokens(pointer: str) -> list[str]:
    if not pointer.startswith("/") or pointer == "/":
        raise ValueError(f"Invalid JSON Pointer '{pointer}'.")
    return [token.replace("~1", "/").replace("~0", "~")
            for token in pointer[1:].split("/")]


def pointer_get(document: object, pointer: str) -> object:
    value = document
    for token in _pointer_tokens(pointer):
        if isinstance(value, list):
            value = value[int(token)]
        elif isinstance(value, dict) and token in value:
            value = value[token]
        else:
            raise KeyError(pointer)
    return value


def pointer_set(document: object, pointer: str, new_value: object) -> None:
    tokens = _pointer_tokens(pointer)
    value = document
    for token in tokens[:-1]:
        if isinstance(value, list):
            value = value[int(token)]
        elif isinstance(value, dict) and token in value:
            value = value[token]
        else:
            raise KeyError(pointer)
    token = tokens[-1]
    if isinstance(value, list):
        value[int(token)] = copy.deepcopy(new_value)
    elif isinstance(value, dict) and token in value:
        value[token] = copy.deepcopy(new_value)
    else:
        raise KeyError(pointer)


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_controls(data: dict, issues: list[ValidationIssue]) -> None:
    for control in CONTROL_FIELDS:
        if control.kind in ("table", "text") or control.read_only:
            continue
        try:
            value = pointer_get(data, control.path)
        except (KeyError, IndexError, ValueError):
            continue  # Compatible defaults are applied by BedSpec.from_dict.
        valid_type = (
            isinstance(value, bool) if control.kind == "boolean" else
            isinstance(value, int) and not isinstance(value, bool)
            if control.kind == "integer" else
            _number(value) if control.kind == "number" else
            value in control.choices
        )
        if not valid_type:
            _issue(issues, "error", control.path, "type",
                   f"{control.label} has the wrong type or an unknown value.")
            continue
        if control.kind in ("number", "integer"):
            numeric = float(value)
            if control.minimum is not None and numeric < control.minimum or \
                    control.maximum is not None and numeric > control.maximum:
                _issue(issues, "error", control.path, "renderable_range",
                       f"{control.label} must be between {control.minimum} and {control.maximum}.")
                continue
            if ((control.safe_minimum is not None and numeric < control.safe_minimum) or
                    (control.safe_maximum is not None and numeric > control.safe_maximum)):
                safe = min(max(numeric, control.safe_minimum
                               if control.safe_minimum is not None else numeric),
                           control.safe_maximum
                           if control.safe_maximum is not None else numeric)
                if control.kind == "integer":
                    safe = int(round(safe))
                _issue(issues, "warning", control.path, "production_range",
                       f"{control.label} is renderable but outside the production-v1 range.",
                       safe_value=safe, has_safe_value=True)


def _validate_structure(spec: BedSpec, issues: list[ValidationIssue]) -> None:
    try:
        steps_per_bar = spec.steps_per_bar
        grid = Grid.from_spec(spec)
        if not math.isfinite(grid.bar) or grid.bar <= 0:
            raise ValueError("Bar duration must be positive and finite.")
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        _issue(issues, "error", "/beats_per_bar", "meter", str(exc))
        return
    if spec.beat_unit != 4:
        _issue(issues, "warning", "/beat_unit", "production_meter",
               "production-v1 uses quarter-note beats; this meter is experimental.")
    if spec.profile_version != "production-v1":
        _issue(issues, "warning", "/profile_version", "profile",
               f"Profile '{spec.profile_version}' is renderable but is not production-v1.")
    if spec.scale not in SCALES:
        _issue(issues, "error", "/scale", "enum", f"Unknown scale '{spec.scale}'.")
    if not isinstance(spec.progression, list) or not 1 <= len(spec.progression) <= 16 or \
            any(not isinstance(value, int) or isinstance(value, bool) or
                not -14 <= value <= 14 for value in spec.progression):
        _issue(issues, "error", "/progression", "progression",
               "Progression must contain 1–16 integer scale degrees between -14 and 14.")
    for path, pattern in (("/drums/kick", spec.drums.kick),
                          ("/drums/rim", spec.drums.rim)):
        if not isinstance(pattern, str) or not pattern or set(pattern) - {"x", "."}:
            _issue(issues, "error", path, "pattern",
                   "Patterns must be non-empty strings containing only 'x' and '.'.")
    for path, pair in (("/lead/register", spec.lead.register),
                       ("/lead/velocity", spec.lead.velocity)):
        if not isinstance(pair, tuple) or len(pair) != 2 or not all(_number(v) for v in pair):
            _issue(issues, "error", path, "pair", "Expected a two-value numeric range.")
        elif pair[0] > pair[1]:
            _issue(issues, "error", path, "range_order", "Range minimum exceeds maximum.")
    phrase = spec.phrase
    if phrase is None:
        _issue(issues, "warning", "/phrase", "legacy_phrase",
               "This legacy BedSpec has no fully resolved phrase provenance.")
        return
    if phrase.family not in STYLES:
        _issue(issues, "error", "/phrase/family", "family",
               f"Unknown resolved family '{phrase.family}'.")
    elif (spec.profile_version == "production-v1" and
          phrase.family not in get_profile("production-v1").families):
        _issue(issues, "warning", "/phrase/family", "production_family",
               f"Family '{phrase.family}' is outside the production-v1 family set.")
    if not isinstance(phrase.loop_bars, int) or not 1 <= phrase.loop_bars <= 32:
        _issue(issues, "error", "/phrase/loop_bars", "renderable_range",
               "Loop bars must be an integer between 1 and 32.")
        return
    phrase_steps = phrase.loop_bars * steps_per_bar
    if phrase.round_robin_strategy != "cyclic":
        _issue(issues, "error", "/phrase/round_robin_strategy", "enum",
               "Resolved sample variation strategy must be 'cyclic'.")
    for name, events in (("chords", phrase.chords), ("bass", phrase.bass),
                         ("lead", phrase.lead)):
        for index, event in enumerate(events):
            base = f"/phrase/{name}/{index}"
            if not isinstance(event.step, int) or not 0 <= event.step < phrase_steps:
                _issue(issues, "error", f"{base}/step", "event_step",
                       f"Step must be between 0 and {phrase_steps - 1}.")
            if not _number(event.duration_steps) or not 0.05 <= event.duration_steps <= phrase_steps:
                _issue(issues, "error", f"{base}/duration_steps", "event_duration",
                       f"Duration must be between 0.05 and {phrase_steps} steps.")
            if not _number(event.velocity) or not 0 <= event.velocity <= 1.5:
                _issue(issues, "error", f"{base}/velocity", "velocity",
                       "Velocity must be between 0 and 1.5.")
            if (not isinstance(event.sample_variation, int) or
                    isinstance(event.sample_variation, bool) or
                    event.sample_variation < 0):
                _issue(issues, "error", f"{base}/sample_variation",
                       "round_robin", "Sample variation must be a non-negative integer.")
            if not isinstance(event.articulation, str) or not event.articulation:
                _issue(issues, "error", f"{base}/articulation", "articulation",
                       "Articulation must be a non-empty label.")
            notes = event.midi_notes if name == "chords" else [event.midi_note]
            if not notes or any(not isinstance(note, int) or isinstance(note, bool) or
                                not 0 <= note <= 127 for note in notes):
                _issue(issues, "error", f"{base}/midi_notes" if name == "chords"
                       else f"{base}/midi_note", "midi", "MIDI notes must be integers from 0 to 127.")
    for index, lane in enumerate(phrase.percussion):
        base = f"/phrase/percussion/{index}"
        if not isinstance(lane.pattern, str) or len(lane.pattern) != phrase_steps or \
                set(lane.pattern) - {"x", "."}:
            _issue(issues, "error", f"{base}/pattern", "pattern_length",
                   f"Pattern must contain exactly {phrase_steps} 'x' or '.' steps.")
        for name, value, low, high in (
            ("level", lane.level, 0, 2), ("probability", lane.probability, 0, 1),
            ("humanize", lane.humanize, 0, 0.25), ("pan", lane.pan, -1, 1),
        ):
            if not _number(value) or not low <= value <= high:
                _issue(issues, "error", f"{base}/{name}", "renderable_range",
                       f"{name} must be between {low} and {high}.")


def _event_density(spec: BedSpec) -> float:
    if not spec.phrase:
        return 0.0
    phrase = spec.phrase
    hits = sum(lane.pattern.count("x") for lane in phrase.percussion)
    events = len(phrase.chords) + len(phrase.bass) + len(phrase.lead) + hits
    return events / max(phrase.loop_bars, 1)


def validate_bed_spec(data_or_spec: dict | BedSpec, *, analyze: bool = True,
                      profile_name: str = "production-v1") -> tuple[BedSpec | None,
                                                                    ExplorerValidationReport]:
    data = asdict(data_or_spec) if isinstance(data_or_spec, BedSpec) else copy.deepcopy(data_or_spec)
    issues = _validate_shape(data)
    if not isinstance(data, dict):
        return None, ExplorerValidationReport("invalid", tuple(issues), {})
    _validate_controls(data, issues)
    spec: BedSpec | None = None
    try:
        spec = BedSpec.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        _issue(issues, "error", "/", "parse", f"Could not parse BedSpec: {exc}")
    if spec is not None:
        _validate_structure(spec, issues)
    if any(issue.severity == "error" for issue in issues) or spec is None or not analyze:
        state: ValidationState = ("invalid" if any(issue.severity == "error" for issue in issues)
                                  else "experimental" if issues else "production-safe")
        return spec, ExplorerValidationReport(state, tuple(issues),
                                              {"density_per_bar": _event_density(spec)
                                               if spec else 0.0})

    measurements: dict[str, float] = {"density_per_bar": _event_density(spec)}
    fingerprint_dict: dict | None = None
    try:
        profile = get_profile(profile_name)
        bars = max(spec.phrase.loop_bars if spec.phrase else 4, 4)
        stems = render_stems(spec, bars)
        audio = sum(stems.values(), np.zeros_like(next(iter(stems.values()))))
        quality, fingerprint = evaluate_preview(audio, stems, spec, profile)
        measurements.update(quality.measurements)
        try:
            loudness = float(pyln.Meter(SR).integrated_loudness(audio))
            if math.isfinite(loudness):
                measurements["loudness_lufs"] = loudness
        except (ValueError, ZeroDivisionError):
            pass
        fingerprint_dict = asdict(fingerprint)
        existing = {(issue.path, issue.code) for issue in issues}
        reason_paths = {
            "swing": "/swing", "percussion": "/drums/level",
            "metrical": "/phrase/percussion", "quality": "/",
        }
        for reason in quality.rejection_reasons:
            path = next((value for word, value in reason_paths.items() if word in reason), "/")
            if (path, "quality") not in existing:
                _issue(issues, "warning", path, "quality", reason)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        _issue(issues, "error", "/", "render", str(exc))

    state = ("invalid" if any(issue.severity == "error" for issue in issues)
             else "experimental" if issues else "production-safe")
    return spec, ExplorerValidationReport(state, tuple(issues), measurements,
                                          fingerprint_dict)


def apply_safe_repairs(data_or_spec: dict | BedSpec,
                       *, profile_name: str = "production-v1") -> tuple[BedSpec | None,
                                                                        ExplorerValidationReport]:
    data = asdict(data_or_spec) if isinstance(data_or_spec, BedSpec) else copy.deepcopy(data_or_spec)
    _, report = validate_bed_spec(data, analyze=True, profile_name=profile_name)
    if report.state == "invalid":
        return None, report
    for issue in report.issues:
        if issue.has_safe_value:
            pointer_set(data, issue.path, issue.safe_value)
    return validate_bed_spec(data, analyze=True, profile_name=profile_name)


def validate_lock_paths(paths: list[str], base: dict) -> None:
    if len(paths) > MAX_LOCKS:
        raise ValueError(f"At most {MAX_LOCKS} lock paths are allowed.")
    for path in paths:
        if path not in LOCKABLE_PATHS:
            raise ValueError(f"Field '{path}' cannot be locked.")
        try:
            pointer_get(base, path)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"Locked field '{path}' is absent from the base BedSpec.") from exc


def randomize_unlocked(base: dict | BedSpec, locked_paths: list[str], *,
                       seed: int | None = None,
                       request: MusicRequest | None = None,
                       progress_callback: ProgressCallback | None = None,
                       cancel_check: CancelCheck | None = None) -> RandomizationResult:
    base_data = asdict(base) if isinstance(base, BedSpec) else copy.deepcopy(base)
    base_spec, base_report = validate_bed_spec(base_data, analyze=False)
    if base_spec is None or base_report.state == "invalid":
        raise ValueError("The base BedSpec must be structurally valid before randomization.")
    validate_lock_paths(locked_paths, base_data)
    chosen_seed = secrets.randbits(64) if seed is None else seed
    if not 0 <= chosen_seed < 2 ** 64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    family = base_spec.phrase.family if base_spec.phrase else "auto"
    profile = (base_spec.profile_version if base_spec.profile_version in PROFILES
               else "production-v1")
    if family != "auto" and family not in get_profile(profile).families:
        profile = "exploration-v1"
    if family != "auto" and family not in get_profile(profile).families:
        family = "auto"
    if request is None:
        has_natural = bool(base_spec.phrase and (
            base_spec.phrase.lead_sample or base_spec.phrase.pad_sample or
            base_spec.phrase.lead_instrument or base_spec.phrase.pad_instrument or
            base_spec.phrase.bass_instrument or
            any(lane.sample for lane in base_spec.phrase.percussion)))
        request = MusicRequest(family=family, palette="hybrid" if has_natural else "electronic",
                               seed=chosen_seed, profile=profile)
    else:
        request = MusicRequest(**{**request.to_dict(), "seed": chosen_seed})
    generated = resolve_music(request, progress_callback=progress_callback,
                              cancel_check=cancel_check)
    candidate = asdict(generated.bed_spec)
    for path in locked_paths:
        try:
            pointer_set(candidate, path, pointer_get(base_data, path))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"Locked field '{path}' is incompatible with the new phrase.") from exc
    spec, report = validate_bed_spec(candidate, analyze=True,
                                     profile_name=request.profile)
    if spec is None or report.state == "invalid":
        details = "; ".join(issue.message for issue in report.issues
                            if issue.severity == "error")
        raise ValueError(f"Locked values prevent a renderable result: {details}")
    return RandomizationResult(spec, report, chosen_seed)


def generation_metadata(result: MusicGenerationResult) -> dict:
    return result.to_dict()


class RenderBusyError(RuntimeError):
    pass


class RenderCancelledError(RuntimeError):
    pass


class ArtifactStore:
    """Checksum-addressed WAV cache constrained to one managed directory."""

    def __init__(self, config: ExplorerConfig | None = None):
        self.config = config or ExplorerConfig.from_environment()
        self.root = self.config.output_root.resolve()
        self._render_slots = threading.BoundedSemaphore(1)
        self._queue_slots = threading.BoundedSemaphore(self.config.max_pending_renders + 1)
        self._locks_guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _cache_key(spec: BedSpec, duration_seconds: float) -> str:
        payload = {
            "bed_spec": asdict(spec), "engine_version": ENGINE_VERSION,
            "sample_rate": SR, "duration_seconds": round(duration_seconds, 6),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _lock_for(self, key: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def path_for(self, artifact_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", artifact_id):
            raise FileNotFoundError("Unknown render artifact.")
        path = self.root / f"{artifact_id}.wav"
        if not path.is_file() or path.parent != self.root:
            raise FileNotFoundError("Unknown render artifact.")
        return path

    def write_spec(self, spec: BedSpec) -> Path:
        """Write a checksum-named BedSpec download beneath the managed root."""
        self._ensure_root()
        text = spec.to_json() + "\n"
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        path = self.root / f"lexibeat-{digest}.bed.json"
        temporary = self.root / f".{path.name}.partial"
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
        return path

    def render(self, spec: BedSpec, duration_seconds: float, *, preview: bool = False,
               progress: ProgressCallback | None = None,
               cancelled: CancelCheck | None = None) -> RenderArtifact:
        limit = PREVIEW_MAX_DURATION_SECONDS if preview else self.config.max_duration_seconds
        if not math.isfinite(duration_seconds) or not 0 < duration_seconds <= limit:
            raise ValueError(f"duration_seconds must be between 0 and {limit:g}")
        _, report = validate_bed_spec(spec, analyze=False)
        if report.state == "invalid":
            raise ValueError("Cannot render an invalid BedSpec: " + "; ".join(
                issue.message for issue in report.issues if issue.severity == "error"))
        key = self._cache_key(spec, duration_seconds)
        self._ensure_root()
        path = self.root / f"{key}.wav"
        if path.exists():
            path.touch()
            return self._artifact(path, duration_seconds, True)
        if not self._queue_slots.acquire(blocking=False):
            raise RenderBusyError("The render queue is full; try again shortly.")
        try:
            with self._lock_for(key):
                if path.exists():
                    path.touch()
                    return self._artifact(path, duration_seconds, True)
                if cancelled and cancelled():
                    raise RenderCancelledError("Render cancelled.")
                if progress:
                    progress(0.1, "Rendering music stems")
                with self._render_slots:
                    if cancelled and cancelled():
                        raise RenderCancelledError("Render cancelled.")
                    grid = Grid.from_spec(spec)
                    bars = max(1, round(max(duration_seconds - 1.0, grid.bar) / grid.bar))
                    try:
                        audio = render_bed(
                            spec, bars,
                            progress_callback=(
                                (lambda value, message: progress(0.1 + value * 0.78, message))
                                if progress else None),
                            cancel_check=cancelled,
                        )
                    except InterruptedError as exc:
                        raise RenderCancelledError("Render cancelled.") from exc
                if cancelled and cancelled():
                    raise RenderCancelledError("Render cancelled.")
                if not len(audio) or audio.ndim != 2 or audio.shape[1] != 2 or \
                        not np.isfinite(audio).all():
                    raise RuntimeError("Renderer produced invalid stereo audio.")
                if float(np.abs(audio).max()) > 0.97 + 1e-7:
                    raise RuntimeError("Renderer exceeded the 0.97 peak safety limit.")
                temporary = self.root / f".{key}.{os.getpid()}.{threading.get_ident()}.partial.wav"
                try:
                    sf.write(temporary, audio, SR, subtype="PCM_16")
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
                if progress:
                    progress(1.0, "Render ready")
                self._trim_cache(keep=path)
                return self._artifact(path, len(audio) / SR, False)
        finally:
            self._queue_slots.release()

    def _artifact(self, path: Path, duration_seconds: float, cache_hit: bool) -> RenderArtifact:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifact_id = path.stem
        return RenderArtifact(artifact_id, f"/api/audio/{artifact_id}",
                              float(duration_seconds), SR, digest, cache_hit)

    def _trim_cache(self, *, keep: Path) -> None:
        files = sorted((path for path in self.root.glob("*.wav") if path != keep),
                       key=lambda path: path.stat().st_mtime)
        total = sum(path.stat().st_size for path in self.root.glob("*.wav"))
        for path in files:
            if total <= self.config.max_cache_bytes:
                break
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            total -= size


def preview_duration(spec: BedSpec) -> float:
    grid = Grid.from_spec(spec)
    bars = spec.phrase.loop_bars if spec.phrase else 4
    return min(max(grid.bar, bars * grid.bar), PREVIEW_MAX_DURATION_SECONDS)


def midi_name(note: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[note % 12]}{note // 12 - 1}"


def step_position(spec: BedSpec, step: int) -> str:
    bar, within = divmod(step, spec.steps_per_bar)
    steps_per_beat = 16 // spec.beat_unit
    beat, subdivision = divmod(within, steps_per_beat)
    return f"bar {bar + 1}, beat {beat + 1}, sixteenth {subdivision + 1}"


def logical_id(ref: SampleRef) -> str:
    return f"{ref.collection}:{ref.asset_id}"


def parse_logical_id(value: str) -> SampleRef:
    parts = value.split(":", 1)
    if len(parts) != 2 or not all(_LOGICAL_PART.fullmatch(part) for part in parts):
        raise ValueError("Sample logical ID must be 'collection:asset_id'.")
    return SampleRef(parts[0], parts[1])


class SampleService:
    def __init__(self, config: ExplorerConfig | None = None,
                 library: SampleLibrary | None = None):
        self.config = config or ExplorerConfig.from_environment()
        self.library = library or SampleLibrary()

    def _metadata(self, asset: SampleAsset) -> dict:
        try:
            path = self.library.resolve(asset.ref).resolve()
            if path.is_relative_to(BUNDLED_ROOT.resolve()):
                availability = "bundled"
            elif path.is_relative_to(self.library.local.resolve()):
                availability = "local"
            else:
                availability = "external"
        except FileNotFoundError:
            availability = "unavailable"
        source = COLLECTIONS.get(asset.collection)
        metadata = asdict(asset)
        metadata["articulation"] = (asset.articulation or
                                    infer_articulation(asset.relative_path))
        metadata["round_robin"] = (asset.round_robin
                                    if asset.round_robin is not None else
                                    infer_round_robin(asset.relative_path))
        return {
            **metadata, "logical_id": logical_id(asset.ref),
            "availability": availability,
            "promoted": self.library.is_promoted(asset),
            "source_name": source.name if source else asset.collection,
            "source_license": asset.license,
            "attribution": source.attribution if source else "",
        }

    def list(self, *, category: str | None = None, collection: str | None = None,
             availability: str | None = None, offset: int = 0,
             limit: int = 100) -> dict:
        if not 0 <= offset <= 1_000_000 or not 1 <= limit <= 200:
            raise ValueError("offset must be non-negative and limit must be 1–200.")
        if collection is not None and collection not in COLLECTIONS:
            raise ValueError(f"Unknown collection '{collection}'.")
        if category is not None and category not in {"pitched", "percussion", "loop", "texture"}:
            raise ValueError(f"Unknown category '{category}'.")
        assets = self.library.assets(category=category,
                                     collections=(collection,) if collection else None)
        if availability:
            rows = [self._metadata(asset) for asset in assets]
            rows = [row for row in rows if row["availability"] == availability]
            selected = rows[offset:offset + limit]
            total = len(rows)
        else:
            selected = [self._metadata(asset)
                        for asset in assets[offset:offset + limit]]
            total = len(assets)
        return {"items": selected, "total": total,
                "offset": offset, "limit": limit}

    def get(self, value: str) -> dict:
        ref = parse_logical_id(value)
        return self._metadata(self.library.asset(ref))

    def promote(self, value: str) -> dict:
        if self.config.hosted:
            raise PermissionError("Sample promotion is disabled on hosted deployments.")
        ref = parse_logical_id(value)
        asset = self.library.asset(ref)
        path = self.library.promote([asset.ref])[0]
        return {**self._metadata(asset), "availability": "local",
                "promoted": True, "managed_name": path.name}
