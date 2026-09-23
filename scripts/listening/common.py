"""What the listening tools share: the production bed, its facts, and a clip of it.

Everything here goes through the **service path** — `resolve_request` with the production profile
and the hybrid palette, and `build_bed` for a rebuilt candidate — because the point of a listening
round is to judge exactly what a host gets. A round that listened to something else would tune the
wrong thing.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from lexibeat.api import MusicRequest
from lexibeat.bedspec import SCALES, BedSpec
from lexibeat.generator import (_safe_inventory, build_bed, render_resolved,
                                resolve_request)
from lexibeat.library import SampleLibrary
from lexibeat.loop import write_mp3
from lexibeat.mix import limit, loudness_normalize
from lexibeat.music import SR
from lexibeat.profiles import ListenerPolicy, get_profile

PROFILE = "production-v1"
PALETTE = "hybrid"
# Every clip at one loudness, so "too loud" and "too quiet" are about the parts inside a clip and
# never about which clip happened to be mastered hotter.
CLIP_LUFS = -20.0
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
NUMERALS = ("I", "II", "III", "IV", "V", "VI", "VII")
# The parts a listener judges, in the order the page shows them.
PARTS = ("lead", "pad", "bass", "percussion", "harmony", "feel")


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


class Production:
    """The production catalogue, opened the way the service opens it — and never a local one.

    A local catalogue on this machine (anything under `LEXIBEAT_CACHE`) would otherwise decide what
    the round sounds like, which is how a laptop once listened to beds with no catalogue samples.
    """

    def __init__(self) -> None:
        self._scratch = tempfile.TemporaryDirectory(prefix="lexibeat-listening-")
        root = Path(self._scratch.name)
        self.library = SampleLibrary(root / "external", root / "local", use_bundled=True)
        if not self.library.uses_bundled_catalog:
            raise SystemExit("The production sample bundle is not readable here; "
                             "run `git lfs pull` or set LEXIBEAT_BUNDLE_ROOT.")
        self.policy = self.library.expansion_policy()
        self.accepted = {bank["name"] for bank in self.policy.get("accepted_banks", [])}
        self.assets, self.instruments, _ = _safe_inventory(self.library)
        self.paths = {asset.asset_id: f"{asset.collection}: {asset.relative_path}"
                      for asset in self.assets}
        self.profile = get_profile(PROFILE)

    def resolve(self, seed: int, family: str = "auto",
                listener: ListenerPolicy | None = None):
        request = MusicRequest(family=family, seed=seed, palette=PALETTE, profile=PROFILE)
        return resolve_request(request, library=self.library, listener=listener)

    def rebuild(self, spec: BedSpec, listener: ListenerPolicy) -> BedSpec:
        """The same winning candidate, built again with a different policy."""
        request = MusicRequest(family=spec.phrase.family, seed=spec.seed, palette=PALETTE,
                               profile=PROFILE)
        return build_bed(spec.phrase.family, spec.seed, request, self.profile, self.assets,
                         self.instruments, expansion_policy=self.policy, listener=listener)

    def rebuild_unrendered(self, family: str, bed_seed: int) -> BedSpec:
        """One candidate as production would build it, without rendering or scoring it."""
        request = MusicRequest(family=family, seed=bed_seed, palette=PALETTE, profile=PROFILE)
        return build_bed(family, bed_seed, request, self.profile, self.assets,
                         self.instruments, expansion_policy=self.policy)

    def defects(self, spec: BedSpec) -> dict[str, bool]:
        """Which of the three known defects this bed shows; see `ListenerPolicy`."""
        lead = spec.phrase.lead_instrument
        return {
            "unapproved_lead": bool(lead and lead.name not in self.accepted),
            "high_lead": max((event.midi_note for event in spec.phrase.lead), default=0) > 88,
            "off_key_fifth": any(off_key_fifth(spec, degree) for degree in spec.progression),
        }


def off_key_fifth(spec: BedSpec, degree: int) -> bool:
    in_scale = {(spec.root + step) % 12 for step in spec.scale_steps()}
    return (spec.chord_root(degree) + 7) % 12 not in in_scale


def numeral(spec: BedSpec, degree: int) -> str:
    steps = SCALES[spec.scale]
    n = len(steps)
    third = (steps[(degree + 2) % n] - steps[degree % n]) % 12
    fifth = (steps[(degree + 4) % n] - steps[degree % n]) % 12
    name = NUMERALS[degree % n]
    name = name if third == 4 else name.lower()
    return name + ("°" if fifth == 6 else "")


def _instrument(ref) -> str:
    name = ref.name.split(":", 1)[-1]
    return name.split("#", 1)[0].replace("/", " › ")


def facts(production: Production, spec: BedSpec) -> dict[str, Any]:
    """What the engine chose for each part, in words the page can show beside the rating."""
    phrase = spec.phrase
    lead_notes = [event.midi_note for event in phrase.lead]
    lead = (_instrument(phrase.lead_instrument) if phrase.lead_instrument
            else f"{spec.lead.instrument} (pack)" if spec.lead.instrument != "synth"
            else "synth bell")
    if phrase.lead_instrument and phrase.lead_instrument.name not in production.accepted:
        lead += " — not listener-approved"
    pad = (_instrument(phrase.pad_instrument) if phrase.pad_instrument
           else "violin section (pack)" if spec.pad.instrument == "strings"
           else f"synth {phrase.pad_timbre}")
    bass = (_instrument(phrase.bass_instrument) if phrase.bass_instrument
            else f"synth {phrase.bass_timbre}")
    lanes = []
    for lane in phrase.percussion:
        source = (production.paths.get(lane.sample.asset_id, lane.sample.asset_id)
                  if lane.sample else lane.sound.replace("synth:", "synth "))
        lanes.append(f"{lane.role or 'lane'}: {source}")
    progression = [numeral(spec, degree) + (" ♯5" if off_key_fifth(spec, degree) else "")
                   for degree in spec.progression]
    return {
        "lead": {"source": lead, "register": (
            f"{note_name(min(lead_notes))}–{note_name(max(lead_notes))}" if lead_notes else "silent"),
            "top": max(lead_notes, default=None)},
        "pad": {"source": pad, "texture": phrase.harmony_texture,
                "extension": spec.chord_extension},
        "bass": {"source": bass, "grammar": phrase.bass_grammar},
        "percussion": {"lanes": lanes},
        "harmony": {"key": f"{NOTE_NAMES[spec.root % 12]} {spec.scale.replace('_', ' ')}",
                    "progression": progression},
        "feel": {"family": phrase.family, "bpm": round(spec.bpm),
                 "meter": f"{spec.beats_per_bar}/{spec.beat_unit}",
                 "swing": round(spec.swing, 3)},
    }


def render_clip(spec: BedSpec, path: Path, seconds: float = 30.0) -> None:
    """The bed alone, about `seconds` long, at one loudness for every clip."""
    audio = render_resolved(spec, duration_seconds=seconds)
    stereo = audio if audio.ndim == 2 else np.stack([audio, audio], axis=1)
    write_mp3(path, limit(loudness_normalize(stereo, CLIP_LUFS, SR), sr=SR), SR)


def write_round(out: Path, kind: str, clips: list[dict], key: dict | None = None) -> None:
    """`round.json` is what the page may see; `key.json`, if any, is what it must not."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "round.json").write_text(json.dumps({"kind": kind, "clips": clips}, indent=2,
                                               ensure_ascii=False), encoding="utf-8")
    if key is not None:
        (out / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")


def bed_json(spec: BedSpec) -> dict:
    return asdict(spec)
