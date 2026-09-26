"""Reusable audio/video machinery for the README demonstration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from .arrange import SOURCE, TARGET, Event
from .formats import Format, FormatError, renderable
from .programme import plan
from .bedspec import STYLES, BedSpec
from .language import ENGLISH, SPANISH, Language
from .loop import build_timeline
from .music import SR, Grid
from .vocab import Item
from .voice import Delivery, Speaker

DEMO_WIDTH = 1280
DEMO_HEIGHT = 720
DEMO_FPS = 24
DEMO_AUDIO_BITRATE = 96_000
DEMO_VIDEO_CRF = 24


@dataclass(frozen=True)
class DemoVariant:
    name: str
    style: str
    seed: int


@dataclass(frozen=True)
class DemoConfig:
    title: str
    format: Format
    bpm: float
    beats_per_bar: int
    beat_unit: int
    items: tuple[Item, ...]
    bars_per_utterance: tuple[int, ...]
    variants: tuple[DemoVariant, ...]
    source_language: Language = SPANISH
    target_language: Language = ENGLISH


def load_demo_config(path: Path) -> DemoConfig:
    """Load and validate the portable demo manifest."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Demo manifest schema_version must be 1.")
    title = str(data.get("title") or "").strip()
    if not title:
        raise ValueError("Demo manifest title cannot be empty.")
    try:
        fmt = renderable(str(data.get("format") or ""))
    except FormatError as exc:
        raise ValueError(f"Demo format: {exc}") from exc
    bpm = float(data.get("bpm") or 0)
    if bpm <= 0:
        raise ValueError("Demo manifest bpm must be positive.")
    try:
        beats_text, unit_text = str(data.get("meter") or "").split("/", 1)
        beats_per_bar, beat_unit = int(beats_text), int(unit_text)
    except (TypeError, ValueError) as exc:
        raise ValueError("Demo manifest meter must look like '4/4'.") from exc
    if beats_per_bar <= 0 or beat_unit <= 0:
        raise ValueError("Demo manifest meter values must be positive.")
    raw_items = data.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("Demo manifest needs at least one vocabulary item.")
    items: list[Item] = []
    bars_per_utterance: list[int] = []
    for index, row in enumerate(raw_items, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Demo item {index} must be an object.")
        item = Item(str(row.get("source") or ""), str(row.get("target") or ""),
                    str(row.get("direction") or ""))
        if not item:
            raise ValueError(f"Demo item {index} needs source and target text.")
        items.append(item)
        span = int(row.get("bars_per_utterance", 1))
        if span not in (1, 2):
            raise ValueError(
                f"Demo item {index} bars_per_utterance must be 1 or 2.")
        bars_per_utterance.append(span)
    raw_variants = data.get("variants")
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("Demo manifest needs at least one music variant.")
    variants: list[DemoVariant] = []
    names: set[str] = set()
    for index, row in enumerate(raw_variants, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Demo variant {index} must be an object.")
        variant = DemoVariant(str(row.get("name") or "").strip(),
                              str(row.get("style") or "").strip(),
                              int(row.get("seed")))
        if not variant.name or variant.name in names:
            raise ValueError("Demo variant names must be non-empty and unique.")
        if variant.style not in STYLES:
            raise ValueError(f"Unknown bed style '{variant.style}'.")
        names.add(variant.name)
        variants.append(variant)
    return DemoConfig(title, fmt, bpm, beats_per_bar, beat_unit,
                      tuple(items), tuple(bars_per_utterance), tuple(variants),
                      Language.from_value(data.get("source_language") or "es"),
                      Language.from_value(data.get("target_language") or "en"))


def resolve_demo_specs(config: DemoConfig) -> dict[str, BedSpec]:
    """Resolve deterministic beds and require one shared speech grid."""
    specs: dict[str, BedSpec] = {}
    for variant in config.variants:
        spec = BedSpec.from_style(variant.style, variant.seed)
        if (spec.beats_per_bar, spec.beat_unit) != (
                config.beats_per_bar, config.beat_unit):
            raise ValueError(
                f"Demo bed '{variant.name}' resolves to "
                f"{spec.beats_per_bar}/{spec.beat_unit}; choose a seed that "
                f"naturally resolves to {config.beats_per_bar}/{config.beat_unit}.")
        # Phrase events are step-based, so tempo can be safely unified after the
        # meter-compatible phrase has been completely resolved.
        spec.bpm = config.bpm
        specs[variant.name] = spec
    grids = {
        (spec.bpm, spec.beats_per_bar, spec.beat_unit) for spec in specs.values()
    }
    if len(grids) != 1:
        detail = ", ".join(
            f"{name}={spec.bpm:g} BPM {spec.beats_per_bar}/{spec.beat_unit}"
            for name, spec in specs.items())
        raise ValueError(f"Demo beds must share one timing grid ({detail}).")
    return specs


def cache_key(speaker: Speaker, text: str, language: Language, delivery: Delivery,
              target_seconds: float | None, slot_seconds: float | None = None) -> str:
    """Return a stable key for one fully directed, post-fit utterance."""
    backend = speaker.backend
    backend_name = getattr(backend, "name", type(backend).__name__)
    if backend_name == "gemini" and getattr(backend, "vertex", False):
        backend_name = "gemini-vertex"
    payload = {
        "schema_version": 1,
        "backend": backend_name,
        "model": getattr(backend, "model_id", ""),
        "voices": getattr(backend, "voices", None),
        "voice_seed": speaker.voice_seed,
        "text": text,
        "lang": language.code,
        "prosody": asdict(delivery.prosody),
        "take": delivery.take,
        "direction": delivery.direction,
        "target_seconds": target_seconds,
        "sample_rate": SR,
    }
    # A take fitted to a slot has a faded tail, so it is a different take; one fitted without a slot
    # keeps the key it always had.
    if slot_seconds is not None:
        payload["slot_seconds"] = slot_seconds
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PersistentSpeaker:
    """Speaker adapter that commits every completed take to disk atomically."""

    def __init__(self, speaker: Speaker, cache_dir: Path, *,
                 refresh: bool = False, max_fit_ratio: float = 1.35) -> None:
        self.speaker = speaker
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.refresh = refresh
        self.max_fit_ratio = max_fit_ratio
        self.prosody_strength = speaker.prosody_strength
        self.backend = speaker.backend

    @property
    def stats(self) -> list[dict[str, Any]]:
        return self.speaker.stats

    def say(self, text: str, language: Language, delivery: Delivery,
            target_seconds: float | None = None, *, slot_seconds: float | None = None,
            retry: bool = False) -> np.ndarray:
        key = cache_key(self.speaker, text, language, delivery, target_seconds, slot_seconds)
        wav_path = self.cache_dir / f"{key}.wav"
        metadata_path = self.cache_dir / f"{key}.json"
        if not self.refresh and not retry and wav_path.is_file() and metadata_path.is_file():
            audio, rate = sf.read(wav_path, dtype="float32")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if rate != SR or audio.ndim != 1 or not len(audio) or \
                    not np.isfinite(audio).all():
                raise RuntimeError(f"Cached speech take is invalid: {wav_path}")
            expected_provider = None
            if getattr(self.backend, "name", None) == "gemini":
                expected_provider = ("vertex-ai" if getattr(self.backend, "vertex", False)
                                     else "gemini-api")
            cached_provider = metadata.get("controls", {}).get("provider")
            if expected_provider is None or cached_provider == expected_provider:
                metadata["cache_hit"] = True
                self.speaker.stats.append(metadata)
                # Keep positional seeds stable when a partially cached run resumes.
                self.speaker._call_index += 1
                return audio

        before = len(self.speaker.stats)
        audio = self.speaker.say(text, language, delivery, target_seconds,
                                 slot_seconds=slot_seconds, retry=retry)
        if len(self.speaker.stats) <= before:
            raise RuntimeError("Speech backend did not record take metadata.")
        metadata = dict(self.speaker.stats[-1])
        if not len(audio) or not np.isfinite(audio).all():
            raise RuntimeError(f"Speech backend produced invalid audio for '{text}'.")
        target = metadata.get("target_seconds")
        original = metadata.get("duration_before_fit")
        if target and original and float(original) / float(target) > self.max_fit_ratio:
            raise RuntimeError(
                f"Speech take for '{text}' is {float(original):.2f}s, beyond the "
                f"safe {float(target):.2f}s × {self.max_fit_ratio:g} fit range.")
        metadata["cache_hit"] = False
        temporary_wav = wav_path.with_suffix(f".{os.getpid()}.partial.wav")
        temporary_json = metadata_path.with_suffix(f".{os.getpid()}.partial.json")
        try:
            sf.write(temporary_wav, audio, SR, subtype="PCM_16",
                     format="WAV")
            temporary_json.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")
            os.replace(temporary_wav, wav_path)
            os.replace(temporary_json, metadata_path)
        finally:
            temporary_wav.unlink(missing_ok=True)
            temporary_json.unlink(missing_ok=True)
        return audio

    def close(self) -> None:
        self.speaker.close()


def arrange_demo(config: DemoConfig, speaker: PersistentSpeaker, grid: Grid, *,
                 intro_bars: int = 2,
                 outro_bars: int = 2) -> tuple[list[Event], int]:
    """Arrange the demo's format with optional longer per-item speech slots."""
    events: list[Event] = []
    bar = intro_bars
    for index, (item, span) in enumerate(
            zip(config.items, config.bars_per_utterance), 1):
        print(f"  [{index}/{len(config.items)}] {item.source} — {item.target}  "
              f"({item.direction or 'plain'}, "
              f"{span} bar{'s' if span != 1 else ''}/utterance)", flush=True)
        for segment in plan(config.format, [item], source_language=config.source_language,
                            target_language=config.target_language).segments:
            if not segment.spoken:
                bar += segment.bars
                continue
            # The plan was made for this one word, so its item index is always 0.
            segment = replace(segment, item=index - 1)
            delivery = Delivery.for_take(segment.take, item.direction,
                                         strength=speaker.prosody_strength)
            audio = speaker.say(
                segment.text, segment.language, delivery,
                target_seconds=grid.bar * span * 0.92)
            events.append(Event(grid.bar_start(bar), audio,
                                f"{segment.side}:{segment.text}", segment))
            bar += span
    return events, bar + outro_bars


def write_tracklist(path: Path, variant: str, config: DemoConfig,
                    timeline: Sequence[dict[str, Any]], spec: BedSpec) -> None:
    lines = [
        f"{variant} — {len(config.items)} items, {spec.bpm:g} BPM, "
        f"format '{config.format.id}'",
        "",
    ]
    for row in timeline:
        at = float(row["start"])
        lines.append(f"{int(at)//60:02d}:{int(at)%60:02d}  "
                     f"{row['source']} — {row['target']}"
                     f"{' (' + row['direction'] + ')' if row['direction'] else ''}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _font_candidates() -> tuple[Path, ...]:
    return (
        Path("/System/Library/Fonts/Avenir Next Condensed.ttc"),
        Path("/System/Library/Fonts/Avenir.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )


def resolve_font(path: Path | None = None) -> Path | None:
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(f"Font not found: {path}")
        return path
    return next((candidate for candidate in _font_candidates()
                 if candidate.is_file()), None)


def _load_font(path: Path | None, size: int):
    from PIL import ImageFont

    if path:
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def _fit_font(draw: Any, text: str, path: Path | None, maximum: int,
              minimum: int, width: int):
    for size in range(maximum, minimum - 1, -2):
        font = _load_font(path, size)
        box = draw.textbbox((0, 0), text, font=font)
        if box[2] - box[0] <= width:
            return font
    return _load_font(path, minimum)


def _centered_text(draw: Any, text: str, y: float, font: Any,
                   fill: tuple[int, ...], *, width: int = DEMO_WIDTH) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((width - (box[2] - box[0])) / 2, y), text,
              font=font, fill=fill)


def _rounded_rectangle(draw: Any, box: tuple[int, int, int, int], radius: int,
                       fill: tuple[int, ...], outline: tuple[int, ...] | None = None,
                       width: int = 1) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline,
                           width=width)


def _argentina_flag(draw: Any, x: int, y: int, w: int, h: int) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=(255, 255, 255))
    draw.rectangle((x, y, x + w, y + h // 3), fill=(116, 172, 223))
    draw.rectangle((x, y + h * 2 // 3, x + w, y + h), fill=(116, 172, 223))
    r = max(3, h // 9)
    draw.ellipse((x + w // 2 - r, y + h // 2 - r,
                  x + w // 2 + r, y + h // 2 + r), fill=(246, 183, 54))


def _spain_flag(draw: Any, x: int, y: int, w: int, h: int) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=(198, 11, 30))
    draw.rectangle((x, y + h // 4, x + w, y + h - h // 4), fill=(255, 196, 0))


def _language_badge(draw: Any, x: int, y: int, w: int, h: int, code: str,
                    font_path: Path | None) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=(86, 104, 140))
    font = _load_font(font_path, max(12, h // 2))
    label = code.split("-")[0].upper()[:3]
    box = draw.textbbox((0, 0), label, font=font)
    draw.text((x + (w - (box[2] - box[0])) / 2, y + (h - (box[3] - box[1])) / 2 - box[1]),
              label, font=font, fill=(240, 244, 255))


def _flag(draw: Any, code: str, x: int, y: int, w: int, h: int,
          font_path: Path | None) -> None:
    """A flag for a language as the voices here speak it: Spain's for Spanish (the voices are from
    mainland Spain), Britain's for English, and a badge with the code for anything else."""
    base = code.split("-")[0].lower()
    if base == "es":
        _spain_flag(draw, x, y, w, h)
    elif base == "en":
        _british_flag(draw, x, y, w, h)
    else:
        _language_badge(draw, x, y, w, h, code, font_path)


def _british_flag(draw: Any, x: int, y: int, w: int, h: int) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=(35, 55, 116))
    thick = max(4, h // 7)
    draw.line((x, y, x + w, y + h), fill=(255, 255, 255), width=thick)
    draw.line((x + w, y, x, y + h), fill=(255, 255, 255), width=thick)
    draw.rectangle((x, y + h // 2 - thick, x + w, y + h // 2 + thick),
                   fill=(255, 255, 255))
    draw.rectangle((x + w // 2 - thick, y, x + w // 2 + thick, y + h),
                   fill=(255, 255, 255))
    red = max(3, thick // 2)
    draw.rectangle((x, y + h // 2 - red, x + w, y + h // 2 + red),
                   fill=(201, 43, 59))
    draw.rectangle((x + w // 2 - red, y, x + w // 2 + red, y + h),
                   fill=(201, 43, 59))


def _background() -> np.ndarray:
    y, x = np.mgrid[0:DEMO_HEIGHT, 0:DEMO_WIDTH]
    horizontal = x / max(DEMO_WIDTH - 1, 1)
    vertical = y / max(DEMO_HEIGHT - 1, 1)
    glow = np.exp(-(((horizontal - 0.82) / 0.42) ** 2 +
                    ((vertical - 0.18) / 0.58) ** 2))
    base = np.empty((DEMO_HEIGHT, DEMO_WIDTH, 3), dtype=np.float32)
    base[..., 0] = 13 + 12 * horizontal + 4 * glow
    base[..., 1] = 17 + 22 * horizontal + 31 * glow
    base[..., 2] = 38 + 29 * horizontal + 27 * glow
    return np.clip(base, 0, 255).astype(np.uint8)


def _active_item(timeline: Sequence[dict[str, Any]], at: float) -> dict[str, Any] | None:
    return next((row for row in timeline
                 if float(row["start"]) <= at < float(row["end"])), None)


def demo_timeline(items: Sequence[Item], events: Sequence[Event], grid: Grid,
                  total_bars: int) -> list[dict[str, Any]]:
    """One row per word, each with the lines it is heard in: the shape the video frames draw from.

    The loop's own result keeps words and lines apart (`items` and `cues`); a frame wants to know
    which side of the card is speaking, so the demo puts them back together here.
    """
    rows, cues = build_timeline(items, events, grid, total_bars)
    for row in rows:
        row["utterances"] = [{"role": cue["side"], "start": cue["start"], "end": cue["end"]}
                             for cue in cues if cue["item"] == row["index"] and cue["side"]]
    return rows


def _active_role(row: dict[str, Any], at: float) -> str | None:
    active = next((utterance for utterance in row["utterances"]
                   if float(utterance["start"]) <= at < float(utterance["end"])), None)
    return str(active["role"]) if active else None


def frame_bytes(title: str, timeline: Sequence[dict[str, Any]], duration: float,
                grid: Grid, frame_index: int, *, font_path: Path | None = None,
                background: np.ndarray | None = None) -> bytes:
    """Render one deterministic RGB frame."""
    from PIL import Image, ImageDraw

    at = frame_index / DEMO_FPS
    image = Image.fromarray((background if background is not None else _background()).copy())
    draw = ImageDraw.Draw(image, "RGBA")
    body_font_path = resolve_font(font_path)
    small = _load_font(body_font_path, 24)
    medium = _load_font(body_font_path, 31)
    _chrome(draw, title, "SPANISH  /  ENGLISH", at, grid, body_font_path)

    row = _active_item(timeline, at)
    if row is None:
        headline = _load_font(body_font_path, 72)
        _centered_text(draw, "Vocabulary, set to a beat.", 252, headline,
                       (244, 246, 255, 245))
        _centered_text(draw, "Expressive Gemini voices · deterministic procedural music",
                       355, medium, (174, 204, 216, 220))
    else:
        active = _active_role(row, at)
        source_visible = at >= float(row["source_reveal"])
        target_visible = at >= float(row["target_reveal"])
        card = (105, 145, DEMO_WIDTH - 105, 592)
        _rounded_rectangle(draw, card, 34, (14, 21, 47, 206),
                           (126, 155, 186, 50), 2)

        source_alpha = 255 if active == SOURCE else 222
        target_alpha = 255 if active == TARGET else 218
        if active == SOURCE:
            _rounded_rectangle(draw, (132, 178, DEMO_WIDTH - 132, 337), 25,
                               (45, 94, 124, 125), (102, 220, 194, 125), 2)
        if active == TARGET:
            _rounded_rectangle(draw, (132, 378, DEMO_WIDTH - 132, 537), 25,
                               (72, 61, 118, 125), (170, 139, 242, 125), 2)
        if source_visible:
            _argentina_flag(draw, 164, 205, 67, 44)
            source_font = _fit_font(draw, str(row["source"]), body_font_path,
                                    70, 42, 810)
            _centered_text(draw, str(row["source"]), 218, source_font,
                           (247, 250, 255, source_alpha))
        if target_visible:
            _british_flag(draw, 164, 405, 67, 44)
            target_font = _fit_font(draw, str(row["target"]), body_font_path,
                                    58, 38, 810)
            _centered_text(draw, str(row["target"]), 420, target_font,
                           (228, 235, 255, target_alpha))
            draw.line((202, 365, DEMO_WIDTH - 202, 365),
                      fill=(154, 180, 207, 55), width=2)

        index = int(row["index"]) + 1
        label = f"{index:02d}  /  {len(timeline):02d}"
        if row.get("direction"):
            label += f"    ·    {str(row['direction']).upper()}"
        _centered_text(draw, label, 619, small, (155, 187, 204, 190))

    _progress(draw, at, duration)
    return image.tobytes()


def _chrome(draw: Any, title: str, languages: str, at: float, grid: Grid,
            font_path: Path | None) -> None:
    """What every frame carries: the title, the language pair, and a pulse on each downbeat."""
    small = _load_font(font_path, 24)
    brand = _load_font(font_path, 30)
    draw.text((56, 39), title, font=brand, fill=(238, 242, 255, 235))
    box = draw.textbbox((0, 0), languages, font=small)
    draw.text((DEMO_WIDTH - 56 - (box[2] - box[0]), 45), languages, font=small,
              fill=(185, 205, 225, 195))
    beat_phase = (at % grid.bar) / grid.bar
    pulse = max(0.0, 1.0 - beat_phase * 5.5)
    radius = int(5 + pulse * 7)
    draw.ellipse((DEMO_WIDTH // 2 - radius, 72 - radius,
                  DEMO_WIDTH // 2 + radius, 72 + radius),
                 fill=(77, 222, 181, int(90 + 120 * pulse)))


def _progress(draw: Any, at: float, duration: float) -> None:
    progress = min(max(at / max(duration, 0.001), 0.0), 1.0)
    draw.rounded_rectangle((56, 678, DEMO_WIDTH - 56, 685), radius=4,
                           fill=(104, 126, 156, 70))
    draw.rounded_rectangle((56, 678, 56 + int((DEMO_WIDTH - 112) * progress), 685),
                           radius=4, fill=(77, 222, 181, 205))


# -- a finished loop's own cues ------------------------------------------------------------------

LANGUAGE_NAMES = {"es": "SPANISH", "en": "ENGLISH", "ru": "RUSSIAN", "fr": "FRENCH",
                  "de": "GERMAN", "it": "ITALIAN", "pt": "PORTUGUESE", "zh": "CHINESE"}
# What a group is, under the card, when it is not simply a word's own lines.
KIND_LABELS = {"example": "EXAMPLE", "remark": "REMARK", "story": "STORY", "header": "TOPIC",
               "callback": "CALLBACK", "pronounce": "SAY IT SLOWLY", "intro": "", "outro": ""}
SECTION_LABELS = {"quiz": "QUIZ", "review": "REVIEW"}
WORD_KINDS = ("say", "pronounce", "cue")
CARD = (105, 145, DEMO_WIDTH - 105, 592)


def cue_groups(cues: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The loop's cues as the cards a viewer sees: one per `group`, each with its distinct lines.

    A line said several times in a drill is one row, with every time it is heard. A card is shown
    from its first line's start until the next card's first line starts.
    """
    groups: dict[int, dict[str, Any]] = {}
    for cue in cues:
        group = groups.setdefault(int(cue["group"]), {
            "start": float(cue["start"]), "kind": cue["kind"], "section": cue["section"],
            "item": cue["item"], "rows": {}})
        row = group["rows"].setdefault((cue["text"], cue["language"]), {
            "text": cue["text"], "language": cue["language"], "kind": cue["kind"],
            "side": cue["side"], "heard": []})
        row["heard"].append((float(cue["start"]), float(cue["end"])))
    ordered = [groups[key] for key in sorted(groups)]
    for index, group in enumerate(ordered):
        group["end"] = ordered[index + 1]["start"] if index + 1 < len(ordered) else math.inf
        group["rows"] = list(group["rows"].values())
    return ordered


def _wrap(draw: Any, text: str, font_path: Path | None, maximum: int, minimum: int,
          width: int, lines: int) -> tuple[Any, list[str]]:
    """The largest font from `maximum` down at which `text` wraps into at most `lines` lines."""
    words = text.split()
    for size in range(maximum, minimum - 1, -2):
        font = _load_font(font_path, size)
        wrapped: list[str] = []
        current = ""
        for word in words:
            trial = f"{current} {word}".strip()
            if draw.textbbox((0, 0), trial, font=font)[2] <= width or not current:
                current = trial
            else:
                wrapped.append(current)
                current = word
        wrapped.append(current)
        if len(wrapped) <= lines and all(draw.textbbox((0, 0), line, font=font)[2] <= width
                                         for line in wrapped):
            return font, wrapped
    return _load_font(font_path, minimum), wrapped


def _card_layout(draw: Any, group: dict[str, Any], font_path: Path | None) -> list[dict]:
    """Where each row of a card goes, and at what size: worked out once per card, not per frame."""
    rows = group["rows"]
    height = (CARD[3] - CARD[1] - 40) / len(rows)
    layout = []
    for index, row in enumerate(rows):
        word = row["kind"] in WORD_KINDS
        biggest = (70 if row["side"] == "source" else 58) if word else 46
        font, lines = _wrap(draw, row["text"], font_path, biggest, 26, 820,
                            1 if word else (3 if len(rows) == 1 else 2))
        line_height = font.size * 1.22
        top = CARD[1] + 20 + index * height
        text_top = top + (height - line_height * len(lines)) / 2
        layout.append({"row": row, "font": font, "lines": lines, "top": top, "height": height,
                       "text_top": text_top, "line_height": line_height})
    return layout


def _card_label(group: dict[str, Any], count: int) -> str:
    parts = [SECTION_LABELS.get(group["section"], "")]
    if group["kind"] not in WORD_KINDS or group["section"] not in SECTION_LABELS:
        parts.append(KIND_LABELS.get(group["kind"], ""))
    if group["item"] is not None:
        parts.append(f"{int(group['item']) + 1:02d}  /  {count:02d}")
    return "    ·    ".join(part for part in parts if part)


def cue_frames(title: str, cues: Sequence[dict[str, Any]], items: Sequence[dict[str, Any]],
               duration: float, grid: Grid, *, font_path: Path | None = None):
    """A frame function for a finished loop: `frame(index, background) -> RGB bytes`.

    Every spoken line is shown, card by card (`cue_groups`): a word with its translation, an example
    with its translation, a remark, a line of the story. A row appears when it is first heard and
    is lit while it is heard; each carries the flag of its language.
    """
    from PIL import Image, ImageDraw

    groups = cue_groups(cues)
    body_font_path = resolve_font(font_path)
    native = next((c["language"] for c in cues if c.get("role") == "native"), "")
    guide = next((c["language"] for c in cues if c.get("role") == "guide"), "")
    languages = "  /  ".join(LANGUAGE_NAMES.get(code.split("-")[0], code.upper())
                             for code in (native, guide) if code)
    layouts: dict[int, list[dict]] = {}
    small = _load_font(body_font_path, 24)
    headline = _load_font(body_font_path, 72)
    medium = _load_font(body_font_path, 31)

    def frame(index: int, background: np.ndarray | None = None) -> bytes:
        at = index / DEMO_FPS
        image = Image.fromarray((background if background is not None
                                 else _background()).copy())
        draw = ImageDraw.Draw(image, "RGBA")
        _chrome(draw, title, languages, at, grid, body_font_path)
        current = next((n for n, g in enumerate(groups) if g["start"] <= at < g["end"]), None)
        if current is None:
            _centered_text(draw, "Vocabulary, set to a beat.", 252, headline,
                           (244, 246, 255, 245))
            _centered_text(draw, "Expressive Gemini voices · deterministic procedural music",
                           355, medium, (174, 204, 216, 220))
        else:
            group = groups[current]
            if current not in layouts:
                layouts[current] = _card_layout(draw, group, body_font_path)
            _rounded_rectangle(draw, CARD, 34, (14, 21, 47, 206), (126, 155, 186, 50), 2)
            for slot in layouts[current]:
                row = slot["row"]
                if at < row["heard"][0][0]:
                    continue
                lit = any(start <= at < end for start, end in row["heard"])
                top, height = slot["top"], slot["height"]
                if lit:
                    native_row = row["language"] == native
                    _rounded_rectangle(
                        draw, (132, int(top + 6), DEMO_WIDTH - 132, int(top + height - 6)), 25,
                        (45, 94, 124, 125) if native_row else (72, 61, 118, 125),
                        (102, 220, 194, 125) if native_row else (170, 139, 242, 125), 2)
                _flag(draw, row["language"], 164, int(top + height / 2 - 22), 67, 44,
                      body_font_path)
                fill = (247, 250, 255, 255 if lit else 222)
                for n, line in enumerate(slot["lines"]):
                    box = draw.textbbox((0, 0), line, font=slot["font"])
                    x = 250 + (DEMO_WIDTH - 250 - 164 - (box[2] - box[0])) / 2
                    draw.text((x, slot["text_top"] + n * slot["line_height"]), line,
                              font=slot["font"], fill=fill)
            _centered_text(draw, _card_label(group, len(items)), 619, small,
                           (155, 187, 204, 190))
        _progress(draw, at, duration)
        return image.tobytes()

    return frame


def _ffmpeg() -> str:
    command = shutil.which("ffmpeg")
    if not command:
        raise RuntimeError("FFmpeg is required to generate the README MP4.")
    return command


def _write_frames(process: subprocess.Popen[bytes], frame, duration: float) -> None:
    assert process.stdin is not None
    background = _background()
    total = math.ceil(duration * DEMO_FPS)
    try:
        for index in range(total):
            process.stdin.write(frame(index, background))
        process.stdin.close()
        process.stdin = None
        _, stderr = process.communicate()
    except BrokenPipeError:
        _, stderr = process.communicate()
        raise RuntimeError(stderr.decode("utf-8", errors="replace")) from None
    if process.returncode:
        raise RuntimeError("FFmpeg video encoding failed: " +
                           stderr.decode("utf-8", errors="replace")[-2000:])


def encode_visual_track(title: str, timeline: Sequence[dict[str, Any]],
                        duration: float, grid: Grid, output: Path, *,
                        font_path: Path | None = None, frame=None,
                        crf: int = DEMO_VIDEO_CRF) -> None:
    """Quality-encode the shared silent H.264 visual stream.

    `frame(index, background)` draws each frame; without one it is the README demo's word card.
    """
    if frame is None:
        def frame(index, background):
            return frame_bytes(title, timeline, duration, grid, index,
                               font_path=font_path, background=background)
    ffmpeg = _ffmpeg()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".{os.getpid()}.partial.mp4")
    try:
        process = subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s:v", f"{DEMO_WIDTH}x{DEMO_HEIGHT}",
             "-r", str(DEMO_FPS), "-i", "-", "-an", "-c:v", "libx264",
             "-preset", "slow", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        _write_frames(process, frame, duration)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def mux_audio(visual: Path, audio: Path, output: Path, *,
              bitrate: int = DEMO_AUDIO_BITRATE) -> None:
    """Copy the shared H.264 stream and add one AAC music variant."""
    ffmpeg = _ffmpeg()
    temporary = output.with_suffix(f".{os.getpid()}.partial.mp4")
    try:
        result = subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(visual), "-i", str(audio), "-map", "0:v:0",
            "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
            "-b:a", str(bitrate), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-shortest", str(temporary),
        ], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("FFmpeg mux failed: " + result.stderr[-2000:])
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def audio_summary(audio: np.ndarray, sample_rate: int = SR) -> dict[str, Any]:
    return {
        "sample_rate": sample_rate,
        "channels": int(audio.shape[1]) if audio.ndim == 2 else 1,
        "duration_seconds": len(audio) / sample_rate,
        "peak": float(np.abs(audio).max()) if len(audio) else 0.0,
        "finite": bool(np.isfinite(audio).all()),
    }
