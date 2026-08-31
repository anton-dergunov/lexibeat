"""Reusable audio/video machinery for the README demonstration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from .arrange import Event, PATTERNS
from .bedspec import BedSpec, STYLES
from .emotion import for_item
from .music import SR, Grid
from .voice import Prosody, Speaker
from .vocab import Item

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
    pattern: str
    items: tuple[Item, ...]
    bars_per_utterance: tuple[int, ...]
    variants: tuple[DemoVariant, ...]


def load_demo_config(path: Path) -> DemoConfig:
    """Load and validate the portable demo manifest."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Demo manifest schema_version must be 1.")
    title = str(data.get("title") or "").strip()
    if not title:
        raise ValueError("Demo manifest title cannot be empty.")
    pattern = str(data.get("pattern") or "")
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown demo pattern '{pattern}'.")
    raw_items = data.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("Demo manifest needs at least one vocabulary item.")
    items: list[Item] = []
    bars_per_utterance: list[int] = []
    for index, row in enumerate(raw_items, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Demo item {index} must be an object.")
        item = Item(str(row.get("source") or ""), str(row.get("target") or ""),
                    str(row.get("emoji") or ""))
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
    return DemoConfig(title, pattern, tuple(items), tuple(bars_per_utterance),
                      tuple(variants))


def resolve_demo_specs(config: DemoConfig) -> dict[str, BedSpec]:
    """Resolve deterministic beds and require one shared speech grid."""
    specs = {
        variant.name: BedSpec.from_style(variant.style, variant.seed)
        for variant in config.variants
    }
    grids = {
        (spec.bpm, spec.beats_per_bar, spec.beat_unit) for spec in specs.values()
    }
    if len(grids) != 1:
        detail = ", ".join(
            f"{name}={spec.bpm:g} BPM {spec.beats_per_bar}/{spec.beat_unit}"
            for name, spec in specs.items())
        raise ValueError(f"Demo beds must share one timing grid ({detail}).")
    return specs


def cache_key(speaker: Speaker, text: str, lang: str, prosody: Prosody,
              emotion_name: str, target_seconds: float | None) -> str:
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
        "lang": lang,
        "prosody": asdict(prosody),
        "emotion": emotion_name,
        "target_seconds": target_seconds,
        "sample_rate": SR,
    }
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

    def say(self, text: str, lang: str, prosody: Prosody, emotion: Any,
            target_seconds: float | None = None, *, retry: bool = False) -> np.ndarray:
        key = cache_key(self.speaker, text, lang, prosody, emotion.name,
                        target_seconds)
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
        audio = self.speaker.say(text, lang, prosody, emotion, target_seconds,
                                 retry=retry)
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


def arrange_demo(config: DemoConfig, speaker: PersistentSpeaker,
                 grid: Grid, *, intro_bars: int = 2,
                 outro_bars: int = 2) -> tuple[list[Event], int]:
    """Arrange the retrieval pattern with optional longer per-item speech slots."""
    events: list[Event] = []
    bar = intro_bars
    for index, (item, span) in enumerate(
            zip(config.items, config.bars_per_utterance), 1):
        emotion = for_item(item.source, item.emoji)
        print(f"  [{index}/{len(config.items)}] {item.emoji or ' '} "
              f"{item.source} — {item.target}  ({emotion.name}, "
              f"{span} bar{'s' if span != 1 else ''}/utterance)", flush=True)
        for kind, repetition in PATTERNS[config.pattern]:
            if kind in ("gap", "rest"):
                bar += 1
                continue
            text = item.source if kind == "es" else item.target
            prosody = Prosody.for_repeat(repetition, speaker.prosody_strength)
            prosody = prosody.with_emotion(emotion, speaker.prosody_strength)
            audio = speaker.say(
                text, kind, prosody, emotion,
                target_seconds=grid.bar * span * 0.92)
            events.append(Event(grid.bar_start(bar), audio, f"{kind}:{text}"))
            bar += span
    return events, bar + outro_bars


def build_timeline(items: Sequence[Item], events: Sequence[Event], grid: Grid,
                   total_bars: int, pattern: str) -> list[dict[str, Any]]:
    """Describe progressive reveals and active utterances from arranged events."""
    slots = PATTERNS[pattern]
    spoken_slots = [(kind, rep) for kind, rep in slots if kind in ("es", "en")]
    expected = len(items) * len(spoken_slots)
    if len(events) != expected:
        raise ValueError(f"Expected {expected} speech events, received {len(events)}.")
    timeline: list[dict[str, Any]] = []
    cursor = 0
    for item_index, item in enumerate(items):
        utterances = []
        for language, repetition in spoken_slots:
            event = events[cursor]
            expected_text = item.source if language == "es" else item.target
            if event.label != f"{language}:{expected_text}":
                raise ValueError("Speech events do not match the configured vocabulary.")
            utterances.append({
                "language": language,
                "repetition": repetition,
                "start": float(event.start),
                "end": float(event.start + len(event.audio) / grid.sr),
            })
            cursor += 1
        source_reveal = next(row["start"] for row in utterances
                             if row["language"] == "es")
        target_reveal = next(row["start"] for row in utterances
                             if row["language"] == "en")
        next_start = (float(events[cursor].start) if cursor < len(events)
                      else float(total_bars * grid.bar))
        timeline.append({
            "index": item_index,
            "source": item.source,
            "target": item.target,
            "emoji": item.emoji,
            "emotion": for_item(item.source, item.emoji).name,
            "start": source_reveal,
            "source_reveal": source_reveal,
            "target_reveal": target_reveal,
            "end": next_start,
            "utterances": utterances,
        })
    return timeline


def write_tracklist(path: Path, variant: str, config: DemoConfig,
                    timeline: Sequence[dict[str, Any]], spec: BedSpec) -> None:
    lines = [
        f"{variant} — {len(config.items)} items, {spec.bpm:g} BPM, "
        f"pattern '{config.pattern}'",
        "",
    ]
    for row in timeline:
        at = float(row["start"])
        lines.append(f"{int(at)//60:02d}:{int(at)%60:02d}  {row['emoji']} "
                     f"{row['source']} — {row['target']} ({row['emotion']})")
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


def _active_language(row: dict[str, Any], at: float) -> str | None:
    active = next((utterance for utterance in row["utterances"]
                   if float(utterance["start"]) <= at < float(utterance["end"])), None)
    return str(active["language"]) if active else None


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
    brand = _load_font(body_font_path, 30)
    draw.text((56, 39), title, font=brand, fill=(238, 242, 255, 235))
    draw.text((DEMO_WIDTH - 274, 45), "SPANISH  /  ENGLISH", font=small,
              fill=(185, 205, 225, 195))

    beat_phase = (at % grid.bar) / grid.bar
    pulse = max(0.0, 1.0 - beat_phase * 5.5)
    radius = int(5 + pulse * 7)
    draw.ellipse((DEMO_WIDTH // 2 - radius, 72 - radius,
                  DEMO_WIDTH // 2 + radius, 72 + radius),
                 fill=(77, 222, 181, int(90 + 120 * pulse)))

    row = _active_item(timeline, at)
    if row is None:
        headline = _load_font(body_font_path, 72)
        _centered_text(draw, "Vocabulary, set to a beat.", 252, headline,
                       (244, 246, 255, 245))
        _centered_text(draw, "Expressive Gemini voices · deterministic procedural music",
                       355, medium, (174, 204, 216, 220))
    else:
        active = _active_language(row, at)
        source_visible = at >= float(row["source_reveal"])
        target_visible = at >= float(row["target_reveal"])
        card = (105, 145, DEMO_WIDTH - 105, 592)
        _rounded_rectangle(draw, card, 34, (14, 21, 47, 206),
                           (126, 155, 186, 50), 2)

        source_alpha = 255 if active == "es" else 222
        target_alpha = 255 if active == "en" else 218
        if active == "es":
            _rounded_rectangle(draw, (132, 178, DEMO_WIDTH - 132, 337), 25,
                               (45, 94, 124, 125), (102, 220, 194, 125), 2)
        if active == "en":
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
        label = f"{index:02d}  /  {len(timeline):02d}    ·    {str(row['emotion']).upper()}"
        _centered_text(draw, label, 619, small, (155, 187, 204, 190))

    progress = min(max(at / max(duration, 0.001), 0.0), 1.0)
    draw.rounded_rectangle((56, 678, DEMO_WIDTH - 56, 685), radius=4,
                           fill=(104, 126, 156, 70))
    draw.rounded_rectangle((56, 678, 56 + int((DEMO_WIDTH - 112) * progress), 685),
                           radius=4, fill=(77, 222, 181, 205))
    return image.tobytes()


def _ffmpeg() -> str:
    command = shutil.which("ffmpeg")
    if not command:
        raise RuntimeError("FFmpeg is required to generate the README MP4.")
    return command


def _write_frames(process: subprocess.Popen[bytes], title: str,
                  timeline: Sequence[dict[str, Any]], duration: float,
                  grid: Grid, font_path: Path | None) -> None:
    assert process.stdin is not None
    background = _background()
    total = math.ceil(duration * DEMO_FPS)
    try:
        for index in range(total):
            process.stdin.write(frame_bytes(
                title, timeline, duration, grid, index,
                font_path=font_path, background=background))
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
                        font_path: Path | None = None) -> None:
    """Quality-encode the shared silent H.264 visual stream."""
    ffmpeg = _ffmpeg()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".{os.getpid()}.partial.mp4")
    try:
        process = subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s:v", f"{DEMO_WIDTH}x{DEMO_HEIGHT}",
             "-r", str(DEMO_FPS), "-i", "-", "-an", "-c:v", "libx264",
             "-preset", "slow", "-crf", str(DEMO_VIDEO_CRF),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        _write_frames(process, title, timeline, duration, grid, font_path)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def mux_audio(visual: Path, audio: Path, output: Path) -> None:
    """Copy the shared H.264 stream and add one AAC music variant."""
    ffmpeg = _ffmpeg()
    temporary = output.with_suffix(f".{os.getpid()}.partial.mp4")
    try:
        result = subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(visual), "-i", str(audio), "-map", "0:v:0",
            "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
            "-b:a", str(DEMO_AUDIO_BITRATE), "-pix_fmt", "yuv420p",
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
