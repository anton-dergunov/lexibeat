"""Validated two-phase lesson rendering shared by local and hosted UIs."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import soundfile as sf

from .api import MusicRequest, resolve_music
from .arrange import Event, arrange, render_speech
from .bedspec import BedSpec
from .explorer import ExplorerConfig, validate_bed_spec
from .generator import ENGINE_VERSION
from .mix import mix_stems
from .music import SR, Grid, render_stems
from .vocab import Item
from .voice import Backend, Speaker

LESSON_MODEL = "chatterbox-multilingual"
LESSON_PATTERN = "retrieval"
LESSON_ENGINE_VERSION = "2"
MAX_LESSON_ITEMS = 6
MAX_CELL_CHARACTERS = 120
DEFAULT_LESSON_ROWS = [
    ["hola", "hello"],
    ["gracias", "thank you"],
    ["por favor", "please"],
]

_locks_guard = threading.Lock()
_locks: dict[str, threading.Lock] = {}


@dataclass(frozen=True)
class LessonSpeechArtifact:
    speech_path: str
    metadata_path: str
    bed_spec: dict[str, Any]
    items: list[dict[str, str]]
    subtitles: list[dict[str, Any]]
    total_bars: int
    voice_seed: int
    cache_hit: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LessonRenderArtifact:
    audio_path: str
    bed_spec: dict[str, Any]
    subtitles: list[dict[str, Any]]
    duration_seconds: float
    cache_hit: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _plain_rows(value: object) -> list[list[object]]:
    if value is None:
        return []
    if hasattr(value, "values"):
        value = value.values.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("Vocabulary must be a two-column table.")
    return [list(row) for row in value if isinstance(row, (list, tuple))]


def normalize_lesson_rows(value: object) -> list[Item]:
    """Validate the public table shape and return normalized bilingual items."""
    items: list[Item] = []
    for index, row in enumerate(_plain_rows(value), 1):
        source = str(row[0] or "").strip() if row else ""
        target = str(row[1] or "").strip() if len(row) > 1 else ""
        if not source and not target:
            continue
        if not source or not target:
            raise ValueError(
                f"Vocabulary row {index} needs both Spanish and English text.")
        if len(source) > MAX_CELL_CHARACTERS or len(target) > MAX_CELL_CHARACTERS:
            raise ValueError(
                f"Vocabulary row {index} exceeds the {MAX_CELL_CHARACTERS}-character limit.")
        item = Item(source, target)
        if not item:
            raise ValueError(f"Vocabulary row {index} contains no speakable text.")
        items.append(item)
    if not items:
        raise ValueError("Enter at least one Spanish/English pair.")
    if len(items) > MAX_LESSON_ITEMS:
        raise ValueError(f"Lessons are limited to {MAX_LESSON_ITEMS} vocabulary pairs.")
    return items


def resolve_lesson_spec(state: dict[str, Any] | None, *,
                        palette: str = "electronic") -> BedSpec:
    """Use the last applied bed, or resolve a safe automatic default."""
    data = (state or {}).get("bed_spec")
    if data:
        spec, report = validate_bed_spec(data, analyze=False)
        if spec is None or report.state == "invalid":
            raise ValueError("The current BedSpec is invalid; validate it in Lab first.")
        return spec
    if palette not in {"hybrid", "electronic"}:
        palette = "electronic"
    return resolve_music(MusicRequest(
        family="auto", energy="balanced", rhythm="steady", palette=palette,
    )).bed_spec


def lesson_gpu_duration(rows: object, _model: object = None,
                        _state: object = None,
                        _progress: object = None) -> int:
    """Reserve realistic ZeroGPU time without rejecting malformed UI input."""
    try:
        count = len(normalize_lesson_rows(rows))
    except ValueError:
        count = 1
    return min(120, 60 + 10 * count)


def _cache_lock(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def _trim_lesson_cache(root: Path, *, keep: set[Path], max_bytes: int) -> None:
    files = [path for path in root.iterdir()
             if path.is_file() and ".partial." not in path.name]
    total = sum(path.stat().st_size for path in files)
    for path in sorted(files, key=lambda item: item.stat().st_mtime):
        if total <= max_bytes:
            break
        if path in keep:
            continue
        size = path.stat().st_size
        path.unlink(missing_ok=True)
        total -= size


def _speech_key(items: Sequence[Item], spec: BedSpec, model: str,
                voice_seed: int) -> str:
    payload = {
        "items": [{"source": item.source, "target": item.target} for item in items],
        "bed_spec": asdict(spec),
        "pattern": LESSON_PATTERN,
        "model": model,
        "voice_seed": voice_seed,
        "engine_version": ENGINE_VERSION,
        "lesson_engine_version": LESSON_ENGINE_VERSION,
        "sample_rate": SR,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _subtitles(events: Sequence[Event], grid: Grid,
               total_bars: int, *, outro_bars: int = 2) -> list[dict[str, Any]]:
    rows = []
    final_hold = grid.bar_start(max(total_bars - outro_bars, 0))
    for index, event in enumerate(events):
        language, _, text = event.label.partition(":")
        spoken_end = float(event.start + len(event.audio) / SR)
        next_start = (float(events[index + 1].start)
                      if index + 1 < len(events) else float(final_hold))
        rows.append({
            "text": text,
            "timestamp": [float(event.start), max(spoken_end, next_start)],
            "language": language,
        })
    return rows


def render_lesson_speech(
    rows: object,
    model: str,
    state: dict[str, Any] | None,
    *,
    backend: Backend,
    config: ExplorerConfig | None = None,
    palette: str = "electronic",
    voice_seed: int | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """GPU phase: synthesize and cache a mono speech stem and timing metadata."""
    if progress:
        progress(0.03, "Validating vocabulary")
    if model != LESSON_MODEL:
        raise ValueError(f"Unsupported lesson model '{model}'.")
    config = config or ExplorerConfig.from_environment()
    items = normalize_lesson_rows(rows)
    if progress:
        progress(0.05, "Resolving the applied music bed")
    spec = resolve_lesson_spec(state, palette=palette)
    voice_seed = (int(spec.seed % (2**31 - 1)) if voice_seed is None
                  else int(voice_seed))
    runtime_model = f"{model}:{getattr(backend, 'model_id', backend.name)}"
    key = _speech_key(items, spec, runtime_model, voice_seed)
    root = config.output_root.resolve() / "lessons"
    speech_path = root / f"{key}.speech.wav"
    metadata_path = root / f"{key}.speech.json"

    def artifact(cache_hit: bool) -> LessonSpeechArtifact:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return LessonSpeechArtifact(
            str(speech_path), str(metadata_path), metadata["bed_spec"],
            metadata["items"], metadata["subtitles"], metadata["total_bars"],
            metadata["voice_seed"], cache_hit)

    root.mkdir(parents=True, exist_ok=True)
    with _cache_lock(key):
        if speech_path.is_file() and metadata_path.is_file():
            if progress:
                progress(1.0, "Using cached speech")
            return artifact(True).to_dict()
        temporary_audio = root / f".{key}.{os.getpid()}.speech.partial.wav"
        temporary_metadata = root / f".{key}.{os.getpid()}.speech.partial.json"
        speaker = Speaker(
            backend="chatterbox", backend_instance=backend,
            voice_seed=voice_seed)
        try:
            grid = Grid.from_spec(spec)
            if progress:
                progress(0.08, "Preparing Chatterbox synthesis")
            events, total_bars = arrange(
                items, speaker, grid, pattern=LESSON_PATTERN, progress=False,
                progress_callback=(
                    (lambda completed, total, message:
                     progress(0.10 + 0.84 * completed / max(total, 1), message))
                    if progress else None))
            speech = render_speech(events, total_bars, grid)
            if not len(speech) or not np.isfinite(speech).all():
                raise RuntimeError("Speech renderer produced invalid audio.")
            sf.write(temporary_audio, speech, SR, subtype="PCM_16")
            metadata = {
                "bed_spec": asdict(spec),
                "items": [{"source": item.source, "target": item.target}
                          for item in items],
                "subtitles": _subtitles(events, grid, total_bars),
                "total_bars": total_bars,
                "voice_seed": voice_seed,
            }
            temporary_metadata.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            os.replace(temporary_audio, speech_path)
            os.replace(temporary_metadata, metadata_path)
            _trim_lesson_cache(
                root, keep={speech_path, metadata_path},
                max_bytes=config.max_cache_bytes)
            if progress:
                progress(1.0, "Speech synthesis complete")
        finally:
            temporary_audio.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)
    return artifact(False).to_dict()


def _mix_key(job: dict[str, Any]) -> str:
    payload = {
        "speech": Path(job["speech_path"]).stem,
        "bed_spec": job["bed_spec"],
        "engine_version": ENGINE_VERSION,
        "sample_rate": SR,
        "speech_lufs": -16.0,
        "music_lufs": -26.0,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def finalize_lesson(job: dict[str, Any], *,
                    config: ExplorerConfig | None = None,
                    progress: Callable[[float, str], None] | None = None,
                    ) -> dict[str, Any]:
    """CPU phase: render the music, duck it under speech, and write final WAV."""
    config = config or ExplorerConfig.from_environment()
    spec, report = validate_bed_spec(job["bed_spec"], analyze=False)
    if spec is None or report.state == "invalid":
        raise ValueError("The lesson contains an invalid BedSpec.")
    speech_path = Path(job["speech_path"]).resolve()
    lesson_root = (config.output_root.resolve() / "lessons")
    if speech_path.parent != lesson_root or not speech_path.is_file():
        raise ValueError("The lesson speech artifact is unavailable.")
    key = _mix_key(job)
    path = lesson_root / f"{key}.lesson.wav"
    if path.is_file():
        return LessonRenderArtifact(
            str(path), asdict(spec), job["subtitles"],
            float(sf.info(path).duration), True).to_dict()

    with _cache_lock(key):
        if path.is_file():
            return LessonRenderArtifact(
                str(path), asdict(spec), job["subtitles"],
                float(sf.info(path).duration), True).to_dict()
        temporary = lesson_root / f".{key}.{os.getpid()}.lesson.partial.wav"
        try:
            if progress:
                progress(0.05, "Loading speech stem")
            speech, rate = sf.read(speech_path, dtype="float32")
            if rate != SR or speech.ndim != 1:
                raise RuntimeError("Speech artifact has an unexpected audio format.")
            if progress:
                progress(0.15, "Rendering music stems")
            stems = render_stems(
                spec, int(job["total_bars"]),
                progress_callback=(
                    (lambda value, message: progress(0.15 + 0.55 * value, message))
                    if progress else None))
            depths = {name: getattr(spec, name).duck_db for name in stems}
            if progress:
                progress(0.75, "Mixing lesson")
            track = mix_stems(stems, speech, depths)
            if track.ndim != 2 or track.shape[1] != 2 or \
                    not len(track) or not np.isfinite(track).all():
                raise RuntimeError("Lesson renderer produced invalid stereo audio.")
            if float(np.abs(track).max()) > 0.97 + 1e-7:
                raise RuntimeError("Lesson renderer exceeded the 0.97 peak limit.")
            sf.write(temporary, track, SR, subtype="PCM_16")
            os.replace(temporary, path)
            _trim_lesson_cache(
                lesson_root,
                keep={path, speech_path, Path(job["metadata_path"]).resolve()},
                max_bytes=config.max_cache_bytes)
            if progress:
                progress(1.0, "Lesson ready")
        finally:
            temporary.unlink(missing_ok=True)
    return LessonRenderArtifact(
        str(path), asdict(spec), job["subtitles"], len(track) / SR,
        False).to_dict()
