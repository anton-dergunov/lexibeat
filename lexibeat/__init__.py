"""LexiBeat: reproducible procedural music for language-learning audio."""

from .api import (
    BedFingerprint,
    MusicGenerationResult,
    MusicRequest,
    QualityReport,
    SampleUsage,
    generate_music,
    render_music,
    resolve_music,
)
from .explorer import (
    ExplorerValidationReport,
    RandomizationResult,
    ValidationIssue,
    apply_safe_repairs,
    randomize_unlocked,
    validate_bed_spec,
)
from .language import Language
from .loop import LoopError, LoopRequest, LoopResult, build_timeline, render_loop
from .vocab import Item
from .voice import Backend, BackendCapabilities, Delivery, Prosody, SpeechRequest, SynthesisResult

__all__ = [
    "Backend",
    "BackendCapabilities",
    "BedFingerprint",
    "Delivery",
    "Item",
    "Language",
    "LoopError",
    "LoopRequest",
    "LoopResult",
    "MusicGenerationResult",
    "MusicRequest",
    "Prosody",
    "QualityReport",
    "SampleUsage",
    "SpeechRequest",
    "SynthesisResult",
    "build_timeline",
    "generate_music",
    "render_loop",
    "render_music",
    "resolve_music",
    "ExplorerValidationReport",
    "RandomizationResult",
    "ValidationIssue",
    "apply_safe_repairs",
    "randomize_unlocked",
    "validate_bed_spec",
]

__version__ = "2.0.0"
