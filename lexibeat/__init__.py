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

__all__ = [
    "BedFingerprint",
    "MusicGenerationResult",
    "MusicRequest",
    "QualityReport",
    "SampleUsage",
    "generate_music",
    "render_music",
    "resolve_music",
    "ExplorerValidationReport",
    "RandomizationResult",
    "ValidationIssue",
    "apply_safe_repairs",
    "randomize_unlocked",
    "validate_bed_spec",
]

__version__ = "1.0.0"
