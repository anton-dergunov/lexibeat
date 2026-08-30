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

__all__ = [
    "BedFingerprint",
    "MusicGenerationResult",
    "MusicRequest",
    "QualityReport",
    "SampleUsage",
    "generate_music",
    "render_music",
    "resolve_music",
]

__version__ = "1.0.0"
