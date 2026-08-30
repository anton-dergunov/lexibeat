"""Filesystem locations shared by the local and hosted sample systems."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_BUNDLE_ROOT = (
    Path(__file__).resolve().parents[1] / "assets" / "production-core" / "v1"
)


def configured_bundle_root() -> Path:
    """Return the repository bundle or an explicitly mounted replacement."""
    configured = os.environ.get("LEXIBEAT_BUNDLE_ROOT")
    return Path(configured).expanduser() if configured else REPOSITORY_BUNDLE_ROOT


BUNDLED_ROOT = configured_bundle_root()
