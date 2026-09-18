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

# SQLite writes this at the head of every database it makes.
SQLITE_MAGIC = b"SQLite format 3\x00"
CATALOG_NAME = "catalog.sqlite3"


def bundle_catalog(root: Path | None = None) -> Path | None:
    """The bundle's catalog if there is a real one, else ``None``.

    Checking that the file *exists* is not enough, and the difference is not theoretical. The
    bundle is Git-LFS tracked, so a clone without `git lfs pull` — a fresh checkout, a CI job that
    skips LFS — leaves a 130-byte pointer *at that exact path*. Every "is the bundle here?" test
    then answered yes and the first query died with `sqlite3.DatabaseError: file is not a
    database`, which is a far worse failure than the well-handled one where the bundle is simply
    absent. Reading the header costs nothing and turns the second case back into the first.
    """
    path = (Path(root) if root else configured_bundle_root()) / CATALOG_NAME
    try:
        with path.open("rb") as handle:
            return path if handle.read(len(SQLITE_MAGIC)) == SQLITE_MAGIC else None
    except OSError:
        return None


def bundle_present(root: Path | None = None) -> bool:
    return bundle_catalog(root) is not None
