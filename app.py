"""Hugging Face Space entry point for the LexiBeat explorer."""

from __future__ import annotations

import os
from pathlib import Path


def _configure_attached_sample_bucket() -> None:
    """Use the conventional read-only bucket mount when it is available."""
    if os.environ.get("LEXIBEAT_BUNDLE_ROOT"):
        return
    mount_root = Path(os.environ.get("LEXIBEAT_BUCKET_MOUNT", "/data"))
    mounted = mount_root / "lexibeat-production-core" / "v1"
    if mounted.joinpath("catalog.sqlite3").is_file():
        os.environ["LEXIBEAT_BUNDLE_ROOT"] = str(mounted)


_configure_attached_sample_bucket()

from lexibeat.explorer_web import create_api

app = create_api()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
