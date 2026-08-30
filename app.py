"""Hugging Face Space entry point for the LexiBeat explorer."""

from __future__ import annotations

import os

from lexibeat.explorer_web import create_api

app = create_api()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
