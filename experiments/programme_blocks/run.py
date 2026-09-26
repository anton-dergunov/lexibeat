"""Try programme blocks one at a time, and score them on a tablet.

    uv run --extra hosted-tts --env-file .env python experiments/programme_blocks/run.py all
    uv run --extra hosted-tts --env-file .env python experiments/programme_blocks/run.py all --dry-run
    uv run python experiments/programme_blocks/run.py serve

This runs in the repository's own environment, not one of its own, because the drills are
rendered with LexiBeat's beds, stretch, fit and mix: what is heard is what a loop would do. See
README.md for what each stage asks and why.

Speech goes to Cloud Text-to-Speech with Application Default Credentials, as Acervo's production
voices do. Text goes to the Gemini free tier through `GEMINI_API_KEY`. `--dry-run` buys no speech:
it counts what would be bought and answers with silence. The text calls are free and still made,
because later stages depend on what they return.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tts  # noqa: E402

STAGES = ["drills", "syllables", "mixed", "context", "commentary", "framing", "pronounce"]
# Cloud TTS bills Gemini voices per token. At a list price of a few tens of dollars per million
# audio tokens and 25 tokens a second of speech, a three-second line costs well under a tenth of a
# cent; WaveNet is billed per character and costs less still.
DOLLARS_PER_CALL = 0.002


def stage(name: str) -> None:
    started = time.perf_counter()
    module = __import__(name)
    before = dict(tts.counted)
    print(f"{name}…", flush=True)
    module.run()
    calls = tts.counted["calls"] - before["calls"]
    note = f" · would buy {calls} speech calls (≈ ${calls * DOLLARS_PER_CALL:.2f})" if tts.DRY else ""
    print(f"{name} done in {time.perf_counter() - started:.0f} s{note}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=[*STAGES, "all", "page", "serve"])
    parser.add_argument("--dry-run", action="store_true",
                        help="buy no speech; count it and answer with silence")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    tts.DRY = args.dry_run
    if args.stage == "serve":
        import serve

        return serve.main(args.port)
    for name in STAGES if args.stage == "all" else [] if args.stage == "page" else [args.stage]:
        stage(name)
    if tts.DRY:
        print(f"Dry run: {tts.counted['calls']} speech calls would be bought "
              f"(≈ ${tts.counted['calls'] * DOLLARS_PER_CALL:.2f}). The page was not rebuilt.")
        return 0
    import page

    print(f"Page: {page.build()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
