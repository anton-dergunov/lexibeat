"""A labelling round: production beds, rated part by part on the tablet.

    uv run python -m scripts.listening.make_round --out out/listening/round-01
    uv run python -m scripts.listening.make_round --out out/listening/round-02 \\
        --replay out/listening/round-01
    uv run python -m scripts.listening.serve out/listening/round-01

Beds are resolved exactly as a host's are (see `common`), with whatever the production profile's
`ListenerPolicy` currently is — so a round after a rule was adopted hears the rule. `--replay`
renders every seed of an earlier round again first, which is what makes rounds comparable: the
same music, judged before and after a change, rather than two unrelated samples.
"""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

from .common import Production, facts, render_clip, write_round


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30,
                        help="fresh clips, on top of any replayed ones")
    parser.add_argument("--replay", type=Path, default=None,
                        help="an earlier round whose seeds are rendered again first")
    parser.add_argument("--seed", type=int, default=None,
                        help="where the fresh seeds start; random if omitted, and printed")
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args(argv)

    seeds: list[int] = []
    if args.replay:
        earlier = json.loads((args.replay / "round.json").read_text(encoding="utf-8"))
        seeds += [int(clip["seed"]) for clip in earlier["clips"]]
    replayed = len(seeds)
    start = args.seed if args.seed is not None else secrets.randbelow(2 ** 31)
    print(f"fresh seeds start at {start}")
    seeds += [start + index for index in range(args.count) if start + index not in seeds]

    production = Production()
    clips = []
    for index, seed in enumerate(seeds, 1):
        result = production.resolve(seed)
        clip_id = f"c{index:02d}"
        render_clip(result.bed_spec, args.out / f"{clip_id}.mp3", args.seconds)
        clips.append({"id": clip_id, "file": f"{clip_id}.mp3", "seed": seed,
                      "replayed": index <= replayed,
                      "facts": facts(production, result.bed_spec)})
        print(f"rendered {clip_id} ({index}/{len(seeds)})  {result.bed_spec.phrase.family}")
    write_round(args.out, "parts", clips)
    print(f"Wrote {len(clips)} clips to {args.out}. Serve them with:\n"
          f"  uv run python -m scripts.listening.serve {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
