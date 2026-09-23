"""Round 0: hear each known defect fixed, one at a time, blind.

    uv run python -m scripts.listening.defects --out out/listening/round-00
    uv run python -m scripts.listening.serve out/listening/round-00

For each of the three `ListenerPolicy` switches it finds production beds that show the defect,
rebuilds the *same* winning candidate with only that switch on, and renders both. The two clips of
a pair differ in that one part and nothing else, and the page does not say which is which: the
answer is in `key.json`, which the server never serves. `summarise` reads both afterwards.
"""

from __future__ import annotations

import argparse
import json
import random
import secrets
from dataclasses import asdict, replace
from pathlib import Path

from .common import Production, render_clip, write_round

# Each switch is laid over production's *own* policy, so a pair differs in that switch alone even
# once others have been adopted — and a defect production already rules out is simply not found.
SWITCHES = {
    "unapproved_lead": ({"approved_catalog_only": True}, "the lead instrument"),
    "high_lead": ({"lead_register_cap": 88}, "the highest notes of the melody"),
    "off_key_fifth": ({"diatonic_fifths": True}, "the chords"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("out/listening/round-00"))
    parser.add_argument("--per-defect", type=int, default=6)
    parser.add_argument("--start-seed", type=int, default=None,
                        help="first request seed to try; random if omitted, and printed")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--max-seeds", type=int, default=400)
    args = parser.parse_args(argv)

    start = args.start_seed if args.start_seed is not None else secrets.randbelow(2 ** 31)
    print(f"start seed {start}")
    production = Production()
    found: dict[str, list[dict]] = {kind: [] for kind in SWITCHES}
    for seed in range(start, start + args.max_seeds):
        if all(len(rows) >= args.per_defect for rows in found.values()):
            break
        spec = production.resolve(seed).bed_spec
        # The rebuild must reproduce the winner exactly, or a pair would differ in more than the
        # switch; this is the check that `build_bed` is still what the pool builds.
        if asdict(production.rebuild(spec, production.profile.listener)) != asdict(spec):
            raise SystemExit(f"Rebuilding seed {seed} did not reproduce its bed.")
        shown = [kind for kind, present in production.defects(spec).items()
                 if present and len(found[kind]) < args.per_defect]
        if not shown:
            continue
        # One pair per seed, for the defect with the fewest pairs so far: more distinct music.
        kind = min(shown, key=lambda name: len(found[name]))
        fixed = production.rebuild(spec, replace(production.profile.listener, **SWITCHES[kind][0]))
        if asdict(fixed) == asdict(spec):
            continue
        found[kind].append({"seed": seed, "family": spec.phrase.family, "bed_seed": spec.seed,
                            "baseline": spec, "fixed": fixed})
        print(f"  {kind:<16} seed {seed}  {spec.phrase.family}")

    pairs = [(kind, row) for kind, rows in found.items() for row in rows]
    order = random.Random(start)
    order.shuffle(pairs)
    clips, key = [], {}
    for index, (kind, row) in enumerate(pairs, 1):
        pair_id = f"p{index:02d}"
        fixed_side = order.choice("ab")
        sides = {fixed_side: row["fixed"], ("b" if fixed_side == "a" else "a"): row["baseline"]}
        for side, spec in sides.items():
            render_clip(spec, args.out / f"{pair_id}-{side}.mp3", args.seconds)
        clips.append({"id": pair_id, "listen_for": SWITCHES[kind][1],
                      "a": f"{pair_id}-a.mp3", "b": f"{pair_id}-b.mp3"})
        key[pair_id] = {"defect": kind, "fixed": fixed_side, "seed": row["seed"],
                        "family": row["family"], "bed_seed": row["bed_seed"]}
        print(f"rendered {pair_id} ({index}/{len(pairs)})")
    write_round(args.out, "pairs", clips, key)
    short = {kind: len(rows) for kind, rows in found.items() if len(rows) < args.per_defect}
    if short:
        print(f"Fewer pairs than asked for: {json.dumps(short)}")
    print(f"Wrote {len(clips)} pairs to {args.out}. Serve them with:\n"
          f"  uv run python -m scripts.listening.serve {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
