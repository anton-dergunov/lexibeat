"""What a listening round said, and what it suggests changing.

    uv run python -m scripts.listening.summarise out/listening/round-00
    uv run python -m scripts.listening.summarise out/listening/round-01 [--variety]

For round 0 (blind pairs) it counts, per switch, how often the fixed clip won. For a labelling
round it ranks every concrete choice the engine made — a lead bank, a pad, a percussion sample, a
chord numeral, how high the lead went — by how often the part it belongs to was disliked, and
lists the choices worth ruling out. `--variety` then shows how many options each role still has
per family under the current production policy, so a rule that would leave a family with one lead
is seen before it is adopted.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load(root: Path, name: str) -> dict:
    path = root / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def pairs(root: Path, clips: list[dict], labels: dict) -> None:
    key = _load(root, "key.json")
    tally: dict[str, Counter] = defaultdict(Counter)
    notes: dict[str, list[str]] = defaultdict(list)
    for clip in clips:
        answer = labels.get(clip["id"], {})
        truth = key[clip["id"]]
        choice = answer.get("choice")
        outcome = ("unanswered" if not choice else "same" if choice == "same"
                   else "fixed" if choice == truth["fixed"] else "baseline")
        tally[truth["defect"]][outcome] += 1
        if answer.get("note"):
            notes[truth["defect"]].append(f"{clip['id']} ({outcome}): {answer['note']}")
    print("Round 0 — each switch against the engine as it is\n")
    for defect, counts in tally.items():
        fixed, baseline = counts["fixed"], counts["baseline"]
        verdict = ("ADOPT" if fixed > baseline else "keep as is" if baseline > fixed
                   else "no clear difference")
        print(f"  {defect:<16} fixed better {fixed}  as-is better {baseline}  "
              f"same {counts['same']}  unanswered {counts['unanswered']}  → {verdict}")
        for note in notes[defect]:
            print(f"      {note}")
    print("\nAdopting a switch means setting it on PRODUCTION_V1's `listener` in "
          "lexibeat/profiles.py and bumping ENGINE_VERSION; see docs/plans/music-listening.md.")


def _choices(clip: dict) -> dict[str, list[str]]:
    """The concrete things each part was made of, as the keys a dislike is counted against."""
    facts = clip["facts"]
    top = facts["lead"].get("top")
    band = "silent" if top is None else f"top note {top // 4 * 4}–{top // 4 * 4 + 3}"
    return {
        "lead": [facts["lead"]["source"].replace(" — not listener-approved", ""), band],
        "pad": [facts["pad"]["source"], f"texture {facts['pad']['texture']}",
                f"extension {facts['pad']['extension']}"],
        "bass": [facts["bass"]["source"], f"grammar {facts['bass']['grammar']}"],
        "percussion": facts["percussion"]["lanes"],
        "harmony": [facts["harmony"]["key"].split(" ", 1)[1],
                    *(f"chord {numeral}" for numeral in set(facts["harmony"]["progression"]))],
        "feel": [facts["feel"]["family"], facts["feel"]["meter"]],
    }


def parts(clips: list[dict], labels: dict, minimum: int) -> None:
    rated = [clip for clip in clips if labels.get(clip["id"], {}).get("overall")]
    print(f"{len(rated)} of {len(clips)} clips rated\n")
    if not rated:
        return
    overall = Counter(labels[clip["id"]]["overall"] for clip in rated)
    clean = [clip for clip in rated if not any(
        part.get("verdict") == "bad" for part in labels[clip["id"]].get("parts", {}).values())]
    print(f"Overall: keep {overall['keep']}  ok {overall['ok']}  reject {overall['reject']}")
    print(f"No part disliked: {len(clean)} of {len(rated)} ({len(clean) / len(rated):.0%})")
    replayed = [clip for clip in rated if clip.get("replayed")]
    if replayed:
        clean_replayed = sum(1 for clip in replayed if clip in clean)
        print(f"  of the replayed clips: {clean_replayed} of {len(replayed)} "
              "— compare with the round they came from")

    stats: dict[tuple[str, str], Counter] = defaultdict(Counter)
    reasons: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for clip in rated:
        judged = labels[clip["id"]].get("parts", {})
        for part, values in _choices(clip).items():
            verdict = judged.get(part, {}).get("verdict")
            if not verdict:
                continue
            for value in values:
                stats[(part, value)][verdict] += 1
                reasons[(part, value)].update(judged[part].get("reasons", []))

    print("\nChoices ranked by how often their part was disliked "
          f"(heard at least {minimum} times):\n")
    ranked = sorted(((key, counts) for key, counts in stats.items()
                     if sum(counts.values()) >= minimum),
                    key=lambda row: -row[1]["bad"] / sum(row[1].values()))
    for (part, value), counts in ranked:
        heard = sum(counts.values())
        why = ", ".join(f"{reason} {n}" for reason, n in reasons[(part, value)].most_common(3))
        print(f"  {counts['bad'] / heard:4.0%} disliked  {heard:>2} heard  {part:<10} {value}"
              + (f"   ({why})" if why else ""))

    proposals = [(part, value) for (part, value), counts in ranked
                 if counts["bad"] / sum(counts.values()) >= 0.6]
    if proposals:
        print("\nWorth ruling out (disliked in at least 60% of the clips they were in):")
        for part, value in proposals:
            print(f"  - {part}: {value}")
        print("A rule goes on PRODUCTION_V1's `listener` in lexibeat/profiles.py — a new kind of "
              "rule is a new ListenerPolicy field. Check --variety before adopting it.")


def variety(per_family: int) -> None:
    """How many distinct lead, pad and bass sources each family reaches under today's policy."""
    from lexibeat.profiles import POSITIVE_FAMILIES

    from .common import Production, facts

    production = Production()
    print(f"\nVariety under the current production policy ({per_family} beds per family, "
          "resolved without rendering):\n")
    print(f"  {'family':<17} leads  pads  basses")
    for family in POSITIVE_FAMILIES:
        found: dict[str, set[str]] = defaultdict(set)
        for seed in range(per_family):
            spec = production.rebuild_unrendered(family, seed)
            described = facts(production, spec)
            for part in ("lead", "pad", "bass"):
                found[part].add(described[part]["source"])
        warn = "   ← fewer than 3 leads" if len(found["lead"]) < 3 else ""
        print(f"  {family:<17} {len(found['lead']):>5} {len(found['pad']):>5} "
              f"{len(found['bass']):>7}{warn}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("round", type=Path)
    parser.add_argument("--minimum", type=int, default=2,
                        help="how many times a choice must be heard to be ranked")
    parser.add_argument("--variety", action="store_true")
    parser.add_argument("--per-family", type=int, default=60)
    args = parser.parse_args(argv)
    round_data = _load(args.round, "round.json")
    labels = _load(args.round, "labels.json")
    if round_data["kind"] == "pairs":
        pairs(args.round, round_data["clips"], labels)
    else:
        parts(round_data["clips"], labels, args.minimum)
    if args.variety:
        variety(args.per_family)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
