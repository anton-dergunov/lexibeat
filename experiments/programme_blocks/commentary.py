"""Section E: one line of commentary per word, from one call per lesson.

The owner's constraint: about twelve words a lesson, and one text call for all of them, not one per
kind per word. So one prompt lists every kind of remark, asks for candidates of only the kinds
that fit each word, and picks one (two at most). The candidates are kept and shown, rateable, under
the pick. That is the material a later judge or randomiser would choose from, gathered now without
building one.
"""

from __future__ import annotations

from common import NAMES, brief, learner_language, parallel, prompt, sample, write_report
import llm

SECTION = "commentary"
KINDS = {"register", "contrast", "false_friend", "pronunciation", "mnemonic", "joke", "culture"}
# About fifty words, a lesson's worth per call.
BATCHES = [("es", 13, 1), ("es", 12, 2), ("en", 13, 3), ("en", 12, 4)]


def run() -> dict:
    def one(batch):
        language, count, seed = batch
        rows = sample(language, count + 12, 500 + seed)[12:]  # clear of the lesson samples
        heads = {r["headword"] for r in rows}

        def check(reply):
            kept = []
            for entry in reply["words"]:
                if entry.get("headword") not in heads:
                    continue
                candidates = [c for c in entry.get("candidates", []) if c.get("kind") in KINDS]
                pick = [i for i in entry.get("pick", []) if 0 <= int(i) < len(candidates)]
                kept.append({**entry, "candidates": candidates, "pick": pick})
            llm.need(len(kept) >= len(heads) * 0.8, "too few words answered")
            return kept

        result = llm.ask(prompt("commentary", language=NAMES[language],
                                learner_language=NAMES[learner_language(language)],
                                words=[brief(r, notes=True) for r in rows]), check=check)
        return rows, result

    cards = []
    stats = {"words": 0, "abstained": 0, "two": 0, "factual": 0, "candidates": 0}
    kinds: dict[str, int] = {}
    prompts = []
    for (language, _, seed), (rows, result) in zip(BATCHES, parallel(one, BATCHES, workers=2)):
        gloss = {r["headword"]: r.get("primaryGloss") for r in rows}
        prompts.append({"id": f"E.prompt.{language}.{seed}",
                        "title": f"Call {len(prompts) + 1}: {len(rows)} {NAMES[language]} words",
                        "why": f"Answered by {result['model']} in {result['seconds']} s.",
                        "details": [{"summary": "Prompt", "body": result["prompt"]},
                                    {"summary": "Raw reply", "body": result["text"]}],
                        "rate": False})
        for entry in result["reply"] or []:
            stats["words"] += 1
            stats["candidates"] += len(entry["candidates"])
            stats["abstained"] += not entry["pick"]
            stats["two"] += len(entry["pick"]) > 1
            items = []
            order = entry["pick"] + [i for i in range(len(entry["candidates"]))
                                     if i not in entry["pick"]]
            for i in order:
                candidate = entry["candidates"][i]
                kinds[candidate["kind"]] = kinds.get(candidate["kind"], 0) + 1
                stats["factual"] += bool(candidate.get("factual"))
                items.append({"id": str(i),
                              "text": f"{'★ ' if i in entry['pick'] else ''}{candidate['kind']}"
                                      f"{' · ⚠ factual' if candidate.get('factual') else ''}",
                              "sub": candidate.get("text", ""),
                              "picked": i in entry["pick"]})
            cards.append({"id": f"E.word.{language}.{entry['headword']}",
                          "title": f"{entry['headword']} — {gloss.get(entry['headword'])}",
                          "why": (f"Abstained: {entry.get('abstain')}" if not entry["pick"] else
                                  f"{len(entry['candidates'])} candidates; ★ is the model's pick."),
                          "items": items, "collapse_after": len(entry["pick"]) or 0,
                          "rate": False})
    summary = (f"{stats['words']} words, {stats['candidates']} candidates "
               f"({stats['candidates'] / max(stats['words'], 1):.1f} a word). Picked two for "
               f"{stats['two']}, abstained on {stats['abstained']}. {stats['factual']} candidates "
               f"flagged as factual. By kind: "
               + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))
    report = {"section": "E", "title": "Commentary",
              "intro": "One remark per word, in the learner's language, from one call per lesson. "
                       "The pick (★) is shown first; tap “other candidates” for the rest. Rate the "
                       "pick, and any candidate you would rather have heard. " + summary,
              "groups": [{"id": "words", "title": "About fifty words", "cards": cards},
                         {"id": "prompts", "title": "The prompt and the four calls",
                          "cards": prompts}]}
    write_report(SECTION, report)
    return report
