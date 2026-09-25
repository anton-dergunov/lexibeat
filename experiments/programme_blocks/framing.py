"""Section F: the programme around the words. Order, sections, title, intro, outro, and more ideas.

One call per lesson plans the episode: the teaching order with a reason for each place, the
sections and their spoken headers, the title, the intro and the outro. One more call writes a
concrete example of each further framing block the plan lists, and proposes three blocks of its
own. Encouragement is the one that is written once and reused, rather than once per episode.
"""

from __future__ import annotations

from common import LESSONS, NAMES, brief, learner_language, lesson, parallel, prompt
from common import write_report
import llm

SECTION = "framing"


def plan(name: str) -> dict:
    rows = lesson(name)
    language = rows[0]["language"]
    heads = {r["headword"] for r in rows}

    def check(reply):
        order = [o["headword"] for o in reply["order"]]
        llm.need(set(order) == heads and len(order) == 12, "the order is not the twelve words")
        placed = [h for s in reply["sections"] for h in s["headwords"]]
        llm.need(sorted(placed) == sorted(order), "the sections do not cover the order")
        llm.need(reply["intro"]["text"] and reply["outro"]["text"], "no intro or outro")
        return reply

    return llm.ask(prompt("framing", language=NAMES[language],
                          learner_language=NAMES[learner_language(language)],
                          words=[brief(r) for r in rows]), check=check)


def run() -> dict:
    names = list(LESSONS)
    cards = []
    for name, result in zip(names, parallel(plan, names, workers=3)):
        rows = lesson(name)
        gloss = {r["headword"]: r.get("primaryGloss") for r in rows}
        topics = {r["headword"]: ", ".join(r.get("topics") or []) for r in rows}
        episode = result["reply"]
        if not episode:
            cards.append({"id": f"F.plan.{name}", "title": name, "notes": "No usable plan.",
                          "details": [{"summary": "Why", "body": str(result["passed_over"])}]})
            continue
        why = {o["headword"]: o.get("why", "") for o in episode["order"]}
        script = [{"kind": "line", "speaker": "guide", "text": episode["intro"]["text"],
                   "direction": episode["intro"].get("direction", ""), "label": "intro"}]
        for section in episode["sections"]:
            script.append({"kind": "line", "speaker": "guide", "text": section["header"],
                           "label": "header"})
            for headword in section["headwords"]:
                script.append({"kind": "drill",
                               "text": f"{headword} — {gloss.get(headword)}",
                               "aside": " · ".join(x for x in (topics.get(headword),
                                                               why.get(headword)) if x)})
        script.append({"kind": "line", "speaker": "guide", "text": episode["outro"]["text"],
                       "direction": episode["outro"].get("direction", ""), "label": "outro"})
        items = [{"id": "title", "text": "title", "sub": episode.get("title", "")},
                 {"id": "intro", "text": "intro", "sub": episode["intro"]["text"]},
                 *[{"id": f"header.{i}", "text": "section header", "sub": s["header"]}
                   for i, s in enumerate(episode["sections"])],
                 {"id": "outro", "text": "outro", "sub": episode["outro"]["text"]},
                 {"id": "order", "text": "the order", "sub": "Did it keep look-alikes apart and "
                                                           "group what belongs together?"}]
        cards.append({"id": f"F.plan.{name}", "title": f"“{episode.get('title', '')}” · {name}",
                      "why": f"Answered by {result['model']} in {result['seconds']} s. The "
                             "episode as it would run, with each word's topic and the reason "
                             "for its place.",
                      "script": script, "items": items,
                      "details": [{"summary": "Prompt", "body": result["prompt"]}],
                      "rate": True})

    rows = lesson(names[0])
    language = rows[0]["language"]
    ideas = llm.ask(prompt("framing_ideas", language=NAMES[language],
                           learner_language=NAMES[learner_language(language)],
                           words=[brief(r) for r in rows]),
                    check=lambda reply: [b for b in reply["blocks"] if b.get("lines")])
    idea_cards = []
    for block in ideas["reply"] or []:
        idea_cards.append({
            "id": f"F.idea.{block['name']}", "title": block["name"].replace("_", " "),
            "why": f"{block.get('what', '')} {block.get('why', '')}".strip(),
            "script": [{"kind": "line", "speaker": l.get("speaker", "guide"),
                        "text": l.get("text", ""), "translation": l.get("translation", "")}
                       for l in block["lines"]],
            "rate": True})
    idea_cards.append({"id": "F.ideas.prompt", "title": "The ideas prompt",
                       "why": f"Answered by {ideas['model']} in {ideas['seconds']} s.",
                       "details": [{"summary": "Prompt", "body": ideas["prompt"]}],
                       "rate": False})
    report = {"section": "F", "title": "Framing",
              "intro": "The guide's lines around the words, and the order they come in. Text only.",
              "groups": [{"id": "plans", "title": "Three episodes planned", "cards": cards},
                         {"id": "ideas", "title": f"More framing blocks, for {names[0]}",
                          "cards": idea_cards}]}
    write_report(SECTION, report)
    return report
