"""Section D: the word in use, as text. Punchy examples, joint examples, mini-stories, callbacks.

Nothing here is spoken: the text is enough to judge whether a line would work, and it is where
the prompts get iterated. Every card carries its prompt, the model that answered and how long it
took. Three twelve-word lessons (`common.LESSONS`) are shared with the framing section, so a
lesson's story, callbacks and intro can be read against each other.
"""

from __future__ import annotations

from common import LESSONS, NAMES, brief, learner_language, lesson, parallel, prompt, sample
from common import write_report
import llm

SECTION = "context"

STYLES = {
    "narrator": ("A narrator's serial",
                 "Style: **a serial told by the native presenter**. Every line is spoken by "
                 "`native`, in {language}, as a storyteller: short sentences in the past tense, "
                 "one small event per beat, a little suspense between beats."),
    "dialogue": ("A two-presenter scene",
                 "Style: **a scene between the two presenters**, acted out in {language}. The "
                 "`native` presenter drives it; the `guide` is a slightly clueless companion who "
                 "also speaks {language}, in even shorter lines. It is a tiny sitcom scene, told "
                 "live, as if it were happening to them."),
}


def _listing(rows: list[dict]) -> list[dict]:
    return [brief(row) for row in rows]


def _words_check(heads: set[str]):
    def check(reply):
        entries = reply["words"]
        kept = [e for e in entries if e.get("headword") in heads and e.get("candidates")]
        llm.need(len(kept) >= len(heads) * 0.8, "too few words answered")
        return kept
    return check


def examples() -> list[dict]:
    groups = []
    for version in ("v1", "v2"):
        cards = []
        for language, seed in (("es", 11), ("en", 22)):
            rows = sample(language, 10, seed)
            learner = learner_language(language)
            result = llm.ask(prompt(f"example_punchy_{version}", language=NAMES[language],
                                    learner_language=NAMES[learner], count="three",
                                    words=_listing(rows)),
                             check=_words_check({r["headword"] for r in rows}))
            glosses = {r["headword"]: r.get("primaryGloss") for r in rows}
            for entry in result["reply"] or []:
                cards.append({
                    "id": f"D.example.{version}.{language}.{entry['headword']}",
                    "title": f"{entry['headword']} — {glosses.get(entry['headword'])}",
                    "why": f"{NAMES[language]}, prompt {version}",
                    "items": [{"id": str(i),
                               "text": c.get("text", "")
                                       + (f"  ⟶  {c['reaction']}" if c.get("reaction") else ""),
                               "sub": " · ".join(x for x in (
                                   c.get("translation", ""), c.get("kind", ""),
                                   f"🎙 {c.get('direction', '')}", c.get("why", "")) if x)}
                              for i, c in enumerate(entry["candidates"])],
                    "rate": False})
            cards.append(_prompt_card(f"D.example.{version}.{language}.prompt",
                                      f"Prompt {version}, {NAMES[language]} words", result))
        groups.append({"id": f"examples-{version}",
                       "title": f"Punchy examples, prompt {version}",
                       "intro": {"v1": "One-liners: a moment, not a definition, at most eight "
                                       "words. Three candidates per word; rate each.",
                                 "v2": "The same words, a second prompt: two presenters, five "
                                       "kinds of moment (confession, complaint, boast, slip, "
                                       "reaction), and sometimes a two-word reaction from the "
                                       "guide."}[version],
                       "cards": cards})
    return groups


def _prompt_card(card_id: str, title: str, result: dict) -> dict:
    passed = "; ".join(f"{p['model']}: {p['why'][:80]}" for p in result["passed_over"])
    return {"id": card_id, "title": title,
            "why": f"Answered by {result['model']} in {result['seconds']} s."
                   + (f" Passed over first: {passed}" if passed else ""),
            "details": [{"summary": "Prompt", "body": result["prompt"]},
                        {"summary": "Raw reply", "body": result["text"]}],
            "rate": True, "prompt_card": True}


def joint() -> dict:
    cards = []
    for name in LESSONS:
        rows = lesson(name)
        language = rows[0]["language"]
        heads = {r["headword"] for r in rows}

        def check(reply, heads=heads):
            groups = [g for g in reply["groups"] if set(g.get("words", [])) <= heads
                      and len(g.get("words", [])) >= 2]
            llm.need(bool(groups), "no usable groups")
            return groups

        result = llm.ask(prompt("example_joint", language=NAMES[language],
                                learner_language=NAMES[learner_language(language)],
                                words=_listing(rows)), check=check)
        cards.append({
            "id": f"D.joint.{name}", "title": f"{name}: {', '.join(r['headword'] for r in rows)}",
            "why": f"Answered by {result['model']}.",
            "items": [{"id": "+".join(g["words"]),
                       "text": f"[{' + '.join(g['words'])}]  {g.get('text', '')}",
                       "sub": " · ".join(x for x in (g.get("translation", ""),
                                                      f"🎙 {g.get('direction', '')}",
                                                      g.get("why", "")) if x)}
                      for g in result["reply"] or []],
            "details": [{"summary": "Prompt", "body": result["prompt"]}],
            "rate": True})
    return {"id": "joint", "title": "Joint examples",
            "intro": "One sentence tying two or three of a lesson's words together. The model "
                     "chooses which words combine. Rate each sentence, and the card for whether "
                     "it chose well.",
            "cards": cards}


def stories() -> dict:
    jobs = [(name, style) for name in LESSONS for style in STYLES]

    def one(job):
        name, style = job
        rows = lesson(name)
        language = rows[0]["language"]
        heads = {r["headword"] for r in rows}

        def check(reply):
            llm.need(set(reply["order"]) == heads, "the order is not the lesson's twelve words")
            for beat in reply["beats"]:
                llm.need(beat["after"] in heads and beat["lines"], "a beat is malformed")
            return reply

        return llm.ask(prompt("story", language=NAMES[language],
                              learner_language=NAMES[learner_language(language)],
                              style=STYLES[style][1].format(language=NAMES[language]),
                              words=_listing(rows)), check=check)

    cards = []
    for (name, style), result in zip(jobs, parallel(one, jobs, workers=3)):
        rows = lesson(name)
        gloss = {r["headword"]: r.get("primaryGloss") for r in rows}
        story = result["reply"]
        if not story:
            cards.append({"id": f"D.story.{name}.{style}", "title": f"{name}, {STYLES[style][0]}",
                          "notes": "No usable story came back.",
                          "details": [{"summary": "What went wrong",
                                       "body": str(result["passed_over"])}]})
            continue
        beats = {}
        for beat in story["beats"]:
            beats.setdefault(beat["after"], []).append(beat)
        script = []
        for headword in story["order"]:
            script.append({"kind": "drill", "text": f"{headword} — {gloss.get(headword)}"})
            for beat in beats.get(headword, []):
                for line in beat["lines"]:
                    script.append({"kind": "line", "speaker": line.get("speaker", "native"),
                                   "text": line.get("text", ""),
                                   "translation": line.get("translation", ""),
                                   "direction": line.get("direction", "")})
        words = sum(len(l["text"].split()) for l in script if l["kind"] == "line")
        cards.append({"id": f"D.story.{name}.{style}",
                      "title": f"“{story.get('title', '')}” · {name}, {STYLES[style][0]}",
                      "why": f"{len(story['beats'])} beats, {words} words of story, answered by "
                             f"{result['model']}. Shown as it would play: each drill, then the "
                             "beat that follows it.",
                      "script": script,
                      "details": [{"summary": "Prompt", "body": result["prompt"]}],
                      "rate": True})
    return {"id": "stories", "title": "Mini-stories",
            "intro": "A thread through the whole lesson, told in pieces between the drills. Each "
                     "lesson in two styles.",
            "cards": cards}


def callbacks() -> dict:
    cards = []
    for name in LESSONS:
        rows = lesson(name)
        language = rows[0]["language"]
        heads = [r["headword"] for r in rows]

        def check(reply, heads=heads):
            kept = [c for c in reply["callbacks"]
                    if c.get("at") in heads and c.get("lines")
                    and all(r in heads and heads.index(r) < heads.index(c["at"])
                            for r in c.get("refers_to", []))]
            llm.need(bool(kept), "no callback refers back correctly")
            return kept

        result = llm.ask(prompt("callback", language=NAMES[language],
                                learner_language=NAMES[learner_language(language)],
                                words=_listing(rows)), check=check)
        items = []
        for i, callback in enumerate(result["reply"] or []):
            lines = " / ".join(
                f"{l.get('speaker', '?')}: {l.get('text', '')}"
                + (f" ({l['translation']})" if l.get("translation") else "")
                for l in callback["lines"])
            earlier = f"earlier: “{callback['earlier']}” · " if callback.get("earlier") else ""
            items.append({"id": str(i),
                          "text": f"{callback['kind']} · after {callback['at']} → "
                                  f"{', '.join(callback.get('refers_to', []))}",
                          "sub": f"{earlier}{lines} · {callback.get('why', '')}"})
        cards.append({"id": f"D.callback.{name}", "title": f"{name}, taught in list order",
                      "why": f"Order: {', '.join(heads)}. Answered by {result['model']}.",
                      "items": items,
                      "details": [{"summary": "Prompt", "body": result["prompt"]}],
                      "rate": True})
    return {"id": "callbacks", "title": "Callbacks",
            "intro": "Bringing an earlier word back later so it has to be recalled: reuse, "
                     "contrast, quiz, chain and twist, rather than “remember the desert?”. Rate "
                     "each, and each card as a whole.",
            "cards": cards}


def run() -> dict:
    groups = examples() + [joint(), stories(), callbacks()]
    report = {"section": "D", "title": "Context", "groups": groups,
              "intro": "Text only. Spanish words are glossed in English and English words in "
                       "Russian, as in the vocabulary. Every prompt is under its group."}
    write_report(SECTION, report)
    return report
