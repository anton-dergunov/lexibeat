"""Text from the Gemini free tier, strongest model first.

Acervo's `gemini-free` row names only the two lite models, which are fast and cheap on the daily
allowance. Asked to syllabify, they put the stress in the wrong place about one word in four, so
the stronger free models go first here and the lite ones are the fallback. The model that answered
is recorded on every card.

JSON *mode* and nothing more: `response_mime_type` asks for a JSON object, the shape is stated in
the prompt, and each stage checks the reply with its own parser. There is deliberately no schema.
Both repositories measured constrained decoding and found that it moves a model's drafting into the
fields a reader sees. A reply that will not parse, or that the stage's check refuses, falls through
to the next model. Replies are cached by prompt, so rebuilding the page, or re-running a stage
whose prompt did not change, costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Callable

from common import OUT

STRONG = ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash")
LITE = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
CACHE = OUT / "cache" / "llm"
_client = None


class Unusable(ValueError):
    """The reply parsed, or did not, but is not what the prompt asked for."""


def _gemini():
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def _parse(text: str) -> Any:
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    return json.loads(fenced.group(1) if fenced else text)


def ask(prompt: str, *, check: Callable[[Any], Any] | None = None,
        temperature: float | None = None) -> dict:
    """Ask for a JSON object. Returns `{reply, model, seconds, prompt, text, passed_over}`.

    `check` receives the parsed reply and returns what the stage keeps. It raises `Unusable` (or any
    ValueError, KeyError or TypeError) to pass the reply over to the next model.

    The strong models are overloaded often ("high demand", 503), so each one is given two more
    tries before the next is asked. An answer from a lite model is kept, but it is not final: the
    next run asks the strong models once more before settling for it.
    """
    key = hashlib.sha256(json.dumps([prompt, temperature]).encode()).hexdigest()[:32]
    path = CACHE / f"{key}.json"
    passed_over: list[dict] = []
    fallback = None
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        try:
            stored["reply"] = check(_parse(stored["text"])) if check else _parse(stored["text"])
            if stored["model"] in STRONG:
                return stored
            fallback = stored
        except (ValueError, KeyError, TypeError) as refused:
            passed_over.append({"model": stored["model"], "why": f"cached: {refused}"})
    walk = [(m, 0 if fallback else 2) for m in STRONG] + ([] if fallback else [(m, 1) for m in LITE])
    for model, retries in walk:
        result = _one(model, prompt, check, temperature, retries, passed_over)
        if result is not None:
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({k: v for k, v in result.items() if k != "reply"},
                                       ensure_ascii=False, indent=1), encoding="utf-8")
            return result
    if fallback:
        return fallback
    return {"model": None, "seconds": 0, "prompt": prompt, "text": "", "reply": None,
            "passed_over": passed_over}


def _one(model: str, prompt: str, check, temperature, retries: int,
         passed_over: list[dict]) -> dict | None:
    from google.genai import types

    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            answer = _gemini().models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json",
                                                   temperature=temperature))
        except Exception as failure:  # the SDK raises one class per status
            text = str(failure)
            if attempt < retries and any(code in text for code in
                                         ("429", "500", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED")):
                time.sleep(20 * (attempt + 1))
                continue
            passed_over.append({"model": model, "why": text[:300]})
            return None
        seconds = round(time.perf_counter() - started, 1)
        text = answer.text or ""
        try:
            reply = check(_parse(text)) if check else _parse(text)
        except (ValueError, KeyError, TypeError) as refused:
            passed_over.append({"model": model, "why": f"unusable: {refused}",
                                "text": text[:2000]})
            return None
        return {"model": model, "seconds": seconds, "prompt": prompt, "text": text,
                "passed_over": list(passed_over), "reply": reply}
    return None


def shown(result: dict) -> dict:
    """What a report keeps of a call: the prompt, who answered, how long, and what went wrong."""
    return {"prompt": result["prompt"], "model": result["model"], "seconds": result["seconds"],
            "passed_over": result["passed_over"], "raw": result["text"]}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise Unusable(message)

