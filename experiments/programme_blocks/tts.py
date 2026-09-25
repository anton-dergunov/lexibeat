"""Speech, reached the way Acervo reaches it in production — and a cache so nothing is bought twice.

Every Gemini and WaveNet voice goes to Cloud Text-to-Speech, `v1/text:synthesize`, with Application
Default Credentials and the quota project in `x-goog-user-project`. That is Acervo's `google-tts`
row, the head of both its `audioExpressive` and `audioPlain` chains. A Gemini voice is chosen with
`voice.modelName` and takes its direction in `input.prompt`. A WaveNet voice takes neither but reads
SSML. Cloudflare's Aura is reached by hand, as Acervo's `cloudflare.py` does, and speaks English only.

The answer is the LINEAR16 master, cached losslessly as FLAC under `out/cache/tts/`, keyed on
everything that was sent **and** the take index. The take belongs in the key because two takes of
one line can carry identical requests, and a cache without it would serve one recording twice.
Every call keeps its exact request body, with the project and account redacted, so the page can
show what was asked.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

from common import OUT

SYNTHESIZE = "https://texttospeech.googleapis.com/v1/text:synthesize"
AURA = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/deepgram/aura-1"
CACHE = OUT / "cache" / "tts"

GEMINI_31 = "gemini-3.1-flash-tts-preview"
GEMINI_25 = "gemini-2.5-flash-tts"

LOCALES = {"es": "es-ES", "en": "en-US", "ru": "ru-RU", "fr": "fr-FR", "de": "de-DE",
           "it": "it-IT", "zh": "cmn-CN"}
WAVENET = {"es": "es-ES-Wavenet-F", "en": "en-US-Wavenet-C", "ru": "ru-RU-Wavenet-A",
           "fr": "fr-FR-Wavenet-A", "de": "de-DE-Wavenet-A", "it": "it-IT-Wavenet-A",
           "zh": "cmn-CN-Wavenet-A"}

# When set, nothing is bought: an uncached request is counted and answered with silence.
DRY = False
counted: dict[str, float] = {"calls": 0, "characters": 0}
_would_buy: set[str] = set()
_lock = threading.Lock()
_token: dict = {}


@dataclass
class Take:
    audio: np.ndarray  # mono float32 at `sr`
    sr: int
    request: dict  # what was sent, redacted
    cached: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def seconds(self) -> float:
        return len(self.audio) / self.sr


def _project() -> str:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    if project:
        return project
    adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    if adc.exists():
        project = json.loads(adc.read_text()).get("quota_project_id", "")
    if not project:
        raise SystemExit("No quota project: set GOOGLE_CLOUD_PROJECT, or run "
                         "`gcloud auth application-default set-quota-project <project>`.")
    return project


def _google_headers() -> dict[str, str]:
    import google.auth
    import google.auth.transport.requests

    with _lock:
        if not _token or not _token["credentials"].valid:
            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"])
            credentials.refresh(google.auth.transport.requests.Request())
            _token.update(credentials=credentials, project=_project())
        return {"Authorization": f"Bearer {_token['credentials'].token}",
                "x-goog-user-project": _token["project"],
                "Content-Type": "application/json"}


def _key(provider: str, body: dict, take: int) -> str:
    blob = json.dumps({"provider": provider, "body": body, "take": take}, sort_keys=True,
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def _cached(key: str) -> Take | None:
    path = CACHE / f"{key}.flac"
    if not path.exists():
        return None
    audio, sr = sf.read(path, dtype="float32")
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return Take(audio, int(sr), meta["request"], True, meta)


def _store(key: str, take: Take) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.flac"
    partial = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.partial.flac")
    sf.write(partial, take.audio, take.sr, format="FLAC")
    os.replace(partial, path)
    path.with_suffix(".json").write_text(
        json.dumps({"request": take.request, **take.meta}, ensure_ascii=False, indent=1),
        encoding="utf-8")


def _decode(data: bytes) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return audio.mean(axis=1).astype(np.float32), int(sr)


def _post(url: str, headers: dict, body: dict, *, timeout: float = 90.0) -> httpx.Response:
    """One request, waiting out a 429 or a 5xx a few times. Anything else is a mistake to see."""
    for attempt in range(6):
        answer = httpx.post(url, headers=headers, json=body, timeout=timeout)
        if answer.status_code == 200:
            return answer
        if answer.status_code in (429, 500, 502, 503, 504) and attempt < 5:
            time.sleep(min(60, 4 * 2 ** attempt))
            continue
        raise RuntimeError(f"{url.split('/')[2]} refused ({answer.status_code}): "
                           f"{answer.text[:400]}")
    raise AssertionError("unreachable")


def _silence(key: str, request: dict, characters: int) -> Take:
    with _lock:
        if key not in _would_buy:
            _would_buy.add(key)
            counted["calls"] += 1
            counted["characters"] += characters
    return Take(np.zeros(12_000, dtype=np.float32), 24_000, request, False, {"dry": True})


def google(text: str | None = None, *, ssml: str | None = None, locale: str, voice: str,
           model: str | None = None, prompt: str | None = None, take: int = 0) -> Take:
    """One Cloud TTS call: a Gemini voice when `model` is given, a WaveNet one when it is not."""
    body: dict = {"input": {"ssml": ssml} if ssml else {"text": text},
                  "voice": {"languageCode": locale, "name": voice},
                  "audioConfig": {"audioEncoding": "LINEAR16"}}
    if model:
        body["voice"]["modelName"] = model
    if prompt:
        body["input"]["prompt"] = prompt
    key = _key("google-tts", body, take)
    request = {"POST": SYNTHESIZE, "headers": {"x-goog-user-project": "<quota project>"},
               "json": body}
    if hit := _cached(key):
        return hit
    if DRY:
        return _silence(key, request, len(ssml or text or "") + len(prompt or ""))
    started = time.perf_counter()
    answer = _post(SYNTHESIZE, _google_headers(), body)
    audio, sr = _decode(base64.b64decode(answer.json()["audioContent"]))
    result = Take(audio, sr, request, False,
                  {"provider": "google-tts", "model": model or voice.split("-")[2].lower(),
                   "voice": voice, "take": take,
                   "generation_seconds": round(time.perf_counter() - started, 2)})
    _store(key, result)
    return result


def aura(text: str, *, speaker: str | None = None, take: int = 0) -> Take:
    """Cloudflare's Deepgram Aura-1: English only, no language field, no direction."""
    body: dict = {"text": text}
    if speaker:
        body["speaker"] = speaker
    key = _key("cloudflare", body, take)
    request = {"POST": AURA.format(account="<account>"), "json": body}
    if hit := _cached(key):
        return hit
    if DRY:
        return _silence(key, request, len(text))
    started = time.perf_counter()
    url = AURA.format(account=os.environ["CLOUDFLARE_ACCOUNT_ID"])
    answer = _post(url, {"Authorization": f"Bearer {os.environ['CLOUDFLARE_API_TOKEN']}"}, body)
    data = answer.content
    if answer.headers.get("content-type", "").startswith("application/json"):
        data = base64.b64decode(answer.json()["result"]["audio"])
    audio, sr = _decode(data)
    result = Take(audio, sr, request, False,
                  {"provider": "cloudflare", "model": "@cf/deepgram/aura-1", "take": take,
                   "generation_seconds": round(time.perf_counter() - started, 2)})
    _store(key, result)
    return result


def gemini_api(text: str, *, model: str = "gemini-3.8-flash-tts", voice: str = "Kore",
               take: int = 0) -> Take:
    """The Gemini API's own TTS, on the free key: no language code at all, the language detected.

    Not a production voice (Acervo reaches Gemini voices through Cloud TTS), and its free allowance
    is a few calls a day; it is here because it is the newest model and the only one that is
    documented to detect the language by itself.
    """
    body = {"model": model, "contents": text,
            "config": {"response_modalities": ["AUDIO"],
                       "speech_config": {"voice_config": {"prebuilt_voice_config":
                                                          {"voice_name": voice}}}}}
    key = _key("gemini-api", body, take)
    request = {"POST": f"generativelanguage.googleapis.com … models/{model}:generateContent",
               "json": body}
    if hit := _cached(key):
        return hit
    if DRY:
        return _silence(key, request, len(text))
    from google import genai
    from google.genai import types

    started = time.perf_counter()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for attempt in range(4):
        try:
            answer = client.models.generate_content(
                model=model, contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)))))
            break
        except Exception as failure:
            if attempt < 3 and any(c in str(failure) for c in ("429", "503", "UNAVAILABLE")):
                time.sleep(20 * (attempt + 1))
                continue
            raise
    part = answer.candidates[0].content.parts[0].inline_data
    if part.mime_type.startswith("audio/wav"):
        audio, sr = _decode(part.data)
    else:  # headerless 16-bit PCM, 24 kHz
        audio, sr = np.frombuffer(part.data, dtype="<i2").astype(np.float32) / 32768, 24_000
    result = Take(audio, sr, request, False,
                  {"provider": "gemini-api", "model": model, "voice": voice, "take": take,
                   "generation_seconds": round(time.perf_counter() - started, 2)})
    _store(key, result)
    return result


def write_clip(path: Path, audio: np.ndarray, sr: int) -> Path:
    """A speech-only clip for the page, as MP3 like a loop."""
    from lexibeat.loop import write_mp3

    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if peak > 0:
        audio = audio / peak * 0.9
    write_mp3(path, np.stack([audio, audio], axis=1).astype(np.float32), sr)
    return path
