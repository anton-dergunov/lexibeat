# The loop service — `/api/v1`

LexiBeat makes **loops**: a handful of words, each spoken over a bar grid with a silence in the
middle to recall the answer in, over a bed that replays byte-identically from a style and a seed.
This document is the contract a host integrates against. `docs/openapi-v1.json` is the machine
copy, committed so a wire change shows up in a diff; a test asserts the two agree.

`docs/design.md` and `docs/music-generation.md` are the research behind the engine and are not this.

## The shape of it, and why

**A render is an operation.** Twelve words is seventy-two speech calls plus a bed and a mix — minutes,
not seconds. `POST /api/v1/loops` answers an operation id and the caller follows it. Progress is
watchable, cancellation lands between utterances, and no connection is held open for four minutes.

**The speech backend is injected.** LexiBeat holds no provider credential, no model chain and no
voice map. The host holds all three, and copying them into this process would mean a second copy of
every key on the same machine. `create_service(backend_factory=…)` takes a callable that is handed
the validated request and whatever render-scoped credential travelled with it, and returns anything
satisfying the `Backend` protocol. **Dispatch reads capabilities, never names**, so a backend this
package has never heard of works exactly as well as one it ships.

**Liveness is not readiness.** `/health` answers 200 with no sample bundle present. The engine then
offers only the sample-free `electronic` palette and says so in `production_bundle`. A readiness gate
would fail the installation of a service that is working correctly.

**The catalogues are LexiBeat's.** `/schema` reports the patterns, the families, the limits and the
audio format. A host reads them rather than copying them, so a family added in a later version
appears in its dialog with nothing changing there.

## Routes

| Route | Answers |
|---|---|
| `GET /api/v1/health` | `{status, api_version, engine_version, production_bundle}` |
| `GET /api/v1/schema` | patterns, profiles, families, energy, rhythm, palette, limits, audio |
| `POST /api/v1/loops` | `202` with an operation |
| `GET /api/v1/operations/{id}` | the operation, with `result` once it has completed |
| `DELETE /api/v1/operations/{id}` | cancels between utterances |
| `GET /api/v1/loops/{id}/audio` | the finished track, `audio/mpeg` |

### `POST /api/v1/loops`

```json
{
  "items": [
    {"source": "asco", "target": "disgust", "direction": "repulsed, recoiling slightly"},
    {"source": "la rabia", "target": "rage", "direction": "angry, with heat behind it"}
  ],
  "source_language": {"code": "es", "name": "Spanish"},
  "target_language": {"code": "en", "name": "English"},
  "pattern": "retrieval",
  "family": "auto",
  "seed": 4711,
  "speech": {"token": "…", "delivery": "directed"}
}
```

`direction` is **free text** and may be empty. It is appended to the per-take prosody words to make
the director note a model actually receives. There is no enum, no emoji column and no punctuation
heuristic: the caller has read the word and this service has not. An **empty** direction does not
mean a bare reading: the note then opens with this service's own default, because a word nobody has
written a delivery for is the common case rather than the exceptional one, and an instruction to be
merely clear is an instruction to be flat.

`source_language` and `target_language` are a `code` and a `name`. The code picks the voice; the
name goes into the director note. A bare string is accepted and stands in for both.

`speech.token` is the host's render-scoped credential, passed to `backend_factory` and to nothing
else. It is registered with the provider-text redactor, and it appears in no operation, no result
and no log line.

`speech.delivery` is **opaque to this service** and reaches `backend_factory` as `RenderContext.delivery`,
exactly as it arrived. It exists because only the host knows what its own voice can do, and the
backend it builds declares capabilities accordingly: a voice that takes a director note asks for one
recording per repetition and lets the note vary them, while a voice that cannot asks for one
recording per line and lets this service vary it by pitch and speed. Neither side branches on the
other's vocabulary; the declaration is the whole interface.

### The operation

```json
{
  "operation_id": "1ab747c8bdfc4f93a5e898aa65042091",
  "status": "running",
  "successful": null,
  "error": null,
  "progress": {"fraction": 0.42, "message": "Synthesizing 7 of 12: Spanish — asco"},
  "created_at": 1758150000.0,
  "updated_at": 1758150042.0,
  "result": null
}
```

`status` is `queued`, `running`, `completed`, `failed` or `cancelled`. `successful` is `true` only
for `completed`. A backend that raises fails **the operation**, not the process.

The completed `result`:

```json
{
  "audio_url": "/api/v1/loops/1ab7…/audio",
  "audio_mime": "audio/mpeg",
  "bitrate_kbps": 128,
  "duration_seconds": 124.53,
  "pattern": "retrieval",
  "style_id": "acoustic-flow",
  "seed": 104740,
  "engine_version": "1.4.0",
  "profile_version": "1.4.0",
  "bed_fingerprint": "90c6ad267d159b0e",
  "total_bars": 68,
  "bpm": 80.0,
  "timeline": [ … ]
}
```

The **resolved BedSpec is deliberately not in it**. Style, seed and engine version replay the bed
exactly, and `bed_fingerprint` is what proves a replay produced the same one — so sending kilobytes
of JSON on every poll would only tempt a host into storing an opaque blob it does not need.

A timeline row is one word, with the text denormalised into it on purpose: it records what was
*said*, so editing the word afterwards cannot make a player caption a recording that no longer
matches.

```json
{
  "index": 0, "source": "asco", "target": "disgust",
  "direction": "repulsed, recoiling slightly",
  "start": 8.82, "source_reveal": 8.82, "target_reveal": 17.65, "end": 44.12,
  "utterances": [{"role": "source", "repetition": 0, "start": 8.82, "end": 10.1}, …]
}
```

`source_reveal` and `target_reveal` are what a retrieval display turns on: the answer must not be on
screen before the recall gap has passed.

## Audio

**MP3 at 128 kbps, constant bitrate**, written by `soundfile` over libsndfile with no ffmpeg and no
extra dependency. libsndfile exposes quality as a 0–1 `compression_level` rather than a bitrate, so
the pair is measured rather than assumed: against libsndfile 1.2.2, `bitrate_mode="CONSTANT"` maps
the level onto LAME's own bitrate ladder and `0.65` is the rung that is 128 kbps. Measured on a
124.5-second loop: 1,993,664 bytes, **128.1 kbps**. The constants live together in `lexibeat/loop.py`
and a test re-measures them on every run.

MP3 rather than a better codec because the content is speech over a generated bed, where 128 kbps is
inaudible from the master, and because a phone decodes it in hardware and seeks in it trivially.

## The `Backend` protocol

```python
class Backend(Protocol):
    name: str
    sample_rate: int
    capabilities: BackendCapabilities
    load_seconds: float
    model_id: str

    def synth(self, request: SpeechRequest) -> SynthesisResult: ...
```

`SpeechRequest` carries `text`, a `Language`, a `Delivery` (`take`, `direction`, `prosody`), an
optional `target_seconds` and an optional `seed`. **`take` is on the request**, which matters to a
host that caches recordings: two takes of one word can produce identical director notes at low
prosody strength, and a cache keyed without the take index would hand back one recording for both
and make the repetition sound *more* mechanical.

`BackendCapabilities` is the whole of what dispatch reads:

| Field | Effect |
|---|---|
| `emotion` | `"post-process"` pitch-shifts locally; `"exaggeration"` varies takes model-side |
| `rate` | `"post-process"` time-stretches locally; `"unsupported"` enables the long-take retry |
| `voice` | documentation only |
| `languages` | a declaration; **empty means any**, and a language not named is refused by name |

## Running it

```python
from lexibeat.service import ServiceConfig, create_service

app = create_service(config=ServiceConfig(output_root=Path("/var/lib/lexibeat")),
                     backend_factory=lambda context: MyBackend(context.credentials))
```

Then `uvicorn` it. There is no `lexibeat-service serve`: a process with no backend has nothing to
run, and a default backend would be exactly the provider credential this package refuses to hold.
`lexibeat-service openapi` refreshes the committed snapshot.

`LEXIBEAT_SERVICE_OUT`, `LEXIBEAT_SERVICE_QUEUE` and `LEXIBEAT_SERVICE_RETAIN` configure the output
root, how many renders may be outstanding, and how many finished tracks are kept.
`LEXIBEAT_BUNDLE_ROOT` points at the sample bundle, which is fetched once with
`lexibeat-bundle fetch --into DIR --from URL --sha256 DIGEST`.
