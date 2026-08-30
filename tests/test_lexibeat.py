from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from lexibeat.api import MusicRequest, render_music, resolve_music
from lexibeat.bedspec import BedSpec
from lexibeat.arrange import arrange
from lexibeat.emotion import EMOTIONS, NEUTRAL, VECTOR_ORDER
from lexibeat.instruments import CatalogMultiSampleInstrument, SampledInstrument
from lexibeat.library import (EXTERNAL_LIMIT, InstrumentRef, SampleAsset,
                              SampleLibrary, SampleRef, directory_size,
                              instrument_refs)
from lexibeat.mix import duck_envelope, mix_stems
from lexibeat.music import Grid, SR, filter_curve, render_bed, render_stems
from lexibeat.samples import PACKS, Sample, SamplePack, midi, missing
from lexibeat.voice import (
    CAPABILITIES,
    CloudflareAura2Backend,
    CloudflareMeloBackend,
    DEFAULT_MODELS,
    FISH_TAGS,
    GeminiBackend,
    IndexTTS25Backend,
    MlxAudioBackend,
    Prosody,
    Qwen3Backend,
    Speaker,
    SynthesisResult,
    delivery_instruction,
    gemini_prompt,
    reference_path,
)
from lexibeat.vocab import Item
from benchmark_voices import pressure_snapshot, write_comparison
from compare_gemini_batched import split_on_long_silences
from compare_beds import Candidate, FAMILIES, POSITIVE_FAMILIES, select_balanced
from lexibeat.sfz import parse as parse_sfz


class VoiceTests(unittest.TestCase):
    def test_experimental_capabilities_and_defaults_are_explicit(self) -> None:
        expected = {"indextts25", "voxcpm2", "qwen3", "tada", "fish-s2"}
        self.assertTrue(expected.issubset(DEFAULT_MODELS))
        self.assertTrue(all(CAPABILITIES[name].experimental for name in expected))
        self.assertEqual(CAPABILITIES["tada"].emotion, "unsupported")
        self.assertEqual(CAPABILITIES["indextts25"].emotion, "8-float vector")

    def test_index_emotion_vector_keeps_official_order(self) -> None:
        vector = EMOTIONS["surprised"].vector()
        self.assertEqual(len(vector), 8)
        self.assertEqual(VECTOR_ORDER[6], "surprised")
        self.assertEqual(vector[6], EMOTIONS["surprised"].exaggeration)
        self.assertEqual(sum(value != 0 for value in vector), 1)

    def test_instruction_and_fish_tag_map_emotion_and_prosody(self) -> None:
        text = delivery_instruction(EMOTIONS["sad"],
                                    Prosody(speed=0.96, semitones=-0.4))
        self.assertIn("sad", text)
        self.assertIn("slowly", text)
        self.assertIn("lower", text)
        self.assertEqual(FISH_TAGS["emphatic"], "emphasis")

    def test_hosted_capabilities_and_defaults_are_explicit(self) -> None:
        hosted = {"gemini", "gemini-vertex", "cloudflare-aura2",
                  "cloudflare-melotts"}
        self.assertTrue(hosted.issubset(DEFAULT_MODELS))
        self.assertTrue(all(CAPABILITIES[name].experimental for name in hosted))
        self.assertEqual(CAPABILITIES["gemini"].emotion, "instruction")
        self.assertEqual(CAPABILITIES["cloudflare-aura2"].rate,
                         "post-process")

    def test_gemini_prompt_fences_exact_transcript(self) -> None:
        prompt = gemini_prompt("¿Dónde está?", "es", EMOTIONS["thoughtful"],
                               Prosody(speed=0.96, semitones=-0.35))
        self.assertIn("native Spanish", prompt)
        self.assertIn("thoughtful", prompt)
        self.assertIn("slightly slowly", prompt)
        self.assertIn("<TRANSCRIPT>\n¿Dónde está?\n</TRANSCRIPT>", prompt)
        self.assertIn("Speak only the transcript", prompt)

    def test_gemini_batch_prompt_treats_pause_tags_as_silence(self) -> None:
        prompt = gemini_prompt("uno\n[long pause]\ndos", "es", NEUTRAL,
                               Prosody())
        self.assertIn("silent timing instruction", prompt)
        self.assertIn("do not speak the tag", prompt)

    def test_batched_gemini_split_uses_longest_silences(self) -> None:
        rate = 1000
        tone = np.sin(2 * np.pi * 30 * np.arange(180) / rate).astype(np.float32)
        short_internal_gap = np.zeros(40, dtype=np.float32)
        phrase = np.concatenate((tone, short_internal_gap, tone))
        long_gap = np.zeros(700, dtype=np.float32)
        audio = np.concatenate([piece for index in range(10)
                                for piece in ((phrase, long_gap)
                                              if index < 9 else (phrase,))])
        segments, metadata = split_on_long_silences(
            audio, 10, sample_rate=rate, min_silence_seconds=0.35)
        self.assertEqual(len(segments), 10)
        self.assertGreaterEqual(metadata["shortest_pause_seconds"], 0.35)
        self.assertGreater(metadata["confidence"], 2.0)

    def test_gemini_routes_voice_and_decodes_pcm_with_cost(self) -> None:
        calls: list[dict] = []
        pcm = (np.array([0, 16384, -16384], dtype="<i2")).tobytes()

        class Interactions:
            def create(self, **kwargs):
                calls.append(kwargs)
                output = types.SimpleNamespace(
                    data=base64.b64encode(pcm).decode(), sample_rate=24000,
                    channels=1,
                    mime_type="audio/l16; rate=24000; channels=1")
                usage = types.SimpleNamespace(total_input_tokens=20,
                                              total_output_tokens=30)
                return types.SimpleNamespace(output_audio=output, usage=usage)

        backend = GeminiBackend.__new__(GeminiBackend)
        backend._client = types.SimpleNamespace(interactions=Interactions())
        backend.model_id = "gemini-3.1-flash-tts-preview"
        backend.voices = {"es": "Sulafat", "en": "Achird"}
        backend.sample_rate = 24000
        result = backend.synth("hola", "es", Prosody(), EMOTIONS["warm"], seed=9)
        self.assertEqual(calls[0]["generation_config"]["speech_config"],
                         [{"voice": "Sulafat"}])
        self.assertIn("<TRANSCRIPT>\nhola\n</TRANSCRIPT>", calls[0]["input"])
        np.testing.assert_allclose(result.audio, [0.0, 0.5, -0.5])
        self.assertEqual(result.controls["model"],
                         "gemini-3.1-flash-tts-preview")
        self.assertGreater(result.controls["estimated_cost_usd"], 0)
        self.assertFalse(result.controls["seed_supported"])

    def test_gemini_vertex_routes_adc_model_locale_and_pcm(self) -> None:
        calls: list[dict] = []
        pcm = np.array([0, 8192, -8192], dtype="<i2").tobytes()

        def record(kind):
            return lambda **kwargs: {"kind": kind, **kwargs}

        fake_types = types.SimpleNamespace(
            GenerateContentConfig=record("config"),
            SpeechConfig=record("speech"),
            VoiceConfig=record("voice"),
            PrebuiltVoiceConfig=record("prebuilt"),
        )

        class Models:
            def generate_content(self, **kwargs):
                calls.append(kwargs)
                inline = types.SimpleNamespace(
                    data=pcm, mime_type="audio/L16;rate=24000;channels=1")
                part = types.SimpleNamespace(inline_data=inline)
                candidate = types.SimpleNamespace(
                    content=types.SimpleNamespace(parts=[part]))
                usage = types.SimpleNamespace(prompt_token_count=12,
                                              candidates_token_count=25)
                return types.SimpleNamespace(candidates=[candidate],
                                             usage_metadata=usage)

        backend = GeminiBackend.__new__(GeminiBackend)
        backend._client = types.SimpleNamespace(models=Models())
        backend._types = fake_types
        backend.vertex = True
        backend.location = "global"
        backend.model_id = "gemini-2.5-flash-tts"
        backend.voices = {"es": "Sulafat", "en": "Achird"}
        backend.sample_rate = 24000
        backend.capabilities = CAPABILITIES["gemini-vertex"]
        backend._min_request_interval = 0.0
        backend._last_request_started = None
        result = backend.synth("hola", "es", Prosody(), EMOTIONS["warm"])
        self.assertEqual(calls[0]["model"], "gemini-2.5-flash-tts")
        speech = calls[0]["config"]["speech_config"]
        self.assertEqual(speech["language_code"], "es-US")
        self.assertEqual(speech["voice_config"]["prebuilt_voice_config"]
                         ["voice_name"], "Sulafat")
        np.testing.assert_allclose(result.audio, [0.0, 0.25, -0.25])
        self.assertEqual(result.controls["provider"], "vertex-ai")
        self.assertGreater(result.controls["estimated_cost_usd"], 0)

    def test_gemini_adapts_to_preview_rate_limit(self) -> None:
        class RateLimitError(Exception):
            status_code = 429
            response = types.SimpleNamespace(headers={})

        interactions = mock.Mock()
        interactions.create.side_effect = [RateLimitError("quota"), "ok"]
        backend = GeminiBackend.__new__(GeminiBackend)
        backend._client = types.SimpleNamespace(interactions=interactions)
        backend.model_id = "gemini-3.1-flash-tts-preview"
        backend._min_request_interval = 0.0
        backend._last_request_started = None
        with mock.patch("lexibeat.voice.time.monotonic", side_effect=[0.0, 0.0]), \
                mock.patch("lexibeat.voice.time.sleep") as sleep:
            self.assertEqual(backend._generate("prompt", "voice"), "ok")
        sleep.assert_called_once_with(20.5)
        self.assertEqual(backend._min_request_interval, 20.5)

    def test_gemini_does_not_retry_daily_quota(self) -> None:
        class DailyLimitError(Exception):
            status_code = 429

        interactions = mock.Mock()
        interactions.create.side_effect = DailyLimitError(
            "GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        backend = GeminiBackend.__new__(GeminiBackend)
        backend._client = types.SimpleNamespace(interactions=interactions)
        backend.model_id = "gemini-3.1-flash-tts-preview"
        backend._min_request_interval = 0.0
        backend._last_request_started = None
        with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
            backend._generate("prompt", "voice")
        interactions.create.assert_called_once()

    @staticmethod
    def _wav_bytes(audio: np.ndarray, rate: int = 24000) -> bytes:
        buffer = io.BytesIO()
        sf.write(buffer, audio, rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    def test_cloudflare_aura_routes_language_speaker_and_cost(self) -> None:
        calls: list[tuple[str, dict]] = []
        backend = CloudflareAura2Backend.__new__(CloudflareAura2Backend)
        backend.model_id = "@cf/deepgram/aura-2-{lang}"
        backend.voices = {"es": "aquila", "en": "luna"}
        backend.sample_rate = 24000
        backend._request = lambda model, payload: (
            calls.append((model, payload)) or
            (self._wav_bytes(np.array([0.0, 0.25], dtype=np.float32)),
             "audio/wav"))
        result = backend.synth("hola", "es", Prosody(speed=0.96),
                               EMOTIONS["warm"], seed=4)
        self.assertEqual(calls[0][0], "@cf/deepgram/aura-2-es")
        self.assertEqual(calls[0][1]["speaker"], "aquila")
        self.assertEqual(calls[0][1]["container"], "wav")
        self.assertAlmostEqual(result.controls["estimated_cost_usd"], 0.00012)
        self.assertFalse(result.controls["native_prosody_supported"])

    def test_cloudflare_decodes_json_envelope_and_retries(self) -> None:
        audio = self._wav_bytes(np.array([0.0, 0.2], dtype=np.float32))

        class Response:
            def __init__(self, status, *, envelope=None, content=b"",
                         content_type="application/json"):
                self.status_code = status
                self.ok = status < 400
                self.headers = {"content-type": content_type}
                self.content = content
                self.text = json.dumps(envelope or {})
                self._envelope = envelope

            def json(self):
                return self._envelope

        responses = [Response(429), Response(200, envelope={
            "success": True,
            "result": {"audio": base64.b64encode(audio).decode(),
                       "content_type": "audio/wav"},
        })]
        backend = CloudflareMeloBackend.__new__(CloudflareMeloBackend)
        backend.account_id = "account"
        backend.api_token = "token"
        backend._session = types.SimpleNamespace(
            post=lambda *args, **kwargs: responses.pop(0))
        with mock.patch("lexibeat.voice.time.sleep") as sleep:
            decoded, content_type = backend._request("@cf/model", {"prompt": "hi"})
        self.assertEqual(decoded, audio)
        self.assertEqual(content_type, "audio/wav")
        sleep.assert_called_once()

    def test_cloudflare_melo_rejects_broken_spanish_path_without_call(self) -> None:
        backend = CloudflareMeloBackend.__new__(CloudflareMeloBackend)
        backend.model_id = "@cf/myshell-ai/melotts"
        backend._request = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "AiError 8002"):
            backend.synth("hola", "es", Prosody(), EMOTIONS["warm"])
        backend._request.assert_not_called()

    def test_cloudflare_melo_caches_text_before_local_variation(self) -> None:
        backend = CloudflareMeloBackend.__new__(CloudflareMeloBackend)
        backend.model_id = "@cf/myshell-ai/melotts"
        backend._audio_cache = {}
        backend._request = mock.Mock(return_value=(
            self._wav_bytes(np.array([0.0, 0.25], dtype=np.float32)),
            "audio/wav"))
        first = backend.synth("hello", "en", Prosody(), EMOTIONS["warm"])
        second = backend.synth("hello", "en", Prosody(semitones=0.4),
                               EMOTIONS["warm"])
        backend._request.assert_called_once()
        self.assertFalse(first.controls["cache_hit"])
        self.assertTrue(second.controls["cache_hit"])
        self.assertGreater(first.controls["estimated_cost_usd"], 0)
        self.assertEqual(second.controls["estimated_cost_usd"], 0)

    def test_cloudflare_errors_redact_credentials(self) -> None:
        secret = "super-secret-token"

        class Response:
            status_code = 401
            ok = False
            headers = {"content-type": "application/json"}
            text = f'{{"error":"bad {secret}"}}'

        backend = CloudflareMeloBackend.__new__(CloudflareMeloBackend)
        backend.account_id = "account"
        backend.api_token = secret
        backend._session = types.SimpleNamespace(post=lambda *a, **k: Response())
        with mock.patch.dict(os.environ, {"CLOUDFLARE_API_TOKEN": secret}):
            with self.assertRaisesRegex(RuntimeError, "<redacted>") as caught:
                backend._request("@cf/model", {"prompt": "hi"})
        self.assertNotIn(secret, str(caught.exception))

    def test_cloudflare_post_processing_is_recorded(self) -> None:
        class FakeBackend:
            name = "cloudflare-aura2"
            model_id = "fake"
            sample_rate = SR
            load_seconds = 0.0

            def synth(self, text, lang, prosody, emotion, target_seconds=None,
                      seed=None):
                return SynthesisResult(np.ones(100, dtype=np.float32), SR, 0.1,
                                       {"native_prosody_supported": False})

        with mock.patch("lexibeat.voice.make_backend", return_value=FakeBackend()), \
                mock.patch("lexibeat.voice.warnings.warn"), \
                mock.patch("lexibeat.voice.librosa.effects.time_stretch",
                           return_value=np.ones(90, dtype=np.float32)) as stretch, \
                mock.patch("lexibeat.voice._pitch_shift",
                           side_effect=lambda audio, *_: audio) as pitch:
            speaker = Speaker(backend="cloudflare-aura2", voice_seed=3)
            speaker.say("hola", "es", Prosody(
                speed=1.02, semitones=0.4, gain_db=0.6))
        stretch.assert_called_once()
        pitch.assert_called_once()
        applied = speaker.stats[0]["controls"]["local_post_process"]
        self.assertEqual(applied, {
            "speed": 1.02, "semitones": 0.4, "gain_db": 0.6})

    def test_qwen_uses_language_voice_instruction_and_seed(self) -> None:
        calls = []

        class Result:
            audio = np.ones(32, dtype=np.float32)
            sample_rate = 24000

        class Model:
            def generate(self, **kwargs):
                calls.append(kwargs)
                yield Result()

        backend = Qwen3Backend.__new__(Qwen3Backend)
        backend._model = Model()
        backend.voices = {"es": "Serena", "en": "Ryan"}
        backend.sample_rate = 24000
        result = backend.synth("hola", "es", Prosody(), EMOTIONS["happy"], seed=11)
        self.assertEqual(result.sample_rate, 24000)
        self.assertEqual(calls[0]["voice"], "Serena")
        self.assertEqual(calls[0]["lang_code"], "spanish")
        self.assertIn("happy", calls[0]["instruct"])

    def test_speaker_records_target_fit_and_deterministic_seed(self) -> None:
        class FakeBackend:
            name = "voxcpm2"
            model_id = "fake"
            sample_rate = SR
            load_seconds = 0.0
            calls = []

            def synth(self, text, lang, prosody, emotion, target_seconds=None,
                      seed=None):
                self.calls.append((target_seconds, seed))
                return SynthesisResult(np.ones(SR * 3, dtype=np.float32), SR,
                                       0.1, {"native": True})

        backend = FakeBackend()
        with mock.patch("lexibeat.voice.make_backend", return_value=backend), \
                mock.patch("lexibeat.voice.warnings.warn"):
            speaker = Speaker(backend="voxcpm2", voice_seed=90)
            audio = speaker.say("hola", "es", target_seconds=1.0)
        self.assertLessEqual(len(audio), SR)
        self.assertEqual(backend.calls, [(1.0, 90)])
        self.assertTrue(speaker.stats[0]["post_fit_applied"])

    def test_index_worker_request_carries_vector_and_duration_target(self) -> None:
        class Input:
            def __init__(self):
                self.value = ""

            def write(self, value):
                self.value += value

            def flush(self):
                pass

        class Process:
            stdin = Input()

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "worker.wav"
            sf.write(output, np.ones(100, dtype=np.float32), 22050)
            backend = IndexTTS25Backend.__new__(IndexTTS25Backend)
            backend._tmp = types.SimpleNamespace(name=tmp)
            backend._process = Process()
            backend.ref_audios = {"es": "/ref.wav"}
            backend.sample_rate = 22050
            backend.close = lambda: None
            backend._read_response = lambda: {
                "generation_seconds": 0.2, "passes": 2,
                "peak_memory_bytes": 123,
                "controls": {"duration_factor": 0.8},
            }
            with mock.patch("lexibeat.voice.hashlib.sha1") as sha:
                sha.return_value.hexdigest.return_value = output.stem
                result = backend.synth("hola", "es", Prosody(speed=1.25),
                                       EMOTIONS["happy"], target_seconds=1.1,
                                       seed=3)
            request = json.loads(backend._process.stdin.value)
            self.assertAlmostEqual(request["duration_factor"], 0.8)
            self.assertEqual(request["target_seconds"], 1.1)
            self.assertEqual(request["emotion"], EMOTIONS["happy"].vector())
            self.assertEqual(result.passes, 2)
    def test_reference_cache_is_language_specific(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
                os.environ, {"LEXIBEAT_CACHE": tmp}):
            self.assertEqual(reference_path("say:Paulina", "es").name,
                             "say-Paulina-es.wav")
            self.assertEqual(reference_path("say:Daniel", "en").name,
                             "say-Daniel-en.wav")

    def test_chatterbox_routes_native_reference_by_language(self) -> None:
        calls: list[dict] = []

        class Result:
            audio = np.ones(16, dtype=np.float32)
            sample_rate = 24000

        class Model:
            sample_rate = 24000

            def generate(self, **kwargs):
                calls.append(kwargs)
                yield Result()

        utils = types.ModuleType("mlx_audio.tts.utils")
        utils.load_model = lambda _: Model()
        modules = {
            "mlx_audio": types.ModuleType("mlx_audio"),
            "mlx_audio.tts": types.ModuleType("mlx_audio.tts"),
            "mlx_audio.tts.utils": utils,
        }
        resolved = lambda source, lang: Path(f"/{lang}-{source.split(':')[-1]}.wav")
        with mock.patch.dict(sys.modules, modules), mock.patch(
                "lexibeat.voice.ensure_reference", side_effect=resolved):
            backend = MlxAudioBackend()
            backend.synth("hola", "es", Prosody(), NEUTRAL)
            backend.synth("hello", "en", Prosody(), NEUTRAL)

        self.assertEqual(calls[0]["ref_audio"], "/es-Paulina.wav")
        self.assertEqual(calls[1]["ref_audio"], "/en-Daniel.wav")
        self.assertEqual(calls[0]["lang_code"], "es")
        self.assertEqual(calls[1]["lang_code"], "en")


class BedSpecTests(unittest.TestCase):
    def test_legacy_dict_gets_compatible_defaults(self) -> None:
        spec = BedSpec.from_dict({"bpm": 80, "pad": {"level": 0.4}})
        self.assertEqual((spec.beats_per_bar, spec.beat_unit), (4, 4))
        self.assertEqual(spec.chord_extension, "none")
        self.assertEqual(spec.pad.instrument, "synth")
        self.assertEqual(spec.pad.duck_db, 5.0)

    def test_json_round_trip_preserves_new_fields(self) -> None:
        spec = BedSpec.from_style("nocturne", 9)
        rebuilt = BedSpec.from_dict(json.loads(spec.to_json()))
        self.assertEqual(rebuilt.to_json(), spec.to_json())

    def test_extensions_add_notes(self) -> None:
        base = BedSpec(chord_extension="none").chord(0)
        seventh = BedSpec(chord_extension="seventh").chord(0)
        add9 = BedSpec(chord_extension="add9").chord(0)
        ninth = BedSpec(chord_extension="ninth").chord(0)
        self.assertGreater(len(seventh), len(base))
        self.assertGreater(len(add9), len(base))
        self.assertEqual(set(ninth), set(seventh) | set(add9))

    def test_supported_meters_have_dynamic_grids(self) -> None:
        for beats, expected in ((3, 12), (4, 16), (5, 20)):
            spec = BedSpec(bpm=60, beats_per_bar=beats, beat_unit=4)
            grid = Grid.from_spec(spec)
            self.assertEqual(grid.steps_per_bar, expected)
            self.assertAlmostEqual(grid.bar, float(beats))
            self.assertAlmostEqual(grid.step_time(1, 0), grid.bar)

    def test_swing_uses_meter_specific_step_length(self) -> None:
        grid = Grid(bpm=60, beats_per_bar=3, beat_unit=4, swing=0.25)
        self.assertAlmostEqual(grid.step_time(0, 2), 0.5 + 0.25 * 0.25)

    def test_styles_are_seeded_and_keep_speech_sized_bars(self) -> None:
        meters = set()
        for seed in range(30):
            first = BedSpec.from_style("yoga", seed)
            second = BedSpec.from_style("yoga", seed)
            self.assertEqual(first.to_json(), second.to_json())
            self.assertGreaterEqual(Grid.from_spec(first).bar, 2.2)
            meters.add(first.beats_per_bar)
        self.assertEqual(meters, {3, 4, 5})

    def test_filter_curves_are_bounded_and_deterministic(self) -> None:
        for name in ("sine", "triangle", "random_walk"):
            spec = BedSpec()
            spec.pad.cutoff_curve = name
            a = filter_curve(spec, 8, 100, np.random.default_rng(4))
            b = filter_curve(spec, 8, 100, np.random.default_rng(4))
            np.testing.assert_allclose(a, b)
            self.assertGreaterEqual(float(a.min()), 120.0)
            self.assertLessEqual(float(a.max()), SR * 0.45)

    def test_wide_styles_resolve_and_round_trip_complete_phrases(self) -> None:
        for style in FAMILIES:
            first = BedSpec.from_style(style, 17)
            second = BedSpec.from_style(style, 17)
            self.assertIsNotNone(first.phrase)
            self.assertEqual(first.to_json(), second.to_json())
            rebuilt = BedSpec.from_dict(json.loads(first.to_json()))
            self.assertEqual(rebuilt.to_json(), first.to_json())
            self.assertGreaterEqual(first.phrase.loop_bars, 4)
            self.assertTrue(first.phrase.chords)
            self.assertTrue(first.phrase.bass)


class PublicGenerationApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        request = MusicRequest(
            family="warm-motion", palette="electronic", seed=1234)
        cls.first = resolve_music(request)
        cls.second = resolve_music(request)

    def test_request_validation_and_round_trip(self) -> None:
        request = MusicRequest.from_dict(MusicRequest().to_dict())
        self.assertEqual(request, MusicRequest())
        with self.assertRaisesRegex(ValueError, "Unknown family"):
            MusicRequest(family="not-a-family").validated()

    def test_fixed_request_is_fully_deterministic_and_versioned(self) -> None:
        self.assertEqual(self.first.bed_spec.to_json(), self.second.bed_spec.to_json())
        self.assertEqual(self.first.fingerprint, self.second.fingerprint)
        self.assertEqual(self.first.engine_version, "1.0.0")
        self.assertEqual(self.first.profile_version, "production-v1")
        self.assertEqual(self.first.bed_spec.profile_version, "production-v1")
        self.assertTrue(self.first.quality.accepted)

    def test_rendered_api_audio_is_finite_stereo_and_bounded(self) -> None:
        audio = render_music(self.first, duration_seconds=8.0)
        self.assertEqual(audio.ndim, 2)
        self.assertEqual(audio.shape[1], 2)
        self.assertTrue(np.isfinite(audio).all())
        self.assertLessEqual(float(np.abs(audio).max()), 0.700001)

    def test_resolution_does_not_use_network(self) -> None:
        with mock.patch("urllib.request.urlopen") as network:
            resolve_music(MusicRequest(
                family="playful-minimal", palette="electronic", seed=77))
        network.assert_not_called()

    def test_resolved_synth_phrase_pcm_is_deterministic(self) -> None:
        spec = BedSpec.from_style("meditative", 3)
        spec.pad.instrument = "synth"
        spec.lead.instrument = "synth"
        spec.phrase.pad_sample = None
        spec.phrase.lead_sample = None
        first = render_stems(spec, 1)
        second = render_stems(spec, 1)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])


class SampleTests(unittest.TestCase):
    def test_pack_manifests_cover_requested_instruments(self) -> None:
        self.assertEqual(midi("F#3"), 54)
        self.assertTrue({"salamander", "vsco-marimba", "vsco-glockenspiel",
                         "vsco-strings"}.issubset(PACKS))
        self.assertTrue(all(entry.remote_path for pack in PACKS.values()
                            for entry in pack.entries()))

    def test_sampled_instrument_uses_manifest_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
                os.environ, {"LEXIBEAT_CACHE": tmp}):
            entry = Sample("tone.wav", "remote/tone.wav", 60, 1, "test")
            pack = SamplePack("test", "CC0", "test", "https://example.test",
                              "https://example.test/", (entry,))
            directory = Path(tmp) / "samples" / "test"
            directory.mkdir(parents=True)
            sf.write(directory / entry.filename,
                     np.sin(2 * np.pi * 261.6 * np.arange(SR) / SR), SR)
            self.assertFalse(missing(pack))
            audio = SampledInstrument(pack).render(60, 0.5, 0.2)
            self.assertEqual(len(audio), int(0.2 * SR))
            self.assertAlmostEqual(float(np.abs(audio).max()), 0.5, places=3)


class TieredLibraryTests(unittest.TestCase):
    def test_shipped_bundle_resolves_without_local_or_external_tier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            library = SampleLibrary(
                base / "missing-external", base / "empty-local", use_bundled=True)
            assets = library.assets()
            self.assertTrue(assets)
            resolved = library.resolve(assets[0].ref)
            self.assertTrue(resolved.exists())
            self.assertIn("production-core", str(resolved))

    def test_catalog_can_open_when_default_external_tier_is_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
                os.environ, {"LEXIBEAT_LIBRARY_ROOT": ""}), mock.patch(
                "lexibeat.library.external_root",
                return_value=Path(tmp) / "missing-volume" / "library"):
            external = Path(tmp) / "missing-volume" / "library"
            library = SampleLibrary(local=Path(tmp) / "local")
            self.assertEqual(library.assets(), [])
            self.assertFalse(external.exists())
            self.assertFalse(library.status()["external"]["available"])

    def test_index_resolve_promote_and_external_offline_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            library = SampleLibrary(base / "external", base / "local")
            source = library.collection_path("freepats-world") / "samples" / "hit_C4.wav"
            source.parent.mkdir(parents=True)
            sf.write(source, np.sin(2 * np.pi * 261.6 * np.arange(SR // 10) / SR), SR)
            self.assertEqual(library.index("freepats-world", deep=True), 1)
            asset = library.assets()[0]
            self.assertEqual(asset.midi_note, 60)
            self.assertEqual(library.resolve(asset.ref), source)
            promoted = library.promote([asset.ref])[0]
            self.assertTrue(promoted.exists())
            self.assertEqual(promoted.suffix, ".wav")
            source.unlink()
            self.assertEqual(library.resolve(asset.ref), promoted)
            promoted.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Attach the external SSD"):
                library.resolve(asset.ref)

    def test_external_quota_is_scoped_and_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = SampleLibrary(Path(tmp) / "external", Path(tmp) / "local")
            with mock.patch("lexibeat.library.directory_size",
                            return_value=EXTERNAL_LIMIT):
                with self.assertRaisesRegex(RuntimeError, "exceed"):
                    library._check_budget("external", 1)
            self.assertEqual(directory_size(Path(tmp) / "unmanaged"), 0)

    def test_demo_audio_is_quarantined_from_normal_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            library = SampleLibrary(base / "external", base / "local")
            demo = library.collection_path("vcsl") / "Piano Demo.wav"
            demo.parent.mkdir(parents=True)
            sf.write(demo, np.zeros(SR // 20), SR)
            library.index("vcsl", deep=True)
            self.assertEqual(library.assets(), [])
            quarantined = library.assets(usable_only=False)
            self.assertEqual(len(quarantined), 1)
            self.assertTrue(quarantined[0].quarantined)

    def test_sfz_regions_inherit_group_and_report_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instrument.sfz"
            path.write_text(
                "<global> lovel=1 hivel=100\n"
                "<group> pitch_keycenter=C4 unsupported_filter=2\n"
                "<region> sample=Samples/Tone C4.wav lokey=C4 hikey=C4 seq_length=2 "
                "seq_position=1\n", encoding="utf-8")
            document = parse_sfz(path)
            self.assertEqual(len(document.zones), 1)
            self.assertEqual(document.zones[0].key_center, 60)
            self.assertEqual(document.zones[0].lo_vel, 1)
            self.assertIn("unsupported_filter", document.unsupported_opcodes)

    def test_catalog_samples_form_serializable_multisample_instrument(self) -> None:
        assets = [SampleAsset(
            collection="vcsl", asset_id=f"id-{note}", sha256=f"hash-{note}",
            relative_path=f"Grand Piano/Sustains/Piano_C{note - 60}.wav",
            license="CC0-1.0", category="pitched", midi_note=note,
            duration_seconds=2.0)
            for note in range(60, 66)]
        instruments = instrument_refs(assets)
        self.assertEqual(len(instruments), 1)
        self.assertEqual(len(instruments[0].zones), 6)
        spec = BedSpec.from_style("radiant", 4)
        spec.phrase.lead_instrument = instruments[0]
        spec.phrase.bass_instrument = instruments[0]
        rebuilt = BedSpec.from_dict(json.loads(spec.to_json()))
        self.assertEqual(rebuilt.phrase.lead_instrument, instruments[0])
        self.assertEqual(rebuilt.phrase.bass_instrument, instruments[0])

    def test_front_instrument_groups_include_mbira_but_not_vibrato_sax(self) -> None:
        assets = []
        for note in range(60, 66):
            assets.extend([
                SampleAsset("vcsl", f"mbira-{note}", f"hash-mbira-{note}",
                            f"Idiophones/Mbira Zimbabwe/Mbira_{note}_C{note - 60}.wav",
                            "CC0-1.0", "pitched", midi_note=note,
                            duration_seconds=1.0),
                SampleAsset("vcsl", f"sax-{note}", f"hash-sax-{note}",
                            f"Aerophones/Tenor Saxophone/Vibrato/Sax_{note}_C{note - 60}.wav",
                            "CC0-1.0", "pitched", midi_note=note,
                            duration_seconds=1.0),
            ])
        names = [instrument.name.lower() for instrument in instrument_refs(assets)]
        self.assertTrue(any("mbira" in name for name in names))
        self.assertFalse(any("saxophone" in name for name in names))

    def test_legacy_promotions_are_only_removed_after_checksum_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            library = SampleLibrary(base / "external", base / "local")
            source = library.collection_path("freepats-world") / "samples" / "hit_C4.wav"
            source.parent.mkdir(parents=True)
            sf.write(source, np.ones(100, dtype=np.float32) * 0.1, SR)
            library.index("freepats-world", deep=True)
            asset = library.assets()[0]
            promoted = library.promote([asset.ref])[0]
            legacy = promoted.with_suffix("")
            legacy.write_bytes(promoted.read_bytes())
            result = library.migrate_legacy_promotions()
            self.assertFalse(legacy.exists())
            self.assertEqual(result["removed_duplicates"], [str(legacy)])
            self.assertTrue(promoted.exists())

    def test_multisample_renderer_selects_nearest_zone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            library = SampleLibrary(base / "external", base / "local")
            root = library.collection_path("vcsl") / "Piano"
            root.mkdir(parents=True)
            for index, pitch in enumerate(("C4", "D4", "E4", "F4", "G4", "A4")):
                sf.write(root / f"tone_{pitch}.wav",
                         np.sin(2 * np.pi * (220 + index * 17) *
                                np.arange(SR) / SR), SR)
            library.index("vcsl", deep=True)
            instrument = instrument_refs(library.assets())[0]
            audio = CatalogMultiSampleInstrument(instrument, library).render(
                65, 0.5, 0.15)
            self.assertEqual(len(audio), int(0.15 * SR))
            self.assertTrue(np.isfinite(audio).all())


class BedSelectionTests(unittest.TestCase):
    def test_selector_balances_families_and_is_deterministic(self) -> None:
        candidates = []
        for family_index, family in enumerate(FAMILIES):
            for seed in range(3):
                spec = BedSpec.from_style(family, seed)
                candidates.append(Candidate(
                    family, seed, spec,
                    np.array([family_index, seed, family_index * seed], dtype=float),
                    8.0, ()))
        first = select_balanced(candidates, 12)
        second = select_balanced(candidates, 12)
        self.assertEqual([(row.family, row.seed) for row in first],
                         [(row.family, row.seed) for row in second])
        self.assertEqual({family: sum(row.family == family for row in first)
                          for family in FAMILIES}, {family: 2 for family in FAMILIES})

    def test_positive_families_are_straight_and_anchor_every_bar(self) -> None:
        for family in POSITIVE_FAMILIES:
            spec = BedSpec.from_style(family, 19)
            self.assertLessEqual(spec.swing, 0.025)
            lane = spec.phrase.percussion[0]
            self.assertTrue(all(lane.pattern[bar * spec.steps_per_bar] == "x"
                                for bar in range(spec.phrase.loop_bars)))
            self.assertEqual(lane.role, "low")
            for boundary in range(spec.steps_per_bar, len(lane.pattern),
                                  spec.steps_per_bar):
                self.assertNotEqual(lane.pattern[boundary - 1:boundary + 2], "xxx")
                self.assertEqual(lane.pattern[boundary - 1], ".")
            self.assertLessEqual(spec.drums.level, 0.64)

    def test_positive_piano_phrases_have_a_useful_register(self) -> None:
        checked = 0
        for family in POSITIVE_FAMILIES:
            for seed in range(20):
                spec = BedSpec.from_style(family, seed)
                if spec.lead.instrument != "piano":
                    continue
                notes = [event.midi_note for event in spec.phrase.lead]
                self.assertGreaterEqual(len(set(notes)), 2)
                self.assertGreaterEqual(max(notes) - min(notes), 7)
                self.assertLessEqual(spec.lead.duck_db, 7.0)
                self.assertGreaterEqual(spec.lead.register[1] - spec.lead.register[0], 24)
                checked += 1
        self.assertGreater(checked, 0)


class RenderAndMixTests(unittest.TestCase):
    def _spec(self, beats: int = 4) -> BedSpec:
        spec = BedSpec(bpm=72, beats_per_bar=beats, beat_unit=4, seed=3)
        spec.pad.instrument = "synth"
        spec.lead.instrument = "synth"
        spec.space.reverb_seconds = 0.15
        spec.space.reverb_mix = 0.1
        return spec

    def test_render_stems_sum_to_bed_shape_for_each_meter(self) -> None:
        for beats in (3, 4, 5):
            spec = self._spec(beats)
            stems = render_stems(spec, 1)
            bed = render_bed(spec, 1)
            self.assertEqual(set(stems), {"pad", "bass", "drums", "lead"})
            self.assertEqual(bed.shape, next(iter(stems.values())).shape)
            self.assertTrue(np.isfinite(bed).all())

    def test_sampled_pad_handles_non_integral_tempo_rounding(self) -> None:
        spec = self._spec()
        spec.bpm = 66
        spec.pad.instrument = "strings"
        stems = render_stems(spec, 1)
        self.assertTrue(np.isfinite(stems["pad"]).all())

    def test_per_layer_ducking_and_limiter(self) -> None:
        seconds = 3
        n = seconds * SR
        t = np.arange(n) / SR
        speech = np.zeros(n, dtype=np.float32)
        speech[SR:2 * SR] = 0.2 * np.sin(2 * np.pi * 220 * t[:SR])
        shallow = duck_envelope(speech, 2.0)
        deep = duck_envelope(speech, 8.0)
        self.assertLess(float(deep[int(1.5 * SR)]),
                        float(shallow[int(1.5 * SR)]))
        tone = (0.05 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
        stems = {"pad": np.stack([tone, tone], axis=1),
                 "bass": np.stack([tone, tone], axis=1)}
        track = mix_stems(stems, speech, {"pad": 8.0, "bass": 2.0})
        self.assertTrue(np.isfinite(track).all())
        self.assertLessEqual(float(np.abs(track).max()), 0.97001)

    def test_arrangement_passes_slot_duration_and_uses_downbeats(self) -> None:
        class FakeSpeaker:
            prosody_strength = 1.0
            targets: list[float] = []

            def say(self, text, lang, prosody, emotion, target_seconds=None):
                self.targets.append(target_seconds)
                return np.ones(100, dtype=np.float32)

        grid = Grid(bpm=60, beats_per_bar=4, beat_unit=4)
        speaker = FakeSpeaker()
        events, _ = arrange([Item("hola", "hello")], speaker, grid,
                            progress=False)
        self.assertEqual(len(events), 6)
        self.assertTrue(all(event.start % grid.bar == 0 for event in events))
        self.assertTrue(all(value == grid.bar * 0.92
                            for value in speaker.targets))


class BenchmarkTests(unittest.TestCase):
    def test_pressure_snapshot_parses_swap_for_numeric_delta(self) -> None:
        outputs = iter([
            "System-wide memory free percentage: 27%",
            "total = 4096.00M  used = 1536.50M  free = 2559.50M",
            "Pages free: 42.",
        ])
        with mock.patch("benchmark_voices.command_output",
                        side_effect=lambda _: next(outputs)):
            snapshot = pressure_snapshot()
        self.assertEqual(snapshot["free_percent"], 27)
        self.assertEqual(snapshot["swap_used_bytes"], int(1536.5 * 2**20))

    def test_comparison_schema_handles_success_and_failure_without_models(self) -> None:
        rows = [
            {
                "backend": "indextts25", "success": True,
                "timing": {"model_load_seconds": 1.0,
                           "model_generation_seconds": 2.0},
                "benchmark_wall_seconds": 3.0,
                "audio": {"duration_seconds": 4.0},
                "memory": {"mlx_peak_bytes": 1024},
                "system_memory": {"peak_process_tree_rss_bytes": 2048,
                                  "lowest_free_percent": 25},
            },
            {"backend": "tada", "success": False},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_comparison(directory, rows)
            payload = json.loads((directory / "comparison.json").read_text())
            markdown = (directory / "comparison.md").read_text()
        self.assertEqual([row["backend"] for row in payload],
                         ["indextts25", "tada"])
        self.assertIn("| tada | failed |", markdown)
        self.assertIn("does not certify pronunciation", markdown)


if __name__ == "__main__":
    unittest.main()
