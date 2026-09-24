"""The voice inside a loop: render it the same way twice, measure what the mix does to it, and pair.

    uv run python -m scripts.listening.voice_check render --label before
    # … change the speech path …
    uv run python -m scripts.listening.voice_check render --label after
    uv run python -m scripts.listening.voice_check pair before after
    uv run python -m scripts.listening.serve out/listening/voice-pairs

A render needs no provider. The voice is a directory of real recorded takes (the README demo's
speech cache by default), handed out by a file-backed backend, so the same loop, bed and takes come
out of every run and the only thing that differs between two labels is the code in between.

Each render makes the loop in both deliveries — `directed`, which the host uses by default and which
leaves a take's pitch and speed alone, and `plain`, where the engine varies them itself — and writes,
per delivery: the finished MP3, the speech stem on its own, and what the master limiter did:
how much of the track it turned down, and by how much at most.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

import lexibeat.mix as mixing
from lexibeat.language import ENGLISH, SPANISH
from lexibeat.loop import LoopRequest, render_loop
from lexibeat.music import SR
from lexibeat.vocab import Item
from lexibeat.voice import BackendCapabilities, SpeechRequest, SynthesisResult

TAKES = Path("out/readme-demo/speech-cache")
OUT = Path("out/listening/voice")
DELIVERIES = {
    "directed": BackendCapabilities("instruction", "instruction", "fixed", languages=()),
    "plain": BackendCapabilities("post-process", "post-process", "fixed", languages=()),
}


class FileVoice:
    """Real takes from disk. Each line gets the same recording on every run; the longest ones go to
    the first words, so a fast bed has to squeeze and overflow some of them."""

    name = "file-voice"
    load_seconds = 0.0
    model_id = "recorded-takes"

    def __init__(self, takes: Path, items: list[Item], capabilities: BackendCapabilities) -> None:
        files = sorted(takes.glob("*.wav"), key=lambda path: (-sf.info(path).duration, path.name))
        if len(files) < 2 * len(items):
            raise SystemExit(f"Need at least {2 * len(items)} takes in {takes}.")
        self.by_text = {}
        long, rest = files[:len(items)], files[len(items):]
        random.Random(7).shuffle(rest)
        for index, item in enumerate(items):
            self.by_text[item.source] = long[index] if index < 3 else rest[2 * index]
            self.by_text[item.target] = rest[2 * index + 1]
        self.capabilities = capabilities
        self.sample_rate = sf.info(files[0]).samplerate

    def synth(self, request: SpeechRequest) -> SynthesisResult:
        audio, rate = sf.read(self.by_text[request.text], dtype="float32", always_2d=True)
        return SynthesisResult(audio=audio.mean(axis=1), sample_rate=rate, generation_seconds=0.0)


ITEMS = [Item(f"palabra {index}", f"word {index}", "curious") for index in range(6)]


def _limiter_report(before: np.ndarray, after: np.ndarray) -> dict:
    """How hard the master limiter worked: per 5 ms block, the gain it applied."""
    step = int(SR * 0.005)
    blocks = len(before) // step
    peak_in = np.abs(before[:blocks * step]).reshape(blocks, step, -1).max(axis=(1, 2))
    peak_out = np.abs(after[:blocks * step]).reshape(blocks, step, -1).max(axis=(1, 2))
    live = peak_in > 1e-4
    gain = np.ones_like(peak_in)
    gain[live] = peak_out[live] / peak_in[live]
    gain_db = 20 * np.log10(np.clip(gain, 1e-6, None))
    return {
        "blocks_turned_down_share": round(float(np.mean(gain_db[live] < -0.1)), 3),
        "blocks_down_3db_share": round(float(np.mean(gain_db[live] < -3.0)), 3),
        "most_reduction_db": round(float(gain_db[live].min()), 2),
        "samples_at_ceiling": int(np.sum(np.abs(after) >= 0.9699)),
    }


def render(label: str, takes: Path, family: str, seed: int) -> None:
    out = OUT / label
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for delivery, capabilities in DELIVERIES.items():
        captured: dict = {}
        original_limit, original_mix = mixing.limit, render_loop.__globals__["mix_stems"]

        def limit(audio, *args, **kwargs):
            result = original_limit(audio, *args, **kwargs)
            captured["limiter"] = _limiter_report(audio, result)
            return result

        def mix_stems(stems, speech, *args, **kwargs):
            captured["speech"] = speech
            return original_mix(stems, speech, *args, **kwargs)

        original_speaker = render_loop.__globals__["Speaker"]

        class Speaker(original_speaker):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured["speaker"] = self

        mixing.limit = limit
        render_loop.__globals__["mix_stems"] = mix_stems
        render_loop.__globals__["Speaker"] = Speaker
        try:
            voice = FileVoice(takes, ITEMS, capabilities)
            request = LoopRequest(items=tuple(ITEMS), source_language=SPANISH,
                                  target_language=ENGLISH, family=family, seed=seed)
            result = render_loop(request, backend=voice, output=out / f"{delivery}.mp3")
        finally:
            mixing.limit = original_limit
            render_loop.__globals__["mix_stems"] = original_mix
            render_loop.__globals__["Speaker"] = original_speaker
        speech = captured["speech"]
        sf.write(out / f"{delivery}-speech.wav", speech, SR)
        # How many takes had to be squeezed into their bar, and how hard.
        stats = captured["speaker"].stats
        ratios = [row["duration_before_fit"] / row["target_seconds"] for row in stats
                  if row.get("target_seconds") and row["duration_before_fit"] > row["target_seconds"]]
        report[delivery] = {"bpm": result.bpm, "style": result.style_id,
                            "speech_peak": round(float(np.abs(speech).max()), 3),
                            "takes": len(stats), "squeezed": len(ratios),
                            "hardest_squeeze": round(max(ratios, default=1.0), 2),
                            "over_the_cap": sum(1 for ratio in ratios if ratio > 1.35),
                            **captured["limiter"]}
        print(f"{label}/{delivery}: {json.dumps(report[delivery])}")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def pair(before: str, after: str, out: Path) -> None:
    """A blind round of before/after pairs for `serve`, the answer kept in `key.json`."""
    out.mkdir(parents=True, exist_ok=True)
    clips, key = [], {}
    for index, delivery in enumerate(DELIVERIES, 1):
        pair_id = f"p{index:02d}"
        after_side = "a" if int(hashlib.sha256(pair_id.encode()).hexdigest(), 16) % 2 else "b"
        before_side = "b" if after_side == "a" else "a"
        shutil.copy(OUT / after / f"{delivery}.mp3", out / f"{pair_id}-{after_side}.mp3")
        shutil.copy(OUT / before / f"{delivery}.mp3", out / f"{pair_id}-{before_side}.mp3")
        clips.append({"id": pair_id, "a": f"{pair_id}-a.mp3", "b": f"{pair_id}-b.mp3",
                      "listen_for": "the voice: clear or metallic, and whether long words are cut"})
        key[pair_id] = {"defect": f"voice ({delivery})", "fixed": after_side}
    (out / "round.json").write_text(json.dumps({"kind": "pairs", "clips": clips}, indent=2))
    (out / "key.json").write_text(json.dumps(key, indent=2))
    print(f"Wrote {len(clips)} pairs to {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("render")
    one.add_argument("--label", required=True)
    one.add_argument("--takes", type=Path, default=TAKES)
    # A fast family, so a bar is short and long takes have to be squeezed or overflow.
    one.add_argument("--family", default="radiant")
    one.add_argument("--seed", type=int, default=4242)
    two = commands.add_parser("pair")
    two.add_argument("before")
    two.add_argument("after")
    two.add_argument("--out", type=Path, default=Path("out/listening/voice-pairs"))
    args = parser.parse_args(argv)
    if args.command == "render":
        # The same bundled catalogue a host renders with, never a local one on this machine.
        with tempfile.TemporaryDirectory() as cache:
            import os
            os.environ["LEXIBEAT_CACHE"] = cache
            render(args.label, args.takes, args.family, args.seed)
    else:
        pair(args.before, args.after, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
