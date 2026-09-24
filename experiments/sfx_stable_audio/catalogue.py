"""Put each generated word sound next to what the open catalogues return for the same word.

    uv run python catalogue.py                  # fetch what is missing, then build the page
    open out/catalogue/index.html

Sources, each taken by its own ranking so the comparison is not picked by ear:

- **ESC-50**: the first clips of the word's class, fetched one file at a time from the dataset's
  repository. Human-curated, 5 s each; CC BY-NC, except the ESC-10 subset, which is CC BY.
- **FSD50K**: CC0 clips carrying every label the word names, where every rater marked the label as
  the predominant sound, fewest other labels first. Only the 7 MB of labels and metadata are
  downloaded; the audio is each clip's Freesound preview.
- **Freesound**: the first CC0 results of a plain text search, at most 30 s long, as a person searching
  would see them.

Freesound (and so FSD50K's audio) needs an API key from https://freesound.org/apiv2/apply, read
from `FREESOUND_API_KEY` or `~/.config/freesound/api_key`. Without one, only ESC-50 is fetched.
Everything downloaded goes under `$LEXIBEAT_CACHE/catalogue-sfx` (default `~/.cache/lexibeat`).
"""

from __future__ import annotations

import csv
import html
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import soundfile as sf

from compare import level, slug, write_page

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("LEXIBEAT_CACHE", Path.home() / ".cache" / "lexibeat")) / "catalogue-sfx"
OUT = HERE / "out" / "catalogue"
ESC50 = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"
FSD50K = "https://zenodo.org/records/4060432/files"
FREESOUND = "https://freesound.org/apiv2"
CC0 = "http://creativecommons.org/publicdomain/zero/1.0/"
FIELDS = "id,name,username,license,duration,previews,tags"
MAX_SECONDS = 10.0  # what the page plays
SOURCE_SECONDS = 30  # what a source may be: ambiences (waves, a stream) are rarely short


def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "lexibeat-experiment"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        path.write_bytes(data)
    return path


def api_key() -> str | None:
    key = os.environ.get("FREESOUND_API_KEY")
    file = Path.home() / ".config" / "freesound" / "api_key"
    if not key and file.exists():
        key = file.read_text().strip()
    return key or None


def freesound(endpoint: str, key: str, **params: str) -> dict:
    query = urllib.parse.urlencode({**params, "token": key})
    request = urllib.request.Request(f"{FREESOUND}/{endpoint}?{query}",
                                     headers={"User-Agent": "lexibeat-experiment"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def preview(sound: dict) -> Path:
    return fetch(sound["previews"]["preview-hq-mp3"], CACHE / "freesound" / f"{sound['id']}.mp3")


def link(sound_id: int | str, text: str) -> str:
    return f'<a href="https://freesound.org/s/{sound_id}/">{html.escape(text)}</a>'


# --- sources ---------------------------------------------------------------------------------


def esc50_clips(category: str, count: int) -> list[tuple[Path, str]]:
    meta = fetch(f"{ESC50}/meta/esc50.csv", CACHE / "esc50.csv")
    rows, sources = [], set()
    # Several ESC-50 clips are cut from one recording; take one clip per recording.
    for r in sorted(csv.DictReader(meta.open()), key=lambda r: r["filename"]):
        if r["category"] == category and r["src_file"] not in sources and len(rows) < count:
            rows.append(r)
            sources.add(r["src_file"])
    clips = []
    for r in rows:
        path = fetch(f"{ESC50}/audio/{r['filename']}", CACHE / "esc50" / r["filename"])
        licence = "CC BY" if r["esc10"] == "True" else "CC BY-NC"
        clips.append((path, f"{link(r['src_file'], 'ESC-50 ' + r['filename'])} · {licence}"))
    return clips


class Fsd50k:
    """The labels, licences and predominance ratings, loaded once."""

    def __init__(self) -> None:
        for name in ("FSD50K.ground_truth", "FSD50K.metadata"):
            if not (CACHE / name).exists():
                archive = fetch(f"{FSD50K}/{name}.zip?download=1", CACHE / f"{name}.zip")
                zipfile.ZipFile(archive).extractall(CACHE)
        truth = CACHE / "FSD50K.ground_truth"
        self.mids = {r[1]: r[2] for r in csv.reader((truth / "vocabulary.csv").open())}
        self.labels: dict[str, list[str]] = {}
        for split in ("dev", "eval"):
            for r in csv.DictReader((truth / f"{split}.csv").open()):
                self.labels[r["fname"]] = r["labels"].split(",")
        meta = CACHE / "FSD50K.metadata"
        self.info = {**json.loads((meta / "dev_clips_info_FSD50K.json").read_text()),
                     **json.loads((meta / "eval_clips_info_FSD50K.json").read_text())}
        self.ratings = json.loads((meta / "pp_pnp_ratings_FSD50K.json").read_text())
        # Labels propagate to their parents ("Meow" brings "Cat" and "Animal"). A label that is on
        # every clip carrying another is taken as its parent, so a caption can name only the
        # sounds that are actually different.
        self.parents: dict[str, set[str]] = {}
        for labels in self.labels.values():
            for label in labels:
                self.parents[label] = self.parents.get(label, set(labels)) & set(labels)
        for label in self.parents:
            self.parents[label].discard(label)

    def other_sounds(self, clip: str, wanted: list[str]) -> list[str]:
        labels = set(self.labels[clip])
        inferred = set(wanted).union(*(self.parents[l] for l in labels))
        return sorted(labels - inferred)

    def candidates(self, wanted: list[str]) -> list[str]:
        def predominant(clip: str) -> bool:
            # 1.0 is "present and predominant"; every rater must have said so. Parent labels
            # ("Insect" above "Buzz") are inferred rather than rated, so only rated ones count.
            ratings = self.ratings.get(clip, {})
            rated = [ratings[self.mids[w]] for w in wanted if self.mids[w] in ratings]
            return bool(rated) and all(v == 1.0 for votes in rated for v in votes)
        found = [clip for clip, labels in self.labels.items()
                 if set(wanted) <= set(labels) and self.info.get(clip, {}).get("license") == CC0
                 and predominant(clip)]
        return sorted(found, key=lambda clip: (len(self.labels[clip]), int(clip)))

    def clips(self, wanted: list[str], count: int, key: str) -> list[tuple[Path, str]]:
        ids = self.candidates(wanted)[:count * 10]
        if not ids:
            return []
        found = freesound("search/text/", key, query="", fields=FIELDS, page_size=str(len(ids)),
                          filter=f"id:({' OR '.join(ids)}) duration:[0.5 TO {SOURCE_SECONDS}]")
        by_id = {str(s["id"]): s for s in found["results"]}
        clips = []
        for clip in ids:
            if clip in by_id and len(clips) < count:
                sound = by_id[clip]
                others = self.other_sounds(clip, wanted)[:3]
                also = f" · also: {html.escape(', '.join(others))}" if others else ""
                clips.append((preview(sound), f"{link(clip, 'FSD50K ' + clip)}: "
                              f"{html.escape(sound['name'][:40])}{also} · CC0"))
        return clips


def freesound_clips(query: str, count: int, key: str) -> list[tuple[Path, str]]:
    found = freesound("search/text/", key, query=query, fields=FIELDS, page_size=str(count),
                      filter=f'license:"Creative Commons 0" duration:[0.5 TO {SOURCE_SECONDS}]')
    return [(preview(s), f"{link(s['id'], s['name'][:40])} by {html.escape(s['username'])} · CC0")
            for s in found["results"][:count]]


# --- page ------------------------------------------------------------------------------------


def place(source: Path, name: str, caption: str) -> dict:
    audio, sample_rate = sf.read(source, always_2d=True)
    audio = audio.T[:, : int(MAX_SECONDS * sample_rate)]
    audio, metrics = level(audio, sample_rate)
    # MP3 like the previews most of these came from: 230 WAV copies would take 200 MB.
    sf.write(OUT / name, audio.T, sample_rate, format="MP3")
    return {"file": name, "problems": [], "caption": caption,
            "seconds": round(audio.shape[-1] / sample_rate, 1), **metrics}


def main() -> None:
    round_ = json.loads((HERE / "rounds" / "catalogue.json").read_text())
    count = round_.get("per_source", 3)
    key = api_key()
    if key is None:
        print("no Freesound API key: fetching ESC-50 only (see the docstring)", file=sys.stderr)
    fsd = Fsd50k() if key else None
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    columns = [{"id": "generated", "label": "Generated",
                "note": "Stable Audio 3 small-sfx, the clip from the earlier rounds."},
               {"id": "esc50", "label": "ESC-50", "note": "Curated 5 s clips. CC BY-NC; its ESC-10 subset CC BY."},
               {"id": "fsd50k", "label": "FSD50K, CC0",
                "note": "Labels verified by people; predominant sound only."},
               {"id": "freesound", "label": "Freesound search, CC0",
                "note": "The first results for a plain query."}]
    cells: dict[tuple[str, str], list[dict]] = {}
    report = []
    for index, row in enumerate(round_["rows"], 1):
        stem = f"{index:02d}_{slug(row['word'])}"
        sources: dict[str, list[tuple[Path, str]]] = {
            "generated": [(HERE / "out" / row["generated"], "Stable Audio 3 small-sfx")]}
        if row.get("esc50"):
            sources["esc50"] = esc50_clips(row["esc50"], count)
        if key and row.get("fsd50k"):
            sources["fsd50k"] = fsd.clips(row["fsd50k"], count, key)
        if key and row.get("freesound"):
            sources["freesound"] = freesound_clips(row["freesound"], count, key)
        for column, clips in sources.items():
            placed = [place(path, f"{stem}_{column}_{k}.mp3", caption)
                      for k, (path, caption) in enumerate(clips, 1)]
            if placed:
                cells[(row["word"], column)] = placed
        found = {c: len(cells.get((row["word"], c), [])) for c in ("esc50", "fsd50k", "freesound")}
        report.append({**row, "found": found})
        print(f"{row['word']:<24} " + "  ".join(f"{c} {n}" for c, n in found.items()))

    OUT.joinpath("report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    shown = [c for c in columns if any(col == c["id"] for _, col in cells)]
    write_page(OUT, {**round_, "columns": shown,
                     "legend": "Catalogue clips are Freesound's MP3 previews or ESC-50's WAVs, trimmed "
                               f"to {MAX_SECONDS:.0f} s. Each caption links to the source."}, cells)
    print(f"listen: {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
