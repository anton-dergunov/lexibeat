"""Build `out/index.html`: every section's report as cards, each one rateable from 1 to 5.

The page is static apart from the scores. Each score is kept as a draft in the browser and written
to `out/labels.json` by `serve.py`, merged per card by `updatedAt` (the listening pages' pattern),
so a dropped connection on the tablet loses nothing. Everything from a report is escaped, except a
section's `intro_html`, which this experiment writes itself.
"""

from __future__ import annotations

from html import escape

from common import OUT, read_report

SECTIONS = ["drills", "syllables", "mixed", "context", "commentary", "framing", "pronounce"]


def _rating(rid: str, small: bool = False) -> str:
    buttons = "".join(f'<button type="button" data-score="{n}">{n}</button>' for n in range(1, 6))
    if small:  # a note is one tap away, so a card of ten clips is not ten empty boxes tall
        buttons += '<button type="button" class="pen" aria-label="Add a note">✎</button>'
    return (f'<div class="rate{" small" if small else ""}" data-id="{escape(rid)}">'
            f'<div class="scores">{buttons}</div>'
            f'<textarea rows="1" placeholder="Note"></textarea></div>')


def _audio(card: dict) -> str:
    out = []
    for clip in card.get("audio", []):
        player = (f'<audio controls preload="none" src="{escape(clip["src"])}"></audio>'
                  if clip.get("src") else "")
        note = f'<div class="warn">{escape(clip["note"])}</div>' if clip.get("note") else ""
        rating = _rating(f'{card["id"]}@{clip["id"]}', small=True) if clip.get("rate") else ""
        out.append(f'<div class="clip"><div class="label">{escape(clip.get("label", ""))}</div>'
                   f'{player}{note}{rating}</div>')
    return "".join(out)


def _grid(card: dict) -> str:
    rows = card.get("grid")
    if not rows:
        return ""
    body = "".join(
        f'<tr><th>{i + 1}</th>' + "".join(
            f'<td class="{"on" if cell != "·" else ""}">{escape(cell) if cell != "·" else ""}</td>'
            for cell in row) + "</tr>" for i, row in enumerate(rows))
    head = "".join(f"<th>{b}</th>" for b in ("1", "", "2", "", "3", "", "4", ""))
    return (f'<div class="gridwrap"><table class="grid"><tr><th>bar</th>{head}</tr>{body}'
            f'</table></div>')


def _table(card: dict) -> str:
    table = card.get("table")
    if not table:
        return ""
    head = "".join(f"<th>{escape(c)}</th>" for c in table["columns"])
    body = ""
    for row in table["rows"]:
        cells = ""
        for cell in row:
            play = (f'<button type="button" class="play" data-src="{escape(cell["src"])}">▶</button>'
                    if cell.get("src") else "")
            cells += f"<td>{play}{escape(cell['text'])}</td>"
        body += f"<tr>{cells}</tr>"
    return f'<div class="gridwrap"><table class="measure"><tr>{head}</tr>{body}</table></div>'


def _items(card: dict) -> str:
    items = card.get("items")
    if not items:
        return ""
    rendered = [f'<li class="{"picked" if item.get("picked") else ""}">'
                f'<div class="itext">{escape(item["text"])}</div>'
                f'<div class="isub">{escape(item.get("sub", ""))}</div>'
                f'{_rating(card["id"] + "#" + item["id"], small=True)}</li>' for item in items]
    cut = card.get("collapse_after")
    if cut is not None and 0 < cut < len(rendered):
        return (f'<ul class="items">{"".join(rendered[:cut])}</ul>'
                f'<details class="more"><summary>other candidates ({len(rendered) - cut})</summary>'
                f'<ul class="items">{"".join(rendered[cut:])}</ul></details>')
    return f'<ul class="items">{"".join(rendered)}</ul>'


def _script(card: dict) -> str:
    rows = card.get("script")
    if not rows:
        return ""
    out = []
    for row in rows:
        if row["kind"] == "drill":
            aside = f'<span class="aside">{escape(row["aside"])}</span>' if row.get("aside") else ""
            out.append(f'<li class="drill">♪ drill · {escape(row["text"])}{aside}</li>')
        else:
            label = f'<span class="who">{escape(row.get("label") or row.get("speaker", ""))}</span>'
            extra = " · ".join(x for x in (row.get("translation", ""),
                                           f'🎙 {row["direction"]}' if row.get("direction") else "")
                               if x)
            out.append(f'<li class="line {escape(row.get("speaker", ""))}">{label}'
                       f'<span class="said">{escape(row["text"])}</span>'
                       + (f'<span class="gloss">{escape(extra)}</span>' if extra else "") + "</li>")
    return f'<ol class="script">{"".join(out)}</ol>'


def _details(card: dict) -> str:
    return "".join(f'<details><summary>{escape(d["summary"])}</summary>'
                   f'<pre>{escape(d["body"])}</pre></details>' for d in card.get("details", []))


def _card(card: dict) -> str:
    parts = [f'<h3>{escape(card["title"])}</h3>']
    if card.get("why"):
        parts.append(f'<p class="why">{escape(card["why"])}</p>')
    if card.get("facts"):
        parts.append(f'<p class="facts">{escape(card["facts"])}</p>')
    if card.get("notes"):
        parts.append(f'<p class="warn">{escape(card["notes"])}</p>')
    parts += [_audio(card), _grid(card), _table(card), _script(card), _items(card),
              _details(card)]
    if card.get("rate", True):
        parts.append(_rating(card["id"]))
    return f'<article class="card" id="{escape(card["id"])}">{"".join(parts)}</article>'


def build() -> str:
    reports = [r for r in (read_report(name) for name in SECTIONS) if r]
    nav = "".join(f'<a href="#s-{r["section"]}">{r["section"]} · {escape(r["title"])}</a>'
                  for r in reports)
    body = []
    for report in reports:
        groups = "".join(
            f'<section class="group" id="g-{report["section"]}-{escape(g["id"])}">'
            f'<h2>{escape(g["title"])}</h2>'
            + (f'<p class="gintro">{escape(g["intro"])}</p>' if g.get("intro") else "")
            + "".join(_card(c) for c in g["cards"]) + "</section>"
            for g in report["groups"])
        intro = (f'<p class="sintro">{escape(report["intro"])}</p>' if report.get("intro") else "")
        intro += (f'<div class="research">{report["intro_html"]}</div>'
                  if report.get("intro_html") else "")
        body.append(f'<section class="section" id="s-{report["section"]}">'
                    f'<h1>{report["section"]}. {escape(report["title"])}</h1>{intro}{groups}'
                    f'</section>')
    html = TEMPLATE.replace("{{nav}}", nav).replace("{{body}}", "".join(body))
    path = OUT / "index.html"
    path.write_text(html, encoding="utf-8")
    return str(path)


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Programme blocks</title>
<style>
  :root { --bg: #f6f4ef; --card: #fff; --ink: #1e1d1a; --muted: #6d6a62; --line: #e2ded4;
          --accent: #3f6f5a; --warn: #b0453b; --chip: #efece4; --on: #e3efe8; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #151513; --card: #1f1e1b; --ink: #ecebe6; --muted: #9c998f; --line: #34322d;
            --accent: #7fb89c; --warn: #e07a6f; --chip: #2a2925; --on: #25352d; } }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 16px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         -webkit-tap-highlight-color: transparent; }
  header { position: sticky; top: 0; z-index: 3; background: var(--bg);
           border-bottom: 1px solid var(--line); padding: 10px 16px 8px; }
  .bar { display: flex; gap: 12px; align-items: baseline; }
  .bar b { font-size: 17px; flex: 1; }
  #status, #progress { font-size: 13px; color: var(--muted); }
  nav { display: flex; gap: 6px; overflow-x: auto; padding-top: 8px; scrollbar-width: none; }
  nav a { flex: none; font-size: 14px; color: var(--ink); text-decoration: none;
          background: var(--chip); border: 1px solid var(--line); border-radius: 16px;
          padding: 4px 12px; }
  main { max-width: 780px; margin: 0 auto; padding: 8px 16px 80px; }
  .section { scroll-margin-top: 96px; }
  h1 { font-size: 22px; margin: 28px 0 6px; }
  h2 { font-size: 18px; margin: 26px 0 4px; }
  h3 { font-size: 16px; margin: 0 0 4px; }
  .sintro, .gintro { color: var(--muted); margin: 4px 0 12px; }
  .research { background: var(--card); border: 1px solid var(--line); border-radius: 14px;
              padding: 4px 16px; font-size: 15px; }
  .research a { color: var(--accent); }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 14px;
          padding: 14px 16px; margin: 12px 0; scroll-margin-top: 96px; }
  .card.done { border-color: color-mix(in srgb, var(--accent) 50%, var(--line)); }
  .why { color: var(--muted); font-size: 15px; margin: 0 0 8px; }
  .facts { font: 13px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; margin: 0 0 8px;
           overflow-wrap: anywhere; }
  .warn { color: var(--warn); font-size: 14px; margin: 4px 0; }
  audio { width: 100%; height: 40px; margin: 2px 0; }
  .clip { border-top: 1px solid var(--line); padding: 8px 0; }
  .clip:first-of-type { border-top: 0; }
  .label { font-size: 14px; font-weight: 600; }
  .gridwrap { overflow-x: auto; margin: 8px 0; }
  table { border-collapse: collapse; font-size: 12px; }
  .grid td, .grid th { border: 1px solid var(--line); padding: 3px 4px; min-width: 44px;
                       vertical-align: top; }
  .grid td.on { background: var(--on); }
  .grid th { color: var(--muted); font-weight: 500; }
  .measure td, .measure th { border: 1px solid var(--line); padding: 4px 6px; text-align: left; }
  .measure td + td { white-space: nowrap; }
  button { font: inherit; color: inherit; background: var(--chip); border: 1px solid var(--line);
           border-radius: 10px; min-height: 44px; min-width: 44px; cursor: pointer; }
  button.play { min-height: 32px; min-width: 32px; margin-right: 6px; border-radius: 16px; }
  button.playing { background: var(--accent); color: var(--bg); }
  .rate { margin-top: 10px; }
  .scores { display: flex; gap: 6px; }
  .scores button { flex: 1; font-weight: 600; }
  .scores button.on { background: var(--accent); border-color: var(--accent); color: var(--bg); }
  .rate.small .scores button { min-height: 36px; font-size: 14px; }
  .rate.small .scores .pen { flex: 0 0 44px; color: var(--muted); }
  .rate.small textarea { display: none; }
  .rate.small.open textarea { display: block; }
  textarea { width: 100%; font: inherit; font-size: 15px; color: inherit; background: var(--bg);
             border: 1px solid var(--line); border-radius: 10px; padding: 6px 10px; margin-top: 6px;
             resize: vertical; }
  .items { list-style: none; padding: 0; margin: 8px 0; }
  .items li { border-top: 1px solid var(--line); padding: 10px 0; }
  .items li:first-child { border-top: 0; }
  .items li.picked .itext { color: var(--accent); }
  .itext { font-weight: 600; }
  .isub { font-size: 15px; margin-top: 2px; }
  .script { list-style: none; padding: 0; margin: 8px 0; }
  .script li { padding: 5px 0 5px 10px; border-left: 3px solid var(--line); margin: 3px 0; }
  .script li.drill { color: var(--muted); font-size: 14px; border-left-color: var(--accent); }
  .script .aside { display: block; font-size: 12px; }
  .script .who { display: inline-block; min-width: 56px; font-size: 12px; color: var(--muted);
                 text-transform: uppercase; letter-spacing: .04em; }
  .script .said { font-weight: 500; }
  .script .gloss { display: block; font-size: 14px; color: var(--muted); }
  details { margin-top: 8px; }
  summary { cursor: pointer; color: var(--muted); font-size: 14px; min-height: 32px; }
  pre { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; background: var(--bg);
        border: 1px solid var(--line); border-radius: 10px; padding: 10px; max-height: 60vh;
        overflow: auto; }
</style>
</head>
<body>
<header><div class="bar"><b>Programme blocks</b><span id="progress"></span><span id="status"></span></div>
<nav>{{nav}}</nav></header>
<main>{{body}}</main>
<script>
const KEY = "programme-blocks-labels";
let labels = {};
let timer = null;
const status = document.getElementById("status");

function draft() { try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch { return {}; } }
function keep() { try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch {} }

function merge(a, b) {
  const out = { ...a };
  for (const [id, v] of Object.entries(b)) {
    if (!out[id] || (v.updatedAt || 0) > (out[id].updatedAt || 0)) out[id] = v;
  }
  return out;
}

function paint() {
  let rated = 0, total = 0;
  document.querySelectorAll(".rate").forEach(el => {
    total++;
    const v = labels[el.dataset.id] || {};
    if (v.score) rated++;
    el.querySelectorAll("[data-score]").forEach(b =>
      b.classList.toggle("on", Number(b.dataset.score) === v.score));
    const t = el.querySelector("textarea");
    if (document.activeElement !== t) t.value = v.note || "";
    if (v.note) el.classList.add("open");
  });
  document.querySelectorAll(".card").forEach(card => {
    const own = card.querySelectorAll(".rate");
    card.classList.toggle("done", own.length > 0 &&
      [...own].every(r => (labels[r.dataset.id] || {}).score));
  });
  document.getElementById("progress").textContent = `${rated} / ${total} rated`;
}

function save() {
  keep();
  clearTimeout(timer);
  status.textContent = "saving…";
  timer = setTimeout(async () => {
    try {
      const r = await fetch("labels.json", { method: "PUT", body: JSON.stringify(labels),
                                             headers: { "Content-Type": "application/json" } });
      status.textContent = r.ok ? "saved" : "not saved, kept on this device";
    } catch { status.textContent = "offline, kept on this device"; }
  }, 600);
}

function set(id, patch) {
  labels[id] = { ...(labels[id] || {}), ...patch, updatedAt: Date.now() };
  paint(); save();
}

document.addEventListener("click", e => {
  const b = e.target.closest("[data-score]");
  if (b) {
    const id = b.closest(".rate").dataset.id, n = Number(b.dataset.score);
    set(id, { score: (labels[id] || {}).score === n ? null : n });
    return;
  }
  const pen = e.target.closest("button.pen");
  if (pen) { const r = pen.closest(".rate"); r.classList.toggle("open");
             if (r.classList.contains("open")) r.querySelector("textarea").focus(); return; }
  const p = e.target.closest("button.play");
  if (p) play(p);
});
document.addEventListener("input", e => {
  if (e.target.matches(".rate textarea")) set(e.target.closest(".rate").dataset.id, { note: e.target.value });
});

// One sound at a time: a clip that starts stops every other clip, and the table's buttons.
const shared = new Audio();
let playingButton = null;
function stopAll(except) {
  document.querySelectorAll("audio").forEach(a => { if (a !== except) a.pause(); });
  if (except !== shared) { shared.pause(); if (playingButton) playingButton.classList.remove("playing"); }
}
document.addEventListener("play", e => stopAll(e.target), true);
function play(button) {
  const same = playingButton === button && !shared.paused;
  stopAll(shared);
  if (playingButton) playingButton.classList.remove("playing");
  if (same) { shared.pause(); playingButton = null; return; }
  shared.src = button.dataset.src; shared.play();
  playingButton = button; button.classList.add("playing");
}
shared.addEventListener("ended", () => playingButton && playingButton.classList.remove("playing"));

(async () => {
  labels = draft();
  try {
    const r = await fetch("labels.json", { cache: "no-store" });
    if (r.ok) { labels = merge(await r.json(), labels); status.textContent = "loaded"; }
  } catch { status.textContent = "offline, drafts only"; }
  paint();
})();
</script>
</body>
</html>
"""
