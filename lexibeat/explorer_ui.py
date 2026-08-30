"""Gradio interface for the LexiBeat music explorer."""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .api import MusicRequest, resolve_music
from .bedspec import BedSpec
from .explorer import (
    CONTROL_FIELDS,
    TABLE_LOCK_PATHS,
    ArtifactStore,
    ExplorerConfig,
    ExplorerValidationReport,
    SampleService,
    apply_safe_repairs,
    logical_id,
    midi_name,
    parse_logical_id,
    pointer_get,
    pointer_set,
    preview_duration,
    randomize_unlocked,
    step_position,
    validate_bed_spec,
)
from .generator import sample_refs


_TABLE_PATHS = ("/phrase/chords", "/phrase/bass", "/phrase/lead",
                "/phrase/percussion")
_EXTRA_LOCK_PATHS = tuple(sorted(TABLE_LOCK_PATHS - set(_TABLE_PATHS)))


def _initial_state() -> dict:
    return {
        "bed_spec": None,
        "music_request": MusicRequest().to_dict(),
        "validation": None,
        "audio_path": None,
    }


def _json_data(spec: BedSpec) -> dict:
    return asdict(spec)


def _format_control(path: str, value: object) -> object:
    if path == "/progression":
        return ", ".join(str(item) for item in value) if isinstance(value, list) else ""
    if path in ("/lead/register", "/lead/velocity"):
        return ", ".join(str(item) for item in value) if isinstance(value, (list, tuple)) else ""
    return value


def _parse_control(path: str, value: object) -> object:
    if path == "/progression":
        if not isinstance(value, str):
            raise ValueError("Progression must be a comma-separated list of scale degrees.")
        try:
            return [int(part.strip()) for part in value.split(",") if part.strip()]
        except ValueError as exc:
            raise ValueError("Progression must contain integer scale degrees.") from exc
    if path == "/lead/register":
        try:
            values = tuple(int(part.strip()) for part in str(value).split(","))
        except ValueError as exc:
            raise ValueError("Register must contain two comma-separated integers.") from exc
        if len(values) != 2:
            raise ValueError("Register must contain exactly two values.")
        return values
    if path == "/lead/velocity":
        try:
            values = tuple(float(part.strip()) for part in str(value).split(","))
        except ValueError as exc:
            raise ValueError("Velocity must contain two comma-separated numbers.") from exc
        if len(values) != 2:
            raise ValueError("Velocity must contain exactly two values.")
        return values
    return value


def _table_rows(spec: BedSpec) -> tuple[list[list], list[list], list[list], list[list]]:
    if spec.phrase is None:
        return [], [], [], []
    chords = [[event.step, step_position(spec, event.step), event.duration_steps,
               ",".join(str(note) for note in event.midi_notes),
               ", ".join(midi_name(note) for note in event.midi_notes), event.velocity]
              for event in spec.phrase.chords]
    bass = [[event.step, step_position(spec, event.step), event.duration_steps,
             event.midi_note, midi_name(event.midi_note), event.velocity]
            for event in spec.phrase.bass]
    lead = [[event.step, step_position(spec, event.step), event.duration_steps,
             event.midi_note, midi_name(event.midi_note), event.velocity]
            for event in spec.phrase.lead]
    percussion = [[lane.sound, lane.pattern, lane.level, lane.probability,
                   lane.humanize, lane.pan, lane.role,
                   logical_id(lane.sample) if lane.sample else ""]
                  for lane in spec.phrase.percussion]
    return chords, bass, lead, percussion


def _clean_rows(value: object) -> list[list]:
    if value is None:
        return []
    if hasattr(value, "values"):
        value = value.values.tolist()
    rows = value if isinstance(value, list) else []
    return [list(row) for row in rows
            if isinstance(row, (list, tuple)) and any(cell not in (None, "") for cell in row)]


def _apply_tables(data: dict, table_values: list[object]) -> None:
    phrase = data.get("phrase")
    if phrase is None:
        if any(_clean_rows(value) for value in table_values):
            raise ValueError("Legacy BedSpecs without a resolved phrase cannot accept phrase tables.")
        return
    chord_rows, bass_rows, lead_rows, percussion_rows = map(_clean_rows, table_values)
    phrase["chords"] = [{
        "step": int(row[0]), "duration_steps": float(row[2]),
        "midi_notes": [int(note.strip()) for note in str(row[3]).split(",") if note.strip()],
        "velocity": float(row[5]),
    } for row in chord_rows]
    phrase["bass"] = [{
        "step": int(row[0]), "duration_steps": float(row[2]),
        "midi_note": int(row[3]), "velocity": float(row[5]),
    } for row in bass_rows]
    phrase["lead"] = [{
        "step": int(row[0]), "duration_steps": float(row[2]),
        "midi_note": int(row[3]), "velocity": float(row[5]),
    } for row in lead_rows]
    lanes = []
    for row in percussion_rows:
        sample = parse_logical_id(str(row[7])) if len(row) > 7 and row[7] else None
        lanes.append({
            "sound": str(row[0]), "pattern": str(row[1]), "level": float(row[2]),
            "probability": float(row[3]), "humanize": float(row[4]),
            "pan": float(row[5]), "role": str(row[6] or ""),
            "sample": asdict(sample) if sample else None,
        })
    phrase["percussion"] = lanes


def _apply_form(state: dict, scalar_values: list[object],
                table_values: list[object]) -> tuple[dict, BedSpec]:
    if not state.get("bed_spec"):
        raise ValueError("Generate or load a BedSpec first.")
    data = json.loads(json.dumps(state["bed_spec"]))
    scalar_fields = [field for field in CONTROL_FIELDS
                     if field.kind != "table" and not field.read_only]
    for control, value in zip(scalar_fields, scalar_values, strict=True):
        try:
            pointer_get(data, control.path)
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        pointer_set(data, control.path, _parse_control(control.path, value))
    _apply_tables(data, table_values)
    spec, report = validate_bed_spec(data, analyze=False)
    if spec is None or report.state == "invalid":
        raise ValueError("Invalid edits: " + "; ".join(
            issue.message for issue in report.issues if issue.severity == "error"))
    state = {**state, "bed_spec": _json_data(spec), "validation": report.to_dict()}
    return state, spec


def _locked_paths(lock_values: list[object]) -> list[str]:
    scalar_fields = [field for field in CONTROL_FIELDS if field.kind != "table"
                     and not field.read_only]
    paths = [field.path for field, selected in zip(scalar_fields, lock_values[:len(scalar_fields)],
                                                   strict=True) if selected]
    start = len(scalar_fields)
    table_locks = lock_values[start:start + len(_TABLE_PATHS)]
    paths.extend(path for path, selected in zip(_TABLE_PATHS, table_locks, strict=True)
                 if selected)
    start += len(_TABLE_PATHS)
    paths.extend(path for path, selected in zip(_EXTRA_LOCK_PATHS, lock_values[start:], strict=True)
                 if selected)
    return paths


def _status(report: ExplorerValidationReport | dict | None,
            heading: str = "Lab validation") -> str:
    if report is None:
        return f"### {heading}\nNo BedSpec is loaded."
    data = report.to_dict() if isinstance(report, ExplorerValidationReport) else report
    state = data["state"]
    symbol = {"production-safe": "✅", "experimental": "⚠️", "invalid": "❌"}[state]
    issues = data.get("issues", [])
    lines = [f"### {heading}", f"{symbol} **{state.replace('-', ' ').title()}**"]
    lines.extend(f"- `{issue['path']}` — {issue['message']}" for issue in issues[:8])
    if len(issues) > 8:
        lines.append(f"- …and {len(issues) - 8} more issues")
    return "\n".join(lines)


def _identity(spec: BedSpec) -> str:
    return (
        "#### Resolved identity\n"
        f"**Engine:** `{spec.engine_version}` &nbsp;·&nbsp; "
        f"**Profile:** `{spec.profile_version}` &nbsp;·&nbsp; "
        f"**Schema:** `{spec.schema_version}` &nbsp;·&nbsp; "
        f"**Seed:** `{spec.seed}`"
    )


def _exports(spec: BedSpec) -> tuple[str, str, str]:
    text = spec.to_json()
    python = ("import json\nfrom lexibeat.bedspec import BedSpec\n\n"
              f"spec = BedSpec.from_dict(json.loads({text!r}))")
    cli = ("uv run generate.py --bed-only --bed-spec bed.json "
           "--out out/bed.wav")
    return python, cli, text


def _write_spec_download(spec: BedSpec, artifacts: ArtifactStore) -> str:
    return str(artifacts.write_spec(spec))


def _sample_tables(spec: BedSpec, samples: SampleService) -> tuple[list[list], list[list]]:
    rows: list[list] = []
    for ref in sample_refs(spec):
        try:
            item = samples.get(logical_id(ref))
            rows.append([item["logical_id"], item["collection"], item["category"],
                         item["availability"], item["promoted"], item["source_license"],
                         item["sha256"], item["relative_path"]])
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            rows.append([logical_id(ref), ref.collection, "", "unavailable", False,
                         "", ref.sha256, str(exc)])
    zones: list[list] = []
    if spec.phrase:
        for role, instrument in (("pad", spec.phrase.pad_instrument),
                                 ("bass", spec.phrase.bass_instrument),
                                 ("lead", spec.phrase.lead_instrument)):
            if not instrument:
                continue
            for zone in instrument.zones:
                zones.append([role, instrument.name, logical_id(zone.sample),
                              zone.root_note, midi_name(zone.root_note),
                              zone.lo_note, zone.hi_note, zone.lo_velocity,
                              zone.hi_velocity, zone.gain_db])
    return rows, zones


def build_demo(config: ExplorerConfig, *, artifacts: ArtifactStore,
               samples: SampleService):
    import gradio as gr

    schema = __import__("lexibeat.explorer", fromlist=["explorer_schema"]).explorer_schema(config)
    scalar_fields = [field for field in CONTROL_FIELDS
                     if field.kind != "table" and not field.read_only]

    with gr.Blocks(title="LexiBeat Music Explorer", fill_width=True,
                   analytics_enabled=False) as demo:
        current = gr.State(_initial_state())
        gr.Markdown("# LexiBeat Music Explorer\nGenerate and inspect reproducible music beds. Voice generation is not enabled.")

        with gr.Tabs(selected="simple") as tabs:
            with gr.Tab("Simple", id="simple"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        family = gr.Dropdown(
                            schema["simple"]["families"], value="auto", label="Style")
                        energy = gr.Radio(
                            schema["simple"]["energy"], value="balanced", label="Energy")
                        rhythm = gr.Radio(
                            schema["simple"]["rhythm"], value="steady", label="Rhythm")
                        palette = gr.Radio(
                            schema["simple"]["palette"], value="hybrid", label="Palette")
                        generate_button = gr.Button("Generate", variant="primary")
                        open_lab_button = gr.Button("Open in Lab")
                    with gr.Column(scale=2, min_width=420):
                        gr.Markdown(
                            "Choose a direction, then generate one complete phrase. "
                            "Generating again replaces the current clip.")
                        simple_audio = gr.Audio(
                            label="Current music preview", interactive=False,
                            editable=False, autoplay=False, buttons=["download"],
                            elem_id="simple-audio")
                with gr.Row():
                    audio_download = gr.File(label="Download WAV", interactive=False)
                    spec_download = gr.File(label="Download BedSpec JSON", interactive=False)

            with gr.Tab("Lab", id="lab"):
                gr.Markdown(
                    "Edit the current resolved bed directly. Changes are applied when you "
                    "validate, randomize, or render; values are never silently clamped.")
                identity_info = gr.Markdown("#### Resolved identity\nGenerate or load a BedSpec first.")
                gr.Markdown(
                    "**Keep on randomize:** these checkboxes do not disable editing. They only "
                    "preserve the checked value when **Randomize unlocked** creates a new bed.")
                scalar_components: list[Any] = []
                scalar_locks: list[Any] = []
                grouped: dict[str, list] = {}
                for control in scalar_fields:
                    grouped.setdefault(control.group, []).append(control)
                for group, controls in grouped.items():
                    display_group = "Timing and structure" if group == "Identity" else group
                    with gr.Accordion(display_group, open=True):
                        for control in controls:
                            with gr.Row():
                                if control.kind == "enum":
                                    component = gr.Dropdown(list(control.choices), label=control.label,
                                                            interactive=True)
                                elif control.kind == "boolean":
                                    component = gr.Checkbox(label=control.label,
                                                            interactive=True)
                                elif control.kind in ("number", "integer"):
                                    component = gr.Number(label=control.label,
                                                          minimum=control.minimum,
                                                          maximum=control.maximum,
                                                          step=control.step,
                                                          interactive=True)
                                else:
                                    component = gr.Textbox(label=control.label,
                                                           interactive=True)
                                scalar_components.append(component)
                                scalar_locks.append(gr.Checkbox(
                                    label="Keep on randomize", value=False, min_width=160))

                with gr.Accordion("Resolved event tables", open=True):
                    chords = gr.Dataframe(
                        headers=["step", "position", "duration", "MIDI notes", "note names", "velocity"],
                        datatype=["number", "str", "number", "str", "str", "number"],
                        type="array", label="Chords", interactive=True)
                    chord_lock = gr.Checkbox(label="Keep all chord events on randomize")
                    bass_events = gr.Dataframe(
                        headers=["step", "position", "duration", "MIDI note", "note name", "velocity"],
                        datatype=["number", "str", "number", "number", "str", "number"],
                        type="array", label="Bass notes", interactive=True)
                    bass_lock = gr.Checkbox(label="Keep all bass events on randomize")
                    lead_events = gr.Dataframe(
                        headers=["step", "position", "duration", "MIDI note", "note name", "velocity"],
                        datatype=["number", "str", "number", "number", "str", "number"],
                        type="array", label="Lead notes", interactive=True)
                    lead_lock = gr.Checkbox(label="Keep all lead events on randomize")
                    percussion = gr.Dataframe(
                        headers=["sound", "pattern", "level", "probability", "humanize", "pan", "role", "sample logical ID"],
                        datatype=["str", "str", "number", "number", "number", "number", "str", "str"],
                        type="array", label="Percussion lanes", interactive=True)
                    percussion_lock = gr.Checkbox(label="Keep all percussion lanes on randomize")

                with gr.Accordion("Sample provenance", open=False):
                    gr.Markdown(
                        "These tables show only the samples referenced by the current bed, "
                        "including every multisample instrument zone. They do not list the "
                        "whole library.")
                    sample_table = gr.Dataframe(
                        headers=["logical ID", "collection", "category", "availability", "promoted", "license", "SHA-256", "source path"],
                        type="array", interactive=False, label="Selected samples")
                    zone_table = gr.Dataframe(
                        headers=["role", "instrument", "logical ID", "root MIDI", "root note", "low note", "high note", "low velocity", "high velocity", "gain dB"],
                        type="array", interactive=False, label="Instrument zones")
                    gr.Markdown("Sample references can be changed in percussion rows or in the advanced JSON editor. Instrument locks preserve complete checksum-addressed zone maps.")
                    extra_locks = [gr.Checkbox(label=(
                        "Keep " + path.removeprefix("/phrase/").replace("_", " ") +
                        " on randomize"))
                                   for path in _EXTRA_LOCK_PATHS]

                gr.Markdown("### Lab actions")
                with gr.Row():
                    randomize_button = gr.Button("Randomize unlocked", variant="primary")
                    validate_button = gr.Button("Validate")
                    safe_button = gr.Button("Return to safe range")
                gr.Markdown(
                    "- **Randomize unlocked** creates a new deterministic bed with a fresh seed "
                    "while preserving every checked field or event section.\n"
                    "- **Validate** applies the current edits and reports whether they are "
                    "production-safe, experimental, or invalid. It does not render audio.\n"
                    "- **Return to safe range** applies only the explicit repair values shown by "
                    "validation; it does not silently change unrelated fields.")
                with gr.Row():
                    preview_button = gr.Button("Render preview")
                    duration = gr.Number(value=min(30, config.max_duration_seconds),
                                         minimum=1, maximum=config.max_duration_seconds,
                                         label="Full render duration (seconds)")
                    render_button = gr.Button("Render full WAV")
                gr.Markdown(
                    "**Render preview** produces one complete phrase, capped at 15 seconds. "
                    "**Render full WAV** uses the duration above and the current edited values.")
                lab_status = gr.Markdown(_status(None))
                lab_audio = gr.Audio(
                    label="Lab render", interactive=False, editable=False,
                    autoplay=False, buttons=["download"], elem_id="lab-audio")
                validation_json = gr.JSON(label="Validation and quality report")

                with gr.Accordion("Load, save, and advanced JSON", open=False):
                    upload = gr.File(label="Load .bed.json", file_types=[".json"], type="filepath")
                    load_button = gr.Button("Load BedSpec")
                    raw_json = gr.Code(label="Current BedSpec JSON", language="json",
                                       lines=20, interactive=True)
                    apply_json_button = gr.Button("Validate and apply raw JSON")
                    with gr.Tabs():
                        with gr.Tab("Python"):
                            python_code = gr.Code(label="Python", language="python", interactive=False)
                        with gr.Tab("CLI"):
                            cli_code = gr.Code(label="CLI", language="shell", interactive=False)
                        with gr.Tab("JSON"):
                            json_code = gr.Code(label="JSON", language="json", interactive=False)

        table_components = [chords, bass_events, lead_events, percussion]
        lock_components = [*scalar_locks, chord_lock, bass_lock, lead_lock,
                           percussion_lock, *extra_locks]

        def audio_value(path: str | None, *, label: str,
                        autoplay: bool = False):
            return gr.Audio(
                value=path, label=label, autoplay=autoplay, playback_position=0,
                interactive=False, editable=False, buttons=["download"])

        def full_values(state: dict, *, clear_locks: bool,
                        simple_audio_path: str | None = None,
                        autoplay: bool = False) -> list:
            spec = BedSpec.from_dict(state["bed_spec"])
            report = state.get("validation")
            python, cli, json_text = _exports(spec)
            sample_rows, zones = _sample_tables(spec, samples)
            scalar_values = []
            data = state["bed_spec"]
            for control in scalar_fields:
                try:
                    value = pointer_get(data, control.path)
                except (KeyError, IndexError, TypeError, ValueError):
                    value = None
                scalar_values.append(_format_control(control.path, value))
            tables = list(_table_rows(spec))
            lock_values = [False] * len(lock_components) if clear_locks else [gr.skip()] * len(lock_components)
            return [
                state, _identity(spec),
                audio_value(simple_audio_path, label="Current music preview",
                            autoplay=autoplay),
                audio_value(None, label="Lab render"), simple_audio_path,
                _write_spec_download(spec, artifacts),
                _status(report), report, json_text, python, cli, json_text,
                sample_rows, zones,
                *scalar_values, *tables, *lock_values,
            ]

        full_outputs = [
            current, identity_info, simple_audio, lab_audio, audio_download,
            spec_download, lab_status, validation_json, raw_json, python_code,
            cli_code, json_code, sample_table, zone_table, *scalar_components,
            *table_components, *lock_components,
        ]

        def generate(family_value: str, energy_value: str, rhythm_value: str,
                     palette_value: str, progress=gr.Progress()):
            progress(0.01, desc="Preparing variations")
            request = MusicRequest(family=family_value, energy=energy_value,
                                   rhythm=rhythm_value, palette=palette_value,
                                   seed=secrets.randbits(64))
            result = resolve_music(
                request,
                progress_callback=lambda value, message:
                    progress(0.02 + 0.70 * value, desc=message))
            _, report = validate_bed_spec(result.bed_spec, analyze=False)
            artifact = artifacts.render(
                result.bed_spec, preview_duration(result.bed_spec), preview=True,
                progress=lambda value, message:
                    progress(0.72 + 0.28 * value, desc=message))
            path = str(artifacts.path_for(artifact.artifact_id))
            state = _initial_state()
            state.update(bed_spec=_json_data(result.bed_spec),
                         music_request=result.request.to_dict(),
                         validation=report.to_dict(), audio_path=path)
            progress(1.0, desc="Loading preview")
            return full_values(state, clear_locks=True,
                               simple_audio_path=path, autoplay=True)

        def edited(state: dict, *values):
            scalar_values = list(values[:len(scalar_components)])
            table_values = list(values[len(scalar_components):len(scalar_components) + 4])
            return _apply_form(state, scalar_values, table_values)

        edit_inputs = [current, *scalar_components, *table_components]

        def validate_current(state: dict, *values):
            state, spec = edited(state, *values)
            _, report = validate_bed_spec(spec, analyze=True)
            state["validation"] = report.to_dict()
            python, cli, json_text = _exports(spec)
            return (state, _status(report, "Validation"), report.to_dict(), json_text,
                    python, cli, json_text, _write_spec_download(spec, artifacts))

        def render_current(state: dict, requested_duration: float | None,
                           preview: bool, progress, *values):
            state, spec = edited(state, *values)
            _, report = validate_bed_spec(spec, analyze=True)
            if report.state == "invalid":
                raise gr.Error("The current BedSpec is invalid. Validate it for exact fields.")
            seconds = preview_duration(spec) if preview else float(requested_duration or 30)
            artifact = artifacts.render(
                spec, seconds, preview=preview,
                progress=lambda value, message: progress(value, desc=message))
            path = str(artifacts.path_for(artifact.artifact_id))
            state.update(validation=report.to_dict(), audio_path=path)
            return (state, _status(report, "Render result"),
                    audio_value(path, label="Lab render", autoplay=True),
                    path, report.to_dict())

        def randomize_current(state: dict, *values, progress=gr.Progress()):
            edit_count = len(scalar_components) + len(table_components)
            state, _spec = edited(state, *values[:edit_count])
            locks = _locked_paths(list(values[edit_count:]))
            progress(0.02, desc="Generating unlocked fields")
            request = MusicRequest.from_dict(state["music_request"])
            result = randomize_unlocked(state["bed_spec"], locks,
                                        seed=secrets.randbits(64), request=request,
                                        progress_callback=lambda value, message:
                                            progress(value, desc=message))
            state.update(bed_spec=_json_data(result.bed_spec),
                         validation=result.validation.to_dict(), audio_path=None)
            progress(1.0, desc="Randomized bed ready")
            return full_values(state, clear_locks=False)

        def safe_current(state: dict, *values):
            state, spec = edited(state, *values)
            repaired, report = apply_safe_repairs(spec)
            if repaired is None:
                raise gr.Error("Invalid fields must be corrected before safe-range repair.")
            state.update(bed_spec=_json_data(repaired), validation=report.to_dict(),
                         audio_path=None)
            return full_values(state, clear_locks=False)

        def load_file(state: dict, file_path: str | None):
            if not file_path:
                raise gr.Error("Choose a .bed.json file first.")
            source = Path(file_path)
            if source.stat().st_size > 2 * 1024 * 1024:
                raise gr.Error("BedSpec files are limited to 2 MB.")
            try:
                data = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise gr.Error(f"Could not read JSON: {exc}") from exc
            spec, report = validate_bed_spec(data, analyze=True)
            if spec is None or report.state == "invalid":
                raise gr.Error("Invalid BedSpec: " + "; ".join(
                    issue.message for issue in report.issues if issue.severity == "error"))
            state = _initial_state()
            state.update(bed_spec=_json_data(spec), validation=report.to_dict())
            return full_values(state, clear_locks=True)

        def apply_raw(state: dict, text: str):
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise gr.Error(f"Invalid JSON: {exc}") from exc
            spec, report = validate_bed_spec(data, analyze=True)
            if spec is None or report.state == "invalid":
                raise gr.Error("Invalid BedSpec: " + "; ".join(
                    issue.message for issue in report.issues if issue.severity == "error"))
            state.update(bed_spec=_json_data(spec), validation=report.to_dict(),
                         audio_path=None)
            return full_values(state, clear_locks=False)

        generate_event = generate_button.click(
            generate, [family, energy, rhythm, palette], full_outputs,
            api_name=False)
        open_lab_button.click(lambda: gr.Tabs(selected="lab"), outputs=tabs,
                              api_name=False)

        validate_button.click(
            validate_current, edit_inputs,
            [current, lab_status, validation_json, raw_json, python_code, cli_code,
             json_code, spec_download], api_name=False)
        preview_event = preview_button.click(
            lambda state, *values, progress=gr.Progress():
                render_current(state, None, True, progress, *values),
            edit_inputs, [current, lab_status, lab_audio, audio_download, validation_json],
            api_name=False)
        render_event = render_button.click(
            lambda state, seconds, *values, progress=gr.Progress():
                render_current(state, seconds, False, progress, *values),
            [current, duration, *scalar_components, *table_components],
            [current, lab_status, lab_audio, audio_download, validation_json],
            api_name=False)
        randomize_event = randomize_button.click(
            randomize_current, [*edit_inputs, *lock_components], full_outputs,
            api_name=False)
        safe_button.click(safe_current, edit_inputs, full_outputs, api_name=False)
        load_button.click(load_file, [current, upload], full_outputs, api_name=False)
        apply_json_button.click(apply_raw, [current, raw_json], full_outputs,
                                api_name=False)

    return demo.queue(default_concurrency_limit=1, max_size=config.max_pending_renders)
