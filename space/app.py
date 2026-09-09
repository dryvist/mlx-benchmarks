"""
MLX Benchmarks Viewer — Gradio Space

Reads all parquet shards from JacobPEvans/mlx-benchmarks and renders
interactive comparison charts. Auto-refreshes data every 10 minutes.

Deploy to HF Spaces (SDK: gradio, Python 3.11+).
"""

import json
import re
import time
from threading import Lock

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import HfFileSystem

DATASET = "datasets/JacobPEvans/mlx-benchmarks"
RUN_INDEX_PATH = "metadata/run-index-v1.json"
CACHE_TTL = 600  # seconds
EXPECTED_COLUMNS = ["timestamp", "suite", "name", "metric", "model", "value"]
CSS = """
#title { text-align: center; margin-bottom: 4px; }
#subtitle { text-align: center; color: #666; margin-bottom: 20px; }
"""


# ── Data loading ──────────────────────────────────────────────────────────────

_cache: tuple[float, pd.DataFrame] | None = None
_cache_lock = Lock()


def empty_data() -> pd.DataFrame:
    return pd.DataFrame(columns=[*EXPECTED_COLUMNS, "model_short", "evidence_status", "series_key"])


def _dataset_path(uri: str) -> str:
    """Turn an HF URI back into its dataset-relative path."""
    return uri.removeprefix(f"hf://{DATASET}/")


def load_run_index(fs: HfFileSystem) -> tuple[dict[str, dict[str, object]], pd.Timestamp | None]:
    """Load the optional immutable-shard index, keyed by dataset-relative path.

    A missing index entry is intentionally *experimental*: absence must never
    make an unreviewed parquet eligible for the scored/default view.
    """
    try:
        with fs.open(f"{DATASET}/{RUN_INDEX_PATH}") as file:
            payload = json.load(file)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}, None
    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        return {}, None
    cutoff = payload.get("historical_scored_before")
    cutoff_timestamp = pd.to_datetime(cutoff, utc=True, errors="coerce") if cutoff else None
    return (
        {entry["path"]: entry for entry in runs if isinstance(entry, dict) and "path" in entry},
        cutoff_timestamp if pd.notna(cutoff_timestamp) else None,
    )


def add_evidence_metadata(
    df: pd.DataFrame, index: dict[str, dict[str, object]], historical_cutoff: pd.Timestamp | None
) -> pd.DataFrame:
    """Join immutable-shard metadata and form a non-collapsing comparison key."""

    def evidence_status(row: pd.Series) -> str:
        entry = index.get(row["source_path"])
        if entry:
            return str(entry.get("status", "experimental"))
        if historical_cutoff is not None and row["timestamp"] < historical_cutoff:
            return "scored"
        return "experimental"

    df["evidence_status"] = df.apply(evidence_status, axis="columns")
    df["variant"] = df["source_path"].map(lambda path: index.get(path, {}).get("variant", "unindexed"))
    df["context_band"] = df["source_path"].map(
        lambda path: index.get(path, {}).get("context_band", "unspecified")
    )
    df["max_output_tokens"] = df["source_path"].map(
        lambda path: index.get(path, {}).get("max_output_tokens", "unspecified")
    )
    df["completion_state"] = df["source_path"].map(
        lambda path: index.get(path, {}).get("completion_state", "unknown")
    )
    host = df.get("hostname", pd.Series("unknown", index=df.index)).fillna("unknown")
    backend = df.get("serving", pd.Series("unknown", index=df.index)).fillna("unknown")
    concurrency = df.get("concurrency", pd.Series("unspecified", index=df.index)).fillna("unspecified")
    # An arm is weights + quant + effort + serving config, so two efforts are two
    # series. Without this the chart averages `high` and `xhigh` into one line and
    # the effort sweep it exists to show disappears. Historical rows have no
    # column at all, hence the whole-frame default.
    effort = df.get("reasoning_effort", pd.Series("unstated", index=df.index)).fillna("unstated")
    df["series_key"] = [
        " | ".join(map(str, values))
        for values in zip(
            df["model"],
            host,
            backend,
            df["variant"],
            df["context_band"],
            concurrency,
            effort,
            strict=False,
        )
    ]
    return df


def evidence_view(df: pd.DataFrame, status: str = "scored") -> pd.DataFrame:
    """Return only results appropriate for the selected evidence view."""
    if status == "scored":
        return df[df["evidence_status"] == "scored"].copy()
    return df[df["evidence_status"] != "scored"].copy()


def normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Coalesce the two historical result layouts and drop non-measurements.

    Two publisher generations flattened results differently. Newer shards write
    ``name`` / ``metric`` / ``value`` / ``unit`` directly; older shards nested
    each result's metric object, so pandas exploded it into
    ``metric_name`` / ``metric_metric`` / ``metric_value`` / ``metric_unit``.
    The viewer only reads the flat columns, so without coalescing here it
    silently ignores most real measurements (e.g. tool-calling, ttft,
    code-accuracy, math-hard, and older throughput runs).

    Rows that were skipped (CI runs with no MLX server) or carry no numeric
    value are failure records, not comparable results — drop them so a suite
    only appears when it actually has data to chart.
    """
    for flat, nested in (
        ("name", "metric_name"),
        ("metric", "metric_metric"),
        ("value", "metric_value"),
        ("unit", "metric_unit"),
    ):
        if flat not in df.columns:
            df[flat] = pd.NA
        if nested in df.columns:
            df[flat] = df[flat].fillna(df[nested])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if "skipped" in df.columns:
        df = df[~df["skipped"].fillna(False).astype(bool)]
    return df.dropna(subset=["name", "metric", "value"]).reset_index(drop=True)


def load_data() -> pd.DataFrame:
    global _cache
    with _cache_lock:
        if _cache and time.time() - _cache[0] < CACHE_TTL:
            return _cache[1]

        fs = HfFileSystem()
        try:
            paths = sorted(f"hf://{p}" for p in fs.glob(f"{DATASET}/data/*.parquet"))
        except (FileNotFoundError, OSError):
            paths = []

        if not paths:
            df = empty_data()
            _cache = (time.time(), df)
            return df

        frames = []
        for path in paths:
            frame = pd.read_parquet(path)
            frame["source_path"] = _dataset_path(path)
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
        raw_suites = set(df["suite"].dropna().unique())
        df = normalize_rows(df)
        run_index, historical_cutoff = load_run_index(fs)
        df = add_evidence_metadata(df, run_index, historical_cutoff)
        df["model_short"] = df["model"].apply(short_model)
        # When runs carry a hostname (e.g. a Mac Studio vs a MacBook Pro — both
        # Apple M4 Max / 128 GB), fold it into the series label so the same model
        # on two machines compares as two bars/lines instead of collapsing to one.
        if "hostname" in df.columns:
            df["model_short"] = [
                f"{label} @{host}" if isinstance(host, str) and host else label
                for label, host in zip(df["model_short"], df["hostname"], strict=False)
            ]
        # Suites that exist in the dataset but have zero comparable rows today
        # (every run skipped/errored) — surfaced in the UI as "awaiting data" so
        # the capability is visibly tracked, not silently dropped.
        df.attrs["awaiting_suites"] = sorted(raw_suites - set(df["suite"].dropna().unique()))
        _cache = (time.time(), df)
        return df


def short_model(name: str) -> str:
    """Strip common prefixes for axis labels."""
    name = re.sub(r"^mlx-community/", "", name)
    name = re.sub(r"^openrouter/openai/", "openrouter/", name)
    return name


def top_task_metric(df: pd.DataFrame, suite: str) -> tuple[str | None, str | None]:
    """Within a suite, the (task, metric) pair comparing the most models.

    Picking the richest pair as the default guarantees the landing chart is
    populated instead of an arbitrary — possibly empty — combination.
    """
    sub = df[df["suite"] == suite]
    if sub.empty:
        return (None, None)
    counts = sub.groupby(["name", "metric"])["series_key"].nunique()
    name, metric = counts.idxmax()
    return (name, metric)


def best_default(df: pd.DataFrame) -> tuple[str, str | None, str | None]:
    """The (suite, task, metric) triple comparing the most models across all data."""
    if df.empty:
        return ("reasoning", None, None)
    counts = df.groupby(["suite", "name", "metric"])["series_key"].nunique()
    suite, name, metric = counts.idxmax()
    return (suite, name, metric)


# ── Chart builders ────────────────────────────────────────────────────────────


def bar_chart(df: pd.DataFrame, suite: str, task: str, metric: str) -> go.Figure:
    """Latest-run bar chart: one bar per model, sorted by score."""
    sub = df[(df["suite"] == suite) & (df["name"] == task) & (df["metric"] == metric)].copy()
    if sub.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data for this selection",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font_size=18,
        )
        return fig

    # Keep only the latest run per fully comparable series.  A model alone is
    # insufficient: host, backend, variant, context band, concurrency, and
    # reasoning effort can all materially change the measured result.
    sub = sub.sort_values("timestamp").groupby("series_key", as_index=False).last()
    sub["label"] = sub["series_key"]
    sub = sub.sort_values("value", ascending=True)
    value_max = sub["value"].max()
    axis_max = max(1.0, float(value_max) * 1.15) if pd.notna(value_max) else 1.0

    fig = px.bar(
        sub,
        x="value",
        y="label",
        orientation="h",
        text=sub["value"].map("{:.3f}".format),
        color="value",
        color_continuous_scale="Blues",
        labels={"value": metric, "label": "Model"},
        title=f"{task} — {metric}  ({suite})",
    )
    fig.update_traces(textposition="outside")
    fig.update_coloraxes(showscale=False)
    fig.update_layout(
        height=max(350, len(sub) * 44),
        margin={"l": 220, "r": 60, "t": 60, "b": 40},
        yaxis_title="",
        xaxis_range=[0, axis_max],
        font_size=13,
    )
    return fig


def trend_chart(df: pd.DataFrame, suite: str, task: str, metric: str, models: list[str]) -> go.Figure:
    """Score-over-time line chart for selected models."""
    sub = df[
        (df["suite"] == suite)
        & (df["name"] == task)
        & (df["metric"] == metric)
        & (df["series_key"].isin(models))
    ].copy()
    if sub.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data for this selection",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font_size=18,
        )
        return fig

    fig = px.line(
        sub.sort_values("timestamp"),
        x="timestamp",
        y="value",
        color="series_key",
        markers=True,
        labels={"value": metric, "timestamp": "Run time", "series_key": "Comparison series"},
        title=f"{task} — {metric} over time",
    )
    fig.update_layout(height=420, font_size=13, legend_title="")
    return fig


def summary_table(df: pd.DataFrame, suite: str, metric: str) -> pd.DataFrame:
    """Pivot table: models x tasks, latest run only."""
    sub = df[(df["suite"] == suite) & (df["metric"] == metric)].copy()
    if sub.empty:
        return pd.DataFrame({"(no data)": []})

    sub = sub.sort_values("timestamp").groupby(["series_key", "name"], as_index=False).last()
    pivot = sub.pivot(index="series_key", columns="name", values="value")
    pivot = pivot.round(4).reset_index().rename(columns={"series_key": "Comparison series"})
    return pivot


# ── Gradio UI ─────────────────────────────────────────────────────────────────


def build_ui():
    evidence_labels = {"Scored evidence": "scored", "Experimental / excluded": "experimental"}
    selected_evidence = "Scored evidence"
    df = evidence_view(load_data(), evidence_labels[selected_evidence])

    def suite_tasks(d, suite):
        return sorted(d[d["suite"] == suite]["name"].dropna().unique().tolist())

    def suite_metrics(d, suite):
        return sorted(d[d["suite"] == suite]["metric"].dropna().unique().tolist())

    def status_text(d, evidence_label):
        n_models = d["series_key"].nunique() if not d.empty else 0
        msg = f"Showing **{len(d)}** results across **{n_models}** comparison series."
        if evidence_label == "Experimental / excluded":
            msg = (
                "⚠️ **Experimental / excluded evidence** — retained for inspection, "
                "not eligible for rankings or deployment decisions.  \n" + msg
            )
        awaiting = d.attrs.get("awaiting_suites", [])
        if awaiting:
            msg += (
                "  \n⚠️ **Awaiting data** — these suites exist but have no runs yet "
                "(need execution on real MLX hardware): " + ", ".join(awaiting) + "."
            )
        return msg

    suites = sorted(df["suite"].dropna().unique().tolist()) if not df.empty else ["reasoning"]
    model_labels = sorted(df["series_key"].dropna().unique().tolist()) if not df.empty else []

    # Land on the suite/task/metric that compares the most models, and scope the
    # Task/Metric dropdowns to that suite so no default combination is ever empty.
    default_suite, default_task, default_metric = best_default(df)
    tasks = suite_tasks(df, default_suite) if not df.empty else []
    metrics = suite_metrics(df, default_suite) if not df.empty else []

    def filtered_tasks(suite, evidence_label):
        d = evidence_view(load_data(), evidence_labels[evidence_label])
        t = suite_tasks(d, suite) if not d.empty else []
        top, _ = top_task_metric(d, suite) if not d.empty else (None, None)
        return gr.Dropdown(choices=t, value=top or (t[0] if t else None))

    def filtered_metrics(suite, evidence_label):
        d = evidence_view(load_data(), evidence_labels[evidence_label])
        m = suite_metrics(d, suite) if not d.empty else []
        _, top = top_task_metric(d, suite) if not d.empty else (None, None)
        return gr.Dropdown(choices=m, value=top or (m[0] if m else None))

    def update_bar(suite, task, metric, evidence_label):
        return bar_chart(evidence_view(load_data(), evidence_labels[evidence_label]), suite, task, metric)

    def update_trend(suite, task, metric, selected_models, evidence_label):
        d = evidence_view(load_data(), evidence_labels[evidence_label])
        return trend_chart(d, suite, task, metric, selected_models or d["series_key"].tolist())

    def update_table(suite, metric, evidence_label):
        return summary_table(evidence_view(load_data(), evidence_labels[evidence_label]), suite, metric)

    def refresh(evidence_label):
        global _cache
        with _cache_lock:
            _cache = None
        d = evidence_view(load_data(), evidence_labels[evidence_label])
        new_suites = sorted(d["suite"].dropna().unique().tolist()) if not d.empty else ["reasoning"]
        b_suite, b_task, b_metric = best_default(d)
        new_tasks = suite_tasks(d, b_suite) if not d.empty else []
        new_metrics = suite_metrics(d, b_suite) if not d.empty else []
        new_model_labels = sorted(d["series_key"].dropna().unique().tolist()) if not d.empty else []
        return (
            gr.Dropdown(choices=new_suites, value=b_suite if new_suites else None),
            gr.Dropdown(choices=new_tasks, value=b_task if new_tasks else None),
            gr.Dropdown(choices=new_metrics, value=b_metric if new_metrics else None),
            gr.CheckboxGroup(choices=new_model_labels, value=new_model_labels[:6]),
            status_text(d, evidence_label),
        )

    with gr.Blocks(title="MLX Benchmarks") as demo:
        gr.Markdown("# MLX Benchmarks Viewer", elem_id="title")
        gr.Markdown(
            "Compare local MLX models and cloud endpoints across throughput, reasoning, "
            "coding, and capability benchmarks.  \n"
            "Data: [JacobPEvans/mlx-benchmarks](https://huggingface.co/datasets/JacobPEvans/mlx-benchmarks)",
            elem_id="subtitle",
        )

        with gr.Row():
            evidence_dd = gr.Radio(
                choices=list(evidence_labels), value=selected_evidence, label="Evidence set"
            )
            suite_dd = gr.Dropdown(choices=suites, value=default_suite, label="Suite")
            task_dd = gr.Dropdown(choices=tasks, value=default_task, label="Task")
            metric_dd = gr.Dropdown(choices=metrics, value=default_metric, label="Metric")
            refresh_btn = gr.Button("↻ Refresh data", scale=0)

        status = gr.Markdown(status_text(df, selected_evidence))

        with gr.Tabs():
            with gr.Tab("Bar chart — latest run"):
                bar_plot = gr.Plot(
                    value=bar_chart(df, default_suite, default_task or "", default_metric or "")
                )

            with gr.Tab("Trend — over time"):
                model_select = gr.CheckboxGroup(
                    choices=model_labels,
                    value=model_labels[:6],
                    label="Models to show",
                )
                trend_plot = gr.Plot()

            with gr.Tab("Summary table"):
                table_out = gr.DataFrame(
                    value=summary_table(df, default_suite, default_metric or ""),
                    interactive=False,
                )

        # Wire up events
        suite_dd.change(filtered_tasks, [suite_dd, evidence_dd], [task_dd])
        suite_dd.change(filtered_metrics, [suite_dd, evidence_dd], [metric_dd])

        for inp in [suite_dd, task_dd, metric_dd]:
            inp.change(update_bar, [suite_dd, task_dd, metric_dd, evidence_dd], [bar_plot])
            inp.change(update_table, [suite_dd, metric_dd, evidence_dd], [table_out])

        for inp in [suite_dd, task_dd, metric_dd, model_select]:
            inp.change(
                update_trend,
                [suite_dd, task_dd, metric_dd, model_select, evidence_dd],
                [trend_plot],
            )

        evidence_dd.change(
            refresh,
            inputs=[evidence_dd],
            outputs=[suite_dd, task_dd, metric_dd, model_select, status],
        )

        refresh_btn.click(
            refresh,
            inputs=[evidence_dd],
            outputs=[suite_dd, task_dd, metric_dd, model_select, status],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(css=CSS)
