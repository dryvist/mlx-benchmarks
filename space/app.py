"""
MLX Benchmarks Viewer — Gradio Space

Reads all parquet shards from JacobPEvans/mlx-benchmarks and renders
interactive comparison charts. Auto-refreshes data every 10 minutes.

Deploy to HF Spaces (SDK: gradio, Python 3.11+).
"""

import json
import os
import re
import time
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
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
FILTER_SPECS = {
    "machine": ("Machine", ("campaign_hardware_machine", "machine", "hostname", "host")),
    "accelerator": (
        "Accelerator model",
        ("campaign_hardware_accelerator_model", "accelerator_model", "chip", "gpu"),
    ),
    "engine": ("Engine", ("campaign_software_engine", "engine")),
    "backend": ("Backend", ("campaign_software_backend", "backend", "tag_backend")),
    "quant_format": (
        "Quant format",
        ("campaign_model_quantization", "quant_format", "quantization", "tag_quant", "quant"),
    ),
    "agents": (
        "Agents",
        ("campaign_run_concurrent_agents", "concurrent_agents", "concurrency", "tag_concurrency", "agents"),
    ),
    "allocated_context": (
        "Allocated context (tokens)",
        (
            "campaign_run_allocated_context_tokens",
            "allocated_context",
            "context_tokens_target",
            "tag_context_tokens_target",
            "configured_window_tokens",
            "tag_configured_window_tokens",
        ),
    ),
    "context_tokens": (
        "Context (tokens)",
        (
            "campaign_run_depth_tokens",
            "context_tokens_actual",
            "tag_context_tokens_actual",
            "context_tokens",
        ),
    ),
    "prompt_tokens": (
        "Prompt (tokens)",
        (
            "campaign_run_prompt_tokens",
            "prompt_tokens",
            "tag_prompt_tokens",
            "total_input_tokens",
            "tag_total_input_tokens",
            "requested_prompt_tokens",
            "tag_requested_prompt_tokens",
        ),
    ),
    "output_budget_tokens": (
        "Output budget (tokens)",
        (
            "campaign_run_output_tokens",
            "max_output_tokens",
            "output_token_budget",
            "max_new_tokens",
            "max_gen_toks",
            "tag_max_gen_toks",
            "output_reservation_tokens",
            "tag_output_reservation_tokens",
            "reserved_output_tokens",
            "tag_reserved_output_tokens",
        ),
    ),
}
WORKLOAD_FIELDS = (
    "allocated_context",
    "context_tokens",
    "prompt_tokens",
    "output_budget_tokens",
    "completion_state",
)
PLACEHOLDERS = frozenset(
    {"", "n/a", "na", "nan", "none", "null", "unknown", "unspecified", "unindexed", "unstated"}
)
CSS = """
#title { color: #243746; margin-bottom: 2px; }
#subtitle { color: #647783; max-width: 760px; margin: 0 0 18px; line-height: 1.5; }
#selection-status { border-left: 3px solid #126b72; padding: 10px 14px; background: #e8f0f1; }
.gradio-container { background: #f2f5f6; }
body, .gradio-container { font-family: Inter, "Segoe UI", Arial, sans-serif; }
button:focus-visible, input:focus-visible, [role="combobox"]:focus-visible {
    outline: 3px solid #126b72 !important;
    outline-offset: 2px;
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; }
}
"""
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
)


# ── Data loading ──────────────────────────────────────────────────────────────

_cache: tuple[float, pd.DataFrame] | None = None
_cache_lock = Lock()


def empty_data() -> pd.DataFrame:
    columns = [
        *EXPECTED_COLUMNS,
        "unit",
        "display_unit",
        "model_short",
        "evidence_status",
        "series_key",
        "series_label",
        "workload_complete",
        *FILTER_SPECS,
        *WORKLOAD_FIELDS,
    ]
    return pd.DataFrame(columns=list(dict.fromkeys(columns)))


def _dataset_path(uri: str) -> str:
    """Return the original shard path represented by a canonical HF URI."""
    path = uri.removeprefix(f"hf://{DATASET}/")
    canonical_prefix = "data/run-canonical-"
    if path.startswith(canonical_prefix):
        return f"data/{path.removeprefix(canonical_prefix)}"
    return path


def _json_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return None
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _clean_value(value: object, preferred: tuple[str, ...] = ()) -> str | None:
    """Return a readable row value, omitting nulls, placeholders, and raw objects."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Mapping):
        for key in preferred:
            if key in value:
                return _clean_value(value[key], preferred)
        return None
    if isinstance(value, (list, tuple)):
        return _clean_value(value[0], preferred) if len(value) == 1 else None
    if isinstance(value, Real):
        number = float(value)
        return f"{int(number)}" if number.is_integer() else f"{number:g}"
    if not isinstance(value, str):
        return None

    text = value.strip()
    if text.casefold() in PLACEHOLDERS:
        return None
    if text.startswith(("{", "[")):
        return _clean_value(_json_mapping(text), preferred)
    return text


def _row_value(row: pd.Series, aliases: tuple[str, ...], preferred: tuple[str, ...] = ()) -> str | None:
    """Read a direct field or its optional tags_json projection without inventing values."""
    for alias in aliases:
        if alias in row.index:
            value = _clean_value(row[alias], preferred)
            if value is not None:
                return value

    for container_name in ("tags_json", "extra_json"):
        container = _json_mapping(row.get(container_name))
        if container is None:
            continue
        for alias in aliases:
            key = alias.removeprefix("tag_")
            value = _clean_value(container.get(key), preferred)
            if value is not None:
                return value
    return None


def _metric_unit(metric: str, unit: object = None) -> str:
    """Normalize known units and infer only units encoded by the metric contract."""
    explicit = _clean_value(unit)
    if explicit is not None:
        normalized = explicit.casefold()
        return {
            "tokens/s": "tok/s",
            "token/s": "tok/s",
            "seconds": "s",
            "second": "s",
            "ratio": "%",
            "percent": "%",
            "x": "x",
            "round": "rounds",
        }.get(normalized, explicit)

    key = metric.casefold()
    if "requests_per_second" in key or "requests_per_s" in key:
        return "req/s"
    if any(part in key for part in ("tokens_per_second", "tok_s", "throughput")):
        return "tok/s"
    if key.endswith("_ms") or any(part in key for part in ("_p50_ms", "_p90_ms", "_p99_ms")):
        return "ms"
    if key.endswith("_s") or "_duration_" in key or "_latency_" in key:
        return "s"
    if any(part in key for part in ("accuracy", "_rate", "ratio", "pass_at")):
        return "%"
    if "sample_len" in key or "token" in key:
        return "tokens"
    return "unitless"


def format_value(value: object, unit: str) -> str:
    """Format a measurement and its unit together for every user-facing value."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return ""
    number = float(number)
    if unit == "%":
        return f"{number * 100:.1f}%"
    if unit == "x":
        return f"{number:.2f}x"
    if unit == "tok/s":
        return f"{number:.1f} tok/s"
    if unit == "req/s":
        return f"{number:.2f} req/s"
    if unit in {"ms", "s"}:
        return f"{number:.2f} {unit}"
    if unit == "rounds":
        return f"{number:g} rounds"
    if unit == "tokens":
        return f"{number:,.0f} tokens" if number.is_integer() else f"{number:,.1f} tokens"
    if unit == "bool":
        return "true" if number >= 0.5 else "false"
    if number.is_integer():
        return f"{number:,.0f} {unit}"
    return f"{number:.3g} {unit}"


def humanize(value: object) -> str:
    text = str(value).replace("_", " ").replace("-", " ").strip()
    text = re.sub(r"\s+", " ", text)
    words = []
    for word in text.split():
        lowered = word.casefold()
        words.append(
            {
                "arc": "ARC",
                "gsm8k": "GSM8K",
                "itl": "ITL",
                "ttft": "TTFT",
                "tpot": "TPOT",
                "vllm": "vLLM",
            }.get(lowered, word.capitalize())
        )
    return " ".join(words)


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


def load_local_run_index(data_dir: Path) -> tuple[dict[str, dict[str, object]], pd.Timestamp | None]:
    """Read the repository run index beside a local Parquet directory when present."""
    path = data_dir.parent / RUN_INDEX_PATH
    try:
        payload = json.loads(path.read_text())
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
    """Add row-backed dimensions and retain incomplete workloads as separate runs."""
    df = df.copy()

    def run_entry(row: pd.Series) -> dict[str, object]:
        path = _clean_value(row.get("source_path"))
        return index.get(path or "", {})

    def evidence_status(row: pd.Series) -> str:
        entry = run_entry(row)
        if entry:
            return str(entry.get("status", "experimental")).casefold()
        if historical_cutoff is not None and row["timestamp"] < historical_cutoff:
            return "scored"
        return "experimental"

    df["evidence_status"] = df.apply(evidence_status, axis="columns")
    df["variant"] = df.apply(lambda row: _clean_value(run_entry(row).get("variant")), axis="columns")
    df["context_band"] = df.apply(
        lambda row: _clean_value(run_entry(row).get("context_band")), axis="columns"
    )

    preferred = {
        "accelerator": ("model", "name", "chip", "accelerator"),
        "engine": ("engine", "name", "runtime"),
        "backend": ("backend", "name"),
        "quant_format": ("format", "quantization", "quant"),
    }

    def dimension_value(row: pd.Series, name: str, aliases: tuple[str, ...]) -> str | None:
        value = _row_value(row, aliases, preferred.get(name, ()))
        if value is not None or name not in {"engine", "backend"}:
            return value
        serving = _json_mapping(row.get("serving"))
        return _clean_value(serving.get(name), preferred.get(name, ())) if serving else None

    for name, (_, aliases) in FILTER_SPECS.items():
        df[name] = df.apply(
            lambda row, key=name, keys=aliases: dimension_value(row, key, keys),
            axis="columns",
        )
    df["output_budget_tokens"] = [
        value if value is not None else _clean_value(run_entry(row).get("max_output_tokens"))
        for (_, row), value in zip(
            df.iterrows(),
            df["output_budget_tokens"],
            strict=False,
        )
    ]

    def completion(row: pd.Series) -> str | None:
        value = _row_value(
            row,
            ("completion_state", "tag_completion_state", "finish_reason", "tag_finish_reasons"),
            ("state", "status", "finish_reason", "reason"),
        )
        if value is not None:
            return value
        truncated = _row_value(row, ("truncated_rate", "tag_truncated_rate"))
        if truncated is not None:
            return "complete" if float(truncated) == 0 else "truncated"
        return _clean_value(run_entry(row).get("completion_state"))

    df["completion_state"] = df.apply(completion, axis="columns")
    df["display_unit"] = [
        _metric_unit(str(metric), unit)
        for metric, unit in zip(
            df.get("metric", pd.Series("", index=df.index)),
            df.get("unit", pd.Series(None, index=df.index)),
            strict=False,
        )
    ]
    df["model_short"] = df["model"].map(
        lambda value: short_model(str(value)) if _clean_value(value) is not None else ""
    )
    df["workload_complete"] = df.loc[:, list(WORKLOAD_FIELDS)].notna().all(axis="columns")
    efforts = df.get("reasoning_effort", pd.Series(None, index=df.index)).map(_clean_value).fillna("unstated")
    variants = df["variant"].map(_clean_value)
    row_ids = df.get("_row_id", pd.Series(None, index=df.index))
    if "source_path" in df:
        row_ids = row_ids.fillna(df["source_path"].astype(str) + ":" + df.index.astype(str))
    else:
        row_ids = row_ids.fillna(df.index.astype(str))

    identity_fields = (
        "model",
        "machine",
        "accelerator",
        "engine",
        "backend",
        "quant_format",
        "agents",
        "allocated_context",
        "context_tokens",
        "prompt_tokens",
        "output_budget_tokens",
        "completion_state",
        "display_unit",
    )

    def comparison_key(position: int, row: pd.Series) -> str:
        values = [row.get(field) for field in identity_fields]
        values.append(variants.iloc[position])
        if not row["workload_complete"]:
            values.insert(0, f"row:{row_ids.iloc[position]}")
        joined = "|".join("" if _clean_value(value) is None else str(value) for value in values)
        return f"{joined}|{efforts.iloc[position] or 'unstated'}"

    df["series_key"] = [comparison_key(position, row) for position, (_, row) in enumerate(df.iterrows())]
    df["series_label"] = df.apply(
        lambda row: (
            " / ".join(
                value
                for value in (
                    _clean_value(row.get("model_short")),
                    _clean_value(row.get("machine")),
                    _clean_value(row.get("accelerator")),
                    "/".join(
                        value
                        for value in (
                            _clean_value(row.get("engine"), ("engine", "name", "runtime")),
                            _clean_value(row.get("backend"), ("backend", "name")),
                        )
                        if value
                    )
                    or None,
                    _clean_value(row.get("quant_format")),
                )
                if value
            )
            or "Benchmark run"
        ),
        axis="columns",
    )
    return df


def evidence_view(df: pd.DataFrame, status: str = "scored") -> pd.DataFrame:
    """Return only results appropriate for the selected evidence view."""
    normalized = status.casefold()
    aliases = {
        "scored evidence": "scored",
        "experimental / excluded": "experimental",
        "experimental": "experimental",
        "recovered": "recovered",
    }
    return df[df["evidence_status"] == aliases.get(normalized, normalized)].copy()


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

        local_data_dir = (
            Path(os.environ["MLX_BENCHMARKS_LOCAL_DATA_DIR"]).expanduser()
            if os.getenv("MLX_BENCHMARKS_LOCAL_DATA_DIR")
            else None
        )
        fs = None
        if local_data_dir is not None:
            paths = sorted(str(path) for path in local_data_dir.glob("*.parquet"))
        else:
            fs = HfFileSystem()
            try:
                paths = sorted(f"hf://{p}" for p in fs.glob(f"{DATASET}/data/run-canonical-*.parquet"))
            except (FileNotFoundError, OSError):
                paths = []

        if not paths:
            df = empty_data()
            _cache = (time.time(), df)
            return df

        frames = []
        for path in paths:
            frame = pd.read_parquet(path)
            if local_data_dir is not None:
                local_path = Path(path)
                frame["source_path"] = _dataset_path(f"hf://{DATASET}/data/{local_path.name}")
                frame["_row_id"] = [f"{local_path.name}:{position}" for position in range(len(frame))]
            else:
                frame["source_path"] = _dataset_path(path)
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
        raw_suites = set(df["suite"].dropna().unique())
        df = normalize_rows(df)
        if local_data_dir is not None:
            run_index, historical_cutoff = load_local_run_index(local_data_dir)
        else:
            run_index, historical_cutoff = load_run_index(fs)
        df = add_evidence_metadata(df, run_index, historical_cutoff)
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


def row_choices(df: pd.DataFrame, column: str) -> list[str]:
    if df.empty or column not in df:
        return []
    return sorted(
        {value for item in df[column] if (value := _clean_value(item)) is not None},
        key=str.casefold,
    )


def filter_rows(df: pd.DataFrame, selections: dict[str, str | None]) -> pd.DataFrame:
    filtered = df
    for column, selected in selections.items():
        if selected is not None and column in filtered:
            filtered = filtered[filtered[column].astype("string") == selected]
    return filtered


def available_filter_choices(df: pd.DataFrame, column: str, selections: dict[str, str | None]) -> list[str]:
    other_filters = {name: value for name, value in selections.items() if name != column}
    return row_choices(filter_rows(df, other_filters), column)


def suite_choices(df: pd.DataFrame) -> list[str]:
    return row_choices(df, "suite")


def task_choices(df: pd.DataFrame, suite: str | None) -> list[str]:
    if suite is None or df.empty:
        return []
    return row_choices(df[df["suite"] == suite], "name")


def metric_choices(df: pd.DataFrame, suite: str | None, task: str | None) -> list[str]:
    if suite is None or task is None or df.empty:
        return []
    matching = df[(df["suite"] == suite) & (df["name"] == task)]
    return row_choices(matching, "metric")


def unit_choices(df: pd.DataFrame, suite: str | None, task: str | None, metric: str | None) -> list[str]:
    if suite is None or task is None or metric is None or df.empty:
        return []
    matching = df[(df["suite"] == suite) & (df["name"] == task) & (df["metric"] == metric)]
    return row_choices(matching, "display_unit")


def valid_triples(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    if df.empty:
        return []
    return sorted(
        {
            (str(row.suite), str(row.name), str(row.metric))
            for row in df[["suite", "name", "metric"]].dropna().itertuples(index=False)
        }
    )


def top_task_metric(df: pd.DataFrame, suite: str) -> tuple[str | None, str | None]:
    """Choose the populated task and metric with the most distinct models."""
    sub = df[df["suite"] == suite]
    if sub.empty:
        return (None, None)
    counts = sub.groupby(["name", "metric"])["model"].nunique()
    name, metric = counts.idxmax()
    return (str(name), str(metric))


def top_metric(df: pd.DataFrame, suite: str, task: str) -> str | None:
    sub = df[(df["suite"] == suite) & (df["name"] == task)]
    if sub.empty:
        return None
    return str(sub.groupby("metric")["model"].nunique().idxmax())


def best_default(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    """Choose a row-backed suite/task/metric triple with the most models."""
    if df.empty:
        return (None, None, None)
    counts = df.groupby(["suite", "name", "metric"])["model"].nunique()
    suite, task, metric = counts.idxmax()
    return (str(suite), str(task), str(metric))


def format_tokens(value: object) -> str | None:
    cleaned = _clean_value(value)
    if cleaned is None:
        return None
    try:
        number = float(cleaned)
    except ValueError:
        return cleaned
    return f"{number:,.0f}" if number.is_integer() else f"{number:,.1f}"


def workload_details(row: pd.Series) -> str:
    details = []
    for name, label in (
        ("allocated_context", "Allocated context"),
        ("context_tokens", "Context"),
        ("prompt_tokens", "Prompt"),
        ("output_budget_tokens", "Output budget"),
    ):
        value = format_tokens(row.get(name))
        if value is not None:
            details.append(f"{label}: {value} tokens")
    agents = _clean_value(row.get("agents"))
    if agents is not None:
        details.append(f"Agents: {agents}")
    completion = _clean_value(row.get("completion_state"))
    if completion is not None:
        details.append(f"Completion: {completion}")
    if not row.get("workload_complete", False):
        details.append("Workload details incomplete; this run is kept separate")
    return "; ".join(details) or "Workload details were not recorded"


def series_labels(df: pd.DataFrame) -> dict[str, str]:
    if df.empty:
        return {}
    latest = df.sort_values("timestamp").drop_duplicates("series_key", keep="last")
    duplicates = latest["series_label"].value_counts()
    labels = {}
    for row in latest.itertuples(index=False):
        label = row.series_label or "Benchmark run"
        if duplicates.get(label, 0) > 1:
            suffixes = []
            for name, label_part in (
                ("context_tokens", "context"),
                ("prompt_tokens", "prompt"),
                ("output_budget_tokens", "output"),
            ):
                value = format_tokens(getattr(row, name, None))
                if value is not None:
                    suffixes.append(f"{label_part} {value} tokens")
            suffix = ", ".join(suffixes) or pd.Timestamp(row.timestamp).strftime("%Y-%m-%d")
            label = f"{label} ({suffix})"
        labels[row.series_key] = label
    grouped = {}
    for key, label in labels.items():
        grouped.setdefault(label, []).append(key)
    for label, keys in grouped.items():
        if len(keys) > 1:
            for number, key in enumerate(sorted(keys), start=1):
                labels[key] = f"{label} (run {number})"
    return labels


# ── Chart builders ────────────────────────────────────────────────────────────


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font_size=16,
    )
    fig.update_layout(height=360, margin={"l": 24, "r": 24, "t": 32, "b": 32})
    return fig


def selected_rows(
    df: pd.DataFrame, suite: str | None, task: str | None, metric: str | None, unit: str | None
) -> pd.DataFrame:
    if suite is None or task is None or metric is None:
        return df.iloc[0:0].copy()
    selected = df[(df["suite"] == suite) & (df["name"] == task) & (df["metric"] == metric)].copy()
    if unit is not None and "display_unit" in selected:
        selected = selected[selected["display_unit"] == unit]
    return selected


def bar_chart(
    df: pd.DataFrame, suite: str | None, task: str | None, metric: str | None, unit: str | None = None
) -> go.Figure:
    """Show the latest row for each fully identified comparison."""
    sub = selected_rows(df, suite, task, metric, unit)
    if sub.empty:
        return empty_figure("No data matches these filters. Clear a filter or choose another task.")

    sub = sub.sort_values("timestamp").drop_duplicates("series_key", keep="last")
    labels = series_labels(sub)
    sub["label"] = sub["series_key"].map(labels)
    sub["display_value"] = sub.apply(
        lambda row: format_value(row["value"], row["display_unit"]), axis="columns"
    )
    sub["workload"] = sub.apply(workload_details, axis="columns")
    sub["run_date"] = pd.to_datetime(sub["timestamp"], utc=True).dt.strftime("%Y-%m-%d %H:%M UTC")
    sub = sub.sort_values("value", ascending=True)
    units = row_choices(sub, "display_unit")
    axis_unit = units[0] if len(units) == 1 else "unit varies by row"
    state = _clean_value(sub.iloc[0].get("evidence_status")) or "experimental"
    color = {"scored": "#126b72", "experimental": "#a95e19", "recovered": "#675493"}.get(state, "#647783")

    fig = px.bar(
        sub,
        x="value",
        y="label",
        orientation="h",
        text="display_value",
        custom_data=["workload", "evidence_status", "run_date"],
        labels={"value": f"{humanize(metric)} ({axis_unit})", "label": "Model and system"},
        title=f"{humanize(task)} — {humanize(metric)}",
        color_discrete_sequence=[color],
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>%{text}<br>%{customdata[0]}"
            "<br>Evidence: %{customdata[1]}<br>Run: %{customdata[2]}<extra></extra>"
        ),
        cliponaxis=False,
    )
    fig.update_layout(
        height=max(360, len(sub) * 48),
        margin={"l": 250, "r": 70, "t": 62, "b": 58},
        yaxis_title="",
        font_size=13,
        showlegend=False,
    )
    return fig


def trend_chart(
    df: pd.DataFrame,
    suite: str | None,
    task: str | None,
    metric: str | None,
    models: list[str] | None,
    unit: str | None = None,
) -> go.Figure:
    """Plot only compatible rows from the active suite, task, metric, and filters."""
    sub = selected_rows(df, suite, task, metric, unit)
    if sub.empty:
        return empty_figure("No trend rows match these filters.")
    if models is None:
        counts = sub["series_key"].value_counts().sort_values(ascending=False)
        models = counts.head(6).index.tolist()
    elif not models:
        return empty_figure("Select a comparison to show its trend.")
    sub = sub[sub["series_key"].isin(models)].sort_values("timestamp").copy()
    if sub.empty:
        return empty_figure("Choose a comparison with data for this task and metric.")

    labels = series_labels(sub)
    sub["label"] = sub["series_key"].map(labels)
    sub["display_value"] = sub.apply(
        lambda row: format_value(row["value"], row["display_unit"]), axis="columns"
    )
    sub["workload"] = sub.apply(workload_details, axis="columns")
    units = row_choices(sub, "display_unit")
    axis_unit = units[0] if len(units) == 1 else "unit varies by row"
    fig = px.line(
        sub,
        x="timestamp",
        y="value",
        color="label",
        markers=True,
        custom_data=["display_value", "workload", "evidence_status"],
        labels={
            "value": f"{humanize(metric)} ({axis_unit})",
            "timestamp": "Run time (UTC)",
            "label": "Model and system",
        },
        title=f"{humanize(task)} — {humanize(metric)} over time",
        color_discrete_sequence=["#126b72", "#a95e19", "#675493", "#4b718e", "#98723b", "#4e7164"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{fullData.name}</b><br>%{customdata[0]}<br>%{customdata[1]}"
            "<br>Evidence: %{customdata[2]}<br>Run: %{x|%Y-%m-%d %H:%M UTC}<extra></extra>"
        )
    )
    fig.update_layout(height=440, font_size=13, legend_title="", margin={"l": 72, "r": 24, "t": 62, "b": 62})
    return fig


def summary_table(
    df: pd.DataFrame,
    suite: str | None,
    task: str | None,
    metric: str | None,
    unit: str | None = None,
) -> pd.DataFrame:
    """Summarize only the active task and unit with evidence and run context."""
    selected = selected_rows(df, suite, task, metric, unit)
    if selected.empty:
        return pd.DataFrame({"No matching runs": []})

    row_counts = selected.groupby("series_key").size().rename("Runs (count)")
    latest = selected.sort_values("timestamp").drop_duplicates("series_key", keep="last").copy()
    latest["Runs (count)"] = latest["series_key"].map(row_counts)
    labels = series_labels(selected)
    latest["Model and system"] = latest["series_key"].map(labels)
    active_unit = unit or row_choices(latest, "display_unit")[0]
    latest[f"Value ({active_unit})"] = latest["value"].map(lambda value: format_value(value, active_unit))
    latest["Evidence"] = (
        latest["evidence_status"]
        .map({"scored": "Scored", "experimental": "Experimental", "recovered": "Recovered"})
        .fillna("Experimental")
    )
    latest["Run date (UTC)"] = pd.to_datetime(latest["timestamp"], utc=True).dt.strftime("%Y-%m-%d %H:%M")

    columns = ["Model and system"]
    row_backed_fields = (
        ("machine", "Machine"),
        ("accelerator", "Accelerator model"),
        ("engine", "Engine"),
        ("backend", "Backend"),
        ("quant_format", "Quant format"),
        ("agents", "Agents (count)"),
        ("allocated_context", "Allocated context (tokens)"),
        ("context_tokens", "Context (tokens)"),
        ("prompt_tokens", "Prompt (tokens)"),
        ("output_budget_tokens", "Output budget (tokens)"),
        ("completion_state", "Completion"),
    )
    for field, label in row_backed_fields:
        if field in latest and latest[field].map(_clean_value).notna().any():
            if field in {
                "agents",
                "allocated_context",
                "context_tokens",
                "prompt_tokens",
                "output_budget_tokens",
            }:
                value_unit = "agent" if field == "agents" else "tokens"
                latest[label] = latest[field].map(
                    lambda value, suffix=value_unit: (
                        f"{format_tokens(value)} {suffix}" if _clean_value(value) is not None else ""
                    )
                )
            else:
                latest[label] = latest[field].map(lambda value: _clean_value(value) or "")
            columns.append(label)

    sample_value = latest.apply(
        lambda row: _row_value(row, ("n_samples", "sample_count", "n_limit", "tag_n_limit")), axis="columns"
    )
    repeat_value = latest.apply(
        lambda row: _row_value(
            row, ("measured_repetitions", "tag_measured_repetitions", "repetitions", "tag_repetitions")
        ),
        axis="columns",
    )
    if sample_value.map(_clean_value).notna().any():
        latest["Samples (count)"] = sample_value
        columns.append("Samples (count)")
    if repeat_value.map(_clean_value).notna().any():
        latest["Repeats (count)"] = repeat_value
        columns.append("Repeats (count)")

    columns.extend([f"Value ({active_unit})", "Evidence", "Runs (count)", "Run date (UTC)"])
    return latest[columns].reset_index(drop=True)


# ── Gradio UI ─────────────────────────────────────────────────────────────────


def build_ui():
    all_rows = load_data()
    evidence_options = [("Scored", "scored"), ("Experimental", "experimental"), ("Recovered", "recovered")]
    initial_evidence = "scored"
    initial_rows = evidence_view(all_rows, initial_evidence)
    default_suite, default_task, default_metric = best_default(initial_rows)
    initial_filter_values: dict[str, str | None] = dict.fromkeys(FILTER_SPECS)
    filtered_rows = filter_rows(initial_rows, initial_filter_values)
    suites = suite_choices(filtered_rows)
    if default_suite not in suites:
        default_suite, default_task, default_metric = best_default(filtered_rows)
    tasks = task_choices(filtered_rows, default_suite)
    if default_task not in tasks and default_suite is not None:
        default_task, _ = top_task_metric(filtered_rows, default_suite)
    metrics = metric_choices(filtered_rows, default_suite, default_task)
    if default_metric not in metrics and default_suite is not None and default_task is not None:
        default_metric = top_metric(filtered_rows, default_suite, default_task)
    units = unit_choices(filtered_rows, default_suite, default_task, default_metric)
    initial_unit = units[0] if units else None
    active_rows = selected_rows(filtered_rows, default_suite, default_task, default_metric, initial_unit)
    initial_labels = series_labels(active_rows)
    initial_models = active_rows["series_key"].value_counts().head(6).index.tolist()

    def filter_option_label(name: str, value: str) -> str:
        if name == "agents":
            return f"{value} agent" if value == "1" else f"{value} agents"
        if name in {"allocated_context", "context_tokens", "prompt_tokens", "output_budget_tokens"}:
            return f"{format_tokens(value)} tokens"
        return value

    def metric_options(data: pd.DataFrame, suite: str | None, task: str | None):
        options = []
        for metric_value in metric_choices(data, suite, task):
            metric_units = unit_choices(data, suite, task, metric_value)
            unit_text = f" ({', '.join(metric_units)})" if metric_units else ""
            options.append((f"{humanize(metric_value)}{unit_text}", metric_value))
        return options

    def dropdown(
        values: list[str],
        label: str,
        value: str | None = None,
        *,
        human_names: bool = False,
        interactive: bool | None = None,
        elem_id: str | None = None,
        filter_name: str | None = None,
    ):
        choices = [
            (
                filter_option_label(filter_name, item)
                if filter_name
                else humanize(item)
                if human_names
                else item,
                item,
            )
            for item in values
        ]
        enabled = bool(values) if interactive is None else interactive
        return gr.Dropdown(
            choices=choices,
            value=value if value in values else None,
            label=label,
            interactive=enabled,
            allow_custom_value=False,
            elem_id=elem_id,
        )

    def selection_status(
        data: pd.DataFrame,
        rows: pd.DataFrame,
        evidence: str,
        suite: str | None,
        task: str | None,
        metric: str | None,
        unit: str | None,
    ) -> str:
        counts = data["evidence_status"].value_counts() if "evidence_status" in data else {}
        evidence_totals = "; ".join(
            f"{label}: {int(counts.get(key, 0))} rows"
            for key, label in (
                ("scored", "Scored"),
                ("experimental", "Experimental"),
                ("recovered", "Recovered"),
            )
        )
        if rows.empty:
            message = (
                f"No {evidence.capitalize()} rows match these filters. Clear a filter or choose another task."
            )
            return f"{message}  \nEvidence in this dataset: {evidence_totals}."
        model_count = rows["model"].map(_clean_value).nunique() if "model" in rows else 0
        comparison_count = rows["series_key"].nunique() if "series_key" in rows else 0
        comparison = " / ".join(humanize(value) for value in (suite, task, metric) if value)
        message = (
            f"**{len(rows):,} results** across **{comparison_count:,} comparisons** and "
            f"**{model_count:,} models**. {comparison} ({unit or 'unit not recorded'})."
        )
        missing = [label for name, (label, _) in FILTER_SPECS.items() if not row_choices(rows, name)]
        if missing:
            message += "  \nNot recorded for these rows: " + ", ".join(missing) + "."
        incomplete = (
            int((~rows["workload_complete"].fillna(False)).sum()) if "workload_complete" in rows else 0
        )
        if incomplete:
            message += (
                f"  \nWorkload details are incomplete on {incomplete:,} rows; each remains a separate run."
            )
        repeat_counts = rows.groupby("series_key").size() if "series_key" in rows else pd.Series(dtype=int)
        if not repeat_counts.gt(1).any():
            message += "  \nTrend shows individual results until comparable runs repeat over time."
        return f"{message}  \nEvidence in this dataset: {evidence_totals}."

    def default_models(rows: pd.DataFrame) -> list[str]:
        if rows.empty:
            return []
        return rows["series_key"].value_counts().head(6).index.tolist()

    def synchronize(evidence, suite, task, metric, unit, selected_models, *filter_values):
        data = evidence_view(load_data(), str(evidence))
        selections = dict(zip(FILTER_SPECS, filter_values, strict=False))
        filter_updates = []
        for name, (label, _) in FILTER_SPECS.items():
            choices = available_filter_choices(data, name, selections)
            selected = selections.get(name)
            if selected not in choices:
                selected = None
            selections[name] = selected
            filter_updates.append(
                dropdown(
                    choices,
                    label,
                    selected,
                    interactive=bool(choices),
                    elem_id=f"filter-{name}",
                    filter_name=name,
                )
            )

        filtered = filter_rows(data, selections)
        available_suites = suite_choices(filtered)
        if suite not in available_suites:
            suite, task, metric = best_default(filtered)
        available_tasks = task_choices(filtered, suite)
        if task not in available_tasks and suite is not None:
            task, _ = top_task_metric(filtered, suite)
        available_metrics = metric_choices(filtered, suite, task)
        if metric not in available_metrics and suite is not None and task is not None:
            metric = top_metric(filtered, suite, task)
        available_units = unit_choices(filtered, suite, task, metric)
        if unit not in available_units:
            matching = selected_rows(filtered, suite, task, metric, None)
            counts = matching["display_unit"].value_counts() if not matching.empty else {}
            unit = max(available_units, key=lambda item: counts.get(item, 0)) if available_units else None

        active = selected_rows(filtered, suite, task, metric, unit)
        labels = series_labels(active)
        valid_models = list(labels)
        chosen_models = [value for value in (selected_models or []) if value in valid_models]
        if not chosen_models:
            chosen_models = default_models(active)
        model_choices = [(labels[key], key) for key in valid_models]

        suite_update = dropdown(available_suites, "Suite", suite, human_names=True)
        task_update = dropdown(available_tasks, "Task", task, human_names=True)
        metric_update = gr.Dropdown(
            choices=metric_options(filtered, suite, task),
            value=metric if metric in available_metrics else None,
            label="Metric" if available_metrics else "Metric — no matching rows",
            interactive=bool(available_metrics),
            allow_custom_value=False,
            elem_id="metric",
        )
        unit_update = gr.Dropdown(
            choices=[(value, value) for value in available_units],
            value=unit,
            label="Unit" if available_units else "Unit — no matching rows",
            interactive=len(available_units) > 1,
            allow_custom_value=False,
            elem_id="unit",
        )
        bar = bar_chart(filtered, suite, task, metric, unit)
        trend = trend_chart(filtered, suite, task, metric, chosen_models, unit)
        table = summary_table(filtered, suite, task, metric, unit)
        status = selection_status(data, active, str(evidence), suite, task, metric, unit)
        model_update = gr.CheckboxGroup(
            choices=model_choices,
            value=chosen_models,
            label="Compatible comparisons with data",
            elem_id="trend-models",
        )
        return (
            suite_update,
            task_update,
            metric_update,
            unit_update,
            *filter_updates,
            bar,
            model_update,
            trend,
            table,
            status,
        )

    def update_trend_from_models(evidence, suite, task, metric, unit, *values):
        selected_models = values[-1] if values else []
        filter_values = values[:-1]
        selections = dict(zip(FILTER_SPECS, filter_values, strict=False))
        data = filter_rows(evidence_view(load_data(), str(evidence)), selections)
        return trend_chart(data, suite, task, metric, selected_models, unit)

    def synchronize_evidence(evidence, *filter_values):
        return synchronize(evidence, None, None, None, None, None, *filter_values)

    def synchronize_suite(evidence, suite, *filter_values):
        return synchronize(evidence, suite, None, None, None, None, *filter_values)

    def synchronize_task(evidence, suite, task, *filter_values):
        return synchronize(evidence, suite, task, None, None, None, *filter_values)

    def synchronize_metric(evidence, suite, task, metric, *filter_values):
        return synchronize(evidence, suite, task, metric, None, None, *filter_values)

    def synchronize_unit(evidence, suite, task, metric, unit, *filter_values):
        return synchronize(evidence, suite, task, metric, unit, None, *filter_values)

    def synchronize_filters(evidence, *filter_values):
        return synchronize(evidence, None, None, None, None, None, *filter_values)

    def refresh(evidence, suite, task, metric, unit, *filter_values):
        global _cache
        with _cache_lock:
            _cache = None
        return synchronize(evidence, suite, task, metric, unit, None, *filter_values)

    with gr.Blocks(title="MLX Benchmarks") as demo:
        gr.Markdown("# MLX Benchmarks", elem_id="title")
        gr.Markdown(
            "Compare one task and metric at a time. Workload, machine, and evidence stay attached to every run.  \n"
            "Dataset: [MLX Benchmarks](https://huggingface.co/datasets/JacobPEvans/mlx-benchmarks)",
            elem_id="subtitle",
        )
        with gr.Row():
            evidence_dd = gr.Radio(
                choices=evidence_options,
                value=initial_evidence,
                label="Evidence",
                scale=1,
                elem_id="evidence",
            )
        with gr.Row():
            suite_dd = dropdown(suites, "Suite", default_suite, human_names=True, elem_id="suite")
            task_dd = dropdown(tasks, "Task", default_task, human_names=True, elem_id="task")
            metric_dd = gr.Dropdown(
                choices=metric_options(filtered_rows, default_suite, default_task),
                value=default_metric,
                label="Metric",
                interactive=bool(metrics),
                allow_custom_value=False,
                elem_id="metric",
            )
            unit_dd = gr.Dropdown(
                choices=[(value, value) for value in units],
                value=initial_unit,
                label="Unit",
                interactive=len(units) > 1,
                allow_custom_value=False,
                elem_id="unit",
            )
            refresh_btn = gr.Button("Refresh data", scale=0)

        with gr.Accordion("Filter by machine and workload", open=True):
            filter_controls = []
            filter_items = list(FILTER_SPECS.items())
            for start in range(0, len(filter_items), 5):
                with gr.Row():
                    for name, (label, _) in filter_items[start : start + 5]:
                        choices = row_choices(initial_rows, name)
                        filter_controls.append(
                            dropdown(
                                choices,
                                label,
                                elem_id=f"filter-{name}",
                                filter_name=name,
                            )
                        )

        status = gr.Markdown(
            selection_status(
                all_rows,
                active_rows,
                initial_evidence,
                default_suite,
                default_task,
                default_metric,
                initial_unit,
            ),
            elem_id="selection-status",
        )
        with gr.Tabs():
            with gr.Tab("Latest"):
                bar_plot = gr.Plot(
                    value=bar_chart(initial_rows, default_suite, default_task, default_metric, initial_unit),
                    label="Latest comparable run",
                )
            with gr.Tab("Trend"):
                model_select = gr.CheckboxGroup(
                    choices=[(initial_labels[key], key) for key in initial_models],
                    value=initial_models,
                    label="Compatible comparisons with data",
                    elem_id="trend-models",
                )
                trend_plot = gr.Plot(
                    value=trend_chart(
                        initial_rows,
                        default_suite,
                        default_task,
                        default_metric,
                        initial_models,
                        initial_unit,
                    ),
                    label="Results over time",
                )
            with gr.Tab("Summary"):
                table_out = gr.DataFrame(
                    value=summary_table(
                        initial_rows, default_suite, default_task, default_metric, initial_unit
                    ),
                    interactive=False,
                    label="Latest run per comparison",
                )

        filter_controls = tuple(filter_controls)
        sync_outputs = [
            suite_dd,
            task_dd,
            metric_dd,
            unit_dd,
            *filter_controls,
            bar_plot,
            model_select,
            trend_plot,
            table_out,
            status,
        ]
        evidence_dd.input(
            synchronize_evidence,
            inputs=[evidence_dd, *filter_controls],
            outputs=sync_outputs,
        )
        suite_dd.input(
            synchronize_suite,
            inputs=[evidence_dd, suite_dd, *filter_controls],
            outputs=sync_outputs,
        )
        task_dd.input(
            synchronize_task,
            inputs=[evidence_dd, suite_dd, task_dd, *filter_controls],
            outputs=sync_outputs,
        )
        metric_dd.input(
            synchronize_metric,
            inputs=[evidence_dd, suite_dd, task_dd, metric_dd, *filter_controls],
            outputs=sync_outputs,
        )
        unit_dd.input(
            synchronize_unit,
            inputs=[evidence_dd, suite_dd, task_dd, metric_dd, unit_dd, *filter_controls],
            outputs=sync_outputs,
        )
        for control in filter_controls:
            control.input(
                synchronize_filters,
                inputs=[evidence_dd, *filter_controls],
                outputs=sync_outputs,
            )

        model_select.input(
            update_trend_from_models,
            inputs=[evidence_dd, suite_dd, task_dd, metric_dd, unit_dd, *filter_controls, model_select],
            outputs=[trend_plot],
        )
        refresh_btn.click(
            refresh,
            inputs=[evidence_dd, suite_dd, task_dd, metric_dd, unit_dd, *filter_controls],
            outputs=sync_outputs,
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(css=CSS, theme=THEME)
