"""Chart builders return Plotly figures, never raise, even on empty data."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# Add the space/ directory to sys.path before resolving app via importlib so
# ruff's top-of-file import rule (E402) stays satisfied without a suppression.
SPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPACE_ROOT))
app = importlib.import_module("app")

SAMPLE_ROWS = [
    {
        "timestamp": "2026-04-24T18:30:00Z",
        "suite": "reasoning",
        "name": "gsm8k_cot_zeroshot",
        "metric": "exact_match_flexible",
        "model": "mlx-community/Qwen3.5-9B-MLX-4bit",
        "value": 0.8,
    },
    {
        "timestamp": "2026-04-24T19:00:00Z",
        "suite": "reasoning",
        "name": "gsm8k_cot_zeroshot",
        "metric": "exact_match_flexible",
        "model": "mlx-community/gemma-4-e4b-it-4bit",
        "value": 0.6,
    },
]


def _sample_df() -> pd.DataFrame:
    df = pd.DataFrame(SAMPLE_ROWS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["source_path"] = ["data/scored-a.parquet", "data/scored-b.parquet"]
    df = app.add_evidence_metadata(df, {}, pd.Timestamp("2026-08-25T00:00:00Z"))
    df["model_short"] = df["model"].apply(app.short_model)
    return df


def test_empty_data_returns_annotated_figure() -> None:
    fig = app.bar_chart(app.empty_data(), "reasoning", "gsm8k_cot_zeroshot", "exact_match_flexible")
    assert isinstance(fig, go.Figure)
    # Annotation present when no data
    assert len(fig.layout.annotations) == 1
    assert "No data" in fig.layout.annotations[0].text


def test_bar_chart_renders_with_rows() -> None:
    df = _sample_df()
    fig = app.bar_chart(df, "reasoning", "gsm8k_cot_zeroshot", "exact_match_flexible")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1  # single trace
    # Two models => two bars
    bar = fig.data[0]
    assert len(bar.y) == 2


def test_trend_chart_renders_with_rows() -> None:
    df = _sample_df()
    fig = app.trend_chart(
        df,
        "reasoning",
        "gsm8k_cot_zeroshot",
        "exact_match_flexible",
        models=df["series_key"].tolist(),
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_summary_table_returns_dataframe() -> None:
    df = _sample_df()
    pivot = app.summary_table(df, "reasoning", "exact_match_flexible")
    assert isinstance(pivot, pd.DataFrame)
    assert "Comparison series" in pivot.columns or pivot.empty


def test_short_model_strips_common_prefixes() -> None:
    assert app.short_model("mlx-community/Qwen3.5-9B-MLX-4bit") == "Qwen3.5-9B-MLX-4bit"
    assert app.short_model("openrouter/openai/gpt-5-mini") == "openrouter/gpt-5-mini"
    assert app.short_model("plain-name") == "plain-name"


def test_normalize_rows_coalesces_layouts_and_drops_non_measurements() -> None:
    raw = pd.DataFrame(
        [
            # Flat layout, real measurement — kept as-is.
            {"suite": "reasoning", "model": "m/a", "name": "gsm8k", "metric": "exact_match", "value": 0.7},
            # Nested layout (older shards) — must be surfaced via coalescing.
            {
                "suite": "tool-calling",
                "model": "m/b",
                "metric_name": "should-call-tool",
                "metric_metric": "accuracy",
                "metric_value": 0.9,
            },
            # Skipped CI run (no MLX server) — dropped.
            {"suite": "code-accuracy", "model": "m/c", "skipped": True, "metric_value": None},
            # No measurement in either layout — dropped.
            {"suite": "framework-eval", "model": "m/d", "name": None, "value": None},
        ]
    )
    out = app.normalize_rows(raw)
    assert set(out["suite"]) == {"reasoning", "tool-calling"}
    surfaced = out[out["suite"] == "tool-calling"].iloc[0]
    assert surfaced["name"] == "should-call-tool"
    assert surfaced["metric"] == "accuracy"
    assert surfaced["value"] == 0.9


def test_unindexed_current_rows_are_experimental() -> None:
    df = pd.DataFrame(
        [{"timestamp": "2026-08-25T00:00:00Z", "model": "m/a", "source_path": "data/new.parquet"}]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    out = app.add_evidence_metadata(df, {}, pd.Timestamp("2026-08-25T00:00:00Z"))
    assert out.loc[0, "evidence_status"] == "experimental"
    assert app.evidence_view(out).empty


def test_mtp_run_index_retains_all_historical_shards_as_non_scored() -> None:
    index_path = SPACE_ROOT.parent / "metadata" / "run-index-v1.json"
    entries = json.loads(index_path.read_text())["runs"]
    assert len(entries) == 34
    assert {entry["status"] for entry in entries} <= {"experimental", "recovered"}
    assert all(entry["caveat"] for entry in entries)
    assert {entry["context_band"] for entry in entries} >= {"64k", "128k"}


def test_bar_chart_keeps_variants_in_separate_series() -> None:
    df = _sample_df()
    duplicate = df.iloc[[0]].copy()
    duplicate["variant"] = "MTP default"
    duplicate["series_key"] = " | ".join(
        [
            str(duplicate.iloc[0]["model"]),
            "unknown",
            "unknown",
            "MTP default",
            "unspecified",
            "unspecified",
            "unstated",
        ]
    )
    fig = app.bar_chart(
        pd.concat([df, duplicate], ignore_index=True),
        "reasoning",
        "gsm8k_cot_zeroshot",
        "exact_match_flexible",
    )
    assert len(fig.data[0].y) == 3


def test_two_reasoning_efforts_are_two_series() -> None:
    """Effort is part of the arm, so the same weights at `high` and at `xhigh`
    must not collapse into one bar — `groupby("series_key").last()` would keep
    only whichever ran later and silently drop the comparison."""
    rows = pd.DataFrame([SAMPLE_ROWS[0], SAMPLE_ROWS[0]])
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows["source_path"] = ["data/a.parquet", "data/b.parquet"]
    rows["reasoning_effort"] = ["high", "xhigh"]

    out = app.add_evidence_metadata(rows, {}, pd.Timestamp("2026-08-25T00:00:00Z"))
    assert out["series_key"].nunique() == 2
    assert all("high" in key for key in out["series_key"])


def test_rows_without_the_column_still_get_a_series_key() -> None:
    """Every shard published before the field existed lacks the column entirely;
    reading it must default rather than raise, and label as unstated rather than
    claiming an effort nobody recorded."""
    rows = pd.DataFrame(SAMPLE_ROWS)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows["source_path"] = ["data/a.parquet", "data/b.parquet"]
    assert "reasoning_effort" not in rows.columns

    out = app.add_evidence_metadata(rows, {}, pd.Timestamp("2026-08-25T00:00:00Z"))
    assert all(key.endswith("unstated") for key in out["series_key"])
