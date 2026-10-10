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


def test_canonical_paths_resolve_to_original_run_index_paths() -> None:
    base = f"hf://{app.DATASET}/"
    assert app._dataset_path(f"{base}data/run-canonical-run-2026-07-01.parquet") == (
        "data/run-2026-07-01.parquet"
    )
    assert app._dataset_path(f"{base}data/run-canonical-train-00000-of-00001.parquet") == (
        "data/train-00000-of-00001.parquet"
    )


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
    pivot = app.summary_table(df, "reasoning", "gsm8k_cot_zeroshot", "exact_match_flexible")
    assert isinstance(pivot, pd.DataFrame)
    assert "Evidence" in pivot.columns or pivot.empty


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
    # The shard files live in the published dataset, not this repo, so the
    # index is checked for non-empty, one entry per shard path, and the shard name shape.
    paths = [entry["path"] for entry in entries]
    assert paths
    assert len(set(paths)) == len(paths)
    assert all(path.startswith("data/run-") and path.endswith(".parquet") for path in paths)
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


def _workload_rows() -> pd.DataFrame:
    rows = pd.DataFrame(
        [
            {
                **SAMPLE_ROWS[0],
                "unit": "tok/s",
                "hostname": "Machine A",
                "chip": "Accelerator A",
                "engine": "MLX",
                "tag_backend": "Metal",
                "quantization": "4-bit",
                "concurrency": 2,
                "tag_configured_window_tokens": 8192,
                "tag_context_tokens_actual": 4096,
                "tag_prompt_tokens": 1024,
                "tag_max_gen_toks": 128,
                "tag_truncated_rate": 0,
                "source_path": "data/first.parquet",
                "_row_id": "first:0",
            },
            {
                **SAMPLE_ROWS[0],
                "unit": "tok/s",
                "hostname": "Machine A",
                "chip": "Accelerator A",
                "engine": "MLX",
                "tag_backend": "Metal",
                "quantization": "4-bit",
                "concurrency": 2,
                "tag_configured_window_tokens": 8192,
                "tag_context_tokens_actual": 16384,
                "tag_prompt_tokens": 8192,
                "tag_max_gen_toks": 256,
                "tag_truncated_rate": 0,
                "source_path": "data/second.parquet",
                "_row_id": "second:0",
            },
        ]
    )
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    return rows


def test_workload_budgets_are_part_of_comparison_identity() -> None:
    rows = app.add_evidence_metadata(
        _workload_rows(),
        {},
        pd.Timestamp("2026-08-25T00:00:00Z"),
    )
    assert rows["workload_complete"].all()
    assert rows["series_key"].nunique() == 2
    assert rows["prompt_tokens"].tolist() == ["1024", "8192"]
    assert rows["output_budget_tokens"].tolist() == ["128", "256"]


def test_incomplete_workload_rows_remain_distinct_and_labels_hide_placeholders() -> None:
    raw = _workload_rows().drop(
        columns=[
            "tag_configured_window_tokens",
            "tag_context_tokens_actual",
            "tag_prompt_tokens",
            "tag_max_gen_toks",
            "tag_truncated_rate",
        ]
    )
    rows = app.add_evidence_metadata(raw, {}, pd.Timestamp("2026-08-25T00:00:00Z"))
    assert not rows["workload_complete"].any()
    assert rows["series_key"].nunique() == 2
    label = rows.iloc[0]["series_label"]
    assert "unknown" not in label.casefold()
    assert "unspecified" not in label.casefold()
    assert "{" not in label
    assert "Machine A" in label
    assert "Accelerator A" not in label
    assert "Metal" not in label
    visible_labels = app.series_labels(rows)
    assert len(set(visible_labels.values())) == len(rows)
    assert all("(run " in value for value in visible_labels.values())


def test_viewer_labels_show_matching_machine_and_accelerator_once_with_date_and_run() -> None:
    raw = _workload_rows()
    raw["hostname"] = "Apple M4 Max"
    raw["chip"] = "Apple M4 Max"
    rows = app.add_evidence_metadata(raw, {}, pd.Timestamp("2026-08-25T00:00:00Z"))

    labels = app.series_labels(rows)

    assert set(labels.values()) == {
        "Qwen3.5-9B-MLX-4bit / Apple M4 Max (2026-04-24) (run 1)",
        "Qwen3.5-9B-MLX-4bit / Apple M4 Max (2026-04-24) (run 2)",
    }
    assert all(label.count("Apple M4 Max") == 1 for label in labels.values())
    assert list(app.series_labels(rows.iloc[[0]]).values()) == [
        "Qwen3.5-9B-MLX-4bit / Apple M4 Max (2026-04-24) (run 1)"
    ]


def test_cascade_options_are_only_row_backed_and_summary_uses_task_and_unit() -> None:
    df = _sample_df()
    other = df.iloc[[0]].copy()
    other["suite"] = "throughput"
    other["name"] = "short-50"
    other["metric"] = "throughput"
    other["value"] = 63.4
    other["unit"] = "tok/s"
    other["display_unit"] = "tok/s"
    rows = pd.concat([df, other], ignore_index=True)

    assert app.suite_choices(rows) == ["reasoning", "throughput"]
    assert app.task_choices(rows, "throughput") == ["short-50"]
    assert app.metric_choices(rows, "throughput", "short-50") == ["throughput"]
    assert app.valid_triples(rows) == [
        ("reasoning", "gsm8k_cot_zeroshot", "exact_match_flexible"),
        ("throughput", "short-50", "throughput"),
    ]
    table = app.summary_table(rows, "throughput", "short-50", "throughput", "tok/s")
    assert table["Value (tok/s)"].tolist() == ["63.4 tok/s"]
    assert "unitless" not in table.to_string().casefold()


def test_evidence_views_and_unit_formatting_are_distinct() -> None:
    rows = _workload_rows()
    paths = ["data/scored.parquet", "data/experimental.parquet"]
    rows = pd.concat([rows.iloc[[0]], rows.iloc[[1]]], ignore_index=True)
    rows["source_path"] = paths
    index = {
        "data/scored.parquet": {"status": "scored"},
        "data/experimental.parquet": {"status": "experimental"},
    }
    recovered = rows.iloc[[1]].copy()
    recovered["source_path"] = "data/recovered.parquet"
    rows = pd.concat([rows, recovered], ignore_index=True)
    index["data/recovered.parquet"] = {"status": "recovered"}
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    enriched = app.add_evidence_metadata(rows, index, None)

    assert len(app.evidence_view(enriched, "scored")) == 1
    assert len(app.evidence_view(enriched, "experimental")) == 1
    assert len(app.evidence_view(enriched, "recovered")) == 1
    assert app.format_value(0.875, "%") == "87.5%"
    assert app.format_value(63.4, "tok/s") == "63.4 tok/s"


def test_campaign_dimension_columns_drive_filters_and_comparison_identity() -> None:
    rows = _workload_rows()
    rows["campaign_hardware_machine"] = "Benchmark workstation"
    rows["campaign_hardware_accelerator_model"] = "RTX PRO 6000"
    rows["campaign_software_engine"] = "llama.cpp"
    rows["campaign_software_backend"] = "CUDA"
    rows["campaign_model_quantization"] = "Q4_K_M"
    rows["campaign_run_concurrent_agents"] = 4
    rows["campaign_run_allocated_context_tokens"] = 32768
    rows["campaign_run_depth_tokens"] = [4096, 8192]
    rows["campaign_run_prompt_tokens"] = 4096
    rows["campaign_run_output_tokens"] = [128, 256]

    enriched = app.add_evidence_metadata(
        rows,
        {},
        pd.Timestamp("2026-08-25T00:00:00Z"),
    )

    assert app.row_choices(enriched, "machine") == ["Benchmark workstation"]
    assert app.row_choices(enriched, "accelerator") == ["RTX PRO 6000"]
    assert app.row_choices(enriched, "engine") == ["llama.cpp"]
    assert app.row_choices(enriched, "backend") == ["CUDA"]
    assert app.row_choices(enriched, "quant_format") == ["Q4_K_M"]
    assert app.row_choices(enriched, "agents") == ["4"]
    assert app.row_choices(enriched, "allocated_context") == ["32768"]
    assert app.row_choices(enriched, "prompt_tokens") == ["4096"]
    assert enriched["context_tokens"].tolist() == ["4096", "8192"]
    assert enriched["output_budget_tokens"].tolist() == ["128", "256"]
    assert enriched["series_key"].nunique() == 2


def test_request_rate_metric_keeps_request_units() -> None:
    assert app._metric_unit("throughput_requests_per_s") == "req/s"
