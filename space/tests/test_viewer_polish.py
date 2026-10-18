"""Viewer defaults and concise chart labels."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

SPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPACE_ROOT))
app = importlib.import_module("app")


def test_throughput_default_prefers_output_tokens_in_both_modes() -> None:
    records = []
    for prompt_tokens, day in ((4096, "01"), (65536, "02")):
        for machine, value in zip(("Machine A", "Machine B"), (10.0, 20.0), strict=True):
            records.append(
                {
                    "timestamp": f"2026-10-{day}T00:00:00Z",
                    "suite": "throughput",
                    "name": "benchmark_serving",
                    "metric": "throughput_output_toks_per_s",
                    "value": value,
                    "model": "throughput-model",
                    "machine": machine,
                    "prompt_tokens": prompt_tokens,
                    "evidence_status": "scored",
                }
            )
    for machine, value in zip(("Machine A", "Machine B"), (0.6, 0.7), strict=True):
        records.append(
            {
                "timestamp": "2026-10-03T00:00:00Z",
                "suite": "framework-eval",
                "name": "promptfoo-aggregate",
                "metric": "score",
                "value": value,
                "model": "eval-model",
                "machine": machine,
                "evidence_status": "scored",
            }
        )
    rows = pd.DataFrame(records)

    expected = ("throughput", "benchmark_serving", "throughput_output_toks_per_s")
    assert app.top_metric(rows, "throughput", "benchmark_serving") == expected[2]
    for mode in (app.AGENTIC_MODE, app.DIAGNOSTIC_MODE):
        assert app.best_default(app.mode_rows(rows, mode)) == expected

    single_output = app.mode_rows(rows, app.AGENTIC_MODE)
    single_output = single_output[
        (single_output["suite"] == "throughput") & (single_output["metric"] == app.THROUGHPUT_OUTPUT_METRIC)
    ].iloc[[0]]
    assert app.best_default(single_output) == expected
    assert "throughput" in app.suite_choices(single_output)

    eval_rows = rows[rows["suite"] == "framework-eval"]
    assert app.top_metric(eval_rows, "framework-eval", "promptfoo-aggregate") == "score"


def test_default_evidence_is_scored_and_mode_frames_keep_dataset_counts(monkeypatch) -> None:
    rows = pd.DataFrame(
        [
            {"viewer_mode": app.AGENTIC_MODE, "evidence_status": "scored", "suite": "reasoning"},
            {"viewer_mode": app.AGENTIC_MODE, "evidence_status": "recovered", "suite": "reasoning"},
            {"viewer_mode": app.DIAGNOSTIC_MODE, "evidence_status": "experimental", "suite": "throughput"},
        ]
    )
    cache = app._prepare_cache(rows)
    monkeypatch.setattr(app, "_cache", cache)

    def no_parquet_read():
        raise AssertionError("mode changes must use the cached frame")

    monkeypatch.setattr(app, "_load_dataset", no_parquet_read)

    assert app.DEFAULT_EVIDENCE == "scored"
    assert app.cached_viewer_rows(app.AGENTIC_MODE, app.DEFAULT_EVIDENCE).evidence_status.tolist() == [
        "scored"
    ]
    assert app.cached_viewer_rows(app.DIAGNOSTIC_MODE, "all").evidence_status.tolist() == ["experimental"]
    assert cache.evidence_counts == {"scored": 1, "recovered": 1, "experimental": 1}
    assert app.evidence_summary(cache.evidence_counts) == (
        "Evidence in this dataset: Scored: 1 rows; Experimental: 1 rows; Recovered: 1 rows."
    )


def test_default_task_is_the_scored_output_task_with_the_most_rows_in_band() -> None:
    rows = [
        {
            "timestamp": f"2026-10-{number + 1:02d}T00:00:00Z",
            "suite": "throughput",
            "name": "benchmark_serving",
            "metric": app.THROUGHPUT_OUTPUT_METRIC,
            "value": 20 + number,
            "model": f"model-{number % 2}",
            "machine": "RTX PRO 6000 Max-Q 96GB",
            "context_band": "64k",
            "evidence_status": "scored",
        }
        for number in range(6)
    ]
    rows.append(
        {
            "timestamp": "2026-10-20T00:00:00Z",
            "suite": "throughput",
            "name": "throughput_probe",
            "metric": app.THROUGHPUT_OUTPUT_METRIC,
            "value": 30,
            "model": "old-model",
            "machine": "Mac Studio",
            "context_band": "64k",
            "evidence_status": "scored",
        }
    )
    rows.extend(
        {
            "timestamp": f"2026-10-{number + 1:02d}T00:00:00Z",
            "suite": "throughput",
            "name": "experimental_task",
            "metric": app.THROUGHPUT_OUTPUT_METRIC,
            "value": 30,
            "model": f"experimental-{number}",
            "machine": "RTX PRO 6000 Max-Q 96GB",
            "context_band": "64k",
            "evidence_status": "experimental",
        }
        for number in range(8)
    )

    in_band = pd.DataFrame(rows).query("context_band == '64k'")

    assert app.best_default(in_band) == ("throughput", "benchmark_serving", app.THROUGHPUT_OUTPUT_METRIC)


def test_bar_labels_keep_identity_and_only_show_varying_nonempty_knobs() -> None:
    rows = pd.DataFrame(
        [
            {
                "timestamp": f"2026-10-0{number + 1}T00:00:00Z",
                "series_key": f"run-{number}",
                "model": "Qwen3.8-27B-Flash-Next-IQ3_S",
                "machine": "Mac Studio M4 Max 128GB",
                "engine": "llama.cpp",
                "engine_version": "b11669",
                "agents": agents,
                "context_band": "32k",
                "mtp": True,
                "arrival_pattern": None,
                "quant_format": "IQ3_S",
            }
            for number, agents in enumerate((4, 8))
        ]
    )

    labels = app.series_labels(rows)

    assert set(labels) == {"run-0", "run-1"}
    for label, concurrency in zip(labels.values(), (4, 8), strict=True):
        assert len(label) <= app.BAR_LABEL_LIMIT
        assert "Qwen3.8" in label
        assert "Mac Studio M4 Max 128GB" in label
        assert "llama.cpp b11669" in label
        assert f"c {concurrency}" in label
        assert "context band" not in label
        assert "MTP" not in label
        assert "None" not in label


def test_bar_labels_are_unique_bounded_and_exclude_free_text_and_runtime_flags() -> None:
    rows = pd.DataFrame(
        [
            {
                "timestamp": f"2026-10-0{number + 1}T00:00:00Z",
                "series_key": f"run-{number}",
                "model": "mlx-community/Qwen3.8-27B-Flash-Next-IQ3_S",
                "machine": "RTX PRO 6000 Max-Q 96GB",
                "engine": "vLLM",
                "engine_version": "0.31.0",
                "reasoning_effort": "low" if number < 2 else "high",
                "thinking": True,
                "mtp": True,
                "quant_format": "NVFP4",
                "agents": number + 1,
                "variant": f"variant free text {number}",
                "completion_state": f"completion complete {number}/{number + 1}",
                "failed": 0,
                "ignore_eos": True,
            }
            for number in range(4)
        ]
    )

    labels = app.series_labels(rows)

    assert len(set(labels.values())) == len(labels)
    assert all(len(label) <= app.BAR_LABEL_LIMIT for label in labels.values())
    assert all(
        label.startswith("Qwen3.8-27B-Flash-Next-IQ3_S · RTX PRO 6000 Max-Q 96GB · vLLM 0.31.0")
        for label in labels.values()
    )
    assert all("variant" not in label.casefold() for label in labels.values())
    assert all("completion" not in label.casefold() for label in labels.values())
    assert all(
        "failed=" not in label.casefold() and "ignore_eos=" not in label.casefold()
        for label in labels.values()
    )
    assert all(len(label.split(" · ")) <= 7 for label in labels.values())


def test_bar_chart_labels_four_reasoning_efforts_separately() -> None:
    efforts = ("low", "medium", "high", "xhigh")
    rows = pd.DataFrame(
        [
            {
                "timestamp": f"2026-10-0{number + 1}T00:00:00Z",
                "suite": "reasoning",
                "name": "promptfoo-aggregate",
                "metric": "score",
                "value": 0.533 + number * 0.05,
                "display_unit": "%",
                "evidence_status": "scored",
                "model": "nvidia/Qwen3.8-27B-NVFP4",
                "machine": "RTX PRO 6000 Max-Q 96GB",
                "engine": "vLLM",
                "engine_version": "0.31.0",
                "agents": "4",
                "reasoning_effort": effort,
                "series_key": f"reasoning-{effort}",
            }
            for number, effort in enumerate(efforts)
        ]
    )

    figure = app.bar_chart(rows, "reasoning", "promptfoo-aggregate", "score", "%")
    labels = list(figure.data[0].y)

    assert len(labels) == 4
    assert len(set(labels)) == 4
    assert all(f"reasoning {effort}" in label for effort, label in zip(efforts, labels, strict=True))
    assert figure.layout.barmode == "group"


def test_percentage_axis_and_value_labels_share_the_zero_to_hundred_scale() -> None:
    rows = pd.DataFrame(
        [
            {
                "timestamp": f"2026-10-0{number + 1}T00:00:00Z",
                "suite": "coding",
                "name": "coding-replay",
                "metric": "pass_rate",
                "value": value,
                "display_unit": "%",
                "evidence_status": "scored",
                "model": f"model-{number}",
                "machine": "Mac Studio M4 Max 128GB",
                "series_key": f"percent-{number}",
            }
            for number, value in enumerate((0.372, 0.25))
        ]
    )

    figure = app.bar_chart(rows, "coding", "coding-replay", "pass_rate", "%")

    assert sorted(figure.data[0].x) == [25.0, 37.2]
    assert set(figure.data[0].text) == {"25.0%", "37.2%"}
    assert list(figure.layout.xaxis.range) == [0, 100]
    assert figure.layout.yaxis.automargin is True
    assert app.format_value(37.2, "%") == "37.2%"


def test_missing_knob_values_are_omitted_from_the_label() -> None:
    rows = pd.DataFrame(
        [
            {
                "timestamp": "2026-10-01T00:00:00Z",
                "series_key": "missing-mtp",
                "model": "Qwen",
                "machine": "Mac Studio",
                "engine": "vLLM",
                "engine_version": "0.31.0",
                "mtp": None,
            },
            {
                "timestamp": "2026-10-02T00:00:00Z",
                "series_key": "mtp-on",
                "model": "Qwen",
                "machine": "Mac Studio",
                "engine": "vLLM",
                "engine_version": "0.31.0",
                "mtp": True,
            },
        ]
    )

    labels = app.series_labels(rows)

    assert "MTP" not in labels["missing-mtp"]
    assert "MTP true" in labels["mtp-on"]
    assert all("None" not in label for label in labels.values())


def test_hover_retains_full_knob_details() -> None:
    details = app.workload_details(
        pd.Series(
            {
                "workload_complete": True,
                "backend": "Metal",
                "quant_format": "4-bit",
                "agents": 8,
                "context_band": "32k",
                "mtp": True,
                "arrival_pattern": "Staggered",
                "variant": "MTP default",
            }
        )
    )

    assert "Backend: Metal" in details
    assert "Quant: 4-bit" in details
    assert "Agents: 8" in details
    assert "Context band: 32k" in details
    assert "MTP: true" in details
    assert "Arrival: Staggered" in details
    assert "Variant: MTP default" in details
