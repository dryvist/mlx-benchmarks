"""Viewer defaults and concise chart labels."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

SPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPACE_ROOT))
app = importlib.import_module("app")


def test_throughput_default_prefers_output_tokens_and_eval_keeps_score() -> None:
    rows = pd.DataFrame(
        [
            {
                "timestamp": "2026-10-01T00:00:00Z",
                "suite": "throughput",
                "name": "benchmark_serving",
                "metric": metric,
                "value": value,
                "model": f"model-{number}",
                "machine": machine,
                "evidence_status": "scored",
            }
            for metric, values in (
                ("itl_p50_ms", (2.0, 3.0)),
                ("throughput_output_toks_per_s", (10.0, 20.0)),
            )
            for number, (machine, value) in enumerate(zip(("Machine A", "Machine B"), values, strict=True))
        ]
    )

    assert app.top_metric(rows, "throughput", "benchmark_serving") == "throughput_output_toks_per_s"
    assert app.best_default(rows) == ("throughput", "benchmark_serving", "throughput_output_toks_per_s")

    eval_rows = rows.assign(suite="reasoning", name="aggregate", metric="score")
    assert app.top_metric(eval_rows, "reasoning", "aggregate") == "score"


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
