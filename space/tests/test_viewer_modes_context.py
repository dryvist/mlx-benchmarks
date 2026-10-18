"""Decision modes and model context metadata for the viewer."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

SPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPACE_ROOT))
app = importlib.import_module("app")


def test_throughput_split_uses_input_length_and_keeps_evaluations_in_agentic_mode() -> None:
    assert (
        app.classify_viewer_row(
            {
                "suite": "throughput",
                "name": "benchmark_serving",
                "metric": "ttft_p99_ms",
                "prompt_tokens": 32768,
            }
        )
        == app.AGENTIC_MODE
    )
    assert (
        app.classify_viewer_row(
            {
                "suite": "throughput-serving",
                "name": "fixed-length",
                "metric": "output_tokens_per_second",
                "prompt_tokens": 131072,
            }
        )
        == app.AGENTIC_MODE
    )
    assert (
        app.classify_viewer_row(
            {
                "suite": "throughput",
                "name": "benchmark_serving",
                "metric": "ttft_p50_ms",
                "prompt_tokens": 4096,
            }
        )
        == app.DIAGNOSTIC_MODE
    )
    assert (
        app.classify_viewer_row(
            {
                "suite": "throughput",
                "name": "benchmark_serving",
                "metric": "ttft_p50_ms",
                "prompt_tokens": None,
            }
        )
        == app.DIAGNOSTIC_MODE
    )
    assert (
        app.classify_viewer_row({"suite": "repo-qa", "name": "complete", "prompt_tokens": None})
        == app.AGENTIC_MODE
    )


def test_all_valid_evidence_includes_decision_and_experimental_rows_only() -> None:
    rows = pd.DataFrame(
        [
            {"name": name, "evidence_status": status}
            for name, status in (
                ("aggregate-score", "scored"),
                ("long-context-throughput", "experimental"),
                ("recovered-result", "recovered"),
                ("invalid-result", "invalid"),
                ("test_1", "scored"),
            )
        ]
    )

    visible = app.evidence_view(rows, "all")

    assert list(visible["name"]) == ["aggregate-score", "long-context-throughput", "recovered-result"]


def test_agentic_default_context_keeps_long_throughput_and_all_evaluation_suites() -> None:
    rows = pd.DataFrame(
        [
            {
                "suite": suite,
                "name": name,
                "metric": metric,
                "prompt_tokens": prompt_tokens,
                "context_band": context_band,
                "evidence_status": status,
            }
            for suite, name, metric, prompt_tokens, context_band, status in (
                (
                    "throughput",
                    "benchmark_serving",
                    "throughput_output_toks_per_s",
                    65536,
                    "64k",
                    "experimental",
                ),
                (
                    "throughput",
                    "benchmark_serving",
                    "throughput_output_toks_per_s",
                    4096,
                    "4k",
                    "scored",
                ),
                ("repo-qa", "complete", "accuracy", None, None, "scored"),
                ("mmlu-pro", "mmlu-pro", "accuracy", None, None, "experimental"),
                ("invalid-suite", "invalid", "accuracy", None, None, "invalid"),
            )
        ]
    )

    agentic = app.mode_rows(rows, app.AGENTIC_MODE)
    visible = app.evidence_view(agentic, "all")
    context_band = app.default_filter_selections(visible, app.AGENTIC_MODE)["context_band"]
    filtered = app.filter_viewer_rows(visible, {"context_band": context_band}, app.AGENTIC_MODE)

    assert context_band == "64k"
    assert set(filtered["suite"]) == {"throughput", "repo-qa", "mmlu-pro"}
    throughput = filtered[filtered["suite"] == "throughput"]
    assert list(throughput["context_band"]) == ["64k"]


def test_default_context_band_prefers_64k_then_nearest_band_at_least_32k() -> None:
    rows = pd.DataFrame(
        [
            {"suite": "throughput", "context_band": band, "prompt_tokens": tokens}
            for band, tokens in (("4k", 4096), ("32k", 32768), ("64k", 65536), ("128k", 131072))
        ]
    )
    assert app.default_context_band(rows) == "64k"
    assert app.default_context_band(rows[rows["context_band"] != "64k"]) == "32k"


def test_agentic_context_filter_keeps_evaluation_suites_visible() -> None:
    rows = pd.DataFrame(
        [
            {"suite": "reasoning", "name": "mmlu-pro", "context_band": None},
            {"suite": "throughput", "name": "benchmark_serving", "context_band": "64k"},
            {"suite": "throughput", "name": "benchmark_serving", "context_band": "128k"},
        ]
    )

    filtered = app.filter_viewer_rows(rows, {"context_band": "64k"}, app.AGENTIC_MODE)

    assert list(filtered["suite"]) == ["reasoning", "throughput"]
    assert list(filtered["context_band"].fillna("")) == ["", "64k"]


def test_context_mapping_covers_requested_models_and_separates_yarn_limits() -> None:
    assert set(app.MODEL_CONTEXTS) == {
        "nvidia/Qwen3.8-27B-NVFP4",
        "unsloth/Qwen3.8-27B-GGUF",
        "mlx-community/Qwen3.8-27B-8bit",
        "unsloth/Qwen3.8-Flash-Next-GGUF",
        "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF",
    }
    for context in app.MODEL_CONTEXTS.values():
        assert context["native_context_tokens"] == 262144
        assert context["max_recommended_context_tokens"] == (
            context["yarn_extended_context_tokens"] or context["native_context_tokens"]
        )
        assert context["card_url"].startswith("https://huggingface.co/")
        assert context["context_basis"]
    assert app.MODEL_CONTEXTS["unsloth/Qwen3.8-27B-GGUF"]["yarn_extended_context_tokens"] == 1000000
    assert app.MODEL_CONTEXTS["unsloth/Qwen3.8-Flash-Next-GGUF"]["yarn_extended_context_tokens"] == 1000000
    assert app.MODEL_CONTEXTS["nvidia/Qwen3.8-27B-NVFP4"]["yarn_extended_context_tokens"] is None


def test_dataset_card_lists_context_for_every_mapped_model() -> None:
    card = (SPACE_ROOT.parent / "dataset-card" / "README.md").read_text()
    assert all(model in card for model in app.MODEL_CONTEXTS)
    assert "YaRN-extended context" in card
    assert "Max recommended context" in card


def test_summary_and_bar_hover_show_model_limits_measured_input_and_ttft() -> None:
    records = []
    for metric, value, unit in (
        ("throughput_output_toks_per_s", 125.0, "tok/s"),
        ("ttft_p50_ms", 240.0, "ms"),
        ("ttft_p99_ms", 510.0, "ms"),
    ):
        records.append(
            {
                "timestamp": "2026-10-01T00:00:00Z",
                "suite": "throughput",
                "name": "benchmark_serving",
                "metric": metric,
                "value": value,
                "unit": unit,
                "model": "nvidia/Qwen3.8-27B-NVFP4",
                "machine": "RTX PRO 6000 Max-Q 96GB",
                "campaign_run_prompt_tokens": 65536,
                "source_path": "data/context-metrics.parquet",
            }
        )
    records.append(
        {
            "timestamp": "2026-10-02T00:00:00Z",
            "suite": "throughput",
            "name": "benchmark_serving",
            "metric": "throughput_output_toks_per_s",
            "value": 80.0,
            "unit": "tok/s",
            "model": "nvidia/Qwen3.8-27B-NVFP4",
            "machine": "RTX 4080 SUPER 16GB",
            "campaign_run_prompt_tokens": 131072,
            "source_path": "data/other-machine.parquet",
        }
    )
    rows = app.add_evidence_metadata(pd.DataFrame(records), {}, None)
    summary = app.summary_table(
        rows, "throughput", "benchmark_serving", "throughput_output_toks_per_s", "tok/s"
    )
    chart = app.bar_chart(rows, "throughput", "benchmark_serving", "throughput_output_toks_per_s", "tok/s")
    columns = summary.set_index("Machine").loc["RTX PRO 6000 Max-Q 96GB"].to_dict()
    hover_index = next(index for index, label in enumerate(chart.data[0].y) if "RTX PRO 6000" in label)
    hover = chart.data[0].customdata[hover_index][0]

    assert columns["Native model context"] == "262,144 tokens"
    assert columns["YaRN-extended context"] == "Not documented"
    assert columns["Max recommended context"] == "262,144 tokens"
    assert columns["Largest input measured"] == "65,536 tokens"
    assert columns["Output tok/s"] == "125.0 tok/s"
    assert columns["TTFT p50 (ms)"] == "240.00 ms"
    assert columns["TTFT p99 (ms)"] == "510.00 ms"
    assert columns["Model card URL"] == app.MODEL_CONTEXTS["nvidia/Qwen3.8-27B-NVFP4"]["card_url"]
    assert (
        summary.set_index("Machine").loc["RTX 4080 SUPER 16GB", "Largest input measured"] == "131,072 tokens"
    )
    assert "Max recommended context: 262,144 tokens" in hover
    assert "Largest input measured for this model and machine: 65,536 tokens" in hover
