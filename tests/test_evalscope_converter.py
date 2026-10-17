"""EvalScope reports publish aggregate scores without sample data."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from mlx_benchmarks.converters import get_converter
from mlx_benchmarks.converters.base import ConverterContext
from mlx_benchmarks.converters.evalscope import EvalScopeConverter
from mlx_benchmarks.envelope import validate_envelope
from mlx_benchmarks.system import detect_system

FIXTURE = Path(__file__).parent / "fixtures" / "evalscope_report_sample.json"


def test_evalscope_aggregate_metrics_form_score_rows() -> None:
    raw = json.loads(FIXTURE.read_text())
    envelope = get_converter("evalscope").build_envelope(
        raw, ConverterContext(suite="coding", model="example-model", git_sha="a" * 40)
    )

    assert [(row["metric"], row["value"]) for row in envelope["results"]] == [
        ("main_problem_pass_rate", 0.25),
        ("subproblem_pass_rate", 0.3333),
    ]
    assert all("sample_count" in row["tags"] for row in envelope["results"])
    assert "questions" not in envelope and "predictions" not in envelope
    assert all(row["reasoning_arm"] == "unstated" for row in envelope["results"])
    assert envelope["results"][0]["output_tokens_mean"] is None
    assert envelope["results"][0]["tags"]["output_tokens_unavailable_reason"] == "sample_logs_missing"


def test_evalscope_uses_sample_usage_for_token_cost_fields(tmp_path: Path) -> None:
    root = tmp_path / "ifbench"
    report = root / "reports" / "example-model" / "ifbench.json"
    predictions = root / "predictions" / "example-model" / "ifbench_default.jsonl"
    config = root / "configs" / "task_config.yaml"
    report.parent.mkdir(parents=True)
    predictions.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    raw = {
        "dataset_name": "ifbench",
        "metrics": [{"identity": {"key": "prompt_level_strict"}, "num": 4, "score": 0.5}],
    }
    report.write_text(json.dumps(raw))
    config.write_text("generation_config:\n  max_tokens: 100\n")
    predictions.write_text(
        "\n".join(
            json.dumps(
                {
                    "model_output": {
                        "usage": {
                            "output_tokens": count,
                            "completion_tokens_details": {"reasoning_tokens": index + 1},
                        },
                        "choices": [{"finish_reason": reason}],
                    }
                }
            )
            for index, (count, reason) in enumerate(
                zip((10, 20, 100, 5), ("stop", "length", "stop", "stop"), strict=True)
            )
        )
        + "\n"
    )

    converter = EvalScopeConverter(tokenizer_loader=lambda _model: None)
    envelope = converter.build_envelope(
        raw,
        ConverterContext(
            suite="coding",
            model="example-model",
            git_sha="a" * 40,
            system=detect_system(),
            reasoning_effort="xhigh",
            source_path=report,
        ),
    )
    validate_envelope(envelope)
    [row] = envelope["results"]
    assert row["reasoning_arm"] == "xhigh"
    assert row["output_tokens_mean"] == 33.75
    assert row["output_tokens_median"] == 15
    assert row["output_tokens_p90"] == 100
    assert row["tokens_per_correct"] == 67.5
    assert row["reasoning_tokens_mean"] == 2.5
    assert row["truncation_rate"] == 0.25
    assert row["tags"]["output_tokens_method"] == "usage"


def test_evalscope_report_without_finite_scores_is_rejected() -> None:
    raw = json.loads(FIXTURE.read_text())
    raw["metrics"][0]["score"] = math.nan
    raw["metrics"][1]["score"] = math.inf

    with pytest.raises(ValueError, match="no finite aggregate scores"):
        get_converter("evalscope").build_envelope(
            raw, ConverterContext(suite="coding", model="example-model", git_sha="a" * 40)
        )


def test_evalscope_incomplete_report_is_rejected() -> None:
    raw = json.loads(FIXTURE.read_text())
    raw["execution_summary"] = {"incomplete": True}

    with pytest.raises(ValueError, match="report is incomplete"):
        get_converter("evalscope").build_envelope(
            raw, ConverterContext(suite="coding", model="example-model", git_sha="a" * 40)
        )
