"""EvalScope reports publish aggregate scores without sample data."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from mlx_benchmarks.converters import get_converter
from mlx_benchmarks.converters.base import ConverterContext

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
