"""Promptfoo JSON conversion stays within the public score summary contract."""

from __future__ import annotations

import json
from pathlib import Path

from mlx_benchmarks.converters import get_converter
from mlx_benchmarks.converters.base import ConverterContext
from mlx_benchmarks.envelope import validate_envelope
from mlx_benchmarks.system import detect_system


def test_promptfoo_json_emits_test_assertion_and_suite_summaries() -> None:
    fixture = Path(__file__).parent / "fixtures" / "promptfoo_sample.json"
    raw = json.loads(fixture.read_text())
    context = ConverterContext(
        suite="framework-eval",
        model="example/model",
        git_sha="abc1234",
        system=detect_system(),
        extra_tags={"domain": "agentic"},
    )

    envelope = get_converter("promptfoo").build_envelope(raw, context)
    validate_envelope(envelope)

    assert envelope["timestamp"] == "2026-10-16T12:00:00Z"
    assert [(row["name"], row["metric"], row["value"]) for row in envelope["results"]] == [
        ("test_0", "score", 1.0),
        ("test_0_assertion_0", "assertion_pass", 1.0),
        ("test_1", "score", 0.0),
        ("test_1_assertion_0", "assertion_pass", 0.0),
        ("promptfoo_aggregate", "score", 0.5),
    ]
    assert all("domain" in row["tags"] for row in envelope["results"])
