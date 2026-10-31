"""Unit tests for the routing runner, without a live endpoint."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_runner() -> ModuleType:
    path = REPO_ROOT / "harness" / "s1-routing" / "run.py"
    spec = importlib.util.spec_from_file_location("s1_routing_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
COST = tomllib.loads((REPO_ROOT / "configs/s1-routing/s1-routing.toml").read_text())["cost"]


def record(label: str, pred: str | None) -> dict:
    return {"label": label, "pred": pred, "invalid": pred is None, "latency_ms": 10.0}


def test_macro_f1_and_confusion() -> None:
    result = runner.summarize(
        [record("bulk", "bulk"), record("bulk", "standard"), record("standard", "standard")], COST
    )
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 2 / 3) / 6)
    assert result["confusion_matrix"]["bulk"]["standard"] == 1


def test_cost_severe_and_top_group() -> None:
    result = runner.summarize(
        [record("deep", "bulk"), record("plan", "complex"), record("bulk", "standard")], COST
    )
    assert result["mean_cost"] == pytest.approx((9 + 0.5 + 2) / 3)
    assert result["severe_count"] == 1


def test_invalid_is_error_and_standard_for_cost() -> None:
    result = runner.summarize([record("deep", None)], COST)
    assert result["invalid_rate"] == 1
    assert result["accuracy"] == 0
    assert result["mean_cost"] == 3
    assert result["confusion_matrix"]["deep"]["invalid"] == 1


def test_knn() -> None:
    label, confidence = runner.knn_predict([1, 0], [([1, 0], "bulk"), ([0, 1], "deep")], 2)
    assert label == "bulk"
    assert confidence > 0.99


def test_systemone_response() -> None:
    assert runner.parse_systemone({"answers": {"route": {"choice": "plan", "answer_confidence": 0.8}}}) == (
        "plan",
        0.8,
    )
    assert runner.parse_systemone({"answers": {"route": {"choice": "deep"}}}) == ("deep", None)
    assert runner.parse_systemone({"answers": {"route": {"choice": "other"}}}) == (None, None)


def test_chat_response() -> None:
    assert runner.parse_chat('{"label":"complex"}') == "complex"
    assert runner.parse_chat('{"route": "deep"}') == "deep"
    assert runner.parse_chat("Use bulk-complex, then standard.") == "bulk-complex"
    assert runner.parse_chat("unrecognized") is None


def test_redact_url() -> None:
    assert runner.redact_url("https://user:secret@example.com:8443/path?key=value") == "https://REDACTED:8443"
    assert runner.redact_url("http://example.com/path") == "http://REDACTED"
