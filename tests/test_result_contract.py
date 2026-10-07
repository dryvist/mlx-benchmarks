from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mlx_benchmarks.result_contract import (
    PublishedResultValidationError,
    validate_files,
    validate_published_result,
)


@pytest.fixture
def published_result() -> dict[str, object]:
    return {
        "model_id": "owner/model",
        "model_revision": "a" * 40,
        "pipeline_tag": "text-generation",
        "model_task": "text-generation",
        "model_task_source": "card",
        "library_name": "transformers",
        "license": "apache-2.0",
        "license_name": None,
        "license_link": None,
        "base_model": None,
        "base_model_relation": None,
        "parameters_total": 1000,
        "dtype": {"F16": 1000},
        "quant": None,
        "architectures": ["ExampleForCausalLM"],
        "model_type": "example",
        "context_length": 4096,
        "gated": False,
        "dataset_id": "owner/benchmark",
        "dataset_task_id": "default",
        "dataset_revision": "b" * 40,
        "value": 0.5,
        "metric": "accuracy",
        "date": "2026-10-06T12:00:00Z",
        "evaluation_framework": "inspect-ai",
        "config": "default",
        "split": "test",
        "engine": "MLX",
        "engine_version": "1.0",
        "profile": "baseline",
        "hardware": {"chip": "Example CPU"},
        "power_limit_w": 100.0,
        "concurrency": 1,
        "ctx_per_slot": 4096,
        "kv_cache_dtype": "float16",
        "thinking": False,
        "temperature": 0.0,
        "max_tokens": 128,
        "prompt_chars": 100,
        "prompt_tokens": 25,
        "system_prompt_chars": 20,
        "system_prompt_tokens": 5,
        "runner": "mlx-benchmarks",
        "harness": "inspect-ai",
        "router_key_alias": "local",
        "run_id": "run-123",
        "start_utc": "2026-10-06T12:00:00Z",
        "end_utc": "2026-10-06T12:00:00Z",
        "dimension_null_reasons": {
            "base_model": "not_applicable",
            "base_model_relation": "not_applicable",
            "quant": "not_applicable",
        },
    }


def test_valid_result_passes(published_result: dict[str, object]) -> None:
    validate_published_result(published_result)


def test_required_null_without_reason_is_rejected(published_result: dict[str, object]) -> None:
    row = deepcopy(published_result)
    row["ctx_per_slot"] = None
    with pytest.raises(PublishedResultValidationError, match="ctx_per_slot is null"):
        validate_published_result(row)


def test_null_reason_is_explicit_and_backfill_is_date_limited(published_result: dict[str, object]) -> None:
    row = deepcopy(published_result)
    row["ctx_per_slot"] = None
    row["dimension_null_reasons"] = {**row["dimension_null_reasons"], "ctx_per_slot": "na_backfill"}
    validate_published_result(row)

    row["date"] = "2026-10-08T12:00:00Z"
    row["start_utc"] = "2026-10-08T11:59:00Z"
    row["end_utc"] = "2026-10-08T12:00:00Z"
    with pytest.raises(PublishedResultValidationError, match="not allowed after"):
        validate_published_result(row)


def test_not_applicable_is_the_only_future_null_escape(published_result: dict[str, object]) -> None:
    row = deepcopy(published_result)
    row["dataset_revision"] = None
    row["dimension_null_reasons"] = {**row["dimension_null_reasons"], "dataset_revision": "not_applicable"}
    validate_published_result(row)


def test_reason_for_non_null_field_is_rejected(published_result: dict[str, object]) -> None:
    row = deepcopy(published_result)
    row["dimension_null_reasons"] = {**row["dimension_null_reasons"], "concurrency": "not_applicable"}
    with pytest.raises(PublishedResultValidationError, match="has a null reason"):
        validate_published_result(row)


def test_model_task_must_match_pipeline_tag(published_result: dict[str, object]) -> None:
    row = deepcopy(published_result)
    row["model_task"] = "text-classification"
    with pytest.raises(PublishedResultValidationError, match="must equal pipeline_tag"):
        validate_published_result(row)


def test_parquet_validator_reads_named_published_columns(
    tmp_path: Path, published_result: dict[str, object]
) -> None:
    parquet_path = tmp_path / "results.parquet"
    row = deepcopy(published_result)
    for field in ("base_model", "dtype", "hardware", "dimension_null_reasons"):
        row[field] = json.dumps(row[field], sort_keys=True)
    row["gated"] = "false"
    pq.write_table(pa.Table.from_pylist([row]), parquet_path)

    assert validate_files([parquet_path]) == 1
