from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mlx_benchmarks.backfill import _write_parquet, backfill_dataset, backfill_result_row
from mlx_benchmarks.result_contract import PublishedResultValidationError, validate_published_result


def test_backfill_copies_real_values_and_marks_unavailable_values() -> None:
    row = {
        "suite": "throughput",
        "model": "owner/model",
        "timestamp": "2026-04-24T18:30:00Z",
        "metric": "tokens/s",
        "value": 0.0,
        "engine": "MLX",
        "concurrency": 1,
        "tag_thinking": "false",
    }
    result = backfill_result_row(row, api=SimpleNamespace())  # type: ignore[arg-type]

    assert result["model_id"] == "owner/model"
    assert result["value"] == 0.0
    assert result["engine"] == "MLX"
    assert result["thinking"] is False
    assert result["dataset_id"] is None
    assert result["dimension_null_reasons"]["dataset_id"] == "not_applicable"
    assert result["dimension_null_reasons"]["model_revision"] == "na_backfill"
    validate_published_result(result)


def test_backfill_reads_model_metadata_only_at_recorded_revision() -> None:
    expected_revision = "a" * 40
    calls: list[tuple[str, str]] = []

    def model_info(*, repo_id: str, revision: str) -> Any:
        calls.append((repo_id, revision))
        return SimpleNamespace(
            id=repo_id,
            sha=revision,
            pipeline_tag="text-generation",
            card_data={"pipeline_tag": "text-generation", "license": "apache-2.0"},
            library_name="transformers",
            safetensors={"total": 400, "parameters": {"BF16": 400}},
            gguf=None,
            config={"architectures": ["ExampleForCausalLM"], "model_type": "example"},
            gated=False,
            tags=["text-generation"],
        )

    result = backfill_result_row(
        {
            "suite": "throughput",
            "model": "owner/model",
            "model_revision": expected_revision,
            "timestamp": "2026-04-24T18:30:00Z",
            "metric": "tokens/s",
            "value": 12.0,
        },
        api=SimpleNamespace(model_info=model_info),  # type: ignore[arg-type]
    )

    assert calls == [("owner/model", expected_revision)]
    assert result["pipeline_tag"] == "text-generation"
    assert result["parameters_total"] == 400
    assert result["dtype"] == {"BF16": 400}
    assert result["model_task"] == "text-generation"
    validate_published_result(result)


def test_backfill_rejects_na_after_cutover() -> None:
    with pytest.raises(PublishedResultValidationError, match="not allowed after"):
        backfill_result_row(
            {
                "suite": "throughput",
                "model": "owner/model",
                "timestamp": "2026-10-07T04:20:00Z",
                "metric": "tokens/s",
                "value": 12.0,
            },
            api=SimpleNamespace(),  # type: ignore[arg-type]
        )


def test_dataset_backfill_dry_run_plans_all_rows_without_calling_create_commit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shard = tmp_path / "history.parquet"
    pq.write_table(
        pa.table(
            {
                "suite": ["throughput"],
                "model": ["owner/model"],
                "timestamp": ["2026-04-24T18:30:00Z"],
                "metric": ["tokens/s"],
                "value": [12.0],
            }
        ),
        shard,
    )

    class ReadOnlyHub:
        def list_repo_files(self, *, repo_id: str, repo_type: str) -> list[str]:
            return ["data/history.parquet"]

        def hf_hub_download(self, *, repo_id: str, repo_type: str, filename: str) -> str:
            return str(shard)

        def create_commit(self, **kwargs: Any) -> None:
            pytest.fail("dry-run must never call create_commit")

    result = backfill_dataset("owner/dataset", apply=False, api=ReadOnlyHub())  # type: ignore[arg-type]
    output = capsys.readouterr().out
    assert result["shards"] == 1
    assert result["rows"] == 1
    assert result["shards_to_rewrite"] == 1
    assert result["operations"] == 1
    assert "mode=dry-run; no Hub files changed" in output

    projected_schema = pq.read_table(shard).schema
    assert "published_result_json" not in projected_schema.names
    assert "model_id" not in projected_schema.names


def test_backfill_parquet_projection_removes_host_identifiers() -> None:
    table = pa.table(
        {
            "hostname": ["REDACTED"],
            "topology": ['{"nodes":[{"hostname":"REDACTED","node_ip":"REDACTED","chip":"GPU"}]}'],
        }
    )
    row = backfill_result_row(
        {
            "suite": "throughput",
            "model": "owner/model",
            "timestamp": "2026-04-24T18:30:00Z",
            "metric": "tokens/s",
            "value": 12.0,
        },
        api=SimpleNamespace(),  # type: ignore[arg-type]
    )

    projected = pq.read_table(pa.BufferReader(_write_parquet(table, [row]))).to_pylist()[0]

    assert projected["hostname"] is None
    assert json.loads(projected["topology"])["nodes"][0] == {"chip": "GPU"}
    assert projected["model_id"] == row["model_id"]
    assert json.loads(projected["dimension_null_reasons"]) == row["dimension_null_reasons"]


def test_backfill_direct_columns_are_idempotent_without_hub_lookup() -> None:
    source = pa.table(
        {
            "suite": ["throughput"],
            "model": ["owner/model"],
            "timestamp": ["2026-04-24T18:30:00Z"],
            "metric": ["tokens/s"],
            "value": [12.0],
        }
    )
    first = backfill_result_row(
        source.to_pylist()[0],
        api=SimpleNamespace(),  # type: ignore[arg-type]
    )
    migrated = pq.read_table(pa.BufferReader(_write_parquet(source, [first])))
    stored = migrated.to_pylist()[0]

    second = backfill_result_row(
        stored,
        api=SimpleNamespace(model_info=lambda **_kwargs: pytest.fail("metadata was already copied")),
    )
    reprojected = pq.read_table(pa.BufferReader(_write_parquet(migrated, [second])))

    assert second["model_id"] == "owner/model"
    assert reprojected.to_pylist() == migrated.to_pylist()
