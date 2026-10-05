from __future__ import annotations

import json

import pyarrow as pa
from scripts.normalize_dataset_schema import canonical_path

from mlx_benchmarks.dataset_schema import PARQUET_ROW_SCHEMA, normalize_legacy_rows


def test_canonical_path_covers_run_and_aggregate_shards() -> None:
    assert canonical_path("data/run-2026-07-01.parquet") == "data/run-canonical-run-2026-07-01.parquet"
    assert canonical_path("data/train-00000-of-00001.parquet") == (
        "data/run-canonical-train-00000-of-00001.parquet"
    )


def test_historical_row_variants_normalize_to_one_schema() -> None:
    common = {
        "schema_version": "1",
        "timestamp": "2026-07-01T00:00:00Z",
        "git_sha": "abc1234",
        "trigger": "local",
        "suite": "throughput",
        "model": "example/model",
        "name": "request",
        "metric": "throughput",
        "value": 12.5,
        "unit": "tokens_per_second",
    }
    older, newer = normalize_legacy_rows(
        [
            {**common, "tag_filter": "strict"},
            {
                **common,
                "decode_tokens_per_second": 11.5,
                "tag_concurrency": "4",
                "tag_future_dimension": "preserved",
                "tags_json": '{"future_dimension":"json value","json_only":"preserved"}',
                "extra_json": '{"previous_extra":"preserved"}',
                "legacy_extra": "preserved",
            },
        ]
    )

    older_table = pa.Table.from_pylist([older], schema=PARQUET_ROW_SCHEMA)
    newer_table = pa.Table.from_pylist([newer], schema=PARQUET_ROW_SCHEMA)
    assert older_table.schema.equals(newer_table.schema, check_metadata=False)
    assert older_table.schema.equals(PARQUET_ROW_SCHEMA, check_metadata=False)
    assert older["decode_tokens_per_second"] is None
    assert newer["decode_tokens_per_second"] == 11.5
    assert older["tag_filter"] == "strict"
    assert newer["tag_concurrency"] == "4"
    assert json.loads(newer["tags_json"]) == {
        "concurrency": "4",
        "future_dimension": "preserved",
        "json_only": "preserved",
    }
    assert json.loads(newer["extra_json"]) == {
        "legacy_extra": "preserved",
        "previous_extra": "preserved",
    }
