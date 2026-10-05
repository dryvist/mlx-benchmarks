from __future__ import annotations

import pyarrow as pa
import pytest

from mlx_benchmarks.dataset_schema import (
    PARQUET_ROW_SCHEMA,
    canonical_shard_needs_refresh,
    normalize_legacy_rows,
)


def _historical_table() -> pa.Table:
    legacy_schema = pa.schema(
        [field for field in PARQUET_ROW_SCHEMA if not field.name.startswith("campaign_")]
    )
    row = {
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
    return pa.Table.from_pylist([row], schema=legacy_schema)


def test_old_canonical_schema_is_refreshed_when_rows_match_source() -> None:
    old = _historical_table()
    expected = pa.Table.from_pylist(normalize_legacy_rows(old.to_pylist()), schema=PARQUET_ROW_SCHEMA)

    assert canonical_shard_needs_refresh(old, expected)
    assert expected.schema.equals(PARQUET_ROW_SCHEMA, check_metadata=False)
    assert not canonical_shard_needs_refresh(expected, expected)


def test_old_canonical_schema_with_different_rows_is_not_refreshed() -> None:
    old = _historical_table()
    expected_rows = normalize_legacy_rows(old.to_pylist())
    expected_rows[0]["value"] = 99.0
    expected = pa.Table.from_pylist(expected_rows, schema=PARQUET_ROW_SCHEMA)

    with pytest.raises(RuntimeError, match="data differs from its normalized source"):
        canonical_shard_needs_refresh(old, expected)
