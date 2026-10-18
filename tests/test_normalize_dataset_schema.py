from __future__ import annotations

import pyarrow as pa
import pytest

from mlx_benchmarks.dataset_schema import (
    APPLE_MACHINE_FALLBACK,
    PARQUET_ROW_SCHEMA,
    campaign_dimension_column,
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


def test_canonical_rows_refresh_from_fallback_to_configured_label() -> None:
    source = pa.Table.from_pylist(
        [{"hostname": "machine-id-one", "chip": "Apple M4 Max", "memory_gb": 128}],
        schema=PARQUET_ROW_SCHEMA,
    )
    fallback = pa.Table.from_pylist(
        normalize_legacy_rows(source.to_pylist()),
        schema=PARQUET_ROW_SCHEMA,
    )
    labels = {"machine-id-one": "MacBook Pro M4 Max 128GB"}
    expected = pa.Table.from_pylist(
        normalize_legacy_rows(source.to_pylist(), machine_labels=labels),
        schema=PARQUET_ROW_SCHEMA,
    )

    assert canonical_shard_needs_refresh(source, expected, machine_labels=labels)
    assert canonical_shard_needs_refresh(
        fallback,
        expected,
        machine_labels=labels,
        prior_projection=fallback,
    )


def test_canonical_projection_prefers_specific_tag_label_over_generic_hostname() -> None:
    source = pa.Table.from_pylist(
        [
            {
                "chip": "Apple M4 Max",
                "memory_gb": 128,
                "hostname": APPLE_MACHINE_FALLBACK,
                "tag_published_from": "synthetic-machine-id",
                "tags_json": '{"published_from":"synthetic-machine-id"}',
            }
        ],
        schema=PARQUET_ROW_SCHEMA,
    )
    labels = {"synthetic-machine-id": "Mac Studio M4 Max 128GB"}
    expected = pa.Table.from_pylist(
        normalize_legacy_rows(source.to_pylist(), machine_labels=labels),
        schema=PARQUET_ROW_SCHEMA,
    )
    prior_projection = pa.Table.from_pylist(
        normalize_legacy_rows(source.to_pylist()),
        schema=PARQUET_ROW_SCHEMA,
    )
    existing_rows = normalize_legacy_rows(source.to_pylist(), machine_labels=labels)
    existing_rows[0]["hostname"] = APPLE_MACHINE_FALLBACK
    existing = pa.Table.from_pylist(existing_rows, schema=PARQUET_ROW_SCHEMA)

    assert canonical_shard_needs_refresh(
        existing,
        expected,
        machine_labels=labels,
        prior_projection=prior_projection,
    )
    refreshed = pa.Table.from_pylist(
        normalize_legacy_rows(existing.to_pylist(), machine_labels=labels),
        schema=PARQUET_ROW_SCHEMA,
    )
    assert refreshed.to_pylist() == expected.to_pylist()


def test_cache_configuration_projection_backfills_each_row_and_records_unknowns() -> None:
    dimensions = {
        "hardware": {"host_ram_limit_gib": None},
        "run": {
            "prefix_cache": True,
            "kv_offload_backend": "none",
            "kv_offload_cpu_gib": 0.0,
            "kv_offload_disk_gib": 0.0,
            "weight_cpu_offload_gib": 0.0,
            "llamacpp_cache_ram_mib": 8192,
            "gpu_memory_utilization": 0.88,
            "max_num_batched_tokens": None,
            "kv_pool_tokens": None,
        },
    }
    null_reasons = {
        "hardware.host_ram_limit_gib": "NOT_SOURCED",
        "run.max_num_batched_tokens": "NOT_CAPTURED",
        "run.kv_pool_tokens": "STARTUP_LOG_NOT_CAPTURED",
    }
    source_rows = _historical_table().to_pylist() * 2

    normalized = normalize_legacy_rows(
        source_rows,
        campaign_dimensions=dimensions,
        dimension_null_reasons=null_reasons,
    )

    assert len(normalized) == 2
    assert all(row[campaign_dimension_column("hardware", "host_ram_limit_gib")] is None for row in normalized)
    assert all(row[campaign_dimension_column("run", "prefix_cache")] is True for row in normalized)
    assert all(row[campaign_dimension_column("run", "kv_offload_backend")] == "none" for row in normalized)
    assert all(row[campaign_dimension_column("run", "kv_offload_cpu_gib")] == 0.0 for row in normalized)
    assert all(row[campaign_dimension_column("run", "llamacpp_cache_ram_mib")] == 8192 for row in normalized)
    assert all(row[campaign_dimension_column("run", "max_num_batched_tokens")] is None for row in normalized)
    assert all(row[campaign_dimension_column("run", "kv_pool_tokens")] is None for row in normalized)
    assert all(row[campaign_dimension_column("speed", "prefix_cache_hit_rate")] is None for row in normalized)
    assert all(row[campaign_dimension_column("resource", "host_ram_peak_gib")] is None for row in normalized)
    assert all(
        row["campaign_dimension_null_reasons_json"]
        == '{"hardware.host_ram_limit_gib": "NOT_SOURCED", "run.kv_pool_tokens": "STARTUP_LOG_NOT_CAPTURED", "run.max_num_batched_tokens": "NOT_CAPTURED"}'
        for row in normalized
    )
