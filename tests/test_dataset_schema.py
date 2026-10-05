from __future__ import annotations

import json

import pyarrow as pa
import pytest

from mlx_benchmarks.dataset_schema import (
    APPLE_MACHINE_FALLBACK,
    PARQUET_ROW_SCHEMA,
    normalize_legacy_rows,
    parse_machine_labels,
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


def test_machine_labels_are_validated_and_apple_fallback_is_generic() -> None:
    assert parse_machine_labels(None) == {}
    assert parse_machine_labels('{"machine-id-one":"MacBook Pro M4 Max 128GB"}') == {
        "machine-id-one": "MacBook Pro M4 Max 128GB"
    }
    assert parse_machine_labels('{"studio-id":"Mac Studio M4 Max 128GB"}') == {
        "studio-id": "Mac Studio M4 Max 128GB"
    }
    with pytest.raises(ValueError, match="invalid entry"):
        parse_machine_labels('{"machine-id-one":"machine-id-one"}')

    [row] = normalize_legacy_rows([{"hostname": "machine-id-one", "chip": "Apple M4 Max", "memory_gb": 128}])
    assert row["hostname"] == APPLE_MACHINE_FALLBACK


def test_machine_label_projection_updates_identity_fields_only() -> None:
    source = {
        "hostname": "machine-id-one",
        "chip": "Apple M4 Max",
        "memory_gb": 128,
        "machine": "machine-id-one",
        "campaign_hardware_machine": "machine-id-one",
        "topology": '{"host":{"node_name":"machine-id-one"}}',
        "tags_json": '{"campaign_hardware_machine":"machine-id-one","other":"preserved"}',
        "extra_json": '{"host":"machine-id-one","other":"preserved"}',
    }
    [row] = normalize_legacy_rows(
        [source],
        machine_labels={"machine-id-one": "MacBook Pro M4 Max 128GB"},
    )
    assert source["hostname"] == "machine-id-one"
    assert row["hostname"] == "MacBook Pro M4 Max 128GB"
    assert row["campaign_hardware_machine"] == "MacBook Pro M4 Max 128GB"
    assert "machine-id-one" not in row["topology"]
    assert "MacBook Pro M4 Max 128GB" in row["topology"]
    assert json.loads(row["extra_json"]) == {
        "host": "MacBook Pro M4 Max 128GB",
        "machine": "MacBook Pro M4 Max 128GB",
        "other": "preserved",
    }
    assert json.loads(row["tags_json"]) == {
        "campaign_hardware_machine": "MacBook Pro M4 Max 128GB",
        "other": "preserved",
    }


def test_nvidia_hardware_fields_produce_public_machine_labels() -> None:
    rows = normalize_legacy_rows(
        [
            {
                "hostname": "machine-id-pro",
                "campaign_hardware_accelerator_model": "RTX PRO 6000 Max-Q",
                "campaign_hardware_accelerator_memory_gb": 96,
            },
            {
                "hostname": "machine-id-4080",
                "gpu": "NVIDIA RTX 4080 SUPER",
                "campaign_hardware_accelerator_memory_gb": 16,
            },
        ]
    )

    assert [row["hostname"] for row in rows] == [
        "RTX PRO 6000 Max-Q 96GB",
        "RTX 4080 SUPER 16GB",
    ]
