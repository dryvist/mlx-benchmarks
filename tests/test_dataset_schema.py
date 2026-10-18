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
                "output_tokens_mean": 123.5,
                "output_tokens_median": 100.0,
                "output_tokens_p90": 250.0,
                "reasoning_tokens_mean": 18.5,
                "tokens_per_correct": 205.8,
                "truncation_rate": 0.01,
                "reasoning_arm": "xhigh",
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
    assert older["campaign_hardware_interconnect"] is None
    assert older["campaign_hardware_world_size"] is None
    assert newer["decode_tokens_per_second"] == 11.5
    assert newer["output_tokens_mean"] == 123.5
    assert newer["reasoning_tokens_mean"] == 18.5
    assert newer["tokens_per_correct"] == 205.8
    assert newer["truncation_rate"] == 0.01
    assert newer["reasoning_arm"] == "xhigh"
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


def test_cluster_hardware_dimensions_normalize_and_legacy_columns_stay_null() -> None:
    [legacy] = normalize_legacy_rows([{}])
    [cluster] = normalize_legacy_rows(
        [{}],
        campaign_dimensions={
            "hardware": {
                "interconnect": "Thunderbolt 5 RDMA",
                "world_size": 2,
            }
        },
    )

    assert legacy["campaign_hardware_interconnect"] is None
    assert legacy["campaign_hardware_world_size"] is None
    assert cluster["campaign_hardware_interconnect"] == "Thunderbolt 5 RDMA"
    assert cluster["campaign_hardware_world_size"] == 2


def test_machine_labels_are_validated_and_apple_fallback_is_generic() -> None:
    assert parse_machine_labels(None) == {}
    assert parse_machine_labels('{"machine-id-one":"MacBook Pro M4 Max 128GB"}') == {
        "machine-id-one": "MacBook Pro M4 Max 128GB"
    }
    assert parse_machine_labels('{"studio-id":"Mac Studio M4 Max 128GB"}') == {
        "studio-id": "Mac Studio M4 Max 128GB"
    }
    cluster_label = "2x Apple M4 Max 128GB (Thunderbolt RDMA)"
    assert parse_machine_labels(json.dumps({"cluster-id": cluster_label})) == {"cluster-id": cluster_label}
    with pytest.raises(ValueError, match="invalid entry"):
        parse_machine_labels('{"machine-id-one":"machine-id-one"}')

    [row] = normalize_legacy_rows([{"hostname": "machine-id-one", "chip": "Apple M4 Max", "memory_gb": 128}])
    assert row["hostname"] == APPLE_MACHINE_FALLBACK


def test_cluster_machine_label_normalizes_from_private_machine_id() -> None:
    label = "2x Apple M4 Max 128GB (Thunderbolt RDMA)"
    [row] = normalize_legacy_rows(
        [
            {
                "hostname": "cluster-id",
                "machine": "cluster-id",
                "campaign_hardware_machine": "cluster-id",
                "chip": "Apple M4 Max",
                "memory_gb": 128,
            }
        ],
        machine_labels={"cluster-id": label},
    )

    assert row["hostname"] == label
    assert row["campaign_hardware_machine"] == label


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


def test_published_from_machine_id_overrides_generic_machine_label() -> None:
    machine_id = "synthetic-machine-id"
    [row] = normalize_legacy_rows(
        [
            {
                "hostname": APPLE_MACHINE_FALLBACK,
                "chip": "Apple M4 Max",
                "memory_gb": 128,
                "tag_published_from": machine_id,
                "published_from": machine_id,
            }
        ],
        machine_labels={machine_id: "MacBook Pro M4 Max 128GB"},
    )

    assert row["hostname"] == "MacBook Pro M4 Max 128GB"
    assert row["tag_published_from"] == "MacBook Pro M4 Max 128GB"
    assert json.loads(row["tags_json"])["published_from"] == "MacBook Pro M4 Max 128GB"
    assert json.loads(row["extra_json"])["published_from"] == "MacBook Pro M4 Max 128GB"
    assert machine_id not in json.dumps(row, sort_keys=True)


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
