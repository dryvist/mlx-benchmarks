from __future__ import annotations

import json
from typing import Any

import pytest

from mlx_benchmarks.converters import get_converter
from mlx_benchmarks.converters.base import ConverterContext
from mlx_benchmarks.envelope import validate_envelope
from mlx_benchmarks.publish import envelope_to_rows

PROPS = {
    "build_info": "b11457-5ad1c5da0",
    "model_path": "/models/example/Example-27B-UD-Q4_K_M.gguf",
    "total_slots": 4,
    "n_ctx": 65536,
}
SERVED = [
    {
        "id": "example/Example-27B-GGUF",
        "meta": {"n_params": 27_000_000_000, "size": 16_000_000_000, "n_ctx_train": 262144},
    }
]


def _record(concurrency: int, index: int, rate: float, series: str = "S1") -> dict[str, Any]:
    return {
        "series": series,
        "power_limit_w": 325,
        "concurrency": concurrency,
        "request_index": index,
        "start_utc": f"2026-10-07T00:25:{10 + index:02d}.250Z",
        "n_predict": 512,
        "temperature": 0.7,
        "thinking": False,
        "prompt": "Write an essay.",
        "client": "curl",
        "timings": {
            "prompt_n": 4,
            "prompt_ms": 100.0,
            "prompt_per_second": 40.0,
            "predicted_n": 512,
            "predicted_ms": 512 / rate * 1000,
            "predicted_per_second": rate,
        },
        "usage": {
            "completion_tokens": 512,
            "prompt_tokens": 25,
            "prompt_tokens_details": {"cached_tokens": 21},
        },
        "finish_reason": "length",
        "server_props": PROPS,
        "served_models": SERVED,
    }


def _records() -> list[dict[str, Any]]:
    return [
        _record(0, 0, 70.0),
        _record(1, 0, 74.0),
        _record(2, 0, 67.0),
        _record(2, 1, 66.0),
        _record(8, 0, 51.0),
    ]


def _ctx(**overrides: Any) -> ConverterContext:
    return ConverterContext(
        suite="throughput",
        model="example/Example-27B-GGUF",
        git_sha="abc1234",
        system={
            "os": "macOS 26",
            "chip": "Publisher CPU",
            "memory_gb": 64,
            "hostname": "publisher-host",
            "kernel": "25.0",
            "gpu": {"model": "Example GPU", "vram_gb": 96.0},
        },
        **overrides,
    )


def test_warmups_are_metadata_not_results() -> None:
    envelope = get_converter("llamacpp-server").build_envelope(_records(), _ctx())
    validate_envelope(envelope)

    assert {r["tags"]["concurrency"] for r in envelope["results"]} == {"1", "2", "8"}
    decode = [r for r in envelope["results"] if r["metric"] == "throughput_output_toks_per_s"]
    assert sorted(r["value"] for r in decode) == [51.0, 66.0, 67.0, 74.0]
    assert all(r["tags"]["warmup_predicted_per_second"] == "70.0" for r in envelope["results"])
    assert all(r["tags"]["max_tokens"] == "512" for r in envelope["results"] if "n_requests" not in r["tags"])


def test_aggregate_only_where_requests_decode_together() -> None:
    envelope = get_converter("llamacpp-server").build_envelope(_records(), _ctx())

    aggregate = {
        r["tags"]["concurrency"]: r["value"]
        for r in envelope["results"]
        if r["metric"] == "throughput_aggregate_output_toks_per_s"
    }
    assert aggregate == {"1": 74.0, "2": 133.0}  # width 8 exceeds the 4 slots: no sum


def test_envelope_derives_campaign_dimensions_from_the_records() -> None:
    declared = {"hardware": {"machine": "Example GPU"}, "software": {"driver_version": "1.2.3"}}
    envelope = get_converter("llamacpp-server").build_envelope(
        _records(),
        _ctx(campaign_dimensions=declared, dimension_null_reasons={"hardware.host_cpu": "NOT_SOURCED"}),
    )
    validate_envelope(envelope)

    dims = envelope["campaign_dimensions"]
    assert dims["hardware"] == {"machine": "Example GPU", "power_cap_w": 325.0}
    assert dims["software"]["engine"] == "llama.cpp"
    assert (dims["software"]["engine_version"], dims["software"]["engine_commit"]) == ("b11457", "5ad1c5da0")
    assert dims["software"]["driver_version"] == "1.2.3"
    assert dims["model"]["quantization"] == "UD-Q4_K_M"
    assert dims["model"]["file_size_gb"] == 16.0
    assert dims["run"]["parallel_slots"] == 4
    assert dims["run"]["warm_cold"] == "warm"
    assert dims["run"]["concurrent_agents"] is None
    assert dims["provenance"]["run_id"] == "S1"
    reasons = envelope["dimension_null_reasons"]
    assert reasons["run.concurrent_agents"] == "VARIES_PER_ROW"
    assert reasons["hardware.host_cpu"] == "NOT_SOURCED"
    assert envelope["system"] == {
        "os": None,
        "chip": None,
        "memory_gb": None,
        "gpu": {"model": "Example GPU", "vram_gb": 96.0},
        "engine": {"name": "llama.cpp", "version": "b11457"},
        "power_limit_w": 325.0,
    }
    assert envelope["quantization"] == "UD-Q4_K_M"
    assert envelope["reasoning_effort"] == "off"
    assert envelope["timestamp"] == "2026-10-07T00:25:10Z"
    assert envelope["campaign"] == {"cell_id": "S1"}


def test_server_answers_ride_each_row_without_the_model_path() -> None:
    envelope = get_converter("llamacpp-server").build_envelope(_records(), _ctx())

    for row in envelope_to_rows(envelope):
        tags = json.loads(row["tags_json"])
        if tags.get("phase") and "server_json" in tags:
            server = json.loads(tags["server_json"])
            assert server["server_props"]["build_info"] == "b11457-5ad1c5da0"
            assert server["server_props"]["model_path"] == "Example-27B-UD-Q4_K_M.gguf"
            assert tags["usage_cached_tokens"] == "21"
            assert tags["timings_prompt_n"] == "4"
        assert row["campaign_hardware_power_cap_w"] == 325.0


def test_mixed_series_are_refused() -> None:
    records = [_record(1, 0, 70.0, "S1"), _record(1, 0, 71.0, "S2")]

    with pytest.raises(ValueError, match="one series"):
        get_converter("llamacpp-server").build_envelope(records, _ctx())
