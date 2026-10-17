"""End-to-end: vllm benchmark_serving sample -> envelope -> passes schema validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlx_benchmarks.converters import get_converter
from mlx_benchmarks.converters.base import ConverterContext
from mlx_benchmarks.envelope import validate_envelope
from mlx_benchmarks.publish import envelope_to_rows
from mlx_benchmarks.system import detect_system

POWER_LIMIT_FIXTURE = Path(__file__).parent / "fixtures/nvidia-smi-enforced-power-limit.csv"


def test_vllm_round_trip(vllm_sample: dict) -> None:
    converter = get_converter("vllm")
    ctx = ConverterContext(
        suite="throughput",
        model="mlx-community/Qwen3.5-9B-MLX-4bit",
        git_sha="deadbeef",
        system=detect_system(),
    )
    envelope = converter.build_envelope(vllm_sample, ctx)

    validate_envelope(envelope)

    assert envelope["suite"] == "throughput"
    assert envelope["model"] == "mlx-community/Qwen3.5-9B-MLX-4bit"

    results = envelope["results"]
    assert len(results) > 0, "expected at least one result"

    metric_names = {r["metric"] for r in results}
    assert "throughput_output_toks_per_s" in metric_names
    assert "ttft_p50_ms" in metric_names
    assert "ttft_p99_ms" in metric_names
    assert "itl_p50_ms" in metric_names

    # All results share the same task name
    assert all(r["name"] == "benchmark_serving" for r in results)

    # Duration propagated from the raw sample (60.12 s)
    assert all(r.get("duration_seconds") == 60.12 for r in results)

    # Metadata tags propagated
    for result in results:
        tags = result.get("tags", {})
        assert tags.get("completed_requests") == "100"
        assert tags.get("total_input_tokens") == "25600"
        assert tags.get("total_output_tokens") == "25600"

    speed = envelope["campaign_dimensions"]["speed"]
    assert speed["potential_output_tok_s_p50"] is None
    assert envelope["dimension_null_reasons"]["speed.potential_output_tok_s_p50"] == (
        "max_concurrency_unavailable"
    )
    assert speed["peak_output_tok_s_1s_max"] is None


def test_vllm_date_defaults_timestamp_and_campaign_run_times(vllm_bench_serve_nvidia_sample: dict) -> None:
    ctx = ConverterContext(
        suite="throughput",
        model="example/model",
        git_sha="deadbeef",
        system=detect_system(),
        campaign_dimensions={"run": {"parallel_slots": 4}},
    )
    converter = get_converter("vllm")

    envelope = converter.build_envelope(vllm_bench_serve_nvidia_sample, ctx)
    validate_envelope(envelope)

    assert envelope["timestamp"] == "2026-10-04T12:00:00Z"
    run = envelope["campaign_dimensions"]["run"]
    assert run["started_at"] == "2026-10-04T12:00:00Z"
    assert run["ended_at"] == "2026-10-04T12:00:41.300000Z"
    assert run["parallel_slots"] == 4

    override_ctx = ConverterContext(
        suite="throughput",
        model="example/model",
        git_sha="deadbeef",
        system=detect_system(),
        timestamp_override="2026-10-04T13:00:00Z",
    )
    override_envelope = converter.build_envelope(vllm_bench_serve_nvidia_sample, override_ctx)
    assert override_envelope["timestamp"] == "2026-10-04T13:00:00Z"


def test_vllm_published_rows_include_the_live_enforced_power_limit(vllm_sample: dict, monkeypatch) -> None:
    fields = [value.strip() for value in POWER_LIMIT_FIXTURE.read_text(encoding="utf-8").strip().split(",")]
    monkeypatch.setenv("MLX_BENCH_GPU_MODEL", fields[0])
    monkeypatch.setenv("MLX_BENCH_POWER_LIMIT_W", fields[3])
    detect_system.cache_clear()
    ctx = ConverterContext(
        suite="throughput",
        model="example/model",
        git_sha="deadbeef",
        system=detect_system(),
    )
    detect_system.cache_clear()

    envelope = get_converter("vllm").build_envelope(vllm_sample, ctx)
    validate_envelope(envelope)
    rows = envelope_to_rows(envelope)

    assert rows
    assert all(row["power_limit_w"] == float(fields[3]) for row in rows)


def test_vllm_nvidia_results_require_a_runtime_power_limit(vllm_sample: dict, monkeypatch) -> None:
    monkeypatch.setenv("MLX_BENCH_GPU_MODEL", "NVIDIA Example GPU")
    monkeypatch.delenv("MLX_BENCH_POWER_LIMIT_W", raising=False)
    detect_system.cache_clear()
    system = detect_system()
    detect_system.cache_clear()

    with pytest.raises(ValueError, match="MLX_BENCH_POWER_LIMIT_W"):
        get_converter("vllm").build_envelope(
            vllm_sample,
            ConverterContext(
                suite="throughput",
                model="example/model",
                git_sha="deadbeef",
                system=system,
            ),
        )


def test_vllm_extra_tags(vllm_sample: dict) -> None:
    converter = get_converter("vllm")
    ctx = ConverterContext(
        suite="throughput",
        model="mlx-community/Qwen3.5-9B-MLX-4bit",
        git_sha="deadbeef",
        system=detect_system(),
        extra_tags={"input_len": "256", "output_len": "256"},
    )
    envelope = converter.build_envelope(vllm_sample, ctx)
    validate_envelope(envelope)
    for result in envelope["results"]:
        tags = result.get("tags", {})
        assert tags.get("input_len") == "256"
        assert tags.get("output_len") == "256"


@pytest.mark.parametrize("suite", ["throughput", "throughput-serving"])
def test_vllm_derived_throughput_dimensions_use_chunk_timings(suite: str) -> None:
    raw = {
        "max_concurrency": 4,
        "completed": 3,
        "duration": 4.0,
        "output_throughput": 100.0,
        "p50_tpot_ms": 20.0,
        "p90_tpot_ms": 40.0,
        "p99_tpot_ms": 80.0,
        "request_rate": 2.0,
        "start_times": [100.0, 101.0, 103.0],
        "ttfts": [0.75, 0.25, 0.1],
        "itls": [[0.5], [0.75], []],
        "output_lens": [4, 4, 2],
    }
    envelope = get_converter("vllm").build_envelope(
        raw,
        ConverterContext(
            suite=suite,
            model="example/model",
            git_sha="deadbeef",
            system=detect_system(),
        ),
    )
    validate_envelope(envelope)

    speed = envelope["campaign_dimensions"]["speed"]
    assert speed["potential_output_tok_s_p50"] == pytest.approx(200.0)
    assert speed["potential_output_tok_s_p90"] == pytest.approx(100.0)
    assert speed["potential_output_tok_s_p99"] == pytest.approx(50.0)
    assert speed["measured_mean_concurrency"] == pytest.approx(0.5875)
    assert speed["potential_output_tok_s_p50_at_mean_concurrency"] == pytest.approx(29.375)
    assert speed["potential_output_tok_s_p90_at_mean_concurrency"] == pytest.approx(14.6875)
    assert speed["potential_output_tok_s_p99_at_mean_concurrency"] == pytest.approx(7.34375)
    assert speed["peak_output_tok_s_1s_max"] == 4.0
    assert speed["peak_output_tok_s_1s_p99"] == 4.0
    assert speed["potential_efficiency"] == pytest.approx(0.5)
    metric_values = {result["metric"]: result["value"] for result in envelope["results"]}
    assert metric_values["tpot_p50_ms"] == 20.0
    assert metric_values["tpot_p90_ms"] == 40.0

    rows = envelope_to_rows(envelope)
    assert rows[0]["campaign_speed_peak_output_tok_s_1s_max"] == 4.0
    assert all(row["campaign_speed_potential_output_tok_s_p50"] == 200.0 for row in rows)


def test_vllm_non_detailed_and_generic_rows_keep_reasoned_nulls(
    vllm_bench_serve_nvidia_sample: dict,
) -> None:
    context = ConverterContext(
        suite="throughput",
        model="example/model",
        git_sha="deadbeef",
        system=detect_system(),
    )
    envelope = get_converter("vllm").build_envelope(vllm_bench_serve_nvidia_sample, context)
    speed = envelope["campaign_dimensions"]["speed"]
    assert speed["peak_output_tok_s_1s_max"] is None
    assert envelope["dimension_null_reasons"]["speed.peak_output_tok_s_1s_max"] == (
        "per_request_timing_arrays_unavailable"
    )
    row_reasons = json.loads(envelope_to_rows(envelope)[0]["campaign_dimension_null_reasons_json"])
    assert row_reasons["speed.peak_output_tok_s_1s_max"] == "per_request_timing_arrays_unavailable"

    generic = dict(envelope)
    generic.pop("campaign_dimensions")
    generic.pop("dimension_null_reasons")
    generic["results"] = [
        {
            "name": "llama.cpp",
            "metric": "throughput_output_toks_per_s",
            "value": 25.0,
            "unit": "tok/s",
            "tags": {},
        }
    ]
    generic_row = envelope_to_rows(generic)[0]
    assert generic_row["campaign_speed_peak_output_tok_s_1s_max"] is None
    generic_reasons = json.loads(generic_row["campaign_dimension_null_reasons_json"])
    assert generic_reasons["speed.peak_output_tok_s_1s_max"] == "derived_input_unavailable"


def test_vllm_missing_optional_metrics(vllm_sample: dict) -> None:
    converter = get_converter("vllm")
    ctx = ConverterContext(
        suite="throughput",
        model="some/model",
        git_sha="deadbeef",
        system=detect_system(),
    )
    sparse = {k: vllm_sample[k] for k in ("output_throughput", "median_ttft_ms", "p99_ttft_ms")}
    envelope = converter.build_envelope(sparse, ctx)
    validate_envelope(envelope)
    metric_names = {r["metric"] for r in envelope["results"]}
    assert "throughput_output_toks_per_s" in metric_names
    assert "tpot_p50_ms" not in metric_names  # not in sparse sample
