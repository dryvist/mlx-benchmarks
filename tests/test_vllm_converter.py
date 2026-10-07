"""End-to-end: vllm benchmark_serving sample -> envelope -> passes schema validation."""

from __future__ import annotations

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
