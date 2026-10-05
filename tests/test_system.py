"""Runtime system detection — smoke test that it returns a schema-shaped dict."""

from __future__ import annotations

import json

import pytest

from mlx_benchmarks.envelope import validate_envelope
from mlx_benchmarks.system import _detect_accelerator, _detect_topology, detect_system

_TOPOLOGY_VARS = (
    "MLX_BENCH_WORLD_SIZE",
    "MLX_BENCH_PARALLELISM",
    "MLX_BENCH_INTERCONNECT",
    "MLX_BENCH_NODES",
)


_ACCELERATOR_VARS = (
    "MLX_BENCH_GPU_MODEL",
    "MLX_BENCH_GPU_VRAM_GB",
    "MLX_BENCH_GPU_DRIVER",
    "MLX_BENCH_GPU_CUDA",
    "MLX_BENCH_ENGINE_NAME",
    "MLX_BENCH_ENGINE_VERSION",
    "MLX_BENCH_POWER_LIMIT_W",
    "MLX_BENCH_CONTAINER",
)


def _clear_accelerator_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ACCELERATOR_VARS:
        monkeypatch.delenv(var, raising=False)


def test_detect_system_returns_required_fields() -> None:
    system = detect_system()
    for key in ("os", "chip", "memory_gb"):
        assert key in system, f"detect_system() must always set {key!r}"

    assert isinstance(system["os"], str) and system["os"]
    assert isinstance(system["chip"], str) and system["chip"]
    assert isinstance(system["memory_gb"], int)
    assert system["memory_gb"] >= 0


def test_detect_system_includes_python_version() -> None:
    import platform

    system = detect_system()
    assert system.get("python_version") == platform.python_version()


def test_detect_topology_absent_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _TOPOLOGY_VARS:
        monkeypatch.delenv(var, raising=False)
    assert _detect_topology() is None


def test_detect_topology_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLX_BENCH_WORLD_SIZE", "2")
    monkeypatch.setenv("MLX_BENCH_PARALLELISM", "pipeline")
    monkeypatch.setenv("MLX_BENCH_INTERCONNECT", "tb5-rdma")
    monkeypatch.setenv(
        "MLX_BENCH_NODES",
        json.dumps([{"hostname": "node-0", "chip": "Apple M3 Ultra", "memory_gb": 256}]),
    )
    assert _detect_topology() == {
        "world_size": 2,
        "parallelism": "pipeline",
        "interconnect": "tb5-rdma",
        "nodes": [{"hostname": "node-0", "chip": "Apple M3 Ultra", "memory_gb": 256}],
    }


def test_detect_topology_ignores_malformed_nodes_json(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _TOPOLOGY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MLX_BENCH_WORLD_SIZE", "2")
    monkeypatch.setenv("MLX_BENCH_NODES", "{not json")
    # Bad nodes JSON is dropped (warning logged), not fatal — world_size survives.
    assert _detect_topology() == {"world_size": 2}


def test_detect_topology_rejects_bad_parallelism(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _TOPOLOGY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MLX_BENCH_PARALLELISM", "quantum")
    # Not in the schema enum, so it's ignored rather than emitted invalid.
    assert _detect_topology() is None


def test_detect_accelerator_absent_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_accelerator_env(monkeypatch)
    assert _detect_accelerator() == {}


def test_detect_accelerator_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_accelerator_env(monkeypatch)
    monkeypatch.setenv("MLX_BENCH_GPU_MODEL", "NVIDIA Example GPU")
    monkeypatch.setenv("MLX_BENCH_GPU_VRAM_GB", "95.6")
    monkeypatch.setenv("MLX_BENCH_GPU_DRIVER", "570.172.08")
    monkeypatch.setenv("MLX_BENCH_GPU_CUDA", "12.8")
    monkeypatch.setenv("MLX_BENCH_ENGINE_NAME", "vllm")
    monkeypatch.setenv("MLX_BENCH_ENGINE_VERSION", "0.30.0")
    monkeypatch.setenv("MLX_BENCH_POWER_LIMIT_W", "300.00")
    monkeypatch.setenv("MLX_BENCH_CONTAINER", "vllm/vllm-openai:v0.30.0")
    detected = _detect_accelerator()
    assert detected == {
        "gpu": {"model": "NVIDIA Example GPU", "vram_gb": 95.6, "driver": "570.172.08", "cuda": "12.8"},
        "engine": {"name": "vllm", "version": "0.30.0"},
        "power_limit_w": 300.0,
        "container": "vllm/vllm-openai:v0.30.0",
    }
    # What the env declares is exactly what schema.json accepts.
    envelope = {
        "schema_version": "1",
        "timestamp": "2026-10-04T12:00:00Z",
        "git_sha": "abc1234",
        "trigger": "local",
        "suite": "throughput",
        "model": "m",
        "system": {"os": "Linux", "chip": "x86_64", "memory_gb": 128, **detected},
        "results": [{"name": "n", "metric": "m", "value": 1.0, "unit": "u"}],
    }
    validate_envelope(envelope)


def test_detect_accelerator_drops_non_numeric_values(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_accelerator_env(monkeypatch)
    monkeypatch.setenv("MLX_BENCH_GPU_MODEL", "  NVIDIA Example GPU  ")
    monkeypatch.setenv("MLX_BENCH_GPU_VRAM_GB", "lots")
    monkeypatch.setenv("MLX_BENCH_POWER_LIMIT_W", "300W")
    # Strings are trimmed; numbers that are not numbers are dropped, not published as strings.
    assert _detect_accelerator() == {"gpu": {"model": "NVIDIA Example GPU"}}


def test_detect_accelerator_ignores_blank_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_accelerator_env(monkeypatch)
    for var in _ACCELERATOR_VARS:
        monkeypatch.setenv(var, "  ")
    assert _detect_accelerator() == {}
