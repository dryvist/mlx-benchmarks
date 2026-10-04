"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def lm_eval_sample() -> dict[str, Any]:
    return json.loads((FIXTURES / "lm_eval_results_sample.json").read_text())


@pytest.fixture
def vllm_sample() -> dict[str, Any]:
    return json.loads((FIXTURES / "vllm_benchmark_serving_sample.json").read_text())


@pytest.fixture
def agentic_sample() -> dict[str, Any]:
    return json.loads((FIXTURES / "agentic_sample.json").read_text())


@pytest.fixture
def promptstack_sample() -> dict[str, Any]:
    return json.loads((FIXTURES / "promptstack_sample.json").read_text())


@pytest.fixture
def factual_sample() -> dict[str, Any]:
    return json.loads((FIXTURES / "factual_sample.json").read_text())


@pytest.fixture
def coding_replay_sample() -> list[dict[str, Any]]:
    return json.loads((FIXTURES / "coding_replay_sample.json").read_text())


@pytest.fixture
def vllm_bench_serve_nvidia_sample() -> dict[str, Any]:
    return json.loads((FIXTURES / "vllm_bench_serve_nvidia_synthetic.json").read_text())


@pytest.fixture
def fio_sample() -> dict[str, Any]:
    return json.loads((FIXTURES / "fio_randread_synthetic.json").read_text())


@pytest.fixture
def nvbandwidth_sample() -> dict[str, Any]:
    return json.loads((FIXTURES / "nvbandwidth_synthetic.json").read_text())


@pytest.fixture
def mbw_sample() -> dict[str, Any]:
    # Same shape the CLI builds from a .txt file: {"output": <raw text>}.
    return {"output": (FIXTURES / "mbw_synthetic.txt").read_text()}


@pytest.fixture
def gpu_burn_sample() -> dict[str, Any]:
    # Stored as JSON because the real output is CR-redrawn and a text fixture would
    # lose its carriage returns to the repo's line-ending hook.
    return json.loads((FIXTURES / "gpu_burn_synthetic.json").read_text())


@pytest.fixture
def valid_envelope() -> dict[str, Any]:
    return json.loads((EXAMPLES / "envelope.valid.json").read_text())


@pytest.fixture
def cluster_envelope() -> dict[str, Any]:
    return json.loads((EXAMPLES / "envelope.cluster.json").read_text())


@pytest.fixture
def nvidia_envelope() -> dict[str, Any]:
    return json.loads((EXAMPLES / "envelope.nvidia.json").read_text())


@pytest.fixture
def invalid_envelope() -> dict[str, Any]:
    return json.loads((EXAMPLES / "envelope.invalid.json").read_text())
