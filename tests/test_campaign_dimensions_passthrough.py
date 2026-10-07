"""Campaign dimensions survive conversion: no converter drops them.

``publish.envelope_to_rows`` flattens ``campaign_dimensions`` into fixed Parquet columns
when the envelope has them, so a converter that builds an envelope without them
publishes every campaign row with those columns empty. These tests pin that every
converter keeps the dimensions it is handed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from mlx_benchmarks.cli import build_parser
from mlx_benchmarks.converters import get_converter
from mlx_benchmarks.converters.base import ConverterContext
from mlx_benchmarks.envelope import validate_envelope
from mlx_benchmarks.publish import envelope_to_rows

DIMENSIONS: dict[str, Any] = {
    "hardware": {"machine": "Example GPU", "power_cap_w": 300.0, "ups_circuit": None},
    "software": {"engine": "vLLM", "engine_version": "0.30.0", "backend": "CUDA"},
    "model": {"hf_repo": "example/model", "quantization": "NVFP4", "revision_sha": None},
    "run": {"concurrent_agents": 4, "allocated_context_tokens": 196608},
    "quality": {"benchmark_name": None, "score": None},
}
NULL_REASONS = {
    "hardware.ups_circuit": "NOT_SOURCED",
    "model.revision_sha": "MODEL_REVISION_UNKNOWN",
    "quality.benchmark_name": "NOT_RUN",
    "quality.score": "NOT_RUN",
}
SYSTEM = {"os": "Ubuntu 24.04", "chip": "Example x86-64 CPU", "memory_gb": 128}


def _bench_serve() -> list[dict]:
    return [
        {
            "model_id": "example/model",
            "prompt_set": "short",
            "concurrency": 1,
            "repetition": 0,
            "gen_tps": 100.0,
            "throughput_tps": 110.0,
            "ttft_ms": 70.0,
            "tpot_ms": 7.5,
        }
    ]


def _agentic_partial(agentic_sample: dict) -> list[dict]:
    return [
        {"kind": "cell", **agentic_sample["cells"][0]},
        {"kind": "multiturn", **agentic_sample["multiturn"][0]},
    ]


def _throughput_probe() -> dict:
    return {
        "model": "example/model",
        "thinking": "off",
        "sequential": {"cumulative_tok_s": {"median": 1.0}},
    }


# kind -> (suite, builder of the raw tool output). Builders take the shared fixtures.
RAW_BY_KIND: dict[str, tuple[str, Callable[[pytest.FixtureRequest], Any]]] = {
    "agentic": ("tool-calling", lambda r: r.getfixturevalue("agentic_sample")),
    "agentic-partial": (
        "tool-calling",
        lambda r: _agentic_partial(r.getfixturevalue("agentic_sample")),
    ),
    "bench-serve": ("throughput", lambda _: _bench_serve()),
    "coding-replay": ("coding", lambda r: r.getfixturevalue("coding_replay_sample")),
    "factual": ("grounded-summary", lambda r: r.getfixturevalue("factual_sample")),
    "fio": ("fio", lambda r: r.getfixturevalue("fio_sample")),
    "gpu-burn": ("gpu-burn", lambda r: r.getfixturevalue("gpu_burn_sample")),
    "llamacpp-server": ("throughput", lambda _: []),
    "lm-eval": ("reasoning", lambda r: r.getfixturevalue("lm_eval_sample")),
    "mbw": ("mbw", lambda r: r.getfixturevalue("mbw_sample")),
    "nvbandwidth": ("nvbandwidth", lambda r: r.getfixturevalue("nvbandwidth_sample")),
    "promptstack": ("promptstack", lambda r: r.getfixturevalue("promptstack_sample")),
    "throughput-probe": ("throughput", lambda _: _throughput_probe()),
    "vllm": ("throughput", lambda r: r.getfixturevalue("vllm_sample")),
}


def _context(suite: str, **overrides: Any) -> ConverterContext:
    return ConverterContext(suite=suite, model="example/model", git_sha="abc1234", system=SYSTEM, **overrides)


def test_every_cli_kind_is_covered() -> None:
    """A converter added later must join this matrix, not escape it."""
    kind_action = next(action for action in build_parser()._actions if action.dest == "kind")
    assert set(RAW_BY_KIND) == set(kind_action.choices or ())


@pytest.mark.parametrize("kind", sorted(RAW_BY_KIND))
def test_converter_keeps_declared_campaign_dimensions(kind: str, request: pytest.FixtureRequest) -> None:
    suite, build_raw = RAW_BY_KIND[kind]
    envelope = get_converter(kind).build_envelope(
        build_raw(request),
        _context(suite, campaign_dimensions=DIMENSIONS, dimension_null_reasons=NULL_REASONS),
    )

    assert envelope["campaign_dimensions"] == DIMENSIONS
    assert envelope["dimension_null_reasons"] == NULL_REASONS
    validate_envelope(envelope)


@pytest.mark.parametrize("kind", sorted(RAW_BY_KIND))
def test_converter_leaves_dimensions_absent_when_none_are_declared(
    kind: str, request: pytest.FixtureRequest
) -> None:
    suite, build_raw = RAW_BY_KIND[kind]
    envelope = get_converter(kind).build_envelope(build_raw(request), _context(suite))

    assert "campaign_dimensions" not in envelope
    assert "dimension_null_reasons" not in envelope


def test_converter_dimensions_reach_the_published_rows(vllm_sample: dict) -> None:
    envelope = get_converter("vllm").build_envelope(
        vllm_sample,
        _context("throughput", campaign_dimensions=DIMENSIONS, dimension_null_reasons=NULL_REASONS),
    )

    rows = envelope_to_rows(envelope)

    assert rows
    for row in rows:
        assert row["campaign_hardware_machine"] == "Example GPU"
        assert row["campaign_hardware_power_cap_w"] == 300.0
        assert row["campaign_run_concurrent_agents"] == 4
        assert row["campaign_hardware_ups_circuit"] is None
        assert json.loads(row["campaign_dimension_null_reasons_json"]) == NULL_REASONS


def test_throughput_probe_keeps_dimensions_written_into_its_result_file() -> None:
    raw = {**_throughput_probe(), "campaign_dimensions": DIMENSIONS, "dimension_null_reasons": NULL_REASONS}

    envelope = get_converter("throughput-probe").build_envelope(raw, _context("throughput"))

    assert envelope["campaign_dimensions"] == DIMENSIONS
    assert envelope["dimension_null_reasons"] == NULL_REASONS
    validate_envelope(envelope)


def test_declared_dimensions_replace_the_ones_in_the_result_file() -> None:
    declared = {"hardware": {"machine": "Declared GPU"}}
    raw = {**_throughput_probe(), "campaign_dimensions": DIMENSIONS}

    envelope = get_converter("throughput-probe").build_envelope(
        raw, _context("throughput", campaign_dimensions=declared)
    )

    assert envelope["campaign_dimensions"] == declared
