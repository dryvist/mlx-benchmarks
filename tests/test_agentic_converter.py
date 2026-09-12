"""End-to-end: agentic runner sample -> envelope -> passes schema validation."""

from __future__ import annotations

import pytest

from mlx_benchmarks.converters import get_converter
from mlx_benchmarks.converters.base import ConverterContext
from mlx_benchmarks.envelope import validate_envelope
from mlx_benchmarks.system import detect_system


def _ctx(**overrides: object) -> ConverterContext:
    defaults: dict = {
        "suite": "tool-calling",
        "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "git_sha": "deadbeef",
        "system": detect_system(),
    }
    defaults.update(overrides)
    return ConverterContext(**defaults)


def test_agentic_round_trip(agentic_sample: dict) -> None:
    converter = get_converter("agentic")
    envelope = converter.build_envelope(agentic_sample, _ctx())

    validate_envelope(envelope)

    assert envelope["suite"] == "tool-calling"
    assert envelope["timestamp"] == "2026-07-08T04:15:00Z"

    results = envelope["results"]
    assert all(r["name"] == "tool_calling" for r in results)

    # 2 cells x (3 rate + 2 latency + 1 aggregate throughput) + 2 multiturn rows
    metric_names = [r["metric"] for r in results]
    assert metric_names.count("valid_tool_call_rate") == 2
    assert metric_names.count("finish_reason_tool_calls_rate") == 2
    assert metric_names.count("request_latency_p50_ms") == 2
    assert metric_names.count("request_latency_p95_ms") == 2
    assert metric_names.count("aggregate_tokens_per_second") == 2
    assert metric_names.count("first_degraded_round") == 2


def test_agentic_cell_dimensions_and_measurements(agentic_sample: dict) -> None:
    converter = get_converter("agentic")
    envelope = converter.build_envelope(agentic_sample, _ctx())
    validate_envelope(envelope)

    gate_cell = [
        r
        for r in envelope["results"]
        if r["metric"] == "valid_tool_call_rate" and r["tags"]["cell"] == "conc4_think-on_ctx-large_stream"
    ]
    assert len(gate_cell) == 1
    result = gate_cell[0]
    assert result["value"] == 0.9
    assert result["unit"] == "ratio"
    # Sweep dimensions travel as string tags.
    tags = result["tags"]
    assert tags["concurrency"] == "4"
    assert tags["thinking"] == "on"
    assert tags["context"] == "large"
    assert tags["stream"] == "stream"
    # Failure taxonomy counts as string tags.
    assert tags["failure_empty_function_name"] == "1"
    assert tags["failure_no_tool_call"] == "0"
    # First-class measurement fields populated where measured.
    assert result["duration_seconds"] == 184.2
    assert result["decode_tokens_per_second"] == 21.4
    assert result["first_token_latency_ms"] == 2410.2


def test_agentic_publishes_both_throughput_numbers(agentic_sample: dict) -> None:
    # decode_tokens_per_second keeps the per-request mean (effective, 21.4);
    # the wall-clock aggregate (58.7) rides alongside as its own metric row so
    # neither number silently changes meaning across shards.
    converter = get_converter("agentic")
    envelope = converter.build_envelope(agentic_sample, _ctx())
    validate_envelope(envelope)

    gate = "conc4_think-on_ctx-large_stream"
    decode = next(
        r for r in envelope["results"] if r["metric"] == "valid_tool_call_rate" and r["tags"]["cell"] == gate
    )
    assert decode["decode_tokens_per_second"] == 21.4

    aggregate = next(
        r
        for r in envelope["results"]
        if r["metric"] == "aggregate_tokens_per_second" and r["tags"]["cell"] == gate
    )
    assert aggregate["value"] == 58.7
    assert aggregate["unit"] == "tok/s"
    assert aggregate["duration_seconds"] == 184.2


def test_agentic_nostream_cell_has_no_first_token_latency(agentic_sample: dict) -> None:
    converter = get_converter("agentic")
    envelope = converter.build_envelope(agentic_sample, _ctx())
    nostream = [
        r
        for r in envelope["results"]
        if r["metric"] == "valid_tool_call_rate" and r["tags"]["stream"] == "nostream"
    ]
    assert len(nostream) == 1
    assert "first_token_latency_ms" not in nostream[0]
    assert nostream[0]["value"] == 1.0


def test_agentic_multiturn_mapping(agentic_sample: dict) -> None:
    converter = get_converter("agentic")
    envelope = converter.build_envelope(agentic_sample, _ctx())
    validate_envelope(envelope)

    multiturn = [r for r in envelope["results"] if r["metric"] == "first_degraded_round"]
    by_thinking = {r["tags"]["thinking"]: r for r in multiturn}

    degraded = by_thinking["on"]
    assert degraded["value"] == 5.0
    assert degraded["unit"] == "round"
    assert degraded["tags"]["degraded"] == "true"
    assert degraded["tags"]["track"] == "multiturn"
    assert degraded["tags"]["valid_rounds"] == "4"

    clean = by_thinking["off"]
    assert clean["value"] == 0.0  # never degraded -> 0 with degraded=false
    assert clean["tags"]["degraded"] == "false"
    assert clean["tags"]["rounds"] == "5"


def test_agentic_extra_tags(agentic_sample: dict) -> None:
    converter = get_converter("agentic")
    envelope = converter.build_envelope(agentic_sample, _ctx(extra_tags={"host": "mac-studio"}))
    validate_envelope(envelope)
    assert all(r["tags"]["host"] == "mac-studio" for r in envelope["results"])


def test_agentic_passes_through_run_context(agentic_sample: dict) -> None:
    # env_class / concurrency / serving flow from the context onto the envelope
    # via the shared apply_optional_fields helper.
    converter = get_converter("agentic")
    envelope = converter.build_envelope(
        agentic_sample,
        _ctx(env_class="isolated", concurrency=4, serving={"stack": "mlx_lm.server"}),
    )
    validate_envelope(envelope)
    assert envelope["env_class"] == "isolated"
    assert envelope["concurrency"] == 4
    assert envelope["serving"] == {"stack": "mlx_lm.server"}


def test_agentic_empty_cells_is_an_error() -> None:
    converter = get_converter("agentic")
    with pytest.raises(ValueError, match="no cells and no multiturn"):
        converter.build_envelope({"cells": [], "multiturn": []}, _ctx())


def test_agentic_multiturn_only_is_valid(agentic_sample: dict) -> None:
    converter = get_converter("agentic")
    raw = {"timestamp": agentic_sample["timestamp"], "multiturn": agentic_sample["multiturn"]}
    envelope = converter.build_envelope(raw, _ctx())
    validate_envelope(envelope)
    assert len(envelope["results"]) == 2


def test_dedicated_from_the_config_block_reaches_every_row(agentic_sample: dict) -> None:
    """`dedicated` is recorded in the run's `config` block, which this converter
    never read — so the field existed in the raw file and in no artifact. It is
    stamped on every row because a reader filtering for comparable measurements
    filters row by row, not by finding the run's config."""
    raw = {**agentic_sample, "config": {**agentic_sample.get("config", {}), "dedicated": True}}
    converter = get_converter("agentic")
    envelope = converter.build_envelope(raw, _ctx())
    validate_envelope(envelope)
    assert envelope["results"]
    assert all(r["tags"]["dedicated"] == "True" for r in envelope["results"])


def test_absent_dedicated_reads_as_unstated_not_false(agentic_sample: dict) -> None:
    """Every row written before the flag existed has no value. Publishing `False`
    would assert the endpoint was shared, which nobody measured."""
    raw = {k: v for k, v in agentic_sample.items() if k != "config"}
    converter = get_converter("agentic")
    envelope = converter.build_envelope(raw, _ctx())
    validate_envelope(envelope)
    assert all(r["tags"]["dedicated"] == "unstated" for r in envelope["results"])


def test_a_graded_level_publishes_verbatim_and_off_does_not_become_on(agentic_sample: dict) -> None:
    """The inversion this replaced: the old truthiness test returned "on" for the
    string "off", because a non-empty string is truthy. A graded cell therefore
    published the exact opposite of the effort it ran at."""
    raw = {
        "timestamp": agentic_sample["timestamp"],
        "cells": [
            dict(agentic_sample["cells"][0], name="conc1_think-off_ctx-small_nostream", thinking="off"),
            dict(agentic_sample["cells"][0], name="conc1_think-xhigh_ctx-small_nostream", thinking="xhigh"),
        ],
    }
    converter = get_converter("agentic")
    envelope = converter.build_envelope(raw, _ctx())
    validate_envelope(envelope)

    levels = {r["tags"]["thinking"] for r in envelope["results"]}
    assert levels == {"off", "xhigh"}


def test_a_boolean_thinking_field_still_reads_on_off(agentic_sample: dict) -> None:
    """Every row written before the harness recorded a level carries a bool, and
    those rows must keep publishing the same two values they always have."""
    raw = {
        "timestamp": agentic_sample["timestamp"],
        "cells": [
            dict(agentic_sample["cells"][0], name="a", thinking=True),
            dict(agentic_sample["cells"][0], name="b", thinking=False),
        ],
    }
    converter = get_converter("agentic")
    envelope = converter.build_envelope(raw, _ctx())
    validate_envelope(envelope)
    assert {r["tags"]["thinking"] for r in envelope["results"]} == {"on", "off"}
