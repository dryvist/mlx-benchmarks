"""NVIDIA additions to envelope v1 are optional: legacy envelopes still validate and
a synthetic NVIDIA envelope round-trips to Parquet with the columns the viewer reads."""

from __future__ import annotations

import ast
import io
import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from mlx_benchmarks.converters import get_converter
from mlx_benchmarks.converters.base import ConverterContext
from mlx_benchmarks.envelope import EnvelopeValidationError, load_schema, validate_envelope
from mlx_benchmarks.publish import envelope_to_rows, publish, rows_to_parquet

NVIDIA_SYSTEM_KEYS = ("gpu", "engine", "power_limit_w", "container")
NEW_SUITES = ("gpu-burn", "nvbandwidth", "mbw", "fio")
# The suite enum as it stood before the NVIDIA campaign: none of these may disappear.
LEGACY_SUITES = (
    "throughput",
    "ttft",
    "tool-calling",
    "code-accuracy",
    "framework-eval",
    "capability-comparison",
    "coding",
    "reasoning",
    "knowledge",
    "evalplus",
    "math-hard",
    "promptstack",
    "grounded-summary",
)


def _viewer_columns() -> list[str]:
    """``EXPECTED_COLUMNS`` read from space/app.py's source (importing it needs gradio)."""
    source = (Path(__file__).resolve().parents[1] / "space" / "app.py").read_text()
    match = re.search(r"^EXPECTED_COLUMNS = (\[.*?\])$", source, re.M)
    assert match, "space/app.py no longer defines EXPECTED_COLUMNS"
    return ast.literal_eval(match.group(1))


def _parquet_table(envelope: dict[str, Any]) -> Any:
    return pq.read_table(io.BytesIO(rows_to_parquet(envelope_to_rows(envelope))))  # type: ignore[arg-type]


def test_legacy_envelopes_validate_without_nvidia_fields(
    valid_envelope: dict, cluster_envelope: dict
) -> None:
    for legacy in (valid_envelope, cluster_envelope):
        assert not set(NVIDIA_SYSTEM_KEYS) & set(legacy["system"]), "fixture is not a legacy envelope"
        validate_envelope(legacy)


def test_nvidia_envelope_validates(nvidia_envelope: dict) -> None:
    validate_envelope(nvidia_envelope)
    system = nvidia_envelope["system"]
    assert set(NVIDIA_SYSTEM_KEYS) <= set(system)
    assert set(system["gpu"]) == {"model", "vram_gb", "driver", "cuda"}
    assert set(system["engine"]) == {"name", "version"}


def test_suite_enum_gains_baselines_and_keeps_legacy_values() -> None:
    enum = load_schema()["properties"]["suite"]["enum"]
    assert set(NEW_SUITES) <= set(enum)
    assert set(LEGACY_SUITES) <= set(enum)


@pytest.mark.parametrize("suite", NEW_SUITES)
def test_new_suites_validate(valid_envelope: dict, suite: str) -> None:
    validate_envelope({**valid_envelope, "suite": suite})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("gpu", {"model": "x", "bogus": 1}),
        ("gpu", {"vram_gb": -1}),
        ("gpu", {"vram_gb": "96"}),
        ("engine", {"name": "vllm", "bogus": 1}),
        ("engine", {"version": 0.30}),
        ("power_limit_w", "300W"),
        ("power_limit_w", -5),
        ("container", {"image": "x"}),
    ],
)
def test_nvidia_system_fields_are_typed(valid_envelope: dict, key: str, value: Any) -> None:
    env = {**valid_envelope, "system": {**valid_envelope["system"], key: value}}
    with pytest.raises(EnvelopeValidationError):
        validate_envelope(env)


def test_sweep_tag_keys_are_documented_strings() -> None:
    tags = load_schema()["properties"]["results"]["items"]["properties"]["tags"]
    for key in ("prompt_tokens", "concurrency", "context_len"):
        assert tags["properties"][key]["type"] == "string"
        assert tags["properties"][key]["description"]


@pytest.mark.parametrize("key", ["prompt_tokens", "concurrency", "context_len", "anything_else"])
def test_tag_values_must_be_strings(nvidia_envelope: dict, key: str) -> None:
    nvidia_envelope["results"][0]["tags"][key] = 8192
    with pytest.raises(EnvelopeValidationError):
        validate_envelope(nvidia_envelope)


def test_nvidia_envelope_converts_to_parquet_with_viewer_columns(nvidia_envelope: dict) -> None:
    table = _parquet_table(nvidia_envelope)
    assert set(_viewer_columns()) <= set(table.column_names)
    assert table.num_rows == len(nvidia_envelope["results"])
    row = table.to_pylist()[0]
    # The viewer's columns keep their meaning...
    assert (row["suite"], row["metric"], row["model"]) == (
        "throughput",
        "throughput_total_toks_per_s",
        row["model"],
    )
    assert row["value"] == pytest.approx(3371.91)
    # ...and the new system fields ride along: nested objects as JSON strings, scalars as-is.
    assert json.loads(row["gpu"])["model"] == "NVIDIA Example GPU"
    assert json.loads(row["engine"]) == {"name": "vllm", "version": "0.30.0"}
    assert row["power_limit_w"] == 300
    assert row["container"] == "vllm/vllm-openai:v0.30.0"
    assert (row["tag_prompt_tokens"], row["tag_concurrency"], row["tag_context_len"]) == (
        "8192",
        "4",
        "139264",
    )


def test_nvidia_envelope_publishes_in_dry_run(nvidia_envelope: dict) -> None:
    assert publish(nvidia_envelope, dry_run=True).startswith(
        "data/run-canonical-run-2026-10-04T12-00-00-abc1234-throughput-"
    )


def test_legacy_and_nvidia_envelopes_use_the_same_parquet_schema(
    valid_envelope: dict, nvidia_envelope: dict
) -> None:
    legacy_table = _parquet_table(valid_envelope)
    nvidia_table = _parquet_table(nvidia_envelope)
    assert legacy_table.schema.equals(nvidia_table.schema, check_metadata=False)
    assert set(_viewer_columns()) <= set(legacy_table.column_names)
    legacy_row = legacy_table.to_pylist()[0]
    assert all(legacy_row[key] is None for key in NVIDIA_SYSTEM_KEYS)


def test_vllm_bench_serve_on_nvidia_host_end_to_end(
    vllm_bench_serve_nvidia_sample: dict, nvidia_envelope: dict
) -> None:
    """`vllm bench serve --save-result` JSON -> existing vllm converter, with the NVIDIA
    system block and sweep tags the runbook passes -> valid -> Parquet."""
    ctx = ConverterContext(
        suite="throughput",
        model="example-org/example-32b",
        git_sha="abc1234",
        env_class="isolated",
        concurrency=4,
        system=nvidia_envelope["system"],
        extra_tags={"prompt_tokens": "8192", "concurrency": "4", "context_len": "139264"},
    )
    envelope = get_converter("vllm").build_envelope(vllm_bench_serve_nvidia_sample, ctx)
    validate_envelope(envelope)

    metrics = {r["metric"]: r for r in envelope["results"]}
    assert metrics["throughput_total_toks_per_s"]["value"] == pytest.approx(3372.0, abs=0.5)
    assert {"ttft_p50_ms", "ttft_p99_ms", "itl_p50_ms", "tpot_p50_ms"} <= set(metrics)
    assert metrics["ttft_p50_ms"]["tags"]["prompt_tokens"] == "8192"
    assert metrics["ttft_p50_ms"]["tags"]["completed_requests"] == "16"

    table = _parquet_table(dict(envelope))
    assert set(_viewer_columns()) <= set(table.column_names)
    assert json.loads(table.to_pylist()[0]["gpu"])["driver"] == "570.172.08"
