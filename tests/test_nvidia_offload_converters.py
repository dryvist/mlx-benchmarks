from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from mlx_benchmarks.cli import _read_kvo_results, _read_moe_csv, _validate_run_index_metadata, main
from mlx_benchmarks.converters import KvOffloadConverter, MoeOffloadConverter
from mlx_benchmarks.converters.base import ConverterContext
from mlx_benchmarks.dataset_schema import STANDARD_CACHE_FIELD_DIMENSIONS, campaign_dimension_column
from mlx_benchmarks.envelope import validate_envelope
from mlx_benchmarks.publish import envelope_to_rows

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def _context(suite: str, model: str, version: str) -> ConverterContext:
    return ConverterContext(
        suite=suite,
        model=model,
        git_sha="deadbeef",
        timestamp_override="2026-10-18T00:00:00Z",
        system={"os": "Linux", "chip": "x86_64", "memory_gb": 64},
        engine_version=version,
    )


def _run_index_validator() -> ModuleType:
    path = ROOT / "scripts" / "validate_run_index.py"
    spec = importlib.util.spec_from_file_location("validate_run_index", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_cache_projection_in_rows(envelope: dict[str, Any]) -> None:
    row = envelope_to_rows(envelope)[0]
    dimensions = envelope["campaign_dimensions"]
    reasons = json.loads(row["campaign_dimension_null_reasons_json"])
    for _standard_name, (category, field) in STANDARD_CACHE_FIELD_DIMENSIONS.items():
        column = campaign_dimension_column(category, field)
        assert column in row
        assert row[column] == dimensions[category][field]
        if row[column] is None:
            assert f"{category}.{field}" in reasons


def test_kvo_lines_become_per_cell_envelopes_with_complete_cache_dimensions() -> None:
    rows = _read_kvo_results(FIXTURES / "kvo_result_sample.jsonl")
    converter = KvOffloadConverter()
    envelopes = [
        converter.build_envelope(
            row, _context("lmcache-multi-round-qa", "nvidia/Qwen3.8-27B-NVFP4", "0.32.0")
        )
        for row in rows
    ]

    assert len(envelopes) == 2
    for envelope in envelopes:
        validate_envelope(envelope)
        dimensions = envelope["campaign_dimensions"]
        reasons = envelope["dimension_null_reasons"]
        for standard_name, (category, field) in STANDARD_CACHE_FIELD_DIMENSIONS.items():
            assert field in dimensions[category], standard_name
            if dimensions[category][field] is None:
                assert f"{category}.{field}" in reasons
        assert dimensions["hardware"]["machine"] == "RTX PRO 6000 Max-Q 96GB"
        assert dimensions["software"]["engine"] == "vLLM"
        assert dimensions["software"]["engine_version"] == "0.32.0"
        assert envelope["system"]["engine"] == {"name": "vLLM", "version": "0.32.0"}
        assert dimensions["run"]["moe_cpu_offload"] is None
        assert reasons["run.moe_cpu_offload"] == "NOT_APPLICABLE_ENGINE"

    assert envelopes[0]["campaign_dimensions"]["run"]["kv_offload_backend"] == "native"
    assert envelopes[1]["campaign_dimensions"]["run"]["kv_offload_backend"] == "none"
    assert envelopes[1]["dimension_null_reasons"]["speed.external_prefix_cache_hit_rate"] == "NOT_CAPTURED"
    rows = envelope_to_rows(envelopes[0])
    metrics = {row["metric"]: row["value"] for row in rows}
    assert metrics == {
        "ttft_first_round_p50_ms": 1240.5,
        "ttft_later_rounds_p50_ms": 432.8,
        "ttft_later_rounds_p99_ms": 911.2,
        "tpot_p50_ms_per_token": 24.5,
        "tpot_p99_ms_per_token": 39.1,
        "throughput_tokens_per_s": 172.6,
        "requests_per_s": 4.27,
        "prefix_cache_hit_rate": 0.81,
        "external_prefix_cache_hit_rate": 0.34,
        "kv_cache_usage_peak": 99.1,
        "host_ram_peak_gib": 52.0,
        "preemptions": 0.0,
    }
    assert rows[0]["campaign_hardware_machine"] == "RTX PRO 6000 Max-Q 96GB"
    _assert_cache_projection_in_rows(envelopes[0])
    _assert_cache_projection_in_rows(envelopes[1])


def test_moe_csv_maps_engine_metrics_and_explicitly_null_cache_fields() -> None:
    rows = _read_moe_csv(FIXTURES / "moe_offload_sample.csv")
    converter = MoeOffloadConverter()
    envelopes = [
        converter.build_envelope(row, _context("moe-offload", row["model"], "b11669")) for row in rows
    ]

    assert len(envelopes) == 2
    for envelope in envelopes:
        validate_envelope(envelope)
        dimensions = envelope["campaign_dimensions"]
        reasons = envelope["dimension_null_reasons"]
        for _standard_name, (category, field) in STANDARD_CACHE_FIELD_DIMENSIONS.items():
            assert field in dimensions[category]
            if dimensions[category][field] is None:
                assert f"{category}.{field}" in reasons
        assert dimensions["hardware"]["machine"] == "RTX 4080 SUPER 16GB"
        assert dimensions["software"]["engine"] == "llama.cpp"
        assert dimensions["run"]["parallel_slots"] == 4
        assert "semaphore_task" not in str(envelope)

    first_run = envelopes[0]["campaign_dimensions"]["run"]
    assert first_run["moe_cpu_offload"] == "ncmoe"
    assert first_run["moe_cpu_layers"] == 12
    assert first_run["llamacpp_cache_ram_mib"] == 8192
    assert first_run["kv_offload_cpu_gib"] == 8.0
    second_run = envelopes[1]["campaign_dimensions"]["run"]
    assert second_run["moe_cpu_offload"] == "cpumoe"
    assert second_run["moe_cpu_layers"] is None
    assert envelopes[1]["dimension_null_reasons"]["run.moe_cpu_layers"] == "NOT_CAPTURED"
    metrics = {row["metric"]: row["value"] for row in envelope_to_rows(envelopes[0])}
    assert metrics == {
        "throughput_output_toks_per_s": 52.4,
        "ttft_p50_ms": 840.2,
        "ttft_p99_ms": 1214.5,
        "gpu_vram_used_gib": 15.1,
        "host_ram_peak_gib": 26.3,
    }
    for envelope in envelopes:
        _assert_cache_projection_in_rows(envelope)


def test_kvo_cli_publishes_each_cell_and_appends_run_index_entries(
    tmp_path: Path,
    published_metadata: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = FIXTURES / "kvo_result_sample.jsonl"
    metadata_path = tmp_path / "published-metadata.json"
    metadata_path.write_text(json.dumps(published_metadata))
    index_path = tmp_path / "run-index-v1.json"
    index_path.write_text('{\n  "schema_version": "run-index-v1",\n  "runs": [\n  ]\n}\n', encoding="utf-8")
    published: list[dict[str, Any]] = []

    def fake_publish(envelope: dict[str, Any], **_: Any) -> str:
        published.append(envelope)
        return f"data/run-2026-10-18T00-00-0{len(published)}-cell.parquet"

    monkeypatch.setattr(
        "mlx_benchmarks.cli.detect_system", lambda: pytest.fail("offload import must not probe host")
    )
    monkeypatch.setattr("mlx_benchmarks.cli.publish", fake_publish)
    assert (
        main(
            [
                str(source),
                "--kind",
                "kv-offload",
                "--suite",
                "lmcache-multi-round-qa",
                "--model",
                "nvidia/Qwen3.8-27B-NVFP4",
                "--model-revision",
                "a" * 40,
                "--engine-version",
                "0.32.0",
                "--git-sha",
                "deadbeef",
                "--published-metadata",
                str(metadata_path),
                "--run-index",
                str(index_path),
            ]
        )
        == 0
    )

    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert len(published) == 2
    assert [entry["status"] for entry in index["runs"]] == ["experimental", "experimental"]
    assert [entry["variant"] for entry in index["runs"]] == ["A2 W2", "A0 W2"]
    assert all(entry["machine"] == "RTX PRO 6000 Max-Q 96GB" for entry in index["runs"])
    _run_index_validator().validate_entries(index["runs"])


def test_moe_cli_publishes_csv_cells_and_indexes_them(
    tmp_path: Path,
    published_metadata: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_path = tmp_path / "published-metadata.json"
    metadata_path.write_text(json.dumps(published_metadata))
    index_path = tmp_path / "run-index-v1.json"
    index_path.write_text('{"schema_version":"run-index-v1","runs":[]}', encoding="utf-8")
    published: list[dict[str, Any]] = []

    def fake_publish(envelope: dict[str, Any], **_: Any) -> str:
        published.append(envelope)
        return f"data/run-2026-10-18T00-00-0{len(published)}-cell.parquet"

    monkeypatch.setattr(
        "mlx_benchmarks.cli.detect_system", lambda: pytest.fail("offload import must not probe host")
    )
    monkeypatch.setattr("mlx_benchmarks.cli.publish", fake_publish)
    assert (
        main(
            [
                str(FIXTURES / "moe_offload_sample.csv"),
                "--kind",
                "moe-offload",
                "--suite",
                "moe-offload",
                "--model-revision",
                "a" * 40,
                "--engine-version",
                "b11669",
                "--git-sha",
                "deadbeef",
                "--published-metadata",
                str(metadata_path),
                "--run-index",
                str(index_path),
            ]
        )
        == 0
    )

    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert len(published) == 2
    assert [envelope["model"] for envelope in published] == ["Qwen3-30B-A3B-Q4_K_M.gguf"] * 2
    assert [entry["status"] for entry in index["runs"]] == ["experimental", "experimental"]
    assert all(entry["machine"] == "RTX 4080 SUPER 16GB" for entry in index["runs"])
    _run_index_validator().validate_entries(index["runs"])


def test_offload_cli_rejects_invalid_run_index_before_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "invalid-run-index.json"
    index_path.write_text('{"schema_version":"wrong","runs":[]}', encoding="utf-8")
    monkeypatch.setattr("mlx_benchmarks.cli.publish", lambda *_a, **_kw: pytest.fail("must not publish"))
    assert (
        main(
            [
                str(FIXTURES / "kvo_result_sample.jsonl"),
                "--kind",
                "kv-offload",
                "--suite",
                "lmcache-multi-round-qa",
                "--run-index",
                str(index_path),
            ]
        )
        == 2
    )


def test_offload_cli_requires_an_engine_version() -> None:
    row = _read_kvo_results(FIXTURES / "kvo_result_sample.jsonl")[0]
    with pytest.raises(ValueError, match="engine version is required"):
        KvOffloadConverter().build_envelope(
            row, _context("lmcache-multi-round-qa", "nvidia/Qwen3.8-27B-NVFP4", "")
        )


def test_scored_llama_run_index_requires_release_build_and_commit(
    published_metadata: dict[str, Any],
) -> None:
    from argparse import Namespace

    args = Namespace(run_index_status="scored")
    with pytest.raises(ValueError, match="bNNNNN"):
        _validate_run_index_metadata(
            args,
            {
                **published_metadata,
                "engine": "llama.cpp",
                "engine_version": "unknown",
                "engine_build": "unknown",
                "engine_commit": "bad",
            },
        )
