"""Hardware-baseline converters (fio, nvbandwidth, mbw, gpu-burn) -> valid envelope v1.

Inputs are synthetic but follow the upstream tools' own output formats (fio stat.c JSON,
nvbandwidth json_output.cpp, mbw.c and gpu_burn-drv.cpp printf formats).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mlx_benchmarks.cli import main
from mlx_benchmarks.converters import get_converter
from mlx_benchmarks.converters.base import ConverterContext
from mlx_benchmarks.envelope import Envelope, validate_envelope
from mlx_benchmarks.publish import PublishError, envelope_to_rows, rows_to_parquet

SYSTEM = {"os": "Ubuntu 24.04", "chip": "Example x86-64 CPU", "memory_gb": 128}


def convert(kind: str, raw: Any, **tags: str) -> Envelope:
    ctx = ConverterContext(
        suite=kind, model="hardware-baseline", git_sha="abc1234", system=SYSTEM, extra_tags=tags
    )
    envelope = get_converter(kind).build_envelope(raw, ctx)
    validate_envelope(envelope)
    return envelope


def by_metric(envelope: Envelope) -> dict[str, Any]:
    return {r["metric"]: r for r in envelope["results"]}


# --- fio ---------------------------------------------------------------------


def test_fio_randread(fio_sample: dict) -> None:
    envelope = convert("fio", fio_sample, campaign="nvidia")
    assert envelope["timestamp"] == "2026-10-04T12:00:00Z"
    metrics = by_metric(envelope)
    # The write/trim directions moved no data, so they produce no rows.
    assert set(metrics) == {"read_bandwidth_mib_per_s", "read_iops", "read_clat_p99_ms"}
    assert metrics["read_bandwidth_mib_per_s"]["value"] == pytest.approx(
        fio_sample["jobs"][0]["read"]["bw_bytes"] / 2**20
    )
    assert metrics["read_iops"]["value"] == pytest.approx(4166.67, abs=0.01)
    assert metrics["read_clat_p99_ms"]["value"] == pytest.approx(3.195904)
    row = metrics["read_iops"]
    assert row["name"] == "iops-test-job"
    assert row["duration_seconds"] == 120.0
    assert row["tags"] == {
        "campaign": "nvidia",
        "fio_version": "fio-3.43",
        "rw": "randread",
        "bs": "4k",
        "iodepth": "256",
        "numjobs": "4",
        "ioengine": "libaio",
    }


def test_fio_never_publishes_the_device_path(fio_sample: dict) -> None:
    assert "/dev/nvme0n1" in str(fio_sample)  # the fixture does carry it
    assert "/dev/nvme0n1" not in str(convert("fio", fio_sample))


def test_fio_aliases_and_write_direction() -> None:
    raw = {
        "fio version": "fio-3.43",
        "jobs": [
            {
                "jobname": "seq_read_bw",
                "job options": {"readwrite": "write", "blocksize": "1M", "directory": "/scratch"},
                "read": {"io_bytes": 0},
                "write": {"io_bytes": 1 << 30, "bw_bytes": 1 << 29, "iops": 512.0, "runtime": 2000},
            }
        ],
    }
    envelope = convert("fio", raw)
    assert set(by_metric(envelope)) == {"write_bandwidth_mib_per_s", "write_iops"}  # no percentiles in input
    tags = envelope["results"][0]["tags"]
    assert (tags["rw"], tags["bs"]) == ("write", "1M")
    assert "directory" not in tags


# --- nvbandwidth ---------------------------------------------------------------


def test_nvbandwidth_bandwidth_latency_and_waived(nvbandwidth_sample: dict) -> None:
    envelope = convert("nvbandwidth", nvbandwidth_sample)
    rows = {r["name"]: r for r in envelope["results"]}
    assert set(rows) == {
        "host_to_device_memcpy_ce",
        "device_to_host_memcpy_ce",
        "device_local_copy",
        "device_to_device_latency_sm",
    }
    assert (rows["host_to_device_memcpy_ce"]["metric"], rows["host_to_device_memcpy_ce"]["unit"]) == (
        "bandwidth",
        "GB/s",
    )
    assert rows["host_to_device_memcpy_ce"]["value"] == pytest.approx(25.3)
    latency = rows["device_to_device_latency_sm"]
    assert (latency["metric"], latency["unit"]) == ("latency", "ns")
    assert latency["value"] == pytest.approx(1480.0)  # mean of the two measured directions
    assert rows["device_local_copy"]["tags"]["nvbandwidth_version"] == "v0.10.0"
    # A waived test case is recorded, not silently dropped and not scored.
    assert envelope["errors"] == ["device_to_device_memcpy_read_ce: Waived"]


def test_nvbandwidth_averages_measured_pairs_and_skips_na() -> None:
    raw = {
        "nvbandwidth": {
            "version": "v0.10.0",
            "testcases": [
                {
                    "name": "device_to_device_memcpy_write_ce",
                    "status": "Passed",
                    "bandwidth_description": "memcpy CE GPU(row) <- GPU(column) bandwidth (GB/s)",
                    "bandwidth_matrix": [["N/A", "276.1"], ["276.3", "N/A"]],
                }
            ],
        }
    }
    [row] = convert("nvbandwidth", raw)["results"]
    assert row["value"] == pytest.approx(276.2)
    assert row["tags"]["pairs"] == "2"


def test_nvbandwidth_run_level_error_is_recorded() -> None:
    envelope = convert("nvbandwidth", {"nvbandwidth": {"error": "No devices found", "testcases": []}})
    assert envelope["results"] == []
    assert envelope["errors"] == ["No devices found"]


# --- mbw -----------------------------------------------------------------------


def test_mbw_reads_the_average_line_per_method(mbw_sample: dict) -> None:
    envelope = convert("mbw", mbw_sample, campaign="nvidia")
    rows = {r["name"]: r for r in envelope["results"]}
    assert set(rows) == {"memcpy", "dumb", "mcblock"}
    memcpy = rows["memcpy"]
    assert memcpy["metric"] == "copy_bandwidth_mib_per_s"
    assert memcpy["unit"] == "MiB/s"
    assert memcpy["value"] == pytest.approx((6632.124 + 6610.002 + 6625.488) / 3, abs=0.001)
    assert memcpy["tags"] == {
        "campaign": "nvidia",
        "array_mib": "1024",
        "runs": "3",
        "mean_elapsed_s": "0.15462",
    }
    assert envelope["errors"] == []


def test_mbw_without_average_lines_records_why() -> None:
    per_run_only = "Getting down to business... Doing 1 runs per test.\n0\tMethod: MEMCPY\tElapsed: 0.15\tMiB: 1024.0\tCopy: 6600.0 MiB/s\n"
    envelope = convert("mbw", {"output": per_run_only})
    assert envelope["results"] == []
    assert "-a" in envelope["errors"][0]
    # An envelope with no rows must not reach the dataset.
    with pytest.raises(PublishError, match="No result rows"):
        rows_to_parquet(envelope_to_rows(envelope))


# --- gpu-burn --------------------------------------------------------------------


def test_gpu_burn_median_gflops_temp_and_verdict(gpu_burn_sample: dict) -> None:
    envelope = convert("gpu-burn", gpu_burn_sample, mode="fp32")
    metrics = by_metric(envelope)
    # The 0 Gflop/s first interval is "no report yet", so it is not a sample.
    assert metrics["gflops"]["value"] == 5200.0
    assert metrics["gflops"]["unit"] == "Gflop/s"
    assert metrics["gflops"]["tags"] == {"mode": "fp32", "samples": "5"}
    assert metrics["temp_max_c"]["value"] == 66.0
    assert metrics["burn_ok"]["value"] == 1.0
    assert {r["name"] for r in envelope["results"]} == {"gpu0"}
    assert envelope["errors"] == []


def test_gpu_burn_two_gpus_keep_columns_aligned_when_a_temp_is_missing() -> None:
    output = (
        "\r50.0%  proc'd: 100 (400 Gflop/s) - 90 (350 Gflop/s)   errors: 0 - 3  (WARNING!)   temps: -- - 61 C \r"
        "100.0%  proc'd: 200 (420 Gflop/s) - 180 (360 Gflop/s)   errors: 0 - 3  (WARNING!)   temps: -- - 63 C "
        "\ndone\n\nTested 2 GPUs:\n\tGPU 0: OK\n\tGPU 1: FAULTY\n"
    )
    rows = {(r["name"], r["metric"]): r["value"] for r in convert("gpu-burn", {"output": output})["results"]}
    assert rows[("gpu0", "gflops")] == 410.0
    assert rows[("gpu1", "gflops")] == 355.0
    assert ("gpu0", "temp_max_c") not in rows  # "--" is no reading, not a zero
    assert rows[("gpu1", "temp_max_c")] == 63.0
    assert (rows[("gpu0", "burn_ok")], rows[("gpu1", "burn_ok")]) == (1.0, 0.0)


def test_gpu_burn_interrupted_run_has_no_verdict() -> None:
    envelope = convert("gpu-burn", {"output": "\r10.0%  proc'd: 5 (100 Gflop/s)   errors: 0   temps: 40 C "})
    assert "burn_ok" not in by_metric(envelope)
    assert "interrupted" in envelope["errors"][0]


# --- CLI: .txt/.log input --------------------------------------------------------


def test_cli_wraps_text_output_for_mbw(
    tmp_path: Path, mbw_sample: dict, capsys: pytest.CaptureFixture
) -> None:
    source = tmp_path / "mbw.txt"
    source.write_text(mbw_sample["output"])
    argv = [str(source), "--kind", "mbw", "--suite", "mbw", "--model", "hardware-baseline"]
    assert main([*argv, "--git-sha", "deadbeef", "--dry-run"]) == 0
    assert "planned" in capsys.readouterr().err


def test_cli_keeps_gpu_burn_carriage_returns(tmp_path: Path, gpu_burn_sample: dict) -> None:
    source = tmp_path / "gpu-burn.log"
    source.write_bytes(gpu_burn_sample["output"].encode())  # bytes: no newline translation
    argv = [str(source), "--kind", "gpu-burn", "--suite", "gpu-burn", "--model", "hardware-baseline"]
    assert main([*argv, "--git-sha", "deadbeef", "--dry-run"]) == 0


def test_cli_json_input_is_unchanged(tmp_path: Path, nvbandwidth_sample: dict) -> None:
    import json

    source = tmp_path / "nvbandwidth.json"
    source.write_text(json.dumps(nvbandwidth_sample))
    argv = [str(source), "--kind", "nvbandwidth", "--suite", "nvbandwidth", "--model", "hardware-baseline"]
    assert main([*argv, "--git-sha", "deadbeef", "--dry-run"]) == 0
