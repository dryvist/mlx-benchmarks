from __future__ import annotations

import json
from typing import Any

import mlx_benchmarks.cli as cli


def test_cli_normalizes_engine_aliases(
    tmp_path,
    vllm_sample: dict[str, Any],
    published_metadata: dict[str, Any],
    monkeypatch,
) -> None:
    raw_path = tmp_path / "results.json"
    raw_path.write_text(json.dumps(vllm_sample))
    dimensions_path = tmp_path / "dimensions.json"
    dimensions_path.write_text(json.dumps({"campaign_dimensions": {"software": {"engine": "vllm"}}}))
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({**published_metadata, "engine": "mlx"}))
    captured: dict[str, Any] = {}

    def capture_publish(envelope: dict[str, Any], **kwargs: Any) -> str:
        captured["envelope"] = envelope
        captured["metadata"] = kwargs["published_metadata"]
        return "planned.parquet"

    monkeypatch.setattr(cli, "publish", capture_publish)
    monkeypatch.setattr(
        cli,
        "detect_system",
        lambda: {"os": "test", "chip": "test", "memory_gb": 1, "engine": {"name": "mlx"}},
    )
    assert (
        cli.main(
            [
                str(raw_path),
                "--kind",
                "vllm",
                "--suite",
                "throughput",
                "--model",
                "example/model",
                "--serving-stack",
                "vllm",
                "--campaign-dimensions",
                str(dimensions_path),
                "--published-metadata",
                str(metadata_path),
                "--dry-run",
            ]
        )
        == 0
    )

    envelope = captured["envelope"]
    assert envelope["campaign_dimensions"]["software"]["engine"] == "vLLM"
    assert envelope["system"]["engine"]["name"] == "MLX"
    assert envelope["serving"]["stack"] == "vLLM"
    assert captured["metadata"]["engine"] == "MLX"
    assert cli._normalize_engine_name("SGLang") == "SGLang"
