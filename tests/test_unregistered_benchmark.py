from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from huggingface_hub import HfApi

import mlx_benchmarks.cli as cli
from mlx_benchmarks.published_results import DATASET_FIELDS, build_published_results


def test_cli_unregistered_flag_nulls_dataset_fields(
    tmp_path,
    lm_eval_sample: dict[str, Any],
    published_metadata: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(lm_eval_sample))
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(published_metadata))
    captured: dict[str, Any] = {}

    def capture_publish(envelope: dict[str, Any], **kwargs: Any) -> str:
        captured["metadata"] = kwargs["published_metadata"]
        return "planned.parquet"

    monkeypatch.setattr(cli, "publish", capture_publish)
    assert (
        cli.main(
            [
                str(results_path),
                "--kind",
                "lm-eval",
                "--suite",
                "reasoning",
                "--published-metadata",
                str(metadata_path),
                "--unregistered-benchmark",
                "--dry-run",
            ]
        )
        == 0
    )

    metadata = captured["metadata"]
    assert all(metadata[field] is None for field in DATASET_FIELDS)
    assert all(metadata["dimension_null_reasons"][field] == "not_applicable" for field in DATASET_FIELDS)
    assert metadata["engine"] == published_metadata["engine"]


def test_registered_dataset_still_requires_hf_benchmark_tags(
    valid_envelope: dict[str, Any],
    published_metadata: dict[str, Any],
    mock_hf_registry: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unregistered_dataset_info(self: Any, repo_id: str, revision: str | None = None, **kwargs: Any) -> Any:
        return SimpleNamespace(sha=revision, tags=[])

    monkeypatch.setattr(HfApi, "dataset_info", unregistered_dataset_info)
    with pytest.raises(ValueError, match="is not tagged as an HF Benchmark"):
        build_published_results(valid_envelope, metadata=published_metadata, api=HfApi())
