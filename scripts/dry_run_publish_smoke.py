#!/usr/bin/env python3
"""Exercise the installed publisher against a local synthetic Hub registry."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from huggingface_hub import HfApi

from mlx_benchmarks.cli import main

HERE = Path(__file__).resolve().parents[1]
SAMPLE = HERE / "tests/fixtures/lm_eval_results_sample.json"
TIMESTAMP = "2026-04-24T18:30:00Z"
MODEL_SHA = "a" * 40
DATASET_SHA = "b" * 40


def main_smoke() -> int:
    metadata = {
        "dataset_id": "owner/benchmark",
        "dataset_task_id": "default",
        "dataset_revision": DATASET_SHA,
        "evaluation_framework": "inspect-ai",
        "config": "default",
        "split": "test",
        "engine": "MLX",
        "engine_version": "1.0",
        "profile": "synthetic",
        "hardware": {"chip": "Synthetic CPU"},
        "power_limit_w": 100,
        "concurrency": 1,
        "ctx_per_slot": 4096,
        "kv_cache_dtype": "float16",
        "thinking": False,
        "temperature": 0,
        "max_tokens": 128,
        "prompt_chars": 100,
        "prompt_tokens": 25,
        "system_prompt_chars": 20,
        "system_prompt_tokens": 5,
        "runner": "mlx-benchmarks-test",
        "harness": "inspect-ai-test",
        "router_key_alias": "local-test",
        "run_id": "synthetic-run",
        "start_utc": TIMESTAMP,
        "end_utc": "2026-04-24T18:32:03Z",
        "dimension_null_reasons": {"quant": "not_applicable"},
    }

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        raw = json.loads(SAMPLE.read_text())
        raw["model_revision"] = MODEL_SHA
        raw_path = directory / "results.json"
        raw_path.write_text(json.dumps(raw))
        metadata_path = directory / "published-metadata.json"
        metadata_path.write_text(json.dumps(metadata))
        eval_yaml = directory / "eval.yaml"
        eval_yaml.write_text(
            "name: Synthetic benchmark\n"
            "description: Publisher smoke fixture.\n"
            "evaluation_framework: inspect-ai\n"
            "tasks:\n"
            "  - id: default\n"
            "    config: default\n"
            "    split: test\n"
        )

        def model_info(_self: HfApi, repo_id: str, revision: str | None = None, **_kwargs: object) -> object:
            return SimpleNamespace(
                id=repo_id,
                sha=revision,
                pipeline_tag="text-generation",
                card_data={
                    "pipeline_tag": "text-generation",
                    "license": "apache-2.0",
                    "base_model": None,
                    "base_model_relation": None,
                },
                library_name="transformers",
                safetensors={"total": 1000, "parameters": {"F16": 1000}},
                gguf=None,
                config={
                    "architectures": ["SyntheticForCausalLM"],
                    "model_type": "synthetic",
                    "max_position_embeddings": 4096,
                },
                gated=False,
                tags=["text-generation"],
            )

        def dataset_info(
            _self: HfApi, repo_id: str, revision: str | None = None, **_kwargs: object
        ) -> object:
            assert repo_id == metadata["dataset_id"]
            return SimpleNamespace(sha=revision, tags=["benchmark"])

        with (
            patch.object(HfApi, "model_info", model_info),
            patch.object(HfApi, "dataset_info", dataset_info),
            patch.object(HfApi, "list_repo_files", return_value=["eval.yaml"]),
            patch.object(HfApi, "hf_hub_download", return_value=str(eval_yaml)),
            patch.object(HfApi, "create_commit", side_effect=AssertionError("dry run attempted a write")),
        ):
            return main(
                [
                    str(raw_path),
                    "--kind",
                    "lm-eval",
                    "--suite",
                    "reasoning",
                    "--timestamp",
                    TIMESTAMP,
                    "--published-metadata",
                    str(metadata_path),
                    "--dry-run",
                ]
            )


if __name__ == "__main__":
    raise SystemExit(main_smoke())
