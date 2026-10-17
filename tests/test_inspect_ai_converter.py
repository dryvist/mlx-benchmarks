from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from mlx_benchmarks.converters import get_converter
from mlx_benchmarks.converters.base import ConverterContext
from mlx_benchmarks.converters.inspect_ai import read_inspect_header
from mlx_benchmarks.envelope import validate_envelope
from mlx_benchmarks.system import detect_system


def _write_eval(path: Path, header: dict, samples: list[dict]) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("header.json", json.dumps(header))
        for index, sample in enumerate(samples):
            archive.writestr(f"samples/{index}_epoch_1.json", json.dumps(sample))


def _header() -> dict:
    return {
        "status": "success",
        "eval": {
            "task": "toy",
            "model": "test-model",
            "model_generate_config": {"max_tokens": 10, "reasoning_effort": "low"},
        },
        "results": {
            "total_samples": 3,
            "completed_samples": 3,
            "scores": [
                {
                    "name": "match",
                    "scorer": "match",
                    "scored_samples": 3,
                    "metrics": {"accuracy": {"name": "mean", "value": 0.5}},
                }
            ],
        },
    }


def test_inspect_eval_uses_per_sample_usage_and_score_ratio(tmp_path: Path) -> None:
    path = tmp_path / "toy.eval"
    samples = [
        {
            "model_usage": {"test-model": {"output_tokens": 2, "reasoning_tokens": 1}},
            "output": {"choices": [{"stop_reason": "stop"}]},
        },
        {
            "model_usage": {"test-model": {"output_tokens": 4, "reasoning_tokens": 0}},
            "output": {"choices": [{"stop_reason": "max_tokens"}]},
        },
        {
            "model_usage": {"test-model": {"output_tokens": 10, "reasoning_tokens": 3}},
            "output": {"choices": [{"stop_reason": "stop"}]},
        },
    ]
    _write_eval(path, _header(), samples)
    raw = read_inspect_header(path)
    converter = get_converter("inspect-ai")
    envelope = converter.build_envelope(
        raw,
        ConverterContext(
            suite="reasoning",
            model="test-model",
            git_sha="deadbeef",
            system=detect_system(),
            source_path=path,
        ),
    )
    validate_envelope(envelope)

    [row] = envelope["results"]
    assert row["name"] == "toy"
    assert row["metric"] == "match.accuracy"
    assert row["reasoning_arm"] == "low"
    assert row["output_tokens_mean"] == 16 / 3
    assert row["output_tokens_median"] == 4
    assert row["output_tokens_p90"] == 10
    assert row["reasoning_tokens_mean"] == 4 / 3
    assert row["tokens_per_correct"] == 16 / round(0.5 * 3)
    assert row["truncation_rate"] == 1 / 3


def test_inspect_eval_missing_usage_is_null_with_reasons(tmp_path: Path) -> None:
    path = tmp_path / "toy.eval"
    header = _header()
    header["eval"]["model_generate_config"] = {}
    header["results"]["total_samples"] = 1
    header["results"]["completed_samples"] = 1
    header["results"]["scores"][0]["scored_samples"] = 1
    _write_eval(path, header, [{}])

    envelope = get_converter("inspect-ai").build_envelope(
        read_inspect_header(path),
        ConverterContext(suite="reasoning", model="test-model", git_sha="deadbeef", source_path=path),
    )

    [row] = envelope["results"]
    assert row["output_tokens_mean"] is None
    assert row["reasoning_tokens_mean"] is None
    assert row["tokens_per_correct"] is None
    assert row["truncation_rate"] is None
    assert row["tags"]["output_tokens_unavailable_reason"] == "sample_token_usage_missing"
    assert row["tags"]["reasoning_tokens_unavailable_reason"] == "reasoning_usage_missing"
    assert row["tags"]["truncation_rate_unavailable_reason"] == "max_tokens_and_finish_reason_missing"


def test_inspect_cli_reads_eval_header_and_sample_usage(tmp_path: Path, monkeypatch) -> None:
    from mlx_benchmarks import cli

    path = tmp_path / "toy.eval"
    _write_eval(path, _header(), [{"model_usage": {"test-model": {"output_tokens": 2}}}])
    published: dict = {}

    def capture(envelope: dict, **_kwargs: object) -> str:
        published.update(envelope)
        return "dry-run"

    monkeypatch.setattr(cli, "publish", capture)
    assert (
        cli.main(
            [str(path), "--kind", "inspect-ai", "--suite", "reasoning", "--git-sha", "deadbeef", "--dry-run"]
        )
        == 0
    )
    assert published["model"] == "test-model"
