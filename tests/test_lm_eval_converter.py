"""End-to-end: lm-eval sample -> envelope -> passes schema validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mlx_benchmarks.converters import get_converter
from mlx_benchmarks.converters.base import ConverterContext
from mlx_benchmarks.converters.lm_eval import LmEvalConverter
from mlx_benchmarks.envelope import validate_envelope
from mlx_benchmarks.system import detect_system


def test_lm_eval_round_trip(lm_eval_sample: dict) -> None:
    converter = get_converter("lm-eval")
    ctx = ConverterContext(
        suite="reasoning",
        model="mlx-community/Qwen3.5-9B-MLX-4bit",
        git_sha="deadbeef",
        system=detect_system(),
    )
    envelope = converter.build_envelope(lm_eval_sample, ctx)

    # Must validate against schema
    validate_envelope(envelope)

    # Results were built from the sample
    assert envelope["suite"] == "reasoning"
    assert envelope["model"] == "mlx-community/Qwen3.5-9B-MLX-4bit"
    results = envelope["results"]
    assert len(results) == 2, "expected one entry per non-stderr metric"
    metric_names = {r["metric"] for r in results}
    assert metric_names == {"exact_match_flexible", "exact_match_strict"}

    # Gen kwargs propagated
    assert envelope.get("gen_kwargs", {}).get("max_gen_toks") == 4096

    # Tags preserved original lm-eval key
    lm_keys = {r["tags"]["lm_eval_key"] for r in results}
    assert lm_keys == {"exact_match,flexible-extract", "exact_match,strict-match"}


def test_unknown_converter_kind_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unknown converter kind"):
        get_converter("no-such-tool")


@dataclass(slots=True)
class _FakeEncoding:
    ids: list[int]


class _WhitespaceTokenizer:
    """Stand-in for an HF tokenizer in tests — splits on whitespace.

    Avoids any HF network access or filesystem cache dependency. One "token"
    per whitespace-separated word is sufficient to verify the converter's
    accounting math.
    """

    def encode(self, text: str) -> _FakeEncoding:
        return _FakeEncoding(ids=text.split())  # type: ignore[arg-type]


_FIXTURES = Path(__file__).parent / "fixtures"


def test_lm_eval_populates_tokens_per_second(tmp_path: Path) -> None:
    """When source_path points at a results_*.json with a sibling samples_*.jsonl
    and a working tokenizer, the converter writes aggregate tok/s fields to the
    result. Numbers are computed against the fake whitespace tokenizer:

      Per sample (3 total):
        prompt  "What is 2 + 2?"  -> 5 tokens (split on whitespace)
        resp    "The answer is 4." -> 4 tokens
      Totals: prompt=15, response=12, duration=100s
      Expected: prompt_tok_s=0.15, decode_tok_s=0.12, total_tok_s=0.27
    """
    source = _FIXTURES / "lm_eval_with_samples" / "results_2026-05-13T00-00-00.000000.json"
    raw = json.loads(source.read_text())

    converter = LmEvalConverter(tokenizer_loader=lambda _model: _WhitespaceTokenizer())
    ctx = ConverterContext(
        suite="reasoning",
        model="mlx-community/Qwen3.5-9B-MLX-4bit",
        git_sha="deadbeef",
        system=detect_system(),
        source_path=source,
    )

    envelope = converter.build_envelope(raw, ctx)
    validate_envelope(envelope)
    [result] = envelope["results"]
    assert result["duration_seconds"] == 100.0
    assert result["prompt_tokens_per_second"] == 15 / 100.0
    assert result["decode_tokens_per_second"] == 12 / 100.0
    assert result["total_tokens_per_second"] == 27 / 100.0
    assert result["output_tokens_mean"] == 4
    assert result["output_tokens_median"] == 4
    assert result["output_tokens_p90"] == 4
    assert result["tokens_per_correct"] == 6
    assert result["truncation_rate"] == 0
    assert result["reasoning_tokens_mean"] is None
    assert result["tags"]["reasoning_tokens_unavailable_reason"] == "reasoning_usage_missing"
    assert result["reasoning_arm"] == "unstated"


def test_lm_eval_skips_throughput_without_source_path() -> None:
    """Library callers that don't supply ``source_path`` still get a valid
    envelope — tok/s fields are simply absent."""
    source = _FIXTURES / "lm_eval_with_samples" / "results_2026-05-13T00-00-00.000000.json"
    raw = json.loads(source.read_text())

    converter = LmEvalConverter(tokenizer_loader=lambda _model: _WhitespaceTokenizer())
    ctx = ConverterContext(
        suite="reasoning",
        model="mlx-community/Qwen3.5-9B-MLX-4bit",
        git_sha="deadbeef",
        system=detect_system(),
        # source_path intentionally omitted
    )
    envelope = converter.build_envelope(raw, ctx)
    validate_envelope(envelope)
    [result] = envelope["results"]
    assert "duration_seconds" in result
    assert "decode_tokens_per_second" not in result
    assert result["output_tokens_mean"] is None
    assert result["tokens_per_correct"] is None
    assert result["reasoning_tokens_mean"] is None
    assert result["tags"]["output_tokens_unavailable_reason"] == "sample_logs_missing"
    assert result["tags"]["reasoning_tokens_unavailable_reason"] == "sample_logs_missing"


def test_lm_eval_derives_usage_fields_from_per_sample_usage(tmp_path: Path) -> None:
    source = tmp_path / "results_2026-05-13T00-00-00.000000.json"
    source.write_text(
        json.dumps(
            {
                "results": {"toy": {"acc,none": 0.5}},
                "config": {"gen_kwargs": {"max_gen_toks": 10}},
                "model_name": "test-model",
                "total_evaluation_time_seconds_per_task": {"toy": 1.0},
            }
        )
    )
    samples = tmp_path / "samples_toy_2026-05-13T00-00-00.000000.jsonl"
    samples.write_text(
        "\n".join(
            json.dumps(
                {
                    "model_output": {
                        "usage": {
                            "output_tokens": output_tokens,
                            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
                        },
                        "choices": [{"finish_reason": finish_reason}],
                    }
                }
            )
            for output_tokens, reasoning_tokens, finish_reason in (
                (2, 1, "stop"),
                (4, 0, "length"),
                (10, 3, "stop"),
            )
        )
        + "\n"
    )
    converter = LmEvalConverter(tokenizer_loader=lambda _model: None)
    envelope = converter.build_envelope(
        json.loads(source.read_text()),
        ConverterContext(suite="reasoning", model="test-model", git_sha="deadbeef", source_path=source),
    )

    [result] = envelope["results"]
    assert result["output_tokens_mean"] == 16 / 3
    assert result["output_tokens_median"] == 4
    assert result["output_tokens_p90"] == 10
    assert result["reasoning_tokens_mean"] == 4 / 3
    assert result["tokens_per_correct"] == 16 / round(0.5 * 3)
    assert result["truncation_rate"] == 1 / 3


def test_lm_eval_tokenizes_samples_once_per_task_across_metrics(tmp_path: Path) -> None:
    """A task emitting multiple metric rows must tokenize the samples file
    only once. Tokenizing per metric would re-read large JSONL files several
    times for the same data and inflate publish time on real lm-eval runs."""
    source = tmp_path / "results_2026-05-13T00-00-00.000000.json"
    source.write_text(
        json.dumps(
            {
                "results": {
                    "gsm8k_cot_zeroshot": {
                        "alias": "gsm8k_cot_zeroshot",
                        "exact_match,flexible-extract": 0.6,
                        "exact_match_stderr,flexible-extract": 0.155,
                        "exact_match,strict-match": 0.5,
                        "exact_match_stderr,strict-match": 0.158,
                        "acc,none": 0.55,
                    }
                },
                "config": {
                    "model": "local-chat-completions",
                    "model_args": {"model": "test-model"},
                    "gen_kwargs": {"max_gen_toks": 2},
                },
                "model_name": "test-model",
                "date": 1778686200.0,
                "total_evaluation_time_seconds": "100.0",
                "total_evaluation_time_seconds_per_task": {"gsm8k_cot_zeroshot": 100.0},
            }
        )
    )
    samples = tmp_path / "samples_gsm8k_cot_zeroshot_2026-05-13T00-00-00.000000.jsonl"
    samples.write_text(
        '{"doc_id": 0, "arguments": [["q1", {}]], "filtered_resps": ["a1"]}\n'
        '{"doc_id": 1, "arguments": [["q2", {}]], "filtered_resps": ["a2"]}\n'
    )

    encode_calls = 0

    class _CountingTokenizer:
        def encode(self, text: str) -> _FakeEncoding:
            nonlocal encode_calls
            encode_calls += 1
            return _FakeEncoding(ids=text.split())  # type: ignore[arg-type]

    counting_tokenizer = _CountingTokenizer()
    converter = LmEvalConverter(tokenizer_loader=lambda _model: counting_tokenizer)
    ctx = ConverterContext(
        suite="reasoning",
        model="test-model",
        git_sha="deadbeef",
        system=detect_system(),
        source_path=source,
    )

    envelope = converter.build_envelope(json.loads(source.read_text()), ctx)
    results = envelope["results"]

    # Three metric rows on the same task (acc + two exact_match variants)
    assert len(results) == 3
    # Each sample has one prompt + one response = 2 encode() calls per sample.
    # 2 samples * 2 = 4 encodes -- and we must do that count exactly once,
    # not once per metric row.
    assert encode_calls == 4, f"expected 4 encode calls (one pass), got {encode_calls}"
    # All three rows must carry the same computed throughput value
    decode_values = {r["decode_tokens_per_second"] for r in results}
    assert len(decode_values) == 1
    assert {r["output_tokens_mean"] for r in results} == {1}
    assert {r["tokens_per_correct"] for r in results} == {2}
    assert {r["truncation_rate"] for r in results} == {0}


def test_lm_eval_skips_throughput_when_tokenizer_unavailable() -> None:
    """When the tokenizer loader returns None (HF offline, missing repo, etc.)
    the converter degrades gracefully — duration still set, tok/s fields not."""
    source = _FIXTURES / "lm_eval_with_samples" / "results_2026-05-13T00-00-00.000000.json"
    raw = json.loads(source.read_text())

    converter = LmEvalConverter(tokenizer_loader=lambda _model: None)
    ctx = ConverterContext(
        suite="reasoning",
        model="mlx-community/Qwen3.5-9B-MLX-4bit",
        git_sha="deadbeef",
        system=detect_system(),
        source_path=source,
    )
    envelope = converter.build_envelope(raw, ctx)
    [result] = envelope["results"]
    assert result["duration_seconds"] == 100.0
    assert "decode_tokens_per_second" not in result
    assert result["output_tokens_mean"] is None
    assert result["output_tokens_median"] is None
    assert result["output_tokens_p90"] is None
    assert result["tokens_per_correct"] is None
    assert result["tags"]["output_tokens_unavailable_reason"] == "sample_token_count_unavailable"
