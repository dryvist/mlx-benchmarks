"""EvalScope aggregate report JSON -> envelope v1 score rows."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from mlx_benchmarks.converters.base import ConverterContext, simple_envelope
from mlx_benchmarks.converters.token_stats import (
    TokenizerLike,
    TokenizerLoader,
    api_output_tokens,
    api_reasoning_tokens,
    count_tokens,
    default_tokenizer_loader,
    finish_reason,
    summarize_token_counts,
)
from mlx_benchmarks.envelope import Envelope, Result

log = logging.getLogger(__name__)


class EvalScopeConverter:
    kind = "evalscope"

    def __init__(self, tokenizer_loader: TokenizerLoader | None = None) -> None:
        self._tokenizer_loader = tokenizer_loader or default_tokenizer_loader
        self._tokenizer_cache: dict[str, TokenizerLike | None] = {}

    def build_envelope(self, raw: dict[str, Any], ctx: ConverterContext) -> Envelope:
        execution = raw.get("execution_summary")
        if isinstance(execution, Mapping) and execution.get("incomplete") is True:
            raise ValueError("EvalScope report is incomplete")
        metrics = raw.get("metrics")
        if not isinstance(metrics, list):
            raise ValueError("EvalScope report contains no aggregate metrics")

        (
            sample_counts,
            sample_method,
            sample_reason,
            truncation_rate,
            truncation_reason,
            reasoning_tokens_mean,
            reasoning_reason,
        ) = self._sample_metrics(raw, ctx)
        rows: list[Result] = []
        for index, metric in enumerate(metrics):
            if not isinstance(metric, Mapping):
                continue
            score = metric.get("score")
            if not isinstance(score, int | float) or isinstance(score, bool) or not math.isfinite(score):
                continue
            identity = metric.get("identity")
            key = identity.get("key") if isinstance(identity, Mapping) else None
            name = metric.get("legacy_name") or key or f"metric_{index}"
            tags = {"sample_count": str(metric.get("num", ""))}
            row: Result = {
                "name": str(raw.get("dataset_name") or ctx.suite),
                "metric": str(name),
                "value": float(score),
                "unit": "ratio",
                "reasoning_arm": ctx.reasoning_effort or "unstated",
                "output_tokens_mean": None,
                "output_tokens_median": None,
                "output_tokens_p90": None,
                "tokens_per_correct": None,
                "truncation_rate": truncation_rate,
                "reasoning_tokens_mean": reasoning_tokens_mean,
            }
            if sample_counts is None:
                tags["output_tokens_unavailable_reason"] = sample_reason or "sample_logs_missing"
                tags["tokens_per_correct_unavailable_reason"] = "output_tokens_unavailable"
            else:
                mean, median, p90 = summarize_token_counts(sample_counts)
                total_tokens = sum(sample_counts)
                correct_count = _correct_count(score, metric.get("num"), len(sample_counts))
                row["output_tokens_mean"] = mean
                row["output_tokens_median"] = median
                row["output_tokens_p90"] = float(p90)
                if correct_count is not None and correct_count > 0:
                    row["tokens_per_correct"] = total_tokens / correct_count
                elif correct_count == 0:
                    tags["tokens_per_correct_unavailable_reason"] = "zero_correct_items"
                else:
                    tags["tokens_per_correct_unavailable_reason"] = "score_not_accuracy_ratio"
                tags["output_tokens_method"] = sample_method or "unknown"
            if truncation_rate is None:
                tags["truncation_rate_unavailable_reason"] = (
                    truncation_reason or "max_tokens_and_finish_reason_missing"
                )
            if reasoning_tokens_mean is None:
                tags["reasoning_tokens_unavailable_reason"] = reasoning_reason or "reasoning_usage_missing"
            row["tags"] = tags
            rows.append(row)
        if not rows:
            raise ValueError("EvalScope report contains no finite aggregate scores")
        return simple_envelope(ctx, rows)

    def _sample_metrics(
        self, raw: dict[str, Any], ctx: ConverterContext
    ) -> tuple[list[int] | None, str | None, str | None, float | None, str | None, float | None, str | None]:
        if ctx.source_path is None:
            return None, None, "sample_logs_missing", None, "sample_logs_missing", None, "sample_logs_missing"
        samples_path = _find_predictions_file(ctx.source_path)
        if samples_path is None:
            return None, None, "sample_logs_missing", None, "sample_logs_missing", None, "sample_logs_missing"

        samples: list[dict[str, Any]] = []
        try:
            with samples_path.open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    sample = json.loads(line)
                    if isinstance(sample, dict):
                        samples.append(sample)
        except (OSError, ValueError) as exc:
            log.warning("could not read EvalScope sample log %s: %s", samples_path, exc)
            return (
                None,
                None,
                "sample_log_unreadable",
                None,
                "sample_log_unreadable",
                None,
                "sample_log_unreadable",
            )
        if not samples:
            return None, None, "sample_logs_empty", None, "sample_logs_empty", None, "sample_logs_empty"

        reasoning_counts = [api_reasoning_tokens(sample) for sample in samples]
        reasoning_tokens_mean = (
            sum(count for count in reasoning_counts if count is not None) / len(reasoning_counts)
            if all(count is not None for count in reasoning_counts)
            else None
        )
        reasoning_reason = None if reasoning_tokens_mean is not None else "reasoning_usage_missing"

        max_tokens = _max_tokens(raw, ctx.source_path)
        counts: list[int] = []
        methods: set[str] = set()
        truncations: list[bool | None] = []
        for sample in samples:
            count = api_output_tokens(sample)
            if count is not None:
                methods.add("usage")
            else:
                tokenizer = self._get_tokenizer(ctx.model)
                response = _response_text(sample)
                if tokenizer is None or response is None:
                    return (
                        None,
                        None,
                        "sample_token_count_unavailable",
                        None,
                        "sample_token_count_unavailable",
                        reasoning_tokens_mean,
                        reasoning_reason,
                    )
                count = count_tokens(tokenizer, response)
                methods.add("re-tokenized")
            counts.append(count)

            reason = finish_reason(sample)
            if reason in {"length", "max_tokens", "model_length"}:
                truncations.append(True)
            elif reason is not None:
                truncations.append(False)
            elif max_tokens is not None:
                truncations.append(count >= max_tokens)
            else:
                truncations.append(None)

        if all(value is not None for value in truncations):
            truncation_rate: float | None = sum(value is True for value in truncations) / len(truncations)
            truncation_reason = None
        else:
            truncation_rate = None
            truncation_reason = "max_tokens_and_finish_reason_missing"
        method = next(iter(methods)) if len(methods) == 1 else "mixed"
        return (
            counts,
            method,
            None,
            truncation_rate,
            truncation_reason,
            reasoning_tokens_mean,
            reasoning_reason,
        )

    def _get_tokenizer(self, model: str) -> TokenizerLike | None:
        if model not in self._tokenizer_cache:
            self._tokenizer_cache[model] = self._tokenizer_loader(model)
        return self._tokenizer_cache[model]


def _find_predictions_file(report_path: Path) -> Path | None:
    run_dir = report_path.parent.parent.parent
    directory = run_dir / "predictions" / report_path.parent.name
    expected = directory / f"{report_path.stem}_default.jsonl"
    if expected.is_file():
        return expected
    matches = sorted(directory.glob(f"{report_path.stem}_*.jsonl"))
    return matches[0] if len(matches) == 1 else None


def _max_tokens(raw: dict[str, Any], report_path: Path) -> int | None:
    generation = raw.get("generation_config")
    config_path = report_path.parent.parent.parent / "configs" / "task_config.yaml"
    value = generation.get("max_tokens") if isinstance(generation, Mapping) else None
    if (not isinstance(value, int) or isinstance(value, bool) or value <= 0) and config_path.is_file():
        try:
            config = yaml.safe_load(config_path.read_text())
        except (OSError, yaml.YAMLError) as exc:
            log.warning("could not read EvalScope task config %s: %s", config_path, exc)
            config = None
        generation = config.get("generation_config") if isinstance(config, Mapping) else None
        value = generation.get("max_tokens") if isinstance(generation, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _response_text(sample: Mapping[str, Any]) -> str | None:
    model_output = sample.get("model_output")
    choices = model_output.get("choices") if isinstance(model_output, Mapping) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        choice = choices[0]
        message = choice.get("message")
        if isinstance(message, Mapping):
            parts = [_content_text(message.get(key)) for key in ("reasoning_content", "reasoning", "content")]
            return "\n".join(part for part in parts if part is not None)
        text = _content_text(choice.get("text"))
        if text is not None:
            return text
    messages = sample.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, Mapping) and message.get("role") == "assistant":
                return _content_text(message.get("content"))
    return None


def _content_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [
            part if isinstance(part, str) else part.get("text")
            for part in value
            if isinstance(part, (str, Mapping))
        ]
        text = "\n".join(part for part in parts if isinstance(part, str))
        return text or None
    return None


def _correct_count(score: float, metric_num: Any, sample_count: int) -> int | None:
    if not 0 <= score <= 1:
        return None
    denominator = (
        metric_num if isinstance(metric_num, int) and not isinstance(metric_num, bool) else sample_count
    )
    return round(score * denominator) if denominator > 0 else None
