"""Promptfoo JSON export -> envelope v1 score and assertion summaries."""

from __future__ import annotations

import datetime
import math
from collections.abc import Mapping
from typing import Any

from mlx_benchmarks.converters.base import ConverterContext, simple_envelope
from mlx_benchmarks.converters.token_stats import summarize_token_counts
from mlx_benchmarks.envelope import Envelope, Result


class PromptfooConverter:
    kind = "promptfoo"

    def build_envelope(self, raw: dict[str, Any], ctx: ConverterContext) -> Envelope:
        container = raw.get("results")
        if isinstance(container, Mapping):
            outputs = container.get("outputs") or container.get("results") or []
        else:
            outputs = raw.get("outputs") or []
        if not isinstance(outputs, list):
            outputs = []

        timestamp = raw.get("timestamp")
        if not isinstance(timestamp, str) and isinstance(container, Mapping):
            timestamp = container.get("timestamp")
        if not isinstance(timestamp, str) or not timestamp:
            timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        results: list[Result] = []
        scores: list[float] = []
        scored_outputs: list[Mapping[str, Any]] = []
        for index, output in enumerate(outputs):
            if not isinstance(output, Mapping):
                continue
            test_index = _index(output.get("testIdx"), index)
            prompt_index = output.get("promptIdx")
            base_tags = {**ctx.extra_tags, "promptfoo_test_idx": str(test_index)}
            if prompt_index is not None:
                base_tags["promptfoo_prompt_idx"] = str(prompt_index)

            score = _score(output)
            if score is not None:
                scores.append(score)
                scored_outputs.append(output)
                test_tags = dict(base_tags)
                if isinstance(output.get("success"), bool):
                    test_tags["success"] = str(output["success"]).lower()
                results.append(
                    {
                        "name": f"test_{test_index}",
                        "metric": "score",
                        "value": score,
                        "unit": "ratio",
                        "tags": test_tags,
                    }
                )
            results.extend(_assertion_rows(output, test_index, base_tags))

        if scores:
            tags = {**ctx.extra_tags, "test_count": str(len(scores))}
            aggregate: Result = {
                "name": "promptfoo_aggregate",
                "metric": "score",
                "value": sum(scores) / len(scores),
                "unit": "ratio",
                "tags": tags,
            }
            output_counts = [_output_tokens(output) for output in scored_outputs]
            if all(count is not None for count in output_counts):
                counts = [count for count in output_counts if count is not None]
                mean, median, p90 = summarize_token_counts(counts)
                aggregate["output_tokens_mean"] = mean
                aggregate["output_tokens_median"] = median
                aggregate["output_tokens_p90"] = float(p90)
                tags["output_tokens_method"] = "promptfoo_token_usage"
                correct_count = sum(output.get("success") is True for output in scored_outputs)
                if correct_count:
                    aggregate["tokens_per_correct"] = sum(counts) / correct_count
                else:
                    tags["tokens_per_correct_unavailable_reason"] = "zero_correct_items"
            else:
                tags["output_tokens_unavailable_reason"] = "sample_token_usage_missing"
                tags["tokens_per_correct_unavailable_reason"] = "output_tokens_unavailable"

            reasoning_counts = [_reasoning_tokens(output) for output in scored_outputs]
            if all(count is not None for count in reasoning_counts):
                aggregate["reasoning_tokens_mean"] = sum(
                    count for count in reasoning_counts if count is not None
                ) / len(reasoning_counts)
            else:
                tags["reasoning_tokens_unavailable_reason"] = "reasoning_usage_missing"

            finish_reasons = [_finish_reason(output) for output in scored_outputs]
            if all(reason is not None for reason in finish_reasons):
                aggregate["truncation_rate"] = sum(
                    reason in {"length", "max_tokens", "model_length"} for reason in finish_reasons
                ) / len(finish_reasons)
            else:
                tags["truncation_rate_unavailable_reason"] = "finish_reason_missing"

            results.append(aggregate)
        return simple_envelope(ctx, results, timestamp=timestamp)


def _index(value: Any, fallback: int) -> int | str:
    return value if isinstance(value, int | str) else fallback


def _score(output: Mapping[str, Any]) -> float | None:
    score = output.get("score")
    if isinstance(score, int | float) and not isinstance(score, bool) and math.isfinite(score):
        return float(score)
    success = output.get("success")
    return float(success) if isinstance(success, bool) else None


def _usage(output: Mapping[str, Any]) -> Mapping[str, Any] | None:
    usage = output.get("tokenUsage")
    if isinstance(usage, Mapping):
        return usage
    response = output.get("response")
    usage = response.get("tokenUsage") if isinstance(response, Mapping) else None
    return usage if isinstance(usage, Mapping) else None


def _output_tokens(output: Mapping[str, Any]) -> int | None:
    usage = _usage(output)
    value = usage.get("completion") if usage is not None else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _reasoning_tokens(output: Mapping[str, Any]) -> int | None:
    usage = _usage(output)
    details = usage.get("completionDetails") if usage is not None else None
    value = details.get("reasoning") if isinstance(details, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _finish_reason(output: Mapping[str, Any]) -> str | None:
    response = output.get("response")
    reason = response.get("finishReason") if isinstance(response, Mapping) else None
    return reason if isinstance(reason, str) else None


def _assertion_rows(output: Mapping[str, Any], test_index: int | str, tags: dict[str, str]) -> list[Result]:
    grading = output.get("gradingResult")
    components = grading.get("componentResults") if isinstance(grading, Mapping) else None
    if not isinstance(components, list):
        return []

    rows: list[Result] = []
    for index, component in enumerate(components):
        if not isinstance(component, Mapping):
            continue
        passed = component.get("pass")
        score = component.get("score")
        if isinstance(passed, bool):
            value, metric = float(passed), "assertion_pass"
        elif isinstance(score, int | float) and not isinstance(score, bool) and math.isfinite(score):
            value, metric = float(score), "assertion_score"
        else:
            continue
        rows.append(
            {
                "name": f"test_{test_index}_assertion_{index}",
                "metric": metric,
                "value": value,
                "unit": "ratio",
                "tags": {**tags, "promptfoo_assertion_idx": str(index)},
            }
        )
    return rows
