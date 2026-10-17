"""Inspect AI ``.eval`` log -> envelope v1 score rows."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from mlx_benchmarks.converters.base import ConverterContext, simple_envelope
from mlx_benchmarks.converters.token_stats import summarize_token_counts
from mlx_benchmarks.envelope import Envelope, Result

log = logging.getLogger(__name__)


class InspectAIConverter:
    kind = "inspect-ai"

    def build_envelope(self, raw: dict[str, Any], ctx: ConverterContext) -> Envelope:
        result_data = raw.get("results")
        results = result_data if isinstance(result_data, Mapping) else {}
        scores = results.get("scores")
        if not isinstance(scores, list):
            raise ValueError("Inspect eval log contains no aggregate scores")

        sample_metrics = _read_sample_metrics(ctx.source_path, raw)
        eval_header = raw.get("eval")
        eval_header = eval_header if isinstance(eval_header, Mapping) else {}
        task_name = str(eval_header.get("task") or ctx.suite)
        generation = eval_header.get("model_generate_config")
        generation = generation if isinstance(generation, Mapping) else {}
        reasoning_arm = ctx.reasoning_effort or generation.get("reasoning_effort") or "unstated"
        rows: list[Result] = []

        for score in scores:
            if not isinstance(score, Mapping):
                continue
            metrics = score.get("metrics")
            if not isinstance(metrics, Mapping):
                continue
            scorer = str(score.get("name") or score.get("scorer") or "score")
            sample_count = _sample_count(score, results, sample_metrics.sample_count)
            for metric_key, metric in metrics.items():
                value = metric.get("value") if isinstance(metric, Mapping) else metric
                if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
                    continue
                unit = "ratio" if 0 <= value <= 1 else "score"
                row: Result = {
                    "name": task_name,
                    "metric": f"{scorer}.{metric_key}",
                    "value": float(value),
                    "unit": unit,
                    "reasoning_arm": str(reasoning_arm),
                    "output_tokens_mean": None,
                    "output_tokens_median": None,
                    "output_tokens_p90": None,
                    "reasoning_tokens_mean": sample_metrics.reasoning_tokens_mean,
                    "tokens_per_correct": None,
                    "truncation_rate": sample_metrics.truncation_rate,
                }
                tags = {"sample_count": str(sample_count)}
                if sample_metrics.output_counts is None:
                    tags["output_tokens_unavailable_reason"] = sample_metrics.output_reason
                    tags["tokens_per_correct_unavailable_reason"] = "output_tokens_unavailable"
                else:
                    mean, median, p90 = summarize_token_counts(sample_metrics.output_counts)
                    row["output_tokens_mean"] = mean
                    row["output_tokens_median"] = median
                    row["output_tokens_p90"] = float(p90)
                    tags["output_tokens_method"] = "inspect_model_usage"
                    correct_count = round(float(value) * sample_count) if unit == "ratio" else 0
                    if sample_count <= 0:
                        tags["tokens_per_correct_unavailable_reason"] = "score_sample_count_missing"
                    elif correct_count > 0:
                        row["tokens_per_correct"] = sum(sample_metrics.output_counts) / correct_count
                    elif unit == "ratio":
                        tags["tokens_per_correct_unavailable_reason"] = "zero_correct_items"
                    else:
                        tags["tokens_per_correct_unavailable_reason"] = "score_not_accuracy_ratio"
                if sample_metrics.reasoning_tokens_mean is None:
                    tags["reasoning_tokens_unavailable_reason"] = sample_metrics.reasoning_reason
                if sample_metrics.truncation_rate is None:
                    tags["truncation_rate_unavailable_reason"] = sample_metrics.truncation_reason
                row["tags"] = tags
                rows.append(row)

        if not rows:
            raise ValueError("Inspect eval log contains no finite aggregate metrics")
        return simple_envelope(ctx, rows)


@dataclass(slots=True)
class _SampleMetrics:
    sample_count: int = 0
    output_counts: list[int] | None = None
    output_reason: str = "sample_logs_missing"
    reasoning_tokens_mean: float | None = None
    reasoning_reason: str = "sample_logs_missing"
    truncation_rate: float | None = None
    truncation_reason: str = "sample_logs_missing"


def read_inspect_header(path: Path) -> dict[str, Any]:
    """Read an Inspect evaluation archive, using Inspect for unsupported ZIP compression."""
    try:
        with ZipFile(path) as archive:
            header = json.loads(archive.read("header.json"))
    except NotImplementedError:
        header = _read_eval_with_inspect(path)
    except (OSError, BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("invalid Inspect .eval log or missing header.json") from exc
    if not isinstance(header, dict):
        raise ValueError("Inspect .eval header must be a JSON object")
    return header


def _read_eval_with_inspect(path: Path) -> dict[str, Any]:
    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:
        raise ValueError("install the 'inspect' extra to read this Inspect archive") from exc
    return read_eval_log(str(path)).model_dump(mode="json")


def _read_sample_metrics(path: Path | None, header: Mapping[str, Any]) -> _SampleMetrics:
    metrics = _SampleMetrics()
    try:
        samples = header.get("samples")
        if not isinstance(samples, list):
            if path is None:
                return metrics
            with ZipFile(path) as archive:
                sample_names = sorted(
                    name
                    for name in archive.namelist()
                    if name.startswith("samples/") and name.endswith(".json")
                )
                samples = [json.loads(archive.read(name)) for name in sample_names]
        if not samples:
            return metrics

        output_counts: list[int] = []
        reasoning_counts: list[int] = []
        truncations: list[bool | None] = []
        output_missing = False
        reasoning_missing = False
        max_tokens = _max_tokens(header)
        for sample in samples:
            if not isinstance(sample, Mapping):
                raise ValueError("sample entry must be an object")
            output_count, reasoning_count = _model_usage_counts(sample)
            if output_count is None:
                output_missing = True
            else:
                output_counts.append(output_count)
            if reasoning_count is None:
                reasoning_missing = True
            else:
                reasoning_counts.append(reasoning_count)
            truncations.append(_truncated(sample, output_count, max_tokens))
    except (OSError, BadZipFile, KeyError, json.JSONDecodeError, ValueError) as exc:
        log.warning("could not read Inspect sample usage: %s", exc)
        metrics.output_reason = "sample_log_unreadable"
        metrics.reasoning_reason = "sample_log_unreadable"
        metrics.truncation_reason = "sample_log_unreadable"
        return metrics

    metrics.sample_count = len(samples)
    if not output_missing and len(output_counts) == metrics.sample_count:
        metrics.output_counts = output_counts
        metrics.output_reason = ""
    else:
        metrics.output_reason = "sample_token_usage_missing"
    if not reasoning_missing and len(reasoning_counts) == metrics.sample_count:
        metrics.reasoning_tokens_mean = sum(reasoning_counts) / metrics.sample_count
        metrics.reasoning_reason = ""
    else:
        metrics.reasoning_reason = "reasoning_usage_missing"
    if truncations and all(value is not None for value in truncations):
        metrics.truncation_rate = sum(value is True for value in truncations) / len(truncations)
        metrics.truncation_reason = ""
    else:
        metrics.truncation_reason = "max_tokens_and_finish_reason_missing"
    return metrics


def _model_usage_counts(sample: Mapping[str, Any]) -> tuple[int | None, int | None]:
    usage = sample.get("model_usage")
    if not isinstance(usage, Mapping) or not usage:
        return None, None
    output_total = 0
    reasoning_total = 0
    reasoning_complete = True
    for model_usage in usage.values():
        if not isinstance(model_usage, Mapping):
            return None, None
        output = _token_count(model_usage.get("output_tokens"))
        if output is None:
            return None, None
        output_total += output
        reasoning = _token_count(model_usage.get("reasoning_tokens"))
        if reasoning is None:
            reasoning_complete = False
        else:
            reasoning_total += reasoning
    return output_total, reasoning_total if reasoning_complete else None


def _token_count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _max_tokens(header: Mapping[str, Any]) -> int | None:
    eval_header = header.get("eval")
    generation = eval_header.get("model_generate_config") if isinstance(eval_header, Mapping) else None
    plan = header.get("plan")
    config = plan.get("config") if isinstance(plan, Mapping) else None
    for source in (generation, config):
        value = source.get("max_tokens") if isinstance(source, Mapping) else None
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _truncated(sample: Mapping[str, Any], output_tokens: int | None, max_tokens: int | None) -> bool | None:
    output = sample.get("output")
    choices = output.get("choices") if isinstance(output, Mapping) else None
    reasons = (
        [
            choice.get("stop_reason") or choice.get("finish_reason")
            for choice in choices
            if isinstance(choice, Mapping)
        ]
        if isinstance(choices, list)
        else []
    )
    if isinstance(output, Mapping) and isinstance(output.get("stop_reason"), str):
        reasons.append(output["stop_reason"])
    if any(reason in {"length", "max_tokens", "model_length"} for reason in reasons):
        return True
    if reasons and all(isinstance(reason, str) for reason in reasons):
        return False
    if max_tokens is not None and output_tokens is not None:
        return output_tokens >= max_tokens
    return None


def _sample_count(score: Mapping[str, Any], results: Mapping[str, Any], fallback: int) -> int:
    for value in (
        score.get("scored_samples"),
        results.get("completed_samples"),
        results.get("total_samples"),
        fallback,
    ):
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return 0
