"""lm-eval ``results.json`` -> envelope v1 converter."""

from __future__ import annotations

import datetime
import logging
import re
from collections.abc import Iterator
from typing import Any

from mlx_benchmarks.converters.base import ConverterContext, apply_optional_fields
from mlx_benchmarks.converters.lm_eval_samples import SampleMetrics, sample_metrics_for_task
from mlx_benchmarks.converters.token_stats import (
    TokenizerLike,
    TokenizerLoader,
    default_tokenizer_loader,
    summarize_token_counts,
)
from mlx_benchmarks.envelope import Envelope, GenKwargs, Result, System

log = logging.getLogger(__name__)

# Friendly display names for common lm-eval metric keys. Unknown keys fall
# through via a deterministic normalization.
_METRIC_MAP = {
    "exact_match,flexible-extract": "exact_match_flexible",
    "exact_match,strict-match": "exact_match_strict",
    "acc,none": "accuracy",
    "acc_norm,none": "accuracy_normalized",
    "pass@1,none": "pass_at_1",
}


class LmEvalConverter:
    kind = "lm-eval"

    def __init__(self, tokenizer_loader: TokenizerLoader | None = None) -> None:
        self._tokenizer_loader = tokenizer_loader or default_tokenizer_loader
        self._tokenizer_cache: dict[str, TokenizerLike | None] = {}

    def build_envelope(self, raw: dict[str, Any], ctx: ConverterContext) -> Envelope:
        timestamp = ctx.timestamp_override or _extract_timestamp(raw)
        system: System = ctx.system or {}  # type: ignore[assignment]

        envelope: Envelope = {
            "schema_version": "1",
            "timestamp": timestamp,
            "git_sha": ctx.git_sha,
            "trigger": ctx.trigger,
            "suite": ctx.suite,
            "model": ctx.model,
            "system": system,
            "results": list(self._iter_results(raw, ctx)),
            "errors": [],
        }

        gen_kwargs = _extract_gen_kwargs(raw)
        if gen_kwargs:
            envelope["gen_kwargs"] = gen_kwargs

        cfg = raw.get("config") or {}
        seed = cfg.get("seed")
        if isinstance(seed, int):
            envelope["seed"] = seed

        model_args = cfg.get("model_args") or {}
        revision = raw.get("model_source") or model_args.get("revision")
        if revision:
            envelope["model_revision"] = str(revision)

        return apply_optional_fields(envelope, ctx)

    def _iter_results(self, raw: dict[str, Any], ctx: ConverterContext) -> Iterator[Result]:
        raw_results: dict[str, dict[str, Any]] = raw.get("results") or {}
        task_durations = raw.get("total_evaluation_time_seconds_per_task") or {}
        cfg = raw.get("config") or {}
        cfg_gen_kwargs = cfg.get("gen_kwargs") or {}
        task_sample_cache: dict[str, SampleMetrics] = {}

        for task_name, task_metrics in raw_results.items():
            duration = task_durations.get(task_name)
            duration_s = float(duration) if isinstance(duration, int | float) else None
            max_gen_toks = cfg_gen_kwargs.get("max_gen_toks")
            sample_metrics = sample_metrics_for_task(
                task_name,
                max_gen_toks,
                duration_s,
                ctx,
                self._tokenizer_loader,
                self._tokenizer_cache,
                task_sample_cache,
            )
            for metric_key, metric_val in task_metrics.items():
                if "stderr" in metric_key or metric_key == "alias":
                    continue
                if not isinstance(metric_val, int | float):
                    log.debug("skip non-numeric metric %s=%r for task %s", metric_key, metric_val, task_name)
                    continue

                display_metric = _METRIC_MAP.get(metric_key, metric_key.replace(",", "_"))
                result: Result = {
                    "name": task_name,
                    "metric": display_metric,
                    "value": float(metric_val),
                    "unit": _guess_unit(display_metric),
                    "reasoning_arm": ctx.reasoning_effort or "unstated",
                    "output_tokens_mean": None,
                    "output_tokens_median": None,
                    "output_tokens_p90": None,
                    "tokens_per_correct": None,
                    "truncation_rate": sample_metrics.truncation_rate,
                    "reasoning_tokens_mean": sample_metrics.reasoning_tokens_mean,
                }

                tags: dict[str, str] = {
                    "lm_eval_key": metric_key,
                    **{k: str(v) for k, v in ctx.extra_tags.items()},
                }
                total_eval_time = raw.get("total_evaluation_time_seconds")
                if total_eval_time is not None:
                    tags["total_eval_time_s"] = str(total_eval_time)
                limit = cfg.get("limit")
                if limit is not None:
                    tags["n_limit"] = str(limit)
                max_gen_toks = cfg_gen_kwargs.get("max_gen_toks")
                if max_gen_toks is not None:
                    tags["max_gen_toks"] = str(max_gen_toks)
                if sample_metrics.output_tokens is None:
                    tags["output_tokens_unavailable_reason"] = (
                        sample_metrics.output_unavailable_reason or "sample_token_count_unavailable"
                    )
                    tags["tokens_per_correct_unavailable_reason"] = "output_tokens_unavailable"
                else:
                    counts = sample_metrics.output_tokens
                    mean, median, p90 = summarize_token_counts(counts)
                    result["output_tokens_mean"] = mean
                    result["output_tokens_median"] = median
                    result["output_tokens_p90"] = float(p90)
                    tags["output_tokens_method"] = sample_metrics.output_method or "unknown"
                    if result["unit"] == "ratio" and 0 <= float(metric_val) <= 1:
                        correct_count = round(float(metric_val) * sample_metrics.sample_count)
                        if correct_count > 0:
                            result["tokens_per_correct"] = sum(counts) / correct_count
                        else:
                            tags["tokens_per_correct_unavailable_reason"] = "zero_correct_items"
                    else:
                        tags["tokens_per_correct_unavailable_reason"] = "score_not_accuracy_ratio"
                if sample_metrics.truncation_rate is None:
                    tags["truncation_rate_unavailable_reason"] = (
                        sample_metrics.truncation_unavailable_reason
                        or "max_gen_toks_and_finish_reason_missing"
                    )
                if sample_metrics.reasoning_tokens_mean is None:
                    tags["reasoning_tokens_unavailable_reason"] = (
                        sample_metrics.reasoning_unavailable_reason or "reasoning_usage_missing"
                    )
                if duration_s is not None:
                    result["duration_seconds"] = duration_s
                    if (
                        duration_s > 0
                        and sample_metrics.prompt_tokens is not None
                        and sample_metrics.output_tokens is not None
                    ):
                        prompt_total = sample_metrics.prompt_tokens
                        output_total = sum(sample_metrics.output_tokens)
                        result["prompt_tokens_per_second"] = prompt_total / duration_s
                        result["decode_tokens_per_second"] = output_total / duration_s
                        result["total_tokens_per_second"] = (prompt_total + output_total) / duration_s
                result["tags"] = tags

                yield result


_ISO8601_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


def _extract_timestamp(raw: dict[str, Any]) -> str:
    raw_date = raw.get("date")
    if isinstance(raw_date, int | float):
        return datetime.datetime.fromtimestamp(raw_date, datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(raw_date, str) and _ISO8601_UTC.match(raw_date):
        return raw_date
    if isinstance(raw_date, str):
        log.warning(
            "ignoring non-ISO-8601 raw['date']=%r; falling back to current UTC time",
            raw_date,
        )
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_gen_kwargs(raw: dict[str, Any]) -> GenKwargs | None:
    cfg_raw = raw.get("config") or {}
    cfg = cfg_raw.get("gen_kwargs") or {}
    if not isinstance(cfg, dict) or not cfg:
        return None
    result: GenKwargs = {}
    max_gen_toks = cfg.get("max_gen_toks")
    if isinstance(max_gen_toks, int):
        result["max_gen_toks"] = max_gen_toks
    temperature = cfg.get("temperature")
    if isinstance(temperature, int | float):
        result["temperature"] = float(temperature)
    top_p = cfg.get("top_p")
    if isinstance(top_p, int | float):
        result["top_p"] = float(top_p)
    top_k = cfg.get("top_k")
    if isinstance(top_k, int):
        result["top_k"] = top_k
    return result or None


def _guess_unit(display_metric: str) -> str:
    if display_metric in {
        "accuracy",
        "accuracy_normalized",
        "pass_at_1",
        "exact_match_flexible",
        "exact_match_strict",
    }:
        return "ratio"
    if display_metric.endswith("_seconds"):
        return "seconds"
    if display_metric.endswith("_ms"):
        return "ms"
    return "ratio"
