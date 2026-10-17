"""Per-task sample statistics for lm-eval prediction logs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_benchmarks.converters.base import ConverterContext
from mlx_benchmarks.converters.token_stats import (
    TokenizerLike,
    TokenizerLoader,
    api_output_tokens,
    api_reasoning_tokens,
    count_tokens,
    default_tokenizer_loader,
    finish_reason,
)

log = logging.getLogger(__name__)


@dataclass(slots=True)
class SampleMetrics:
    prompt_tokens: int | None
    output_tokens: tuple[int, ...] | None
    sample_count: int
    output_method: str | None
    output_unavailable_reason: str | None
    truncation_rate: float | None
    truncation_unavailable_reason: str | None
    reasoning_tokens_mean: float | None = None
    reasoning_unavailable_reason: str | None = "sample_logs_missing"


def sample_metrics_for_task(
    task_name: str,
    max_gen_toks: Any,
    duration_s: float | None,
    ctx: ConverterContext,
    tokenizer_loader: TokenizerLoader | None,
    tokenizer_cache: dict[str, TokenizerLike | None],
    cache: dict[str, SampleMetrics],
) -> SampleMetrics:
    """Read one task's samples once and calculate token and truncation stats."""
    if task_name in cache:
        return cache[task_name]
    unavailable = SampleMetrics(None, None, 0, None, "sample_logs_missing", None, "sample_logs_missing")
    if ctx.source_path is None:
        cache[task_name] = unavailable
        return unavailable
    samples_file = _find_samples_file(ctx.source_path, task_name)
    if samples_file is None:
        cache[task_name] = unavailable
        return unavailable

    loader = tokenizer_loader or default_tokenizer_loader
    tokenizer = (
        _get_tokenizer(ctx.model, loader, tokenizer_cache)
        if duration_s is not None and duration_s > 0
        else None
    )
    prompt_total = 0 if tokenizer is not None else None
    output_counts: list[int] = []
    methods: set[str] = set()
    truncations: list[bool | None] = []
    reasoning_counts: list[int] = []
    reasoning_count_missing = False
    sample_count = 0
    output_count_missing = False
    valid_max = (
        max_gen_toks
        if isinstance(max_gen_toks, int) and not isinstance(max_gen_toks, bool) and max_gen_toks > 0
        else None
    )
    try:
        with samples_file.open() as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    sample = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not isinstance(sample, dict):
                    continue
                sample_count += 1
                output_count = api_output_tokens(sample)
                if output_count is not None:
                    methods.add("usage")
                else:
                    if tokenizer is None:
                        tokenizer = _get_tokenizer(ctx.model, loader, tokenizer_cache)
                    response = _extract_response(sample)
                    if tokenizer is None:
                        output_count_missing = True
                    else:
                        output_count = count_tokens(tokenizer, response)
                        methods.add("hf_tokenizer")
                reasoning_count = api_reasoning_tokens(sample)
                if reasoning_count is None:
                    reasoning_count_missing = True
                else:
                    reasoning_counts.append(reasoning_count)
                if output_count is not None:
                    output_counts.append(output_count)
                if tokenizer is not None and prompt_total is not None:
                    prompt_total += count_tokens(tokenizer, _extract_prompt(sample))

                reason = finish_reason(sample)
                if reason in {"length", "max_tokens", "model_length"}:
                    truncations.append(True)
                elif reason is not None:
                    truncations.append(False)
                elif valid_max is not None and output_count is not None:
                    truncations.append(output_count >= valid_max)
                else:
                    truncations.append(None)
    except OSError as exc:
        log.warning("could not read %s: %s", samples_file, exc)
        unavailable.output_unavailable_reason = "sample_log_unreadable"
        unavailable.truncation_unavailable_reason = "sample_log_unreadable"
        unavailable.reasoning_unavailable_reason = "sample_log_unreadable"
        cache[task_name] = unavailable
        return unavailable
    if sample_count == 0:
        unavailable.output_unavailable_reason = "sample_logs_empty"
        unavailable.truncation_unavailable_reason = "sample_logs_empty"
        unavailable.reasoning_unavailable_reason = "sample_logs_empty"
        cache[task_name] = unavailable
        return unavailable

    complete_counts = (
        tuple(output_counts) if len(output_counts) == sample_count and not output_count_missing else None
    )
    if truncations and all(value is not None for value in truncations):
        truncation_rate: float | None = sum(value is True for value in truncations) / len(truncations)
        truncation_reason = None
    else:
        truncation_rate = None
        truncation_reason = "max_gen_toks_and_finish_reason_missing"
    reasoning_tokens_mean = (
        sum(reasoning_counts) / sample_count
        if len(reasoning_counts) == sample_count and not reasoning_count_missing
        else None
    )
    result = SampleMetrics(
        prompt_total,
        complete_counts,
        sample_count,
        next(iter(methods)) if len(methods) == 1 else "mixed" if methods else None,
        None if complete_counts is not None else "sample_token_count_unavailable",
        truncation_rate,
        truncation_reason,
        reasoning_tokens_mean,
        None if reasoning_tokens_mean is not None else "reasoning_usage_missing",
    )
    cache[task_name] = result
    return result


def _get_tokenizer(
    model: str,
    loader: TokenizerLoader,
    cache: dict[str, TokenizerLike | None],
) -> TokenizerLike | None:
    if model not in cache:
        cache[model] = loader(model)
    return cache[model]


def _find_samples_file(source_path: Path, task_name: str) -> Path | None:
    """Find this task's sample log beside its lm-eval result file."""
    if not source_path.exists():
        return None
    directory = source_path.parent
    matches = sorted(directory.glob(f"samples_{task_name}_*.jsonl"))
    if not matches:
        return None
    stem = source_path.stem
    if stem.startswith("results_"):
        ts = stem.removeprefix("results_")
        candidate = directory / f"samples_{task_name}_{ts}.jsonl"
        if candidate.is_file():
            return candidate
    return matches[-1]


def _extract_prompt(sample: dict[str, Any]) -> str:
    """Reconstruct the prompt text sent by lm-eval."""
    args = sample.get("arguments")
    if isinstance(args, list):
        chunks: list[str] = []
        for arg in args:
            if isinstance(arg, list) and arg and isinstance(arg[0], str):
                chunks.append(arg[0])
            elif isinstance(arg, dict):
                text = arg.get("text") or arg.get("prompt")
                if isinstance(text, str):
                    chunks.append(text)
            elif isinstance(arg, str):
                chunks.append(arg)
        if chunks:
            return "\n".join(chunks)
    doc = sample.get("doc")
    if isinstance(doc, dict):
        for key in ("question", "input", "prompt"):
            value = doc.get(key)
            if isinstance(value, str):
                return value
    return ""


def _extract_response(sample: dict[str, Any]) -> str:
    """Reconstruct the response text, preferring lm-eval's filtered output."""
    resps = sample.get("filtered_resps") or sample.get("resps")
    if not isinstance(resps, list):
        return ""
    flat: list[str] = []
    for response in resps:
        if isinstance(response, str):
            flat.append(response)
        elif isinstance(response, list):
            flat.extend(str(value) for value in response if isinstance(value, str | int | float))
    return "\n".join(flat) if flat else ""
