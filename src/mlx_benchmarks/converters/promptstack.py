"""promptstack runner JSON (harness/promptstack/run.py) -> envelope v1 converter.

One raw file per (model, surface, run): a ``config`` block and a ``cells``
array. Each cell is one (prompt_variant, thinking) combination and carries a
``probe_results`` array — one block per probe class (reasoning, tool_call,
instruction, homelab_qa). Every probe-class block becomes several metric rows
under the constant result name ``promptstack``; the sweep dimensions
(prompt_variant, probe_class, thinking, concurrency) travel as string tags so
the dataset viewer — and the adoption rule in ``docs/promptstack.md`` — can
pivot ``base_plus_variant`` against ``current`` per probe class.
"""

from __future__ import annotations

import datetime
from typing import Any

from mlx_benchmarks.converters.base import ConverterContext, apply_optional_fields, thinking_level
from mlx_benchmarks.envelope import Envelope, Result, System

# Per-probe-class rate metrics: raw key -> metric name. All are ratios in [0, 1].
_RATE_METRICS: dict[str, str] = {
    "task_success_rate": "task_success_rate",
    "valid_tool_call_rate": "valid_tool_call_rate",
    "unsupported_claim_rate": "unsupported_claim_rate",
    "instruction_adherence_rate": "instruction_adherence_rate",
}

# Per-probe-class token averages: raw key -> metric name.
_TOKEN_METRICS: dict[str, str] = {
    "tokens_prompt_avg": "tokens_prompt",
    "tokens_completion_avg": "tokens_completion",
}

# Per-probe-class latency percentiles: raw key -> metric name.
_LATENCY_METRICS: dict[str, str] = {
    "latency_p50_ms": "request_latency_p50_ms",
    "latency_p95_ms": "request_latency_p95_ms",
}


class PromptstackConverter:
    kind = "promptstack"

    def build_envelope(self, raw: dict[str, Any], ctx: ConverterContext) -> Envelope:
        cells = raw.get("cells") or []
        if not cells:
            raise ValueError("promptstack raw results contain no cells — nothing to publish")

        probe_bank_version = str((raw.get("config") or {}).get("probe_bank_version", ""))
        timestamp = ctx.timestamp_override or _extract_timestamp(raw)
        system: System = ctx.system or {}  # type: ignore[assignment]

        results: list[Result] = []
        for cell in cells:
            results.extend(_cell_results(cell, probe_bank_version, ctx))

        envelope: Envelope = {
            "schema_version": "1",
            "timestamp": timestamp,
            "git_sha": ctx.git_sha,
            "trigger": ctx.trigger,
            "suite": ctx.suite,
            "model": ctx.model,
            "system": system,
            "results": results,
            "errors": [],
        }
        return apply_optional_fields(envelope, ctx)


def _cell_results(cell: dict[str, Any], probe_bank_version: str, ctx: ConverterContext) -> list[Result]:
    results: list[Result] = []
    for block in cell.get("probe_results") or []:
        tags: dict[str, str] = {
            "cell": str(cell.get("name", "")),
            "prompt_variant": str(cell.get("prompt_variant", "")),
            "probe_class": str(block.get("probe_class", "")),
            "probe_bank_version": probe_bank_version,
            "thinking": thinking_level(cell.get("thinking")),
            "concurrency": "1",
            "n_tasks": str(block.get("n_tasks", "")),
            **{k: str(v) for k, v in ctx.extra_tags.items()},
        }

        for raw_key, metric in _RATE_METRICS.items():
            value = block.get(raw_key)
            if not isinstance(value, int | float):
                continue
            results.append(
                {
                    "name": "promptstack",
                    "metric": metric,
                    "value": float(value),
                    "unit": "ratio",
                    "tags": dict(tags),
                }
            )

        for raw_key, metric in _TOKEN_METRICS.items():
            value = block.get(raw_key)
            if not isinstance(value, int | float):
                continue
            results.append(
                {
                    "name": "promptstack",
                    "metric": metric,
                    "value": float(value),
                    "unit": "tokens",
                    "tags": dict(tags),
                }
            )

        for raw_key, metric in _LATENCY_METRICS.items():
            value = block.get(raw_key)
            if not isinstance(value, int | float):
                continue
            results.append(
                {
                    "name": "promptstack",
                    "metric": metric,
                    "value": float(value),
                    "unit": "ms",
                    "tags": dict(tags),
                }
            )
    return results


def _extract_timestamp(raw: dict[str, Any]) -> str:
    ts = raw.get("timestamp")
    if isinstance(ts, str) and ts:
        return ts
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
