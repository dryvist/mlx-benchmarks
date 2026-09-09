"""factual runner JSON (harness/factual/run.py) -> envelope v1 converter.

One raw file per (model, run): a ``config`` block and a ``cells`` array, one
cell per thinking mode. Every cell becomes several metric rows under the
constant result name ``factual``; the thinking dimension and the fixture-bank
version travel as string tags so shards stay comparable only against the bank
they were scored with.
"""

from __future__ import annotations

import datetime
from typing import Any

from mlx_benchmarks.converters.base import ConverterContext, apply_optional_fields, thinking_level
from mlx_benchmarks.envelope import Envelope, Result, System

# Per-cell rate metrics: raw key -> metric name. All are ratios in [0, 1].
# grounded_accuracy is the headline (all four checks passed); the rest decompose
# it so a shard shows *which* way a model failed, not just that it did.
_RATE_METRICS: dict[str, str] = {
    "grounded_accuracy": "grounded_accuracy",
    "fact_recall_rate": "fact_recall_rate",
    "fabricated_number_rate": "fabricated_number_rate",
    "forbidden_claim_rate": "forbidden_claim_rate",
    "tool_syntax_leak_rate": "tool_syntax_leak_rate",
    "error_rate": "error_rate",
}

_LATENCY_METRICS: dict[str, str] = {
    "latency_p50_ms": "request_latency_p50_ms",
    "latency_p95_ms": "request_latency_p95_ms",
}


class FactualConverter:
    kind = "factual"

    def build_envelope(self, raw: dict[str, Any], ctx: ConverterContext) -> Envelope:
        cells = raw.get("cells") or []
        if not cells:
            raise ValueError("factual raw results contain no cells — nothing to publish")

        bank_version = str((raw.get("config") or {}).get("fixture_bank_version", ""))
        system: System = ctx.system or {}  # type: ignore[assignment]

        results: list[Result] = []
        for cell in cells:
            results.extend(_cell_results(cell, bank_version, ctx))

        envelope: Envelope = {
            "schema_version": "1",
            "timestamp": ctx.timestamp_override or _extract_timestamp(raw),
            "git_sha": ctx.git_sha,
            "trigger": ctx.trigger,
            "suite": ctx.suite,
            "model": ctx.model,
            "system": system,
            "results": results,
            "errors": [],
        }
        return apply_optional_fields(envelope, ctx)


def _cell_results(cell: dict[str, Any], bank_version: str, ctx: ConverterContext) -> list[Result]:
    tags: dict[str, str] = {
        "cell": str(cell.get("name", "")),
        "thinking": thinking_level(cell.get("thinking")),
        "fixture_bank_version": bank_version,
        "n_responses": str(cell.get("n_responses", "")),
        # The suite drives one request at a time by design: factual grounding is
        # a single-turn property, and holding concurrency at 1 keeps the latency
        # rows comparable with the agentic suite's conc1 cells.
        "concurrency": "1",
        **{k: str(v) for k, v in ctx.extra_tags.items()},
    }

    results: list[Result] = []
    for raw_key, metric in _RATE_METRICS.items():
        value = cell.get(raw_key)
        if not isinstance(value, int | float):
            continue
        result: Result = {
            "name": "factual",
            "metric": metric,
            "value": float(value),
            "unit": "ratio",
            "tags": dict(tags),
        }
        wall = cell.get("wall_seconds")
        if isinstance(wall, int | float):
            result["duration_seconds"] = float(wall)
        results.append(result)

    for raw_key, metric in _LATENCY_METRICS.items():
        value = cell.get(raw_key)
        if not isinstance(value, int | float):
            continue
        results.append(
            {
                "name": "factual",
                "metric": metric,
                "value": float(value),
                "unit": "ms",
                "tags": dict(tags),
            }
        )

    tokens = cell.get("tokens_completion_avg")
    if isinstance(tokens, int | float):
        results.append(
            {
                "name": "factual",
                "metric": "tokens_completion",
                "value": float(tokens),
                "unit": "tokens",
                "tags": dict(tags),
            }
        )
    return results


def _extract_timestamp(raw: dict[str, Any]) -> str:
    ts = raw.get("timestamp")
    if isinstance(ts, str) and ts:
        return ts
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
