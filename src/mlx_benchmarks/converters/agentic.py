"""agentic runner JSON (harness/agentic/run.py) -> envelope v1 converter.

One raw file per (model, run): a ``config`` block, a ``cells`` array of
single-shot matrix cells, and a ``multiturn`` array (the mlx-lm #1011
degradation track). Every cell becomes several metric rows under the constant
result name ``tool_calling``; the sweep dimensions (concurrency / thinking /
context / stream) travel as string tags so the dataset viewer can pivot on
them.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

from mlx_benchmarks.converters.base import ConverterContext, apply_optional_fields, thinking_level
from mlx_benchmarks.envelope import Envelope, Result, System

log = logging.getLogger(__name__)

# Per-cell rate metrics: raw cell key -> metric name. All are ratios in [0, 1].
_RATE_METRICS: dict[str, str] = {
    "valid_tool_call_rate": "valid_tool_call_rate",
    "finish_reason_tool_calls_rate": "finish_reason_tool_calls_rate",
    "reasoning_present_rate": "reasoning_present_rate",
}

# Per-cell latency percentile metrics: raw cell key -> metric name.
_LATENCY_METRICS: dict[str, str] = {
    "latency_p50_ms": "request_latency_p50_ms",
    "latency_p95_ms": "request_latency_p95_ms",
}


class AgenticConverter:
    kind = "agentic"

    def build_envelope(self, raw: dict[str, Any], ctx: ConverterContext) -> Envelope:
        cells = raw.get("cells") or []
        multiturn = raw.get("multiturn") or []
        if not cells and not multiturn:
            raise ValueError(
                "agentic raw results contain no cells and no multiturn track — nothing to publish"
            )

        timestamp = ctx.timestamp_override or _extract_timestamp(raw)
        system: System = ctx.system or {}  # type: ignore[assignment]

        results: list[Result] = []
        for cell in cells:
            results.extend(_cell_results(cell, ctx))
        for track in multiturn:
            results.append(_multiturn_result(track, ctx))

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


def _cell_results(cell: dict[str, Any], ctx: ConverterContext) -> list[Result]:
    tags: dict[str, str] = {
        "cell": str(cell.get("name", "")),
        "concurrency": str(cell.get("concurrency", "")),
        "thinking": thinking_level(cell.get("thinking")),
        "context": str(cell.get("context", "")),
        "stream": "stream" if cell.get("stream") else "nostream",
        "n_requests": str(cell.get("n_requests", "")),
        **{f"failure_{kind}": str(count) for kind, count in (cell.get("failures") or {}).items()},
        **{k: str(v) for k, v in ctx.extra_tags.items()},
    }

    results: list[Result] = []
    for raw_key, metric in _RATE_METRICS.items():
        value = cell.get(raw_key)
        if not isinstance(value, int | float):
            log.debug("cell %s missing %r — skipping metric", cell.get("name"), raw_key)
            continue
        result: Result = {
            "name": "tool_calling",
            "metric": metric,
            "value": float(value),
            "unit": "ratio",
            "tags": dict(tags),
        }
        _attach_measurements(result, cell)
        results.append(result)

    for raw_key, metric in _LATENCY_METRICS.items():
        value = cell.get(raw_key)
        if not isinstance(value, int | float):
            continue
        results.append(
            {
                "name": "tool_calling",
                "metric": metric,
                "value": float(value),
                "unit": "ms",
                "tags": dict(tags),
            }
        )

    # Wall-clock aggregate throughput (sum of completion tokens / wall seconds),
    # published alongside decode_tokens_per_second. decode_* carries the
    # per-request MEAN and is depressed under concurrency; this row is the honest
    # headline number. Kept as a distinct metric so neither value changes meaning
    # across shards. See _attach_measurements.
    aggregate = cell.get("aggregate_tokens_per_second")
    if isinstance(aggregate, int | float) and aggregate >= 0:
        aggregate_result: Result = {
            "name": "tool_calling",
            "metric": "aggregate_tokens_per_second",
            "value": float(aggregate),
            "unit": "tok/s",
            "tags": dict(tags),
        }
        wall = cell.get("wall_seconds")
        if isinstance(wall, int | float):
            aggregate_result["duration_seconds"] = float(wall)
        results.append(aggregate_result)
    return results


def _attach_measurements(result: Result, cell: dict[str, Any]) -> None:
    """Populate first-class measurement fields where the runner measured them."""
    wall = cell.get("wall_seconds")
    if isinstance(wall, int | float):
        result["duration_seconds"] = float(wall)
    tps = cell.get("effective_tokens_per_second")
    if isinstance(tps, int | float) and tps >= 0:
        # decode_tokens_per_second here is the per-request MEAN completion
        # throughput (effective_tokens_per_second). Under concurrency > 1 it is
        # depressed by contention and is NOT the wall-clock aggregate the schema
        # description implies — but it is what every prior shard put here, so it
        # stays put for cross-shard comparability. The honest wall-clock figure
        # rides alongside as the aggregate_tokens_per_second metric row.
        result["decode_tokens_per_second"] = float(tps)
    ftl = cell.get("first_token_p50_ms")
    if isinstance(ftl, int | float) and ftl >= 0:
        result["first_token_latency_ms"] = float(ftl)


def _multiturn_result(track: dict[str, Any], ctx: ConverterContext) -> Result:
    rounds = track.get("rounds") or []
    first_degraded = track.get("first_degraded_round")
    degraded = isinstance(first_degraded, int)
    return {
        "name": "tool_calling",
        "metric": "first_degraded_round",
        # 0 = never degraded through all rounds (schema needs a number; see tags.degraded)
        "value": float(first_degraded) if degraded else 0.0,
        "unit": "round",
        "tags": {
            "track": "multiturn",
            "thinking": thinking_level(track.get("thinking")),
            "degraded": "true" if degraded else "false",
            "rounds": str(len(rounds)),
            "valid_rounds": str(sum(1 for r in rounds if r.get("outcome") == "valid")),
            **{k: str(v) for k, v in ctx.extra_tags.items()},
        },
    }


def _extract_timestamp(raw: dict[str, Any]) -> str:
    ts = raw.get("timestamp")
    if isinstance(ts, str) and ts:
        return ts
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
