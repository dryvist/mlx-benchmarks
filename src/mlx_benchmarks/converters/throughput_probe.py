from __future__ import annotations

import datetime
import logging
from typing import Any

from mlx_benchmarks.converters.base import ConverterContext, apply_optional_fields
from mlx_benchmarks.envelope import Envelope, Result, System

log = logging.getLogger(__name__)

_METRICS: tuple[tuple[str, str, str], ...] = (
    ("cumulative_tok_s", "throughput_total_toks_per_s", "tok/s"),
    ("decode_tok_s", "throughput_output_toks_per_s", "tok/s"),
    ("prefill_tok_s", "throughput_prompt_toks_per_s", "tok/s"),
    ("ttft_s", "ttft_p50_ms", "ms"),
    ("total_s", "request_duration_p50_s", "s"),
)


class ThroughputProbeConverter:
    kind = "throughput-probe"

    def build_envelope(self, raw: Any, ctx: ConverterContext) -> Envelope:
        timestamp = (
            ctx.timestamp_override
            or raw.get("started_utc")
            or datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        system: System = ctx.system or {}  # type: ignore[assignment]
        envelope: Envelope = {
            "schema_version": "1",
            "timestamp": timestamp,
            "git_sha": ctx.git_sha,
            "trigger": ctx.trigger,
            "suite": ctx.suite,
            "model": ctx.model,
            "system": system,
            "results": self._iter_results(raw, ctx),
            "errors": [],
        }
        for field in ("campaign", "cell_status", "context"):
            value = raw.get(field)
            if value is not None:
                envelope[field] = value
        return apply_optional_fields(envelope, ctx)

    def _iter_results(self, raw: Any, ctx: ConverterContext) -> list[Result]:
        if not isinstance(raw, dict) or not isinstance(raw.get("sequential"), dict):
            log.warning("throughput-probe output has no sequential summary")
            return []

        sequential = raw["sequential"]
        tags = {k: str(v) for k, v in ctx.extra_tags.items()}
        for key in ("n_ok", "n_err", "answered_rate", "truncated_rate"):
            if key in sequential:
                tags[key] = str(sequential[key])
        if isinstance(sequential.get("finish_reasons"), list):
            tags["finish_reasons"] = ",".join(map(str, sequential["finish_reasons"]))
        # ``thinking`` on its own is not readable: a run that passed no kwarg
        # records "off" while the model reasons at its own default (see
        # harness/throughput/run.py). ``think_kwarg_sent`` is what separates
        # "we asked for off" from "we asked for nothing", and ``think_value``
        # carries the graded level the boolean cannot hold — both were recorded
        # on every row and dropped here.
        for key in (
            "max_tokens",
            "thinking",
            "think_kwarg",
            "think_value",
            "think_kwarg_sent",
            "dedicated",
            "context_tokens_target",
        ):
            if key in raw:
                tags[key] = "unstated" if raw[key] is None else str(raw[key])
        sequential_runs = raw.get("sequential_runs")
        if isinstance(sequential_runs, list):
            tags["measured_repetitions"] = str(len(sequential_runs))
            prompt_tokens = {
                run["prompt_tokens"]
                for run in sequential_runs
                if isinstance(run, dict) and isinstance(run.get("prompt_tokens"), int)
            }
            if prompt_tokens:
                tags["context_tokens_actual"] = ",".join(map(str, sorted(prompt_tokens)))

        results: list[Result] = []
        for source_key, metric, unit in _METRICS:
            stats = sequential.get(source_key)
            if not isinstance(stats, dict) or not isinstance(stats.get("median"), int | float):
                continue
            result_tags = dict(tags)
            # Which phase produced the number. Without it the sequential and
            # concurrent rows are indistinguishable once published, and a reader
            # cannot tell a one-at-a-time figure from an aggregate under load.
            result_tags["phase"] = "sequential"
            for stat in ("min", "max"):
                if isinstance(stats.get(stat), int | float):
                    result_tags[f"{source_key}_{stat}"] = str(stats[stat])
            value = float(stats["median"])
            if source_key == "ttft_s":
                value *= 1000.0
            results.append(
                {
                    "name": "throughput_probe",
                    "metric": metric,
                    "value": value,
                    "unit": unit,
                    "tags": result_tags,
                    "raw": raw,
                }
            )
        results.extend(_concurrent_results(raw, tags))
        return results


# The concurrent phase's aggregate throughput: the only figure in this harness
# that answers "what does the endpoint deliver with N requests in flight". The
# runner has always recorded it and this converter dropped it, so every published
# throughput row described one-at-a-time serving no matter what width was driven.
_CONCURRENT_METRICS: tuple[tuple[str, str], ...] = (
    ("aggregate_cumulative_tok_s", "throughput_aggregate_toks_per_s"),
    ("aggregate_decode_tok_s", "throughput_aggregate_output_toks_per_s"),
)


def _concurrent_results(raw: Any, tags: dict[str, str]) -> list[Result]:
    concurrent = raw.get("concurrent") if isinstance(raw, dict) else None
    if not isinstance(concurrent, dict):
        # --skip-concurrent, or a run that failed before the phase. Absent, not zero.
        return []

    base = dict(tags)
    base["phase"] = "concurrent"
    for key in ("width", "n_ok", "n_err", "wall_s"):
        if concurrent.get(key) is not None:
            base[key] = str(concurrent[key])
    errors = concurrent.get("errors")
    if isinstance(errors, list) and errors:
        # A partially-failed batch still publishes; the aggregate is over the
        # requests that returned, so the error count is what stops it reading
        # as a clean measurement.
        base["concurrent_errors"] = str(len(errors))

    results: list[Result] = []
    for source_key, metric in _CONCURRENT_METRICS:
        value = concurrent.get(source_key)
        if not isinstance(value, int | float):
            continue
        results.append(
            {
                "name": "throughput_probe",
                "metric": metric,
                "value": float(value),
                "unit": "tok/s",
                "tags": dict(base),
                "raw": raw,
            }
        )
    return results
