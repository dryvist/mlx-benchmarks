"""vllm ``benchmark_serving`` JSON -> envelope v1 converter."""

from __future__ import annotations

import datetime
import logging
import math
from collections import defaultdict
from typing import Any, cast

from mlx_benchmarks.converters.base import (
    ConverterContext,
    apply_optional_fields,
    require_gpu_power_limit,
)
from mlx_benchmarks.dataset_schema import DERIVED_THROUGHPUT_SPEED_FIELDS, THROUGHPUT_SUITES
from mlx_benchmarks.envelope import CampaignDimensions, Envelope, Result, System

log = logging.getLogger(__name__)

# Map from vllm benchmark_serving JSON key -> (metric_name, unit)
# Only keys listed here are extracted; unknown keys are silently ignored so
# the converter remains forward-compatible with new vllm output fields.
#
# Headline metric policy (2026-07-27, see docs/schema.md): throughput_total_toks_per_s
# (cumulative prompt+completion tok/s, the envelope's total_tokens_per_second
# concept) is the PRIMARY figure to report/compare. throughput_output_toks_per_s
# (decode-only) is supporting detail — it hides prefill-engine gains that are a
# real, felt latency win. vllm's benchmark_serving already reports both; only
# which one is treated as headline downstream (RANKINGS.md, viewer) changes.
_METRIC_MAP: dict[str, tuple[str, str]] = {
    "output_throughput": ("throughput_output_toks_per_s", "tok/s"),
    "total_token_throughput": ("throughput_total_toks_per_s", "tok/s"),
    "request_throughput": ("throughput_requests_per_s", "req/s"),
    "median_ttft_ms": ("ttft_p50_ms", "ms"),
    "p99_ttft_ms": ("ttft_p99_ms", "ms"),
    "median_itl_ms": ("itl_p50_ms", "ms"),
    "p99_itl_ms": ("itl_p99_ms", "ms"),
    "median_tpot_ms": ("tpot_p50_ms", "ms"),
    "p90_tpot_ms": ("tpot_p90_ms", "ms"),
    "p99_tpot_ms": ("tpot_p99_ms", "ms"),
}


class VllmConverter:
    kind = "vllm"

    def build_envelope(self, raw: dict[str, Any], ctx: ConverterContext) -> Envelope:
        started_at = _bench_serve_started_at(raw)
        timestamp = ctx.timestamp_override or (
            started_at.isoformat().replace("+00:00", "Z")
            if started_at is not None
            else datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        system: System = ctx.system or {}  # type: ignore[assignment]
        require_gpu_power_limit(system)

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

        apply_optional_fields(envelope, ctx)
        duration = raw.get("duration")
        if started_at is not None and isinstance(duration, int | float) and not isinstance(duration, bool):
            dimensions = cast("CampaignDimensions", dict(envelope.get("campaign_dimensions") or {}))
            run_dimensions = dimensions.get("run")
            if run_dimensions is None:
                run_dimensions = {}
            if isinstance(run_dimensions, dict):
                run_dimensions = dict(run_dimensions)
                run_dimensions.setdefault("started_at", started_at.isoformat().replace("+00:00", "Z"))
                ended_at = started_at + datetime.timedelta(seconds=float(duration))
                run_dimensions.setdefault("ended_at", ended_at.isoformat().replace("+00:00", "Z"))
                dimensions["run"] = run_dimensions
                envelope["campaign_dimensions"] = dimensions

        if ctx.suite in THROUGHPUT_SUITES:
            _add_derived_throughput_dimensions(envelope, raw)

        return envelope

    def _iter_results(self, raw: dict[str, Any], ctx: ConverterContext) -> list[Result]:
        results: list[Result] = []
        duration = raw.get("duration")
        completed = raw.get("completed")
        num_input = raw.get("total_input_tokens")
        num_output = raw.get("total_output_tokens")

        tags: dict[str, str] = {k: str(v) for k, v in ctx.extra_tags.items()}
        if isinstance(completed, int | float):
            tags["completed_requests"] = str(int(completed))
        if isinstance(num_input, int | float):
            tags["total_input_tokens"] = str(int(num_input))
        if isinstance(num_output, int | float):
            tags["total_output_tokens"] = str(int(num_output))

        for raw_key, (metric_name, unit) in _METRIC_MAP.items():
            raw_val = raw.get(raw_key)
            if metric_name == "tpot_p50_ms":
                raw_val = _first_number(raw, "p50_tpot_ms", "median_tpot_ms")
            numeric_val = _finite_number(raw_val)
            if numeric_val is None:
                log.debug("vllm sample missing key %r — skipping metric %r", raw_key, metric_name)
                continue
            result: Result = {
                "name": "benchmark_serving",
                "metric": metric_name,
                "value": numeric_val,
                "unit": unit,
                "tags": dict(tags),
            }
            numeric_duration = _finite_number(duration)
            if numeric_duration is not None:
                result["duration_seconds"] = numeric_duration
            results.append(result)

        if not results:
            log.warning("vllm sample produced no results; raw keys: %s", list(raw))

        return results


def _bench_serve_started_at(raw: dict[str, Any]) -> datetime.datetime | None:
    raw_date = raw.get("date")
    if not isinstance(raw_date, str):
        return None
    try:
        return datetime.datetime.strptime(raw_date, "%Y%m%d-%H%M%S").replace(tzinfo=datetime.UTC)
    except ValueError:
        log.warning("ignoring non-vLLM bench serve raw['date']=%r", raw_date)
        return None


def _finite_number(value: Any) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _first_number(raw: dict[str, Any], *keys: str) -> float | None:
    return next((number for key in keys if (number := _finite_number(raw.get(key))) is not None), None)


def _request_timing_data(
    raw: dict[str, Any],
) -> list[tuple[float, float, list[float], int]] | None:
    start_times = raw.get("start_times")
    ttfts = raw.get("ttfts")
    itls = raw.get("itls")
    output_lens = raw.get("output_lens")
    if not (
        isinstance(start_times, list)
        and isinstance(ttfts, list)
        and isinstance(itls, list)
        and isinstance(output_lens, list)
    ):
        return None
    if not start_times or not (len(start_times) == len(ttfts) == len(itls) == len(output_lens)):
        return None

    completed = _finite_number(raw.get("completed"))
    if completed is not None and completed != len(start_times):
        return None

    requests: list[tuple[float, float, list[float], int]] = []
    for start, ttft, gaps, output_len in zip(start_times, ttfts, itls, output_lens, strict=True):
        start_time = _finite_number(start)
        first_token_delay = _finite_number(ttft)
        if (
            start_time is None
            or first_token_delay is None
            or first_token_delay < 0
            or not isinstance(gaps, list)
            or not isinstance(output_len, int)
            or isinstance(output_len, bool)
            or output_len < 0
        ):
            return None

        offsets = [first_token_delay]
        latency = first_token_delay
        for gap in gaps:
            inter_chunk_delay = _finite_number(gap)
            if inter_chunk_delay is None or inter_chunk_delay < 0:
                return None
            latency += inter_chunk_delay
            offsets.append(latency)
        requests.append((start_time, latency, offsets, output_len))
    return requests


def _window_rates(
    requests: list[tuple[float, float, list[float], int]],
) -> tuple[float, float] | None:
    token_events = [
        (start_time + offset, output_len / len(offsets))
        for start_time, _latency, offsets, output_len in requests
        if output_len > 0
        for offset in offsets
    ]
    if not token_events:
        return None

    run_start = min(start_time for start_time, _latency, _offsets, _length in requests)
    first_token_time = min(timestamp for timestamp, _tokens in token_events)
    last_token_time = max(timestamp for timestamp, _tokens in token_events)
    buckets: dict[int, float] = defaultdict(float)
    for timestamp, tokens in token_events:
        buckets[math.floor(timestamp - run_start)] += tokens

    first_window = math.floor(first_token_time - run_start)
    last_window = math.floor(last_token_time - run_start)
    window_counts = [buckets.get(index, 0.0) for index in range(first_window, last_window + 1)]
    ordered = sorted(window_counts)
    p99_index = math.ceil(0.99 * len(ordered)) - 1
    return max(window_counts), ordered[p99_index]


def _add_derived_throughput_dimensions(envelope: Envelope, raw: dict[str, Any]) -> None:
    values: dict[str, float | None] = dict.fromkeys(DERIVED_THROUGHPUT_SPEED_FIELDS)
    reasons: dict[str, str] = {}

    max_concurrency = _finite_number(raw.get("max_concurrency"))
    percentiles = (
        ("p50", _first_number(raw, "p50_tpot_ms", "median_tpot_ms")),
        ("p90", _finite_number(raw.get("p90_tpot_ms"))),
        ("p99", _finite_number(raw.get("p99_tpot_ms"))),
    )
    for percentile, tpot_ms in percentiles:
        field = f"potential_output_tok_s_{percentile}"
        if max_concurrency is None or max_concurrency <= 0:
            reasons[f"speed.{field}"] = "max_concurrency_unavailable"
        elif tpot_ms is None or tpot_ms <= 0:
            reasons[f"speed.{field}"] = f"tpot_{percentile}_unavailable"
        else:
            values[field] = max_concurrency * 1000.0 / tpot_ms

    requests = _request_timing_data(raw)
    duration = _finite_number(raw.get("duration"))
    if requests is None:
        reasons["speed.measured_mean_concurrency"] = "per_request_timing_arrays_unavailable"
        reasons["speed.peak_output_tok_s_1s_max"] = "per_request_timing_arrays_unavailable"
        reasons["speed.peak_output_tok_s_1s_p99"] = "per_request_timing_arrays_unavailable"
    else:
        if duration is None or duration <= 0:
            reasons["speed.measured_mean_concurrency"] = "duration_unavailable"
        else:
            mean_concurrency = sum(latency for _start, latency, _offsets, _length in requests) / duration
            values["measured_mean_concurrency"] = mean_concurrency

        window_rates = _window_rates(requests)
        if window_rates is None:
            reasons["speed.peak_output_tok_s_1s_max"] = "no_output_token_events"
            reasons["speed.peak_output_tok_s_1s_p99"] = "no_output_token_events"
        else:
            values["peak_output_tok_s_1s_max"], values["peak_output_tok_s_1s_p99"] = window_rates

    request_rate = _finite_number(raw.get("request_rate"))
    if request_rate is not None and request_rate > 0:
        for percentile, tpot_ms in percentiles:
            field = f"potential_output_tok_s_{percentile}_at_mean_concurrency"
            mean_concurrency = values["measured_mean_concurrency"]
            if mean_concurrency is None:
                reasons[f"speed.{field}"] = "measured_mean_concurrency_unavailable"
            elif tpot_ms is None or tpot_ms <= 0:
                reasons[f"speed.{field}"] = f"tpot_{percentile}_unavailable"
            else:
                values[field] = mean_concurrency * 1000.0 / tpot_ms
    else:
        reason = "request_rate_unavailable" if raw.get("request_rate") is None else "not_a_finite_rate_run"
        for percentile, _tpot_ms in percentiles:
            reasons[f"speed.potential_output_tok_s_{percentile}_at_mean_concurrency"] = reason

    output_throughput = _finite_number(raw.get("output_throughput"))
    potential_p50 = values["potential_output_tok_s_p50"]
    if output_throughput is None or potential_p50 is None or potential_p50 <= 0:
        reasons["speed.potential_efficiency"] = "output_throughput_or_potential_p50_unavailable"
    else:
        values["potential_efficiency"] = output_throughput / potential_p50

    for field in DERIVED_THROUGHPUT_SPEED_FIELDS:
        if values[field] is None:
            reasons.setdefault(f"speed.{field}", "derived_input_unavailable")

    dimensions = cast("CampaignDimensions", dict(envelope.get("campaign_dimensions") or {}))
    speed = dict(dimensions.get("speed") or {})
    speed.update(values)
    dimensions["speed"] = speed
    envelope["campaign_dimensions"] = dimensions

    null_reasons = dict(envelope.get("dimension_null_reasons") or {})
    for field, value in values.items():
        reason_path = f"speed.{field}"
        if value is None:
            null_reasons[reason_path] = reasons[reason_path]
        else:
            null_reasons.pop(reason_path, None)
    envelope["dimension_null_reasons"] = null_reasons
