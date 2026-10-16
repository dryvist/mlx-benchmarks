"""Optional power enrichment from the Prometheus-compatible range query API."""

from __future__ import annotations

import datetime
import json
import math
from collections.abc import Mapping
from itertools import pairwise
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from mlx_benchmarks.envelope import Envelope

POWER_FIELDS = (
    "power_avg_w",
    "power_max_w",
    "energy_j_per_output_token",
    "tokens_per_second_per_watt",
)
METRICS_URL_ENV = "MLX_BENCH_METRICS_URL"
QUERY_STEP_SECONDS = 1


class PowerMetricsError(RuntimeError):
    """Raised when a configured telemetry query cannot be read or interpreted."""


def enrich_power_metrics(envelope: Envelope, metrics_url: str | None = None) -> None:
    """Fill missing power dimensions and their explicit null reasons in place."""
    import os

    url = metrics_url or os.environ.get(METRICS_URL_ENV)
    dimensions = envelope.setdefault("campaign_dimensions", {})
    resource = dimensions.setdefault("resource", {})
    reasons = envelope.setdefault("dimension_null_reasons", {})

    if not url:
        _null_missing(resource, reasons, "METRICS_URL_UNSET")
        return

    window = _run_window(envelope)
    if window is None:
        _null_missing(resource, reasons, "RUN_WINDOW_UNAVAILABLE")
        return
    start, end = window

    samples: list[tuple[float, float]] = []
    if any(
        resource.get(field) is None for field in ("power_avg_w", "power_max_w", "energy_j_per_output_token")
    ):
        samples = _query_power(url, _metric_query(envelope), start, end)
        if samples:
            if resource.get("power_avg_w") is None:
                resource["power_avg_w"] = sum(value for _, value in samples) / len(samples)
                reasons.pop("resource.power_avg_w", None)
            if resource.get("power_max_w") is None:
                resource["power_max_w"] = max(value for _, value in samples)
                reasons.pop("resource.power_max_w", None)
        else:
            for field in ("power_avg_w", "power_max_w"):
                if resource.get(field) is None:
                    reasons.setdefault(f"resource.{field}", "NO_POWER_SAMPLES")

    _derive_energy(resource, reasons, samples, _output_token_count(envelope), end - start)

    for field in POWER_FIELDS:
        resource.setdefault(field, None)
        if resource.get(field) is not None:
            reasons.pop(f"resource.{field}", None)
        elif f"resource.{field}" not in reasons:
            reasons[f"resource.{field}"] = "POWER_METRIC_UNAVAILABLE"


def _null_missing(resource: dict[str, Any], reasons: dict[str, str], reason: str) -> None:
    for field in POWER_FIELDS:
        if resource.get(field) is None:
            resource[field] = None
            reasons.setdefault(f"resource.{field}", reason)


def _metric_query(envelope: Mapping[str, Any]) -> str:
    system = envelope.get("system") or {}
    os_name = str(system.get("os") or "").casefold()
    metric = "macmon_sys_power_watts" if "macos" in os_name else "DCGM_FI_DEV_POWER_USAGE"
    return f"sum({metric})"


def _run_window(envelope: Mapping[str, Any]) -> tuple[float, float] | None:
    dimensions = envelope.get("campaign_dimensions") or {}
    run = dimensions.get("run") or {}
    started_at = run.get("started_at")
    ended_at = run.get("ended_at")
    try:
        if started_at and ended_at:
            start = _timestamp_seconds(started_at)
            end = _timestamp_seconds(ended_at)
            return (start, end) if end > start else None

        duration = run.get("phase_duration_seconds")
        if duration is None:
            durations = [
                result.get("duration_seconds")
                for result in envelope.get("results", [])
                if isinstance(result.get("duration_seconds"), int | float) and result["duration_seconds"] > 0
            ]
            duration = max(durations, default=None)
        if not isinstance(duration, int | float) or duration <= 0:
            return None

        if started_at:
            start = _timestamp_seconds(started_at)
            return start, start + float(duration)
        if ended_at:
            end = _timestamp_seconds(ended_at)
            return end - float(duration), end
        start = _timestamp_seconds(envelope["timestamp"])
        return start, start + float(duration)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _timestamp_seconds(value: str) -> float:
    if not isinstance(value, str):
        raise TypeError("timestamp must be a string")
    parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must have a timezone")
    return parsed.astimezone(datetime.UTC).timestamp()


def _query_power(url: str, query: str, start: float, end: float) -> list[tuple[float, float]]:
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise PowerMetricsError("metrics URL is invalid") from exc
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise PowerMetricsError("metrics URL must use HTTP or HTTPS")
    path = parts.path.rstrip("/")
    if path.endswith("/api/v1/query_range"):
        pass
    elif path.endswith("/api/v1"):
        path += "/query_range"
    else:
        path += "/api/v1/query_range"
    params = urlencode(
        {
            "query": query,
            "start": f"{start:.3f}",
            "end": f"{end:.3f}",
            "step": f"{QUERY_STEP_SECONDS}s",
        }
    )
    query_string = "&".join(value for value in (parts.query, params) if value)
    request = Request(urlunsplit((parts.scheme, parts.netloc, path, query_string, "")))
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read())
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise PowerMetricsError(f"metrics query failed ({type(exc).__name__})") from exc

    if not isinstance(payload, Mapping) or payload.get("status") != "success":
        raise PowerMetricsError("metrics API returned a non-success status")
    data = payload.get("data")
    result = data.get("result") if isinstance(data, Mapping) else None
    if not isinstance(result, list):
        raise PowerMetricsError("metrics API response did not contain a range result")

    samples: list[tuple[float, float]] = []
    for series in result:
        values = series.get("values") if isinstance(series, Mapping) else None
        if not isinstance(values, list):
            continue
        for point in values:
            if not isinstance(point, list) or len(point) < 2:
                continue
            try:
                timestamp, watts = float(point[0]), float(point[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(timestamp) and math.isfinite(watts) and watts >= 0:
                samples.append((timestamp, watts))
    return sorted(samples)


def _output_token_count(envelope: Mapping[str, Any]) -> int | None:
    run = (envelope.get("campaign_dimensions") or {}).get("run") or {}
    value = run.get("measured_output_tokens")
    return value if isinstance(value, int) and value > 0 else None


def _derive_energy(
    resource: dict[str, Any],
    reasons: dict[str, str],
    samples: list[tuple[float, float]],
    output_tokens: int | None,
    duration: float,
) -> None:
    avg_power = resource.get("power_avg_w")
    if output_tokens is None:
        for field in ("energy_j_per_output_token", "tokens_per_second_per_watt"):
            if resource.get(field) is None:
                reasons.setdefault(f"resource.{field}", "OUTPUT_TOKENS_UNAVAILABLE")
        return
    if duration <= 0 or not isinstance(avg_power, int | float) or avg_power <= 0:
        for field in ("energy_j_per_output_token", "tokens_per_second_per_watt"):
            if resource.get(field) is None:
                reasons.setdefault(f"resource.{field}", "POWER_METRIC_UNAVAILABLE")
        return

    if resource.get("energy_j_per_output_token") is None:
        if len(samples) < 2:
            reasons.setdefault("resource.energy_j_per_output_token", "INSUFFICIENT_POWER_SAMPLES")
        else:
            energy_j = sum(
                (left[1] + right[1]) * (right[0] - left[0]) / 2 for left, right in pairwise(samples)
            )
            resource["energy_j_per_output_token"] = energy_j / output_tokens
            reasons.pop("resource.energy_j_per_output_token", None)
    if resource.get("tokens_per_second_per_watt") is None:
        resource["tokens_per_second_per_watt"] = output_tokens / duration / float(avg_power)
        reasons.pop("resource.tokens_per_second_per_watt", None)
