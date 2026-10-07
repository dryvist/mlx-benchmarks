"""llama-server per-request records -> envelope v1 converter.

Input is a list of one JSON object per request, as recorded by a driver that
posts chat completions to ``llama-server`` and keeps the server's own
``timings`` and ``usage`` blocks next to the full ``/props`` and ``/v1/models``
answers. One envelope covers one series (one build, one model file, one power
limit); split a multi-series file before converting.

Records with ``concurrency`` 0 are warm-up requests. They never become result
rows; their timings ride on every result row as ``warmup_*`` tags so the
discarded run stays visible.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import PurePosixPath
from typing import Any

from mlx_benchmarks.converters.base import (
    ConverterContext,
    apply_optional_fields,
    require_gpu_power_limit,
)
from mlx_benchmarks.envelope import CampaignDimensions, Envelope, Result, System

log = logging.getLogger(__name__)

# timings key -> (metric, unit, result column that carries the same number)
_RATE_METRICS: tuple[tuple[str, str, str, str], ...] = (
    ("predicted_per_second", "throughput_output_toks_per_s", "tok/s", "decode_tokens_per_second"),
    ("prompt_per_second", "throughput_prompt_toks_per_s", "tok/s", "prompt_tokens_per_second"),
)
_QUANT_RE = re.compile(r"-((?:UD-)?I?Q\d[A-Za-z0-9_]*|F16|BF16)\.gguf$")
_BUILD_RE = re.compile(r"^(b\d+)(?:-([0-9a-f]+))?$")
_GB = 1_000_000_000


def _timestamp(start_utc: str) -> str:
    """``2026-10-07T00:22:15.390Z`` -> ``2026-10-07T00:22:15Z``."""
    return re.sub(r"\.\d+(?=Z$)", "", start_utc)


def _only(values: set[Any]) -> Any:
    """The single value every record agrees on, else ``None``."""
    return next(iter(values)) if len(values) == 1 else None


def _server_system(ctx: ConverterContext, records: list[dict[str, Any]]) -> System:
    """The serving host's facts, never the machine that ran the publisher.

    The records come from a remote server, so ``ctx.system`` (detected on the
    publishing machine: OS, CPU, memory, hostname, kernel) describes the wrong
    box and is dropped; the three required fields stay explicit nulls. Only the
    accelerator declarations the caller made for the server are kept.
    """
    declared = ctx.system or {}
    system: dict[str, Any] = {"os": None, "chip": None, "memory_gb": None}
    for key in ("gpu", "engine", "container"):
        if key in declared:
            system[key] = declared[key]
    require_gpu_power_limit(declared)
    power = _runtime_power_limit_w(ctx)
    if power is not None:
        system["power_limit_w"] = power
    build = _BUILD_RE.match(str(_first_props(records).get("build_info") or ""))
    if build and "engine" not in system:
        system["engine"] = {"name": "llama.cpp", "version": build.group(1)}
    return system  # type: ignore[return-value]


def _runtime_power_limit_w(ctx: ConverterContext) -> float | None:
    """Use the GPU host's run-time read, never per-request annotations."""
    power = (ctx.system or {}).get("power_limit_w")
    return float(power) if power is not None else None


def _first_props(records: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r["server_props"] for r in records if isinstance(r.get("server_props"), dict)), {})


def _sanitized_props(record: dict[str, Any]) -> dict[str, Any]:
    """The server's own answers, minus where the file lived on disk."""
    props = dict(record.get("server_props") or {})
    if isinstance(props.get("model_path"), str):
        props["model_path"] = PurePosixPath(props["model_path"]).name
    return {"server_props": props, "served_models": record.get("served_models") or []}


class LlamaCppServerConverter:
    kind = "llamacpp-server"

    def build_envelope(self, raw: Any, ctx: ConverterContext) -> Envelope:
        records = [r for r in raw if isinstance(r, dict)] if isinstance(raw, list) else []
        series = {r.get("series") for r in records} - {None}
        if len(series) > 1:
            raise ValueError(f"one envelope covers one series; got {sorted(map(str, series))}")
        measured = [r for r in records if r.get("concurrency")]
        warmups = [r for r in records if r.get("concurrency") == 0]
        if not measured:
            log.warning("llama-server records hold no measured requests")

        system = _server_system(ctx, records)

        starts = [r["start_utc"] for r in measured if r.get("start_utc")]
        timestamp = ctx.timestamp_override or _timestamp(min(starts, default="1970-01-01T00:00:00Z"))
        envelope: Envelope = {
            "schema_version": "1",
            "timestamp": timestamp,
            "git_sha": ctx.git_sha,
            "trigger": ctx.trigger,
            "suite": ctx.suite,
            "model": ctx.model,
            "system": system,
            "results": self._iter_results(measured, warmups, ctx),
            "errors": [],
            "cell_status": "success",
        }
        first = (measured or records or [{}])[0]
        quantization = _model_meta(first).get("quantization")
        if quantization:
            envelope["quantization"] = quantization
        temperature = _only({r.get("temperature") for r in measured} - {None})
        if temperature is not None:
            envelope["gen_kwargs"] = {"temperature": float(temperature)}
            n_predict = _only({r.get("n_predict") for r in measured} - {None})
            if n_predict is not None:
                envelope["gen_kwargs"]["max_gen_toks"] = int(n_predict)
        thinking = _only({r.get("thinking") for r in measured} - {None})
        if isinstance(thinking, bool) and ctx.reasoning_effort is None:
            envelope["reasoning_effort"] = "on" if thinking else "off"
        if series:
            campaign: dict[str, str] = {"cell_id": str(next(iter(series)))}
            for key, tag in (("id", "campaign_id"), ("profile", "profile")):
                if tag in ctx.extra_tags:
                    campaign[key] = ctx.extra_tags[tag]
            envelope["campaign"] = campaign  # type: ignore[typeddict-item]
        apply_optional_fields(envelope, ctx)
        if records:  # nothing to derive from an empty file; declared dimensions pass through as given
            dimensions, reasons = _dimensions(records, measured, ctx, timestamp)
            envelope["campaign_dimensions"] = dimensions
            envelope["dimension_null_reasons"] = reasons
        return envelope

    def _iter_results(
        self, measured: list[dict[str, Any]], warmups: list[dict[str, Any]], ctx: ConverterContext
    ) -> list[Result]:
        base = {k: str(v) for k, v in ctx.extra_tags.items()}
        for index, warm in enumerate(warmups):
            suffix = "" if len(warmups) == 1 else f"_{index}"
            timings = warm.get("timings") or {}
            if warm.get("start_utc"):
                base[f"warmup_start_utc{suffix}"] = str(warm["start_utc"])
            for key in ("predicted_per_second", "prompt_per_second", "predicted_n", "predicted_ms"):
                if isinstance(timings.get(key), int | float):
                    base[f"warmup_{key}{suffix}"] = str(timings[key])

        results: list[Result] = []
        by_width: dict[int, list[dict[str, Any]]] = {}
        for record in sorted(measured, key=lambda r: (r["concurrency"], r.get("request_index", 0))):
            by_width.setdefault(int(record["concurrency"]), []).append(record)
            tags = {**base, **_request_tags(record)}
            timings = record.get("timings") or {}
            for key, metric, unit, column in _RATE_METRICS:
                if isinstance(timings.get(key), int | float):
                    result: Result = {
                        "name": "llamacpp_server",
                        "metric": metric,
                        "value": float(timings[key]),
                        "unit": unit,
                        "tags": dict(tags),
                    }
                    if column == "decode_tokens_per_second" and isinstance(
                        timings.get("predicted_ms"), int | float
                    ):
                        result["duration_seconds"] = float(timings["predicted_ms"]) / 1000.0
                    result[column] = float(timings[key])  # type: ignore[literal-required]
                    results.append(result)
            if isinstance(timings.get("prompt_ms"), int | float):
                results.append(
                    {
                        "name": "llamacpp_server",
                        "metric": "prompt_eval_ms",
                        "value": float(timings["prompt_ms"]),
                        "unit": "ms",
                        "first_token_latency_ms": float(timings["prompt_ms"]),
                        "tags": {**tags, "note": "server-side prompt evaluation; excludes client path"},
                    }
                )
        slots = _only({(r.get("server_props") or {}).get("total_slots") for r in measured} - {None})
        for width, members in by_width.items():
            if slots is not None and width > slots:
                continue  # queued requests do not decode together; a sum would overstate
            rates = [m["timings"]["predicted_per_second"] for m in members if "timings" in m]
            if rates:
                results.append(
                    {
                        "name": "llamacpp_server",
                        "metric": "throughput_aggregate_output_toks_per_s",
                        "value": float(sum(rates)),
                        "unit": "tok/s",
                        "tags": {
                            **base,
                            "phase": "concurrent",
                            "concurrency": str(width),
                            "aggregation": "sum_of_per_request_decode_rates",
                            "n_requests": str(len(rates)),
                        },
                    }
                )
        return results


def _request_tags(record: dict[str, Any]) -> dict[str, str]:
    tags: dict[str, str] = {}
    scalar = (
        "series",
        "power_limit_w",
        "concurrency",
        "request_index",
        "start_utc",
        "n_predict",
        "temperature",
        "thinking",
        "prompt",
        "client",
        "finish_reason",
    )
    for key in scalar:
        value = record.get(key)
        if value is not None:
            tags[key] = str(value).lower() if isinstance(value, bool) else str(value)
    if "n_predict" in tags:
        tags["max_tokens"] = tags["n_predict"]
    tags["phase"] = "concurrent" if int(record.get("concurrency") or 0) > 1 else "sequential"
    for key, value in (record.get("timings") or {}).items():
        if isinstance(value, int | float):
            tags[f"timings_{key}"] = str(value)
    usage = record.get("usage") or {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if isinstance(usage.get(key), int):
            tags[f"usage_{key}"] = str(usage[key])
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if isinstance(cached, int):
        tags["usage_cached_tokens"] = str(cached)
    tags["server_json"] = json.dumps(_sanitized_props(record), sort_keys=True, separators=(",", ":"))
    return tags


def _model_meta(record: dict[str, Any]) -> dict[str, Any]:
    props = record.get("server_props") or {}
    models = record.get("served_models") or [{}]
    served = models[0] if isinstance(models[0], dict) else {}
    path = props.get("model_path")
    quant = _QUANT_RE.search(path) if isinstance(path, str) else None
    out: dict[str, Any] = {"id": served.get("id"), "meta": served.get("meta") or {}}
    if quant:
        out["quantization"] = quant.group(1)
    return out


def _dimensions(
    records: list[dict[str, Any]],
    measured: list[dict[str, Any]],
    ctx: ConverterContext,
    timestamp: str,
) -> tuple[CampaignDimensions, dict[str, str]]:
    """Everything the records carry, laid over what the caller declared."""
    first = (measured or records or [{}])[0]
    props = first.get("server_props") or {}
    model = _model_meta(first)
    meta = model["meta"]
    declared: dict[str, Any] = dict(ctx.campaign_dimensions or {})
    dims: dict[str, dict[str, Any]] = {group: dict(values) for group, values in declared.items()}
    reasons: dict[str, str] = dict(ctx.dimension_null_reasons or {})

    def put(group: str, key: str, value: Any, reason: str = "NOT_RECORDED") -> None:
        fields = dims.setdefault(group, {})
        if value is not None:
            fields[key] = value
            reasons.pop(f"{group}.{key}", None)
        elif fields.get(key) is None:
            fields[key] = None
            reasons.setdefault(f"{group}.{key}", reason)

    power = _runtime_power_limit_w(ctx)
    put("hardware", "power_cap_w", power)

    build = _BUILD_RE.match(str(props.get("build_info") or ""))
    put("software", "engine", "llama.cpp")
    put("software", "engine_version", build.group(1) if build else props.get("build_info"))
    put("software", "engine_commit", build.group(2) if build else None)

    put("model", "id", model["id"])
    put("model", "hf_repo", model["id"])
    put("model", "quantization", model.get("quantization"))
    params, size = meta.get("n_params"), meta.get("size")
    put("model", "total_parameters", params if isinstance(params, int) else None)
    put("model", "file_size_gb", round(size / _GB, 3) if isinstance(size, int) else None)
    if isinstance(params, int) and isinstance(size, int) and params:
        put("model", "bits_per_weight", round(size * 8 / params, 3))
    put("model", "native_max_context_tokens", meta.get("n_ctx_train"))

    put("run", "allocated_context_tokens", props.get("n_ctx"))
    put("run", "parallel_slots", props.get("total_slots"))
    put("run", "temperature", _only({r.get("temperature") for r in measured} - {None}))
    thinking = _only({r.get("thinking") for r in measured} - {None})
    put("run", "thinking", thinking if isinstance(thinking, bool) else None)
    put("run", "output_tokens", _only({r.get("n_predict") for r in measured} - {None}))
    put(
        "run",
        "prompt_tokens",
        _only({(r.get("usage") or {}).get("prompt_tokens") for r in measured} - {None}),
    )
    put("run", "concurrent_agents", None, "VARIES_PER_ROW")
    put("run", "repeats", 1 if measured else None)
    put("run", "warm_cold", "warm" if any(r.get("concurrency") == 0 for r in records) else None)

    put("provenance", "timestamp_utc", timestamp)
    put("provenance", "run_id", _only({r.get("series") for r in records} - {None}))
    put("provenance", "schema_version", "1")
    return dims, reasons  # type: ignore[return-value]
