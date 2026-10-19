"""NVIDIA cache-tier and MoE-offload cells -> envelope v1."""

from __future__ import annotations

import datetime
import math
from collections.abc import Mapping
from typing import Any

import pyarrow as pa

from mlx_benchmarks.converters.base import ConverterContext, apply_optional_fields
from mlx_benchmarks.dataset_schema import STANDARD_CACHE_FIELD_DIMENSIONS, STANDARD_CACHE_FIELD_TYPES
from mlx_benchmarks.envelope import CampaignDimensions, Envelope, Result, System

KV_MACHINE = "RTX PRO 6000 Max-Q 96GB"
MOE_MACHINE = "RTX 4080 SUPER 16GB"
KV_ENGINE = "vLLM"
MOE_ENGINE = "llama.cpp"
UNKNOWN_DIMENSION_REASON = "NOT_CAPTURED"

_KV_RESULTS: dict[str, tuple[str, str]] = {
    "ttft_first_round_p50_ms": ("ttft_first_round_p50_ms", "ms"),
    "ttft_later_rounds_p50_ms": ("ttft_later_rounds_p50_ms", "ms"),
    "ttft_later_rounds_p99_ms": ("ttft_later_rounds_p99_ms", "ms"),
    "tpot_p50_ms_per_token": ("tpot_p50_ms_per_token", "ms/token"),
    "tpot_p99_ms_per_token": ("tpot_p99_ms_per_token", "ms/token"),
    "throughput_tokens_per_s": ("throughput_tokens_per_s", "tok/s"),
    "requests_per_s": ("requests_per_s", "req/s"),
    "prefix_cache_hit_rate": ("prefix_cache_hit_rate", "fraction"),
    "external_prefix_cache_hit_rate": ("external_prefix_cache_hit_rate", "fraction"),
    "kv_cache_usage_peak": ("kv_cache_usage_peak", "%"),
    "host_ram_peak_gib": ("host_ram_peak_gib", "GiB"),
    "preemptions": ("preemptions", "requests"),
}
_MOE_RESULTS: dict[str, tuple[str, str]] = {
    "decode_tok_s": ("throughput_output_toks_per_s", "tok/s"),
    "ttft_p50_ms": ("ttft_p50_ms", "ms"),
    "ttft_p99_ms": ("ttft_p99_ms", "ms"),
    "vram_used_gib": ("gpu_vram_used_gib", "GiB"),
    "guest_ram_peak_gib": ("host_ram_peak_gib", "GiB"),
}


def _typed_value(value: Any, arrow_type: pa.DataType, field: str) -> Any:
    if value is None or value == "":
        return None
    if pa.types.is_boolean(arrow_type):
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        raise ValueError(f"{field} must be true or false")
    if pa.types.is_integer(arrow_type):
        if isinstance(value, bool):
            raise ValueError(f"{field} must be an integer")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be an integer") from exc
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"{field} must be a finite integer")
        return int(number)
    if pa.types.is_floating(arrow_type):
        if isinstance(value, bool):
            raise ValueError(f"{field} must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"{field} must be finite")
        return number
    if pa.types.is_string(arrow_type):
        return str(value)
    raise TypeError(f"unsupported campaign dimension type for {field}: {arrow_type}")


def _cache_dimensions(
    raw: Mapping[str, Any],
    *,
    machine: str,
    accelerator_model: str,
    accelerator_memory_gb: float,
    engine: str,
    engine_version: str,
    engine_build: str | None,
    engine_commit: str | None,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[CampaignDimensions, dict[str, str]]:
    dimensions: dict[str, dict[str, Any]] = {}
    reasons: dict[str, str] = {}
    overrides = overrides or {}
    for name, (category, field) in STANDARD_CACHE_FIELD_DIMENSIONS.items():
        raw_value = overrides[name] if name in overrides else raw.get(name)
        value = _typed_value(raw_value, STANDARD_CACHE_FIELD_TYPES[name], name)
        dimensions.setdefault(category, {})[field] = value
        if value is None:
            reasons[f"{category}.{field}"] = UNKNOWN_DIMENSION_REASON

    dimensions.setdefault("hardware", {}).update(
        {
            "machine": machine,
            "accelerator_model": accelerator_model,
            "accelerator_memory_gb": accelerator_memory_gb,
        }
    )
    software = dimensions.setdefault("software", {})
    software["engine"] = engine
    software["engine_version"] = engine_version
    if engine_build is not None:
        software["engine_build"] = engine_build
    if engine_commit is not None:
        software["engine_commit"] = engine_commit
    return dimensions, reasons  # type: ignore[return-value]


def _result_value(value: Any, field: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _results(
    raw: Mapping[str, Any],
    metric_map: Mapping[str, tuple[str, str]],
    *,
    name: str,
    tags: dict[str, str],
) -> list[Result]:
    results: list[Result] = []
    for source, (metric, unit) in metric_map.items():
        value = _result_value(raw.get(source), source)
        if value is not None:
            results.append(
                {
                    "name": name,
                    "metric": metric,
                    "value": value,
                    "unit": unit,
                    "tags": dict(tags),
                }
            )
    if not results:
        raise ValueError(f"{name} has no numeric benchmark metrics")
    return results


def _base_envelope(
    raw: Mapping[str, Any],
    ctx: ConverterContext,
    *,
    engine: str,
    version: str,
    dimensions: CampaignDimensions,
    reasons: dict[str, str],
    name: str,
    tags: dict[str, str],
    metrics: Mapping[str, tuple[str, str]],
    profile: str,
) -> Envelope:
    timestamp = ctx.timestamp_override or datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    system: System = dict(ctx.system or {})  # type: ignore[assignment]
    system["engine"] = {"name": engine, "version": version}
    power_limit = _result_value(raw.get("gpu_power_limit_w"), "gpu_power_limit_w")
    if power_limit is not None:
        system["power_limit_w"] = power_limit
    envelope: Envelope = {
        "schema_version": "1",
        "timestamp": timestamp,
        "git_sha": ctx.git_sha,
        "trigger": ctx.trigger,
        "suite": ctx.suite,
        "model": ctx.model,
        "campaign": {"cell_id": name, "profile": profile},
        "system": system,
        "results": _results(raw, metrics, name=name, tags=tags),
        "errors": [],
        "campaign_dimensions": dimensions,
        "dimension_null_reasons": reasons,
    }
    return apply_optional_fields(envelope, ctx)


def _string(raw: Mapping[str, Any], field: str, default: str = "") -> str:
    value = raw.get(field)
    return value.strip() if isinstance(value, str) else default


def _version(raw: Mapping[str, Any], ctx: ConverterContext) -> str:
    value = _string(raw, "engine_version") or ctx.engine_version
    if not value:
        raise ValueError("engine version is required in the result line or --engine-version")
    return value


class KvOffloadConverter:
    kind = "kv-offload"
    engine_name = KV_ENGINE
    machine = KV_MACHINE

    def build_envelope(self, raw: dict[str, Any], ctx: ConverterContext) -> Envelope:
        arm = _string(raw, "arm")
        workload = _string(raw, "workload")
        if not arm or not workload:
            raise ValueError("KVO_RESULT requires arm and workload")
        version = _version(raw, ctx)
        dimensions, reasons = _cache_dimensions(
            raw,
            machine=KV_MACHINE,
            accelerator_model="RTX PRO 6000 Max-Q",
            accelerator_memory_gb=96.0,
            engine=KV_ENGINE,
            engine_version=version,
            engine_build=ctx.engine_build,
            engine_commit=ctx.engine_commit,
        )
        ambient_temp = _typed_value(raw.get("ambient_temp_f"), pa.float64(), "ambient_temp_f")
        environment = dimensions.setdefault("environment", {})
        environment["ambient_temp_f"] = ambient_temp
        if ambient_temp is None:
            reasons["environment.ambient_temp_f"] = UNKNOWN_DIMENSION_REASON
        power_limit = _typed_value(raw.get("gpu_power_limit_w"), pa.float64(), "gpu_power_limit_w")
        dimensions.setdefault("hardware", {})["power_cap_w"] = power_limit
        if power_limit is None:
            reasons["hardware.power_cap_w"] = UNKNOWN_DIMENSION_REASON
        fan_percent = _result_value(raw.get("gpu_fan_percent"), "gpu_fan_percent")
        environment["fan_duty_pct"] = {"gpu": fan_percent} if fan_percent is not None else None
        if fan_percent is None:
            reasons["environment.fan_duty_pct"] = UNKNOWN_DIMENSION_REASON
        run = dimensions.setdefault("run", {})
        run["moe_cpu_offload"] = None
        run["moe_cpu_layers"] = None
        reasons["run.moe_cpu_offload"] = "NOT_APPLICABLE_ENGINE"
        reasons["run.moe_cpu_layers"] = "NOT_APPLICABLE_ENGINE"
        tags = {"arm": arm, "workload": workload}
        for field in ("seed", "run_id"):
            if raw.get(field) is not None:
                tags[field] = str(raw[field])
        name = f"{arm}/{workload}"
        return _base_envelope(
            raw,
            ctx,
            engine=KV_ENGINE,
            version=version,
            dimensions=dimensions,
            reasons=reasons,
            name=name,
            tags=tags,
            metrics=_KV_RESULTS,
            profile=name,
        )


def _moe_mode(arm: str, n_cpu_moe: int | None) -> str:
    normalized = arm.casefold().replace("_", "-")
    if "n-cpu-moe" in normalized:
        return "ncmoe"
    if "cpu-moe" in normalized:
        return "cpumoe"
    return "ncmoe" if n_cpu_moe and n_cpu_moe > 0 else "none"


class MoeOffloadConverter:
    kind = "moe-offload"
    engine_name = MOE_ENGINE
    machine = MOE_MACHINE

    def build_envelope(self, raw: dict[str, Any], ctx: ConverterContext) -> Envelope:
        arm = _string(raw, "arm")
        profile = _string(raw, "profile")
        if not arm or not profile:
            raise ValueError("MoE CSV row requires profile and arm")
        version = _version(raw, ctx)
        cache_ram = _typed_value(raw.get("cache_ram_mib"), pa.int64(), "cache_ram_mib")
        n_cpu_moe = _typed_value(raw.get("n_cpu_moe"), pa.int64(), "n_cpu_moe")
        output_tokens = _typed_value(raw.get("output_tokens"), pa.int64(), "output_tokens")
        guest_ram = _typed_value(raw.get("guest_ram_peak_gib"), pa.float64(), "guest_ram_peak_gib")
        moe_mode = _moe_mode(arm, n_cpu_moe)
        moe_layers = n_cpu_moe if moe_mode == "ncmoe" else 0 if moe_mode == "none" else None
        cache_backend = None if cache_ram is None else "llamacpp_cache_ram" if cache_ram > 0 else "none"
        overrides = {
            "host_ram_limit_gib": None,
            "prefix_caching": None,
            "kv_offload_backend": cache_backend,
            "kv_offload_cpu_gib": None if cache_ram is None else cache_ram / 1024,
            "kv_offload_disk_gib": None if cache_ram is None else 0.0,
            "weight_cpu_offload_gib": None,
            "llamacpp_cache_ram_mib": cache_ram,
            "gpu_memory_utilization": None,
            "max_num_batched_tokens": None,
            "gpu_kv_capacity_tokens": None,
            "cache_workload": "none",
            "num_users": None,
            "num_rounds": None,
            "shared_prompt_tokens": None,
            "user_history_tokens": None,
            "answer_tokens": output_tokens,
            "request_rate": None,
            "prefix_cache_hit_rate": None,
            "external_prefix_cache_hit_rate": None,
            "kv_cache_usage_peak": None,
            "kv_block_idle_before_evict_p50_s": None,
            "host_ram_peak_gib": guest_ram,
            "preemptions": None,
            "ttft_first_round_p50_ms": None,
            "ttft_later_rounds_p50_ms": None,
            "ttft_later_rounds_p99_ms": None,
        }
        dimensions, reasons = _cache_dimensions(
            raw,
            machine=MOE_MACHINE,
            accelerator_model="RTX 4080 SUPER",
            accelerator_memory_gb=16.0,
            engine=MOE_ENGINE,
            engine_version=version,
            engine_build=ctx.engine_build,
            engine_commit=ctx.engine_commit,
            overrides=overrides,
        )
        run = dimensions.setdefault("run", {})
        run["prompt_tokens"] = _typed_value(raw.get("prompt_tokens"), pa.int64(), "prompt_tokens")
        run["parallel_slots"] = _typed_value(raw.get("concurrency"), pa.int64(), "concurrency")
        run["moe_cpu_offload"] = moe_mode
        run["moe_cpu_layers"] = moe_layers
        if raw.get("prompt_tokens") in (None, ""):
            reasons["run.prompt_tokens"] = UNKNOWN_DIMENSION_REASON
        if raw.get("concurrency") in (None, ""):
            reasons["run.parallel_slots"] = UNKNOWN_DIMENSION_REASON
        if moe_layers is None:
            reasons["run.moe_cpu_layers"] = UNKNOWN_DIMENSION_REASON
        tags = {
            field: str(raw[field])
            for field in ("profile", "arm", "prompt_tokens", "output_tokens", "concurrency", "n_cpu_moe")
            if raw.get(field) not in (None, "")
        }
        name = f"{profile}/{arm}"
        return _base_envelope(
            raw,
            ctx,
            engine=MOE_ENGINE,
            version=version,
            dimensions=dimensions,
            reasons=reasons,
            name=name,
            tags=tags,
            metrics=_MOE_RESULTS,
            profile=profile,
        )


def engine_metadata(raw: Mapping[str, Any], ctx: ConverterContext, fallback: str) -> dict[str, str]:
    """Return the target engine facts used by published metadata and the run index."""
    version = _version(raw, ctx)
    values = {"engine": fallback, "engine_version": version}
    build = _string(raw, "engine_build") or ctx.engine_build
    commit = _string(raw, "engine_commit") or ctx.engine_commit
    if build:
        values["engine_build"] = build
    if commit:
        values["engine_commit"] = commit
    return values
