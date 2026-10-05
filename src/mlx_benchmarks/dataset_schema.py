"""Stable flat Parquet schema shared by every published run shard."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

import pyarrow as pa

TAG_KEYS = (
    "aggregation",
    "answered_rate",
    "arm",
    "backend",
    "cache_busting",
    "campaign",
    "campaign_id",
    "caveat",
    "caveat2",
    "caveats",
    "cell",
    "cell_id",
    "check",
    "check_rc",
    "clean_reset",
    "clustered",
    "completed_requests",
    "concurrency",
    "configured_window_tokens",
    "context",
    "context_len",
    "context_tokens_actual",
    "context_tokens_target",
    "cumulative_tok_s_max",
    "cumulative_tok_s_min",
    "decode_tok_s_max",
    "decode_tok_s_min",
    "dedicated",
    "degraded",
    "experiment",
    "failure_bad_json_args",
    "failure_bad_response_body",
    "failure_empty_function_name",
    "failure_http_error",
    "failure_no_tool_call",
    "failure_stream_truncated",
    "failure_timeout",
    "failure_unknown_tool",
    "filter",
    "finish_reasons",
    "initial_model_state",
    "lm_eval_key",
    "maturity",
    "max_gen_toks",
    "max_tokens",
    "measured_repetitions",
    "method",
    "mtp",
    "mtp_runtime_commit",
    "n_err",
    "n_limit",
    "n_ok",
    "n_requests",
    "n_runs",
    "n_tasks",
    "output_reservation_tokens",
    "overlap",
    "phase",
    "prefill_tok_s_max",
    "prefill_tok_s_min",
    "prefix_cache_tokens",
    "profile",
    "prompt",
    "prompt_set",
    "prompt_tokens",
    "provisional",
    "published_from",
    "quant",
    "quiesce_window",
    "reasoning_effort",
    "recovered_partial",
    "repetition_penalty",
    "repetitions",
    "repo",
    "requested_prompt_tokens",
    "reserved_output_tokens",
    "rounds",
    "run",
    "run_tag",
    "scoring",
    "serving",
    "speculative_decoding",
    "spread_tok_s",
    "stream",
    "swap_baseline_mb",
    "swap_growth_mb",
    "task",
    "temperature",
    "think_kwarg",
    "think_kwarg_sent",
    "think_value",
    "thinking",
    "total_eval_time_s",
    "total_input_tokens",
    "total_output_tokens",
    "total_s_max",
    "total_s_min",
    "track",
    "truncated_rate",
    "ttft_s_max",
    "ttft_s_min",
    "valid_rounds",
    "validated",
    "wall_s",
    "warmup_runs",
    "width",
)

OPTIONAL_RESULT_COLUMNS = (
    "duration_seconds",
    "prompt_tokens_per_second",
    "decode_tokens_per_second",
    "total_tokens_per_second",
    "first_token_latency_ms",
    "peak_rss_mb",
)

SYSTEM_COLUMNS = (
    "os",
    "chip",
    "memory_gb",
    "hostname",
    "vllm_mlx_version",
    "runner",
    "python_version",
    "mlx_version",
    "lm_eval_version",
    "kernel",
    "gpu",
    "engine",
    "power_limit_w",
    "container",
    "topology",
)

CAMPAIGN_DIMENSION_TYPES = {
    "hardware": {
        "machine": pa.string(),
        "accelerator_model": pa.string(),
        "accelerator_memory_gb": pa.float64(),
        "memory_bandwidth_gbps": pa.float64(),
        "host_cpu": pa.string(),
        "host_ram_gb": pa.float64(),
        "host_ram_speed_mts": pa.float64(),
        "pcie_generation": pa.float64(),
        "pcie_width_lanes": pa.int64(),
        "power_cap_w": pa.float64(),
        "ups_circuit": pa.string(),
        "chassis_container": pa.string(),
    },
    "software": {
        "operating_system": pa.string(),
        "kernel": pa.string(),
        "driver_version": pa.string(),
        "accelerator_runtime": pa.string(),
        "engine": pa.string(),
        "engine_version": pa.string(),
        "engine_commit": pa.string(),
        "backend": pa.string(),
        "gpu_architectures": pa.string(),
        "build_flags": pa.string(),
        "flash_attention": pa.bool_(),
    },
    "model": {
        "family": pa.string(),
        "id": pa.string(),
        "hf_repo": pa.string(),
        "revision_sha": pa.string(),
        "total_parameters": pa.int64(),
        "active_parameters": pa.int64(),
        "architecture": pa.string(),
        "experts_total": pa.int64(),
        "experts_active": pa.int64(),
        "quantization": pa.string(),
        "bits_per_weight": pa.float64(),
        "file_size_gb": pa.float64(),
        "license": pa.string(),
        "native_max_context_tokens": pa.int64(),
        "file_sha256": pa.string(),
    },
    "run": {
        "allocated_context_tokens": pa.int64(),
        "kv_cache_dtype": pa.string(),
        "prompt_tokens": pa.int64(),
        "depth_tokens": pa.int64(),
        "output_tokens": pa.int64(),
        "concurrent_agents": pa.int64(),
        "batch_size": pa.int64(),
        "ubatch_size": pa.int64(),
        "parallel_slots": pa.int64(),
        "prefix_cache": pa.bool_(),
        "speculative_mtp": pa.bool_(),
        "draft_model": pa.string(),
        "temperature": pa.float64(),
        "thinking": pa.bool_(),
        "chat_template": pa.string(),
        "seed": pa.int64(),
        "repeats": pa.int64(),
        "warm_cold": pa.string(),
    },
    "speed": {
        "ttft_p50_ms": pa.float64(),
        "ttft_p90_ms": pa.float64(),
        "ttft_p99_ms": pa.float64(),
        "prefill_tokens_per_second": pa.float64(),
        "decode_tokens_per_second_per_agent": pa.float64(),
        "aggregate_output_tokens_per_second": pa.float64(),
        "total_tokens_per_second": pa.float64(),
        "tpot_ms": pa.float64(),
        "itl_p50_ms": pa.float64(),
        "itl_p99_ms": pa.float64(),
        "mtp_acceptance_rate": pa.float64(),
        "request_success_rate": pa.float64(),
    },
    "resource": {
        "accelerator_memory_peak_gb": pa.float64(),
        "host_ram_peak_gb": pa.float64(),
        "cpu_offload_layers": pa.int64(),
        "gpu_utilization_avg_percent": pa.float64(),
        "gpu_utilization_max_percent": pa.float64(),
        "cpu_performance_utilization_avg_percent": pa.float64(),
        "cpu_efficiency_utilization_avg_percent": pa.float64(),
        "power_avg_w": pa.float64(),
        "power_max_w": pa.float64(),
        "system_power_avg_w": pa.float64(),
        "system_power_max_w": pa.float64(),
        "soc_power_avg_w": pa.float64(),
        "soc_power_max_w": pa.float64(),
        "cpu_power_avg_w": pa.float64(),
        "cpu_power_max_w": pa.float64(),
        "memory_power_avg_w": pa.float64(),
        "memory_power_max_w": pa.float64(),
        "accelerator_memory_power_avg_w": pa.float64(),
        "accelerator_memory_power_max_w": pa.float64(),
        "ane_power_avg_w": pa.float64(),
        "ane_power_max_w": pa.float64(),
        "energy_j_per_output_token": pa.float64(),
        "tokens_per_second_per_watt": pa.float64(),
        "temperature_max_c": pa.float64(),
        "average_clocks_mhz": pa.float64(),
        "cpu_performance_clock_avg_mhz": pa.float64(),
        "cpu_efficiency_clock_avg_mhz": pa.float64(),
        "throttle_reasons": pa.string(),
        "fan_percent": pa.float64(),
    },
    "quality": {
        "benchmark_name": pa.string(),
        "benchmark_version": pa.string(),
        "subset": pa.string(),
        "sample_count": pa.int64(),
        "score": pa.float64(),
        "standard_error": pa.float64(),
        "confidence_interval_low": pa.float64(),
        "confidence_interval_high": pa.float64(),
        "judge_model": pa.string(),
        "contamination_note": pa.string(),
    },
    "provenance": {
        "timestamp_utc": pa.string(),
        "run_id": pa.string(),
        "config_name": pa.string(),
        "config_git_sha": pa.string(),
        "operator_agent": pa.string(),
        "raw_output_path": pa.string(),
        "schema_version": pa.string(),
    },
    "cost": {
        "kwh_per_million_output_tokens": pa.float64(),
        "local_electricity_usd_per_million_tokens": pa.float64(),
        "median_rental_usd_per_million_tokens": pa.float64(),
        "median_api_usd_per_million_tokens": pa.float64(),
    },
}

CAMPAIGN_DIMENSION_JSON_FIELDS = frozenset(
    {
        ("software", "gpu_architectures"),
        ("software", "build_flags"),
        ("resource", "throttle_reasons"),
    }
)


def campaign_dimension_column(category: str, field: str) -> str:
    """Return the fixed Parquet column name for one campaign dimension."""
    suffix = "_json" if (category, field) in CAMPAIGN_DIMENSION_JSON_FIELDS else ""
    return f"campaign_{category}_{field}{suffix}"


JSON_STRING_COLUMNS = frozenset({"serving", "gpu", "engine", "container", "topology"})
APPLE_MACHINE_FALLBACK = "Apple M4 Max"
_MACHINE_LABEL_PATTERNS = (
    re.compile(r"Mac Studio M4 Max 128GB\Z"),
    re.compile(r"MacBook Pro M4 Max [1-9][0-9]*GB\Z"),
    re.compile(r"RTX PRO 6000 Max-Q 96GB\Z"),
    re.compile(r"RTX 4080 SUPER 16GB\Z"),
)
_MACHINE_ID_FIELDS = (
    "campaign_hardware_machine",
    "machine",
    "hostname",
    "host",
    "published_from",
)
_NESTED_MACHINE_ID_FIELDS = frozenset(
    {"campaign_hardware_machine", "machine", "hostname", "host", "node_name"}
)

PARQUET_ROW_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("timestamp", pa.string()),
        ("git_sha", pa.string()),
        ("trigger", pa.string()),
        ("suite", pa.string()),
        ("model", pa.string()),
        ("model_revision", pa.string()),
        ("quantization", pa.string()),
        ("seed", pa.int64()),
        ("env_class", pa.string()),
        ("concurrency", pa.int64()),
        ("reasoning_effort", pa.string()),
        ("serving", pa.string()),
        ("os", pa.string()),
        ("chip", pa.string()),
        ("memory_gb", pa.int64()),
        ("hostname", pa.string()),
        ("vllm_mlx_version", pa.string()),
        ("runner", pa.string()),
        ("python_version", pa.string()),
        ("mlx_version", pa.string()),
        ("lm_eval_version", pa.string()),
        ("kernel", pa.string()),
        ("gpu", pa.string()),
        ("engine", pa.string()),
        ("power_limit_w", pa.float64()),
        ("container", pa.string()),
        ("topology", pa.string()),
        ("system_extra", pa.string()),
        *[
            (campaign_dimension_column(category, field), column_type)
            for category, fields in CAMPAIGN_DIMENSION_TYPES.items()
            for field, column_type in fields.items()
        ],
        ("campaign_dimension_null_reasons_json", pa.string()),
        ("name", pa.string()),
        ("metric", pa.string()),
        ("value", pa.float64()),
        ("unit", pa.string()),
        *[(column, pa.float64()) for column in OPTIONAL_RESULT_COLUMNS],
        *[(f"tag_{key}", pa.string()) for key in TAG_KEYS],
        ("tags_json", pa.string()),
        ("extra_json", pa.string()),
    ]
)


def empty_parquet_row() -> dict[str, Any]:
    """Return one row with every canonical column present and unset."""
    return dict.fromkeys(PARQUET_ROW_SCHEMA.names)


def _machine_label_is_valid(value: object) -> bool:
    return value == APPLE_MACHINE_FALLBACK or (
        isinstance(value, str) and any(pattern.fullmatch(value) for pattern in _MACHINE_LABEL_PATTERNS)
    )


def parse_machine_labels(value: str | None) -> dict[str, str]:
    """Parse and validate the optional hostname-to-label Actions secret."""
    if not value:
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("machine label configuration must be valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("machine label configuration contains an invalid entry")
    labels: dict[str, str] = {}
    for host, label in decoded.items():
        if not isinstance(host, str) or not host.strip() or not _machine_label_is_valid(label):
            raise ValueError("machine label configuration contains an invalid entry")
        labels[host] = label
    return labels


def _machine_label_hardware_text(row: Mapping[str, Any]) -> str:
    values = [row.get("chip"), row.get("gpu"), row.get("campaign_hardware_accelerator_model")]
    return " ".join(
        json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value or "")
        for value in values
    ).casefold()


def _nested_machine_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict):
        found = [
            nested
            for key, item in value.items()
            if str(key).casefold() in _NESTED_MACHINE_ID_FIELDS
            for nested in ([item] if isinstance(item, str) else [])
        ]
        return found + [nested for item in value.values() for nested in _nested_machine_ids(item)]
    if isinstance(value, list):
        return [nested for item in value for nested in _nested_machine_ids(item)]
    return []


def _machine_identity_values(
    row: Mapping[str, Any],
    tags: Mapping[str, Any],
    extras: Mapping[str, Any],
) -> list[Any]:
    direct = [row.get(field) for field in _MACHINE_ID_FIELDS] + [
        container.get(field.removeprefix("tag_"))
        for container in (tags, extras)
        for field in _MACHINE_ID_FIELDS
    ]
    nested = [
        identity for field in ("topology", "system_extra") for identity in _nested_machine_ids(row.get(field))
    ]
    return direct + nested


def _scrub_machine_identifiers(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        for machine_id, label in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
            value = value.replace(machine_id, label)
        return value
    if isinstance(value, dict):
        return {
            _scrub_machine_identifiers(key, replacements): _scrub_machine_identifiers(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_scrub_machine_identifiers(item, replacements) for item in value]
    return value


def _machine_label_for_row(
    row: Mapping[str, Any],
    tags: Mapping[str, Any],
    extras: Mapping[str, Any],
    labels: Mapping[str, str],
) -> str | None:
    identity_values = _machine_identity_values(row, tags, extras)
    for value in identity_values:
        if isinstance(value, str) and value in labels:
            label = labels[value]
            if not _machine_label_is_valid(label):
                raise ValueError("machine label configuration contains an invalid entry")
            break
    else:
        valid_labels = [
            value for value in identity_values if isinstance(value, str) and _machine_label_is_valid(value)
        ]
        label = next(
            (value for value in valid_labels if value != APPLE_MACHINE_FALLBACK),
            valid_labels[0] if valid_labels else "",
        )

    hardware = _machine_label_hardware_text(row)
    if not label:
        if "m4 max" in hardware:
            label = APPLE_MACHINE_FALLBACK
        elif "rtx pro 6000" in hardware:
            label = "RTX PRO 6000 Max-Q 96GB"
        elif "rtx 4080 super" in hardware:
            label = "RTX 4080 SUPER 16GB"
    if not label:
        if any(value not in (None, "") for value in identity_values):
            raise ValueError("a machine identifier has no safe hardware label")
        return None

    if "m4 max" in label.casefold() and "m4 max" not in hardware:
        raise ValueError("a machine label does not match the row hardware")
    if label.startswith("MacBook Pro M4 Max "):
        memory = row.get("memory_gb")
        if memory is None or not label.endswith(f"{int(memory)}GB"):
            raise ValueError("a machine label does not match the row memory")
    if label.startswith("Mac Studio M4 Max ") and row.get("memory_gb") != 128:
        raise ValueError("a machine label does not match the row memory")
    if label.startswith("RTX PRO 6000") and "rtx pro 6000" not in hardware:
        raise ValueError("a machine label does not match the row hardware")
    if label.startswith("RTX 4080 SUPER") and "rtx 4080 super" not in hardware:
        raise ValueError("a machine label does not match the row hardware")
    accelerator_memory = row.get("campaign_hardware_accelerator_memory_gb")
    expected_accelerator_memory = (
        int(label.removesuffix("GB").split()[-1])
        if label.startswith(("RTX PRO 6000", "RTX 4080 SUPER"))
        else None
    )
    if (
        accelerator_memory is not None
        and expected_accelerator_memory is not None
        and float(accelerator_memory) != expected_accelerator_memory
    ):
        raise ValueError("a machine label does not match the row accelerator memory")
    return label


def normalize_legacy_rows(
    rows: list[dict[str, Any]],
    *,
    machine_labels: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Pad historical rows, retain dynamic fields, and project safe machine labels."""
    normalized_rows: list[dict[str, Any]] = []
    schema_names = set(PARQUET_ROW_SCHEMA.names)
    for legacy in rows:
        tags_json = legacy.get("tags_json")
        if tags_json is None:
            tags: dict[str, Any] = {}
        else:
            tags = json.loads(tags_json) if isinstance(tags_json, str) else tags_json
            if not isinstance(tags, dict):
                raise ValueError("historical tags_json must contain a JSON object")
            tags = dict(tags)
        tags.update(
            {
                key.removeprefix("tag_"): value
                for key, value in legacy.items()
                if key.startswith("tag_") and value is not None
            }
        )
        extras = {
            key: value
            for key, value in legacy.items()
            if key not in schema_names and not key.startswith("tag_") and value is not None
        }
        extra_json = legacy.get("extra_json")
        if extra_json is not None:
            decoded_extras = json.loads(extra_json) if isinstance(extra_json, str) else extra_json
            if not isinstance(decoded_extras, dict):
                raise ValueError("historical extra_json must contain a JSON object")
            extras = {**decoded_extras, **extras}
        row = {name: legacy.get(name) for name in PARQUET_ROW_SCHEMA.names}
        for column in JSON_STRING_COLUMNS:
            value = row[column]
            if value is not None and not isinstance(value, str):
                row[column] = json.dumps(value, sort_keys=True)
        system_extra = legacy.get("system_extra")
        row["system_extra"] = (
            system_extra
            if isinstance(system_extra, str)
            else json.dumps(system_extra, sort_keys=True)
            if system_extra is not None
            else "{}"
        )
        label = _machine_label_for_row(row, tags, extras, machine_labels or {})
        if label:
            row["hostname"] = label
            for field in _MACHINE_ID_FIELDS:
                if row.get(field) is not None:
                    row[field] = label
                tag = field.removeprefix("tag_")
                tag_field = f"tag_{tag}"
                if row.get(tag_field) is not None:
                    row[tag_field] = label
                if tag in tags:
                    tags[tag] = label
                if tag in extras:
                    extras[tag] = label
            row["tags_json"] = json.dumps(tags, sort_keys=True)
            row["extra_json"] = json.dumps(extras, sort_keys=True, default=str)
        row["tags_json"] = json.dumps(tags, sort_keys=True)
        row["extra_json"] = json.dumps(extras, sort_keys=True, default=str)
        raw_machine_ids = {
            value
            for value in _machine_identity_values(row, tags, extras)
            if isinstance(value, str) and value and not _machine_label_is_valid(value)
        }
        if raw_machine_ids:
            if label is None:
                raise ValueError("a machine identifier has no safe hardware label")
            replacements = {
                machine_id: (machine_labels or {}).get(machine_id, label) for machine_id in raw_machine_ids
            }
            row = _scrub_machine_identifiers(row, replacements)
        normalized_rows.append(row)
    return normalized_rows


def canonical_shard_needs_refresh(
    existing: pa.Table,
    expected: pa.Table,
    *,
    machine_labels: Mapping[str, str] | None = None,
    prior_projection: pa.Table | None = None,
) -> bool:
    """Check whether a canonical shard can be refreshed from its retained source."""
    if not expected.schema.equals(PARQUET_ROW_SCHEMA, check_metadata=False):
        raise ValueError("expected table does not use the canonical schema")
    if existing.num_rows != expected.num_rows:
        raise RuntimeError("canonical shard row count differs from its source")
    if (
        existing.schema.equals(PARQUET_ROW_SCHEMA, check_metadata=False)
        and existing.to_pylist() == expected.to_pylist()
    ):
        return False
    if prior_projection is not None:
        if not prior_projection.schema.equals(PARQUET_ROW_SCHEMA, check_metadata=False):
            raise ValueError("prior projection does not use the canonical schema")
        if prior_projection.num_rows != expected.num_rows:
            raise RuntimeError("prior projection row count differs from its source")
        if existing.to_pylist() == prior_projection.to_pylist():
            return True

    refreshed = pa.Table.from_pylist(
        normalize_legacy_rows(existing.to_pylist(), machine_labels=machine_labels),
        schema=PARQUET_ROW_SCHEMA,
    )
    if refreshed.to_pylist() != expected.to_pylist():
        raise RuntimeError("canonical shard data differs from its normalized source")
    return True
