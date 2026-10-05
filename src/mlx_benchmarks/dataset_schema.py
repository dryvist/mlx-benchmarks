"""Stable flat Parquet schema shared by every published run shard."""

from __future__ import annotations

import json
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


def normalize_legacy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pad historical flat rows and retain dynamic tags/columns as JSON."""
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
        row["tags_json"] = json.dumps(tags, sort_keys=True)
        row["extra_json"] = json.dumps(extras, sort_keys=True, default=str)
        normalized_rows.append(row)
    return normalized_rows
