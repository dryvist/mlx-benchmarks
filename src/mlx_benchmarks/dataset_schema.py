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
