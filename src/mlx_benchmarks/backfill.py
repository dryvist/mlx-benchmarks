"""One-time, dry-run-first migration for historical Hugging Face score shards."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi

from mlx_benchmarks.dataset_schema import PARQUET_ROW_SCHEMA, PUBLISHED_JSON_COLUMNS
from mlx_benchmarks.privacy import PRIVATE_IDENTIFIER_KEYS, remove_private_identifiers
from mlx_benchmarks.published_results import (
    _model_fields,
    _registered_task,
    eval_results_path,
    eval_results_yaml,
)
from mlx_benchmarks.result_contract import (
    MODEL_FIELDS,
    RESULT_FIELDS,
    row_schema_required_fields,
    validate_published_result,
)

DEFAULT_REPO_ID = "JacobPEvans/mlx-benchmarks"
_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_NON_EVAL_SUITES = {"throughput", "ttft", "gpu-burn", "mbw", "nvbandwidth", "fio"}
_RUN_FIELDS = {
    "engine",
    "engine_version",
    "profile",
    "hardware",
    "power_limit_w",
    "concurrency",
    "ctx_per_slot",
    "kv_cache_dtype",
    "thinking",
    "temperature",
    "max_tokens",
    "prompt_chars",
    "prompt_tokens",
    "system_prompt_chars",
    "system_prompt_tokens",
    "runner",
    "harness",
    "router_key_alias",
    "run_id",
    "start_utc",
    "end_utc",
}
_DATASET_FIELDS = {
    "dataset_id",
    "dataset_task_id",
    "dataset_revision",
    "evaluation_framework",
    "config",
    "split",
}
_PRIVACY_JSON_COLUMNS = {"gpu", "engine", "system_engine", "serving", "topology", "system_extra"}


def _first(row: dict[str, Any], *keys: str) -> Any:
    return next((row[key] for key in keys if row.get(key) is not None), None)


def _json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    return None


def _as_number(value: Any, *, integer: bool = False) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return int(value) if integer and int(value) == value else value
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        if integer:
            return int(parsed) if parsed.is_integer() else None
        return parsed
    return None


def _utc_timestamp(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC).isoformat(timespec="auto").replace("+00:00", "Z")


def _legacy_run_fields(row: dict[str, Any]) -> dict[str, Any]:
    tags = _json_mapping(row.get("tags_json"))

    def tagged(field: str, *row_keys: str) -> Any:
        value = _first(row, *row_keys)
        return tags.get(field) if value is None else value

    hardware = _json_mapping(row.get("hardware"))
    if not hardware:
        hardware = {field: row[field] for field in ("chip", "memory_gb") if row.get(field) is not None}
        gpu = _json_mapping(row.get("gpu"))
        if gpu:
            hardware["gpu"] = gpu

    engine = row.get("engine")
    if isinstance(engine, str):
        engine_value: Any = engine
        engine_data = _json_mapping(engine)
        if engine_data:
            engine_value = engine_data.get("name")
            engine_version = engine_data.get("version")
        else:
            engine_version = None
    elif isinstance(engine, dict):
        engine_value = engine.get("name")
        engine_version = engine.get("version")
    else:
        engine_value = None
        engine_version = None

    direct = {
        "engine": engine_value,
        "engine_version": _first(
            row,
            "engine_version",
            "campaign_software_engine_version",
        )
        or engine_version,
        "profile": tagged("profile", "profile", "tag_profile"),
        "hardware": hardware or None,
        "power_limit_w": _as_number(_first(row, "power_limit_w", "campaign_hardware_power_cap_w")),
        "concurrency": _as_number(_first(row, "concurrency", "tag_concurrency"), integer=True),
        "ctx_per_slot": _as_number(row.get("ctx_per_slot"), integer=True),
        "kv_cache_dtype": _first(row, "kv_cache_dtype", "campaign_run_kv_cache_dtype"),
        "thinking": _as_bool(tagged("thinking", "thinking", "campaign_run_thinking", "tag_thinking")),
        "temperature": _as_number(
            tagged("temperature", "temperature", "campaign_run_temperature", "tag_temperature")
        ),
        "max_tokens": _as_number(
            tagged("max_tokens", "max_tokens", "tag_max_tokens", "tag_max_gen_toks"), integer=True
        ),
        "prompt_chars": row.get("prompt_chars"),
        "prompt_tokens": _as_number(
            tagged("prompt_tokens", "prompt_tokens", "tag_prompt_tokens"), integer=True
        ),
        "system_prompt_chars": row.get("system_prompt_chars"),
        "system_prompt_tokens": row.get("system_prompt_tokens"),
        "runner": row.get("runner"),
        "harness": row.get("harness"),
        "router_key_alias": row.get("router_key_alias"),
        "run_id": _first(row, "run_id", "campaign_provenance_run_id", "tag_run_id", "tag_run"),
        "start_utc": _utc_timestamp(_first(row, "start_utc", "timestamp")),
        "end_utc": _utc_timestamp(row.get("end_utc")),
    }
    return {key: direct[key] for key in _RUN_FIELDS}


def backfill_result_row(row: dict[str, Any], api: HfApi) -> dict[str, Any]:
    """Copy recorded values and exact-revision Hub metadata; mark unknowns explicitly."""
    if row_schema_required_fields() <= row.keys():
        result = {
            field: row[field]
            for field in MODEL_FIELDS
            | RESULT_FIELDS
            | _RUN_FIELDS
            | _DATASET_FIELDS
            | {"license_name", "license_link", "tags", "source_url", "notes", "dimension_null_reasons"}
            if field in row
        }
        for field in PUBLISHED_JSON_COLUMNS:
            value = result.get(field)
            if isinstance(value, str):
                try:
                    result[field] = json.loads(value)
                except json.JSONDecodeError as exc:
                    if field != "base_model":
                        raise ValueError(f"published {field} column is not valid JSON") from exc
        if result.get("gated") in {"true", "false"}:
            result["gated"] = result["gated"] == "true"
        for field in ("tags", "source_url", "notes"):
            if result.get(field) is None:
                result.pop(field, None)
        validate_published_result(result)
        return result

    existing = row.get("published_result_json")
    if isinstance(existing, str):
        try:
            decoded = json.loads(existing)
        except json.JSONDecodeError as exc:
            raise ValueError("published_result_json is not valid JSON") from exc
        if isinstance(decoded, dict):
            validate_published_result(decoded)
            return decoded

    model_id = _first(row, "model_id", "model")
    revision = _first(row, "model_revision", "campaign_model_revision_sha")
    if isinstance(model_id, str) and isinstance(revision, str) and _SHA_RE.fullmatch(revision):
        model = _model_fields(api, str(model_id), revision)
    else:
        model: dict[str, Any] = dict.fromkeys(MODEL_FIELDS)
        model["model_id"] = model_id
        model["tags"] = []

    result: dict[str, Any] = dict(model)
    run_fields = _legacy_run_fields(row)
    result.update(run_fields)
    result.update(
        {
            "dataset_id": row.get("dataset_id"),
            "dataset_task_id": row.get("dataset_task_id"),
            "dataset_revision": row.get("dataset_revision"),
            "evaluation_framework": row.get("evaluation_framework"),
            "config": row.get("config"),
            "split": row.get("split"),
            "value": _as_number(_first(row, "value", "metric_value")),
            "metric": _first(row, "metric", "metric_metric"),
            "date": _utc_timestamp(_first(row, "date", "timestamp")),
            "source_url": row.get("source_url"),
            "notes": row.get("notes"),
        }
    )
    for field in ("source_url", "notes"):
        if result[field] is None:
            result.pop(field)

    reasons: dict[str, str] = {}
    if not isinstance(model_id, str) or not isinstance(revision, str) or not _SHA_RE.fullmatch(revision):
        for field in MODEL_FIELDS - {"model_id", "tags"}:
            reasons[field] = "na_backfill"
        if revision is not None:
            reasons["model_revision"] = "na_backfill"
        if model_id is None:
            reasons["model_id"] = "na_backfill"

    suite = row.get("suite")
    if suite in _NON_EVAL_SUITES and not any(result.get(field) is not None for field in _DATASET_FIELDS):
        reasons.update(dict.fromkeys(_DATASET_FIELDS, "not_applicable"))
    else:
        for field in _DATASET_FIELDS:
            if result.get(field) is None:
                reasons[field] = "na_backfill"

    for field in RESULT_FIELDS | _RUN_FIELDS:
        if result.get(field) is None:
            reasons[field] = "na_backfill"
    if isinstance(run_fields.get("hardware"), dict):
        hardware = run_fields["hardware"]
        if any(
            key.casefold() in {"hostname", "host", "ip", "address", "node", "node_name"} for key in hardware
        ):
            hardware = {
                key: value
                for key, value in hardware.items()
                if key.casefold() not in {"hostname", "host", "ip", "address", "node", "node_name"}
            }
            result["hardware"] = hardware or None
            if result["hardware"] is None:
                reasons["hardware"] = "na_backfill"

    if result.get("base_model") is None and result.get("base_model_relation") is None:
        reasons.setdefault("base_model", "not_applicable")
        reasons.setdefault("base_model_relation", "not_applicable")
    if result.get("license") != "other":
        result["license_name"] = None
        result["license_link"] = None
    for field in ("license_name", "license_link"):
        if field in result and result[field] is None and result.get("license") == "other":
            reasons[field] = "na_backfill"

    for field in row_schema_required_fields():
        if result.get(field) is None:
            reasons.setdefault(field, "na_backfill")

    result["dimension_null_reasons"] = reasons
    validate_published_result(result)
    if result.get("dataset_id") is not None and all(
        result.get(field) is not None for field in _DATASET_FIELDS
    ):
        _registered_task(api, result)
    return result


def _write_parquet(table: pa.Table, rows: list[dict[str, Any]]) -> bytes:
    legacy_fields = [field for field in table.schema if field.name != "published_result_json"]
    field_names = {field.name for field in legacy_fields}
    fields = legacy_fields + [field for field in PARQUET_ROW_SCHEMA if field.name not in field_names]
    schema = pa.schema(fields)
    output_rows = []
    for original, result in zip(table.to_pylist(), rows, strict=True):
        updated = dict(original)
        old_engine = updated.get("engine")
        if old_engine is not None and "system_engine" not in updated:
            updated["system_engine"] = old_engine
        for name in tuple(updated):
            if name.casefold() in PRIVATE_IDENTIFIER_KEYS:
                updated[name] = None
        for name, value in tuple(updated.items()):
            if (
                name not in _PRIVACY_JSON_COLUMNS
                and name not in PUBLISHED_JSON_COLUMNS
                and not name.endswith("_json")
            ):
                continue
            if not isinstance(value, str):
                continue
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                continue
            updated[name] = json.dumps(remove_private_identifiers(decoded), sort_keys=True)
        for name, value in result.items():
            if (
                name in PUBLISHED_JSON_COLUMNS
                and value is not None
                and not (name == "base_model" and isinstance(value, str))
            ):
                value = json.dumps(value, sort_keys=True, allow_nan=False)
            elif name == "gated" and value is not None:
                value = str(value).lower()
            updated[name] = value
        output_rows.append(updated)
    columns: list[pa.Array] = []
    for name, field in zip(schema.names, schema, strict=True):
        values = [row.get(name) for row in output_rows]
        columns.append(pa.array(values, type=field.type))
    result_table = pa.Table.from_arrays(columns, schema=schema)
    buffer = io.BytesIO()
    pq.write_table(result_table, buffer)
    return buffer.getvalue()


def backfill_dataset(repo_id: str, *, apply: bool, api: HfApi | None = None) -> dict[str, int]:
    """Plan or apply a one-time rewrite of every Parquet row in the dataset."""
    token = os.environ.get("HF_TOKEN")
    if apply and not token:
        raise RuntimeError("HF_TOKEN must be set for --apply")
    hub = api or HfApi(token=token)
    paths = sorted(
        path
        for path in hub.list_repo_files(repo_id=repo_id, repo_type="dataset")
        if path.startswith("data/") and path.endswith(".parquet")
    )
    operations: list[CommitOperationAdd] = []
    sidecars: dict[str, bytes] = {}
    changed_shards = 0
    rows_seen = 0
    results_without_sidecars = 0
    for path in paths:
        local_path = hub.hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=path)
        table = pq.read_table(local_path)
        legacy_rows = table.to_pylist()
        enriched = [backfill_result_row(row, hub) for row in legacy_rows]
        rows_seen += len(enriched)
        parquet_bytes = _write_parquet(table, enriched)
        projected = pq.read_table(io.BytesIO(parquet_bytes))
        if (
            not table.schema.equals(projected.schema, check_metadata=False)
            or table.to_pylist() != projected.to_pylist()
        ):
            operations.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=parquet_bytes))
            changed_shards += 1
        for index, result in enumerate(enriched, start=1):
            if any(
                result.get(field) is None for field in _DATASET_FIELDS | {"value", "metric", "date", "run_id"}
            ):
                results_without_sidecars += 1
                continue
            sidecars[eval_results_path(result, index)] = eval_results_yaml(result).encode("utf-8")

    existing_paths = set(hub.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    for path, content in sidecars.items():
        if path in existing_paths:
            local_path = hub.hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=path)
            if Path(local_path).read_bytes() != content:
                raise RuntimeError(f"existing eval result differs from planned migration output: {path}")
            continue
        operations.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=content))

    print(
        f"mode={'apply' if apply else 'dry-run'} shards={len(paths)} rows={rows_seen} "
        f"shards_to_rewrite={changed_shards} eval_results_to_add={len(sidecars)} "
        f"results_without_sidecars={results_without_sidecars} operations={len(operations)}"
    )
    if not apply:
        print("mode=dry-run; no Hub files changed")
    elif operations:
        hub.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message="chore(dataset): backfill published result metadata",
        )
        print(f"published_operations={len(operations)}")
    return {
        "shards": len(paths),
        "rows": rows_seen,
        "shards_to_rewrite": changed_shards,
        "eval_results_to_add": len(sidecars),
        "results_without_sidecars": results_without_sidecars,
        "operations": len(operations),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--apply", action="store_true", help="Write the migration to the Hugging Face dataset"
    )
    args = parser.parse_args(argv)
    try:
        backfill_dataset(args.repo_id, apply=args.apply)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"backfill failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
