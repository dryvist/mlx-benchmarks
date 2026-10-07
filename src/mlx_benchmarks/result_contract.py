"""Validate published score rows against the single schema in ``schema.json``."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
import pyarrow.parquet as pq
from jsonschema import Draft7Validator

from mlx_benchmarks.dataset_schema import PUBLISHED_JSON_COLUMNS
from mlx_benchmarks.envelope import load_schema

CUTOVER_UTC = datetime(2026, 10, 7, 4, 15, tzinfo=UTC)
MODEL_FIELDS = {
    "model_id",
    "model_revision",
    "pipeline_tag",
    "model_task",
    "model_task_source",
    "library_name",
    "license",
    "license_name",
    "license_link",
    "base_model",
    "base_model_relation",
    "tags",
    "parameters_total",
    "dtype",
    "quant",
    "architectures",
    "model_type",
    "context_length",
    "gated",
}
RESULT_FIELDS = {"value", "metric", "date"}


class PublishedResultValidationError(ValueError):
    """Raised when a published score has a missing, silent, or invalid value."""


def _row_validator() -> Draft7Validator:
    root = load_schema()
    Draft7Validator.check_schema(root)
    schema = {
        "$schema": root["$schema"],
        "definitions": root["definitions"],
        "$ref": "#/definitions/published_result",
    }
    return Draft7Validator(schema, format_checker=Draft7Validator.FORMAT_CHECKER)


def validate_published_metadata(metadata: dict[str, Any]) -> None:
    """Validate run metadata using the published row schema as its source of truth."""
    root = load_schema()
    definition = copy.deepcopy(root["definitions"]["published_result"])
    omitted = MODEL_FIELDS | RESULT_FIELDS | {"tags", "license_name", "license_link"}
    definition["required"] = [field for field in definition["required"] if field not in omitted]
    definition["properties"] = {
        field: value for field, value in definition["properties"].items() if field not in omitted
    }
    schema: dict[str, Any] = {"$schema": root["$schema"], "definitions": root["definitions"]}
    schema["definitions"] = {**schema["definitions"], "published_metadata": definition}
    schema["$ref"] = "#/definitions/published_metadata"
    errors = sorted(
        Draft7Validator(schema, format_checker=Draft7Validator.FORMAT_CHECKER).iter_errors(metadata),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        detail = "\n".join(
            f"- {error.message} at $.{'.'.join(map(str, error.absolute_path))}" for error in errors
        )
        raise PublishedResultValidationError(detail)


def _row_time(row: dict[str, Any]) -> datetime | None:
    raw = row.get("start_utc") or row.get("date")
    if not isinstance(raw, str):
        return None
    if len(raw) == 10:
        parsed = datetime.fromisoformat(raw).replace(tzinfo=UTC)
    else:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def validate_published_result(row: dict[str, Any]) -> None:
    """Enforce schema types, explicit null reasons, and cutover semantics."""
    errors = sorted(_row_validator().iter_errors(row), key=lambda err: list(err.absolute_path))
    if errors:
        detail = "\n".join(
            f"- {error.message} at $.{'.'.join(map(str, error.absolute_path))}" for error in errors
        )
        raise PublishedResultValidationError(detail)

    reasons = row.get("dimension_null_reasons", {})
    required_fields = row_schema_required_fields()
    if row.get("license") == "other":
        required_fields.update({"license_name", "license_link"})
    for field, value in row.items():
        if field == "dimension_null_reasons" or field not in required_fields:
            continue
        reason = reasons.get(field)
        if value is None and reason is None:
            raise PublishedResultValidationError(f"{field} is null without dimension_null_reasons[{field!r}]")
        if value is not None and reason is not None:
            raise PublishedResultValidationError(f"{field} has a null reason but is not null")

    for field in reasons:
        if field in row and row[field] is not None:
            raise PublishedResultValidationError(f"{field} has a null reason but is not null")

    if row.get("model_task") != row.get("pipeline_tag") and (
        row.get("model_task") is not None or row.get("pipeline_tag") is not None
    ):
        raise PublishedResultValidationError("model_task must equal pipeline_tag")

    if "na_backfill" in reasons.values():
        measured_at = _row_time(row)
        if measured_at is None:
            raise PublishedResultValidationError("na_backfill requires a dated row")
        if measured_at > CUTOVER_UTC:
            raise PublishedResultValidationError(
                "na_backfill is not allowed after the 2026-10-07T04:15:00Z cutover"
            )

    for field in ("start_utc", "end_utc"):
        value = row.get(field)
        if isinstance(value, str) and not value.endswith("Z"):
            raise PublishedResultValidationError(f"{field} must be expressed in UTC with a Z suffix")

    start = _row_time({"date": row.get("start_utc")})
    end = _row_time({"date": row.get("end_utc")})
    measured = _row_time({"date": row.get("date")})
    if start is not None and measured is not None and start != measured:
        raise PublishedResultValidationError("date must equal start_utc")
    if start is not None and end is not None and end < start:
        raise PublishedResultValidationError("end_utc must be at or after start_utc")


def row_schema_required_fields() -> set[str]:
    """Return required field names from the canonical JSON Schema."""
    return set(load_schema()["definitions"]["published_result"]["required"])


def _decode_json_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        values = value if isinstance(value, list) else value.get("published_results", [value])
    if not values or not all(isinstance(value, dict) for value in values):
        raise ValueError(f"{path}: expected one or more JSON objects")
    return values


def _decode_parquet_rows(path: Path) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    decoded: list[dict[str, Any]] = []
    for row in rows:
        published = {
            field: row.get(field)
            for field in load_schema()["definitions"]["published_result"]["properties"]
            if field in row
        }
        for field in PUBLISHED_JSON_COLUMNS:
            value = published.get(field)
            if isinstance(value, str):
                try:
                    published[field] = json.loads(value)
                except json.JSONDecodeError:
                    if field != "base_model":
                        raise
        gated = published.get("gated")
        if gated in {"true", "false"}:
            published["gated"] = gated == "true"
        decoded.append(published)
    return decoded


def validate_files(paths: list[Path]) -> int:
    """Validate JSON, JSONL, or Parquet files and return the number of rows."""
    count = 0
    failures: list[str] = []
    for path in paths:
        try:
            if path.suffix == ".parquet":
                rows = _decode_parquet_rows(path)
            elif path.suffix in {".json", ".jsonl"}:
                rows = _decode_json_rows(path)
            else:
                raise ValueError(f"unsupported result file: {path}")
            for index, row in enumerate(rows, start=1):
                try:
                    validate_published_result(row)
                except PublishedResultValidationError as exc:
                    failures.append(f"{path}:{index}: {exc}")
            count += len(rows)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            failures.append(f"{path}: {exc}")

    if failures:
        raise PublishedResultValidationError("\n".join(failures))
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        count = validate_files(args.files)
    except (PublishedResultValidationError, jsonschema.SchemaError) as exc:
        print(f"invalid benchmark results:\n{exc}", file=sys.stderr)
        return 1
    print(f"ok: {count} published result row(s) validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
