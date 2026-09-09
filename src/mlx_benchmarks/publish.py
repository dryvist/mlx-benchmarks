"""Serialize envelope -> Parquet and upload to the HuggingFace dataset repo."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import subprocess
from pathlib import PurePosixPath
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import HfHubHTTPError

from mlx_benchmarks.envelope import Envelope, validate_envelope

log = logging.getLogger(__name__)

DEFAULT_REPO_ID = "JacobPEvans/mlx-benchmarks"
DEFAULT_REPO_TYPE = "dataset"


class PublishError(RuntimeError):
    """Raised for publisher-layer failures (auth, upload, empty results, ...)."""


def slugify(model: str) -> str:
    """Filesystem-safe slug of a model identifier.

    ``mlx-community/Qwen3.5-9B-MLX-4bit`` -> ``mlx-community-qwen3-5-9b-mlx-4bit``.
    """
    return re.sub(r"[^a-zA-Z0-9-]", "-", model).lower().strip("-")


def current_git_sha(fallback: str = "unknown") -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, timeout=3).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return fallback


_OPTIONAL_RESULT_COLUMNS: tuple[str, ...] = (
    "duration_seconds",
    "prompt_tokens_per_second",
    "decode_tokens_per_second",
    "total_tokens_per_second",
    "first_token_latency_ms",
    "peak_rss_mb",
)


def envelope_to_rows(envelope: Envelope) -> list[dict[str, Any]]:
    """Explode ``envelope['results']`` into one flat row per measurement.

    Optional result fields (``duration_seconds``, the ``*_tokens_per_second``
    set, etc.) and tag-derived columns are normalized across rows so PyArrow's
    schema inference sees a consistent column set. Without this, a parquet
    written from a mixed envelope where the *first* row lacks a field but a
    later row has it would silently drop the column.
    """
    system = dict(envelope.get("system") or {})
    base: dict[str, Any] = {
        "schema_version": envelope.get("schema_version"),
        "timestamp": envelope.get("timestamp"),
        "git_sha": envelope.get("git_sha"),
        "trigger": envelope.get("trigger"),
        "suite": envelope.get("suite"),
        "model": envelope.get("model"),
    }
    # Flatten every system field (os/chip/memory_gb, versions, kernel, runner,
    # hostname, ...) into its own column. Nested fields (topology) can't be a
    # scalar parquet cell, so they ride as a JSON string.
    for key, value in system.items():
        base[key] = json.dumps(value) if isinstance(value, dict | list) else value

    # Dynamic-key access over a plain mapping view — these optional top-level
    # scalars are copied through verbatim; envelope is a JSON object at runtime.
    env_map = cast("dict[str, Any]", envelope)
    for key in ("model_revision", "quantization", "seed", "env_class", "concurrency", "reasoning_effort"):
        if key in env_map:
            base[key] = env_map[key]
    if "serving" in envelope:
        base["serving"] = json.dumps(envelope["serving"])

    results = envelope.get("results", [])
    present_optionals = {col for col in _OPTIONAL_RESULT_COLUMNS if any(col in r for r in results)}
    present_tags = {f"tag_{k}" for r in results for k in (r.get("tags") or {})}

    rows: list[dict[str, Any]] = []
    for r in results:
        row: dict[str, Any] = {
            **base,
            "name": r.get("name"),
            "metric": r.get("metric"),
            "value": r.get("value"),
            "unit": r.get("unit"),
        }
        for col in present_optionals:
            row[col] = r.get(col)
        tags = r.get("tags") or {}
        for tag_col in present_tags:
            row[tag_col] = tags.get(tag_col.removeprefix("tag_"))
        rows.append(row)
    return rows


def rows_to_parquet(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise PublishError("No result rows to serialize — envelope has empty results[]")
    table = pa.Table.from_pylist(rows)
    buf = io.BytesIO()
    pq.write_table(table, buf)  # type: ignore[unused-ignore,no-untyped-call]
    return buf.getvalue()


def target_path(envelope: Envelope, payload: bytes | None = None) -> str:
    """Deterministic HF-dataset path for this envelope.

    Format: ``data/run-<ts_slug>-<git_sha>-<suite>-<model_slug>-<payload_hash>.parquet``.

    ``payload_hash`` is an 8-char SHA-256 prefix of the serialized parquet,
    which (a) guarantees that two runs producing different metrics in the same
    second cannot collide even if every other dimension matches, and (b)
    content-addresses the shard so a deterministic re-publish is a no-op
    instead of silently overwriting. Pass ``payload=None`` to get the
    legacy-compatible pre-hash prefix (useful only for tests inspecting the
    slug composition).
    """
    timestamp: str = envelope["timestamp"]
    ts_slug = timestamp.replace(":", "-").rstrip("Z")
    model_slug = slugify(envelope["model"])[:50]
    prefix = f"data/run-{ts_slug}-{envelope['git_sha']}-{envelope['suite']}-{model_slug}"
    if payload is None:
        return f"{prefix}.parquet"
    digest = hashlib.sha256(payload).hexdigest()[:8]
    return f"{prefix}-{digest}.parquet"


def publish(
    envelope: Envelope,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = DEFAULT_REPO_TYPE,
    dry_run: bool = False,
    token: str | None = None,
    validate: bool = True,
) -> str:
    """Validate, serialize and optionally upload ``envelope`` to the HF dataset.

    Returns the target path. When ``dry_run`` is True no network I/O happens;
    otherwise ``HF_TOKEN`` (or an explicit ``token`` arg) is required. HF API
    errors propagate as :class:`PublishError` so callers get a single type to
    catch for the full "publish failed for non-local reason" class.
    """
    if validate:
        validate_envelope(envelope)

    rows = envelope_to_rows(envelope)
    parquet_bytes = rows_to_parquet(rows)
    path = target_path(envelope, payload=parquet_bytes)

    if dry_run:
        log.info("dry-run: would publish %d bytes to %s in %s", len(parquet_bytes), path, repo_id)
        return path

    effective_token = token or os.environ.get("HF_TOKEN")
    if not effective_token:
        raise PublishError("HF_TOKEN not set — export HF_TOKEN before publishing, or pass token=...")

    api = HfApi(token=effective_token)
    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=[CommitOperationAdd(path_in_repo=path, path_or_fileobj=parquet_bytes)],
            commit_message=f"feat: add {envelope['suite']} run for {envelope['model']}",
        )
    except HfHubHTTPError as exc:
        raise PublishError(f"HF upload failed for {path}: {exc}") from exc
    log.info("published %d-byte parquet to %s", len(parquet_bytes), path)

    # Local import: events reuses envelope_to_rows from this module.
    from mlx_benchmarks.events import append_events

    try:
        # run_id = content-addressed shard basename, joinable back to HF.
        append_events(envelope, run_id=PurePosixPath(path).stem)
    except OSError as exc:
        # The feed is derived state (replayable from the dataset); never fail
        # a successful publish over it — but say so loudly.
        log.warning("bench-events append failed (publish succeeded, replay later): %s", exc)
    return path
