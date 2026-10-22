from __future__ import annotations

import argparse
import datetime
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

from mlx_benchmarks.dataset_schema import (
    CACHE_CONFIGURATION_FIELDS,
    CACHE_OFFLOAD_BACKENDS,
    RECENT_SHARD_PATH,
    STANDARD_CACHE_FIELD_DIMENSIONS,
    STANDARD_CACHE_FIELD_TYPES,
    normalize_legacy_rows,
)

DEFAULT_REPO_ID = "JacobPEvans/mlx-benchmarks"
INDEX_PATH = Path("metadata/run-index-v1.json")
REQUIRED_FIELDS = {
    "path",
    "status",
    "variant",
    "comparison_group",
    "context_band",
    "max_output_tokens",
    "completion_state",
    "caveat",
}
VALID_STATUSES = {"scored", "experimental", "invalid", "recovered"}
CANONICAL_MACHINES = {
    "Mac Studio M4 Max 128GB",
    "RTX PRO 6000 Max-Q 96GB",
    "RTX 4080 SUPER 16GB",
    "2x Apple M4 Max 128GB (Thunderbolt RDMA)",
}
RUN_DATE = re.compile(r"^data/run-(\d{4}-\d{2}-\d{2})")
LLAMA_CPP_VERSION = re.compile(r"b\d+\Z")
ENGINE_COMMIT = re.compile(r"[0-9a-f]{7,40}\Z")
BACKFILL_START = datetime.date(2026, 10, 11)


def _matches_arrow_type(value: object, arrow_type: pa.DataType) -> bool:
    if pa.types.is_boolean(arrow_type):
        return isinstance(value, bool)
    if pa.types.is_integer(arrow_type):
        return isinstance(value, int) and not isinstance(value, bool)
    if pa.types.is_floating(arrow_type):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if pa.types.is_string(arrow_type):
        return isinstance(value, str)
    return False


def validate_cache_projection(entry: dict[str, object]) -> None:
    """Require typed, caveated cache configuration overrides for backfilled shards."""
    dimensions = entry.get("campaign_dimensions")
    reasons = entry.get("dimension_null_reasons")
    if not isinstance(dimensions, dict) or not isinstance(reasons, dict):
        raise ValueError(
            f"run-index entry must contain cache projections and null reasons: {entry.get('path')}"
        )
    for name in CACHE_CONFIGURATION_FIELDS:
        category, field = STANDARD_CACHE_FIELD_DIMENSIONS[name]
        values = dimensions.get(category)
        if not isinstance(values, dict) or field not in values:
            raise ValueError(f"run-index cache projection is missing {category}.{field}: {entry.get('path')}")
        value = values[field]
        reason_key = f"{category}.{field}"
        if value is None:
            if not isinstance(reasons.get(reason_key), str) or not reasons[reason_key]:
                raise ValueError(
                    f"null cache configuration needs a reason for {reason_key}: {entry.get('path')}"
                )
        elif not _matches_arrow_type(value, STANDARD_CACHE_FIELD_TYPES[name]):
            raise ValueError(f"invalid type for {reason_key}: {entry.get('path')}")
        if name == "kv_offload_backend" and value is not None and value not in CACHE_OFFLOAD_BACKENDS:
            raise ValueError(f"invalid KV offload backend for {entry.get('path')}: {value!r}")


def load_index(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "run-index-v1":
        raise ValueError("run index must declare schema_version=run-index-v1")
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("run index runs must be a list")
    return runs


def _canonical_source_path(path: str) -> str:
    return (
        path.replace("data/run-canonical-run-", "data/run-", 1)
        if path.startswith("data/run-canonical-run-")
        else path
    )


def validate_shard_coverage(entries: list[dict[str, object]], canonical_paths: set[str]) -> None:
    """Require a source-path index entry for each published canonical run shard."""
    indexed_paths = {_canonical_source_path(str(entry["path"])) for entry in entries}
    canonical_sources = {_canonical_source_path(path) for path in canonical_paths}
    missing = sorted(canonical_sources - indexed_paths)
    if missing:
        raise ValueError(
            f"canonical run shards missing from run index ({len(missing)}): {', '.join(missing)}"
        )


def validate_entries(entries: list[dict[str, object]]) -> None:
    paths: set[str] = set()
    for entry in entries:
        missing = REQUIRED_FIELDS - set(entry)
        if missing:
            raise ValueError(f"run index entry missing fields: {sorted(missing)}")
        path = entry["path"]
        if not isinstance(path, str) or not path.startswith("data/run-") or not path.endswith(".parquet"):
            raise ValueError(f"invalid parquet path: {path!r}")
        source_path = _canonical_source_path(path)
        if source_path in paths:
            raise ValueError(f"duplicate parquet path: {path}")
        paths.add(source_path)
        if entry["status"] not in VALID_STATUSES:
            raise ValueError(f"invalid status for {path}: {entry['status']!r}")
        date_match = RUN_DATE.match(path)
        if date_match and datetime.date.fromisoformat(date_match.group(1)) >= BACKFILL_START:
            validate_cache_projection(entry)
        if (
            date_match
            and datetime.date.fromisoformat(date_match.group(1)) >= datetime.date(2026, 9, 1)
            and entry.get("machine") not in CANONICAL_MACHINES
        ):
            raise ValueError(f"missing or invalid machine label for {path}")
        if entry["status"] != "scored":
            continue
        engine = entry.get("engine")
        version = entry.get("engine_version")
        if engine not in {"vLLM", "llama.cpp"} or not isinstance(version, str) or not version:
            raise ValueError(f"scored run is missing a normalized engine or version: {path}")
        if engine == "vLLM" and version not in {"0.31.0", "0.32.0"}:
            raise ValueError(f"unsupported vLLM version for {path}: {version!r}")
        if engine == "llama.cpp":
            if not LLAMA_CPP_VERSION.fullmatch(version):
                raise ValueError(f"llama.cpp version must use its release tag for {path}: {version!r}")
            build = entry.get("engine_build")
            commit = entry.get("engine_commit")
            if build != version or not isinstance(commit, str) or not ENGINE_COMMIT.fullmatch(commit):
                raise ValueError(f"llama.cpp build and commit must be recorded separately for {path}")
    for entry in entries:
        if entry.get("engine") == "llama.cpp" and entry.get("engine_version") is None:
            caveat = entry.get("caveat")
            if entry.get("status") != "experimental" or not isinstance(caveat, str):
                raise ValueError("uncaptured llama.cpp builds must be experimental with a caveat")
            if "engine build not captured" not in caveat:
                raise ValueError("uncaptured llama.cpp build caveat is missing")


def validate_remote(entries: list[dict[str, object]], repo_id: str) -> None:
    api = HfApi()
    remote_paths = {
        item.path
        for item in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True)
        if item.path.startswith("data/")
    }
    # Newer runs are published only as canonical shards (data/run-canonical-run-...).
    missing = sorted(
        str(entry["path"])
        for entry in entries
        if entry["path"] not in remote_paths
        and str(entry["path"]).replace("data/", "data/run-canonical-", 1) not in remote_paths
    )
    if missing:
        raise ValueError(f"indexed parquet files missing from {repo_id}: {', '.join(missing)}")
    canonical_run_paths = {
        path
        for path in remote_paths
        if path.startswith("data/run-canonical-run-") and path.endswith(".parquet")
    }
    validate_shard_coverage(entries, canonical_run_paths)
    recent_path = api.hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=RECENT_SHARD_PATH)
    recent_rows = pq.read_table(recent_path).to_pylist()
    projected_rows = normalize_legacy_rows(recent_rows)
    if len(projected_rows) != len(recent_rows):
        raise ValueError(
            f"normalization changed recent dataset row count: {len(recent_rows)} to {len(projected_rows)}"
        )
    hardware_dimensions = [_hardware_dimensions(row) for row in recent_rows]
    columns = {
        "machine": "campaign_hardware_machine",
        "accelerator_model": "campaign_hardware_accelerator_model",
        "accelerator_memory_gb": "campaign_hardware_accelerator_memory_gb",
    }
    uncovered = {
        field: sum(
            dimensions.get(field) is not None and projected[column] is None
            for dimensions, projected in zip(hardware_dimensions, projected_rows, strict=True)
        )
        for field, column in columns.items()
    }
    if any(uncovered.values()):
        missing = ", ".join(f"{field}={count}" for field, count in uncovered.items() if count)
        raise ValueError(f"recent dataset hardware dimensions lack campaign projections: {missing}")
    print(
        "validated recent dataset hardware projections for "
        + ", ".join(
            f"{field}={sum(dimensions.get(field) is not None for dimensions in hardware_dimensions)}"
            for field in columns
        )
    )


def _hardware_dimensions(row: dict[str, object]) -> dict[str, object]:
    hardware = row.get("hardware")
    if isinstance(hardware, str):
        try:
            hardware = json.loads(hardware)
        except json.JSONDecodeError:
            return {}
    return hardware if isinstance(hardware, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the canonical HF run index")
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--remote", action="store_true", help="Confirm indexed shards exist on Hugging Face")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    args = parser.parse_args()
    entries = load_index(args.index)
    validate_entries(entries)
    if args.remote:
        validate_remote(entries, args.repo_id)
    print(f"validated {len(entries)} run-index entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
