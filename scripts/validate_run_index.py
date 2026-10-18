from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

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


def load_index(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "run-index-v1":
        raise ValueError("run index must declare schema_version=run-index-v1")
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("run index runs must be a list")
    return runs


def validate_entries(entries: list[dict[str, object]]) -> None:
    paths: set[str] = set()
    for entry in entries:
        missing = REQUIRED_FIELDS - set(entry)
        if missing:
            raise ValueError(f"run index entry missing fields: {sorted(missing)}")
        path = entry["path"]
        if not isinstance(path, str) or not path.startswith("data/run-") or not path.endswith(".parquet"):
            raise ValueError(f"invalid parquet path: {path!r}")
        if path in paths:
            raise ValueError(f"duplicate parquet path: {path}")
        paths.add(path)
        if entry["status"] not in VALID_STATUSES:
            raise ValueError(f"invalid status for {path}: {entry['status']!r}")


def validate_remote(entries: list[dict[str, object]], repo_id: str) -> None:
    remote_paths = {
        item.path
        for item in HfApi().list_repo_tree(repo_id, repo_type="dataset", recursive=True)
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
