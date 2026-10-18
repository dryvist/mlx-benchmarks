from __future__ import annotations

import argparse
import datetime
import json
import re
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
CANONICAL_MACHINES = {
    "Mac Studio M4 Max 128GB",
    "RTX PRO 6000 Max-Q 96GB",
    "RTX 4080 SUPER 16GB",
}
RUN_DATE = re.compile(r"^data/run-(\d{4}-\d{2}-\d{2})")
LLAMA_CPP_VERSION = re.compile(r"b\d+\Z")
ENGINE_COMMIT = re.compile(r"[0-9a-f]{7,40}\Z")


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
        date_match = RUN_DATE.match(path)
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
