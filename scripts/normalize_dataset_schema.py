"""Publish schema-normalized copies of historical Hugging Face shards."""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi

from mlx_benchmarks.dataset_schema import PARQUET_ROW_SCHEMA, normalize_legacy_rows
from mlx_benchmarks.publish import canonical_shard_path

DEFAULT_REPO_ID = "JacobPEvans/mlx-benchmarks"
DATASET_CARD_PATH = Path("dataset-card/README.md")


def normalize_legacy_table(table: pa.Table) -> pa.Table:
    """Pad one historical table to the canonical schema without changing rows."""
    rows = normalize_legacy_rows(table.to_pylist())
    return pa.Table.from_pylist(rows, schema=PARQUET_ROW_SCHEMA)


def _parquet_bytes(table: pa.Table) -> bytes:
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def _read_table(api: HfApi, repo_id: str, path: str) -> pa.Table:
    local_path = api.hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=path)
    return pq.read_table(local_path)


def normalize_dataset(repo_id: str, *, apply: bool) -> tuple[int, int, int]:
    """Add canonical copies and the dataset card in one Hub commit.

    Original paths are retained. Existing canonical paths are checked and
    skipped, so rerunning this command never overwrites a published result.
    """
    token = os.environ.get("HF_TOKEN")
    if apply and not token:
        raise RuntimeError("HF_TOKEN must be set for the write operation")
    api = HfApi(token=token)
    repo_paths = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    originals = sorted(
        path
        for path in repo_paths
        if path.startswith("data/")
        and "/" not in path.removeprefix("data/")
        and path.endswith(".parquet")
        and not path.startswith("data/run-canonical-")
    )
    operations: list[CommitOperationAdd] = []
    source_rows = 0
    existing_copies = 0
    canonical_to_add = 0
    for source_path in originals:
        source = _read_table(api, repo_id, source_path)
        target_path = canonical_shard_path(source_path)
        source_rows += source.num_rows
        normalized = normalize_legacy_table(source)
        if normalized.num_rows != source.num_rows:
            raise RuntimeError(f"normalization changed the row count for {source_path}")
        if not normalized.schema.equals(PARQUET_ROW_SCHEMA, check_metadata=False):
            raise RuntimeError(f"normalization produced an unexpected schema for {source_path}")
        if target_path in repo_paths:
            existing = _read_table(api, repo_id, target_path)
            if not existing.schema.equals(PARQUET_ROW_SCHEMA, check_metadata=False):
                raise RuntimeError(f"canonical shard has an unexpected schema: {target_path}")
            if existing.num_rows != source.num_rows:
                raise RuntimeError(f"canonical shard row count differs from its source: {target_path}")
            if existing.to_pylist() != normalized.to_pylist():
                raise RuntimeError(f"canonical shard differs from its source: {target_path}")
            existing_copies += 1
            continue
        operations.append(
            CommitOperationAdd(
                path_in_repo=target_path,
                path_or_fileobj=_parquet_bytes(normalized),
            )
        )
        canonical_to_add += 1

    card = DATASET_CARD_PATH.read_bytes()
    remote_card_path = api.hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="README.md")
    card_changed = Path(remote_card_path).read_bytes() != card
    if card_changed:
        operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card))

    print(
        f"source_shards={len(originals)} source_rows={source_rows} "
        f"canonical_existing={existing_copies} canonical_to_add={canonical_to_add} "
        f"dataset_card_update={str(card_changed).lower()}"
    )
    if not apply:
        print("mode=dry-run; no Hub files changed")
        return len(originals), source_rows, existing_copies
    if operations:
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message="fix(dataset): normalize published parquet schemas",
        )
        print(f"published_operations={len(operations)}")
    else:
        print("mode=apply; canonical dataset already current")
    return len(originals), source_rows, existing_copies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--apply", action="store_true", help="Upload normalized copies and the dataset card")
    args = parser.parse_args()
    normalize_dataset(args.repo_id, apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
