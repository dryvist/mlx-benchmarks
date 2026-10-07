"""Flat bench-events JSONL feed for log-pipeline ingest (issue #119).

The HF dataset stays the canonical, schema-validated store. This module
appends one flat JSON line **per result row** of every published shard to a
persistent local log that the existing log shipper tails — no new transport.
The file is derived state: ``mlx-bench-events replay`` regenerates it from
the HF dataset at any time.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from mlx_benchmarks.envelope import Envelope
from mlx_benchmarks.publish import DEFAULT_REPO_ID, DEFAULT_REPO_TYPE, envelope_to_rows

log = logging.getLogger(__name__)

DEFAULT_EVENTS_PATH = Path.home() / ".local/state/mlx-benchmarks/bench-events.jsonl"


def envelope_events(envelope: Envelope, run_id: str) -> list[dict[str, Any]]:
    """Flat event dicts for every result row of ``envelope``.

    Reuses :func:`envelope_to_rows` (same flattening the parquet shard gets)
    plus ``run_id`` — the original run-shard basename — so events are joinable
    back to their HF shard and idempotently deduplicable.
    """
    return [{**row, "run_id": run_id} for row in envelope_to_rows(envelope)]


def append_events(envelope: Envelope, run_id: str, events_path: Path | None = None) -> int:
    """Append ``envelope``'s events to ``events_path``; return the line count."""
    if events_path is None:
        events_path = DEFAULT_EVENTS_PATH
    events = envelope_events(envelope, run_id)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, sort_keys=True) + "\n")
    return len(events)


def replay(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    events_path: Path | None = None,
    token: str | None = None,
) -> int:
    """Rebuild ``events_path`` from every parquet shard in the HF dataset.

    Rewrites the file atomically (tmp + rename) so a tailing shipper never
    sees a truncated file. Returns the total event-line count.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    api = HfApi(token=token or os.environ.get("HF_TOKEN") or None)
    shard_paths = sorted(
        p
        for p in api.list_repo_files(repo_id=repo_id, repo_type=DEFAULT_REPO_TYPE)
        if p.startswith("data/run-canonical-") and p.endswith(".parquet")
    )

    if events_path is None:
        events_path = DEFAULT_EVENTS_PATH
    tmp_path = events_path.with_suffix(".jsonl.tmp")
    events_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with tmp_path.open("w", encoding="utf-8") as fh:
        for shard in shard_paths:
            run_id = Path(shard).stem.removeprefix("run-canonical-")
            local = api.hf_hub_download(repo_id=repo_id, repo_type=DEFAULT_REPO_TYPE, filename=shard)
            table = pq.read_table(local)
            for row in table.to_pylist():
                fh.write(json.dumps({**row, "run_id": run_id}, sort_keys=True) + "\n")
                total += 1
    tmp_path.replace(events_path)
    log.info("replayed %d events from %d shards to %s", total, len(shard_paths), events_path)
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mlx-bench-events",
        description="Rebuild the bench-events JSONL feed from the HF dataset.",
    )
    parser.add_argument("action", choices=["replay"], help="Only 'replay' is supported")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset repo")
    parser.add_argument(
        "--events-path",
        type=Path,
        default=DEFAULT_EVENTS_PATH,
        help=f"Output JSONL path (default: {DEFAULT_EVENTS_PATH})",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    replay(repo_id=args.repo_id, events_path=args.events_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
