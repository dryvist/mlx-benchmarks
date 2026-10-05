"""Cover events.py — JSONL emission on publish and replay from parquet."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from mlx_benchmarks.envelope import Envelope
from mlx_benchmarks.events import append_events, envelope_events, replay
from mlx_benchmarks.publish import envelope_to_rows, publish, rows_to_parquet


def test_envelope_events_shape(valid_envelope: Envelope) -> None:
    events = envelope_events(valid_envelope, run_id="run-x-abc123")
    assert len(events) == len(valid_envelope["results"])
    event = events[0]
    assert event["run_id"] == "run-x-abc123"
    for field in ("timestamp", "model", "suite", "metric", "value", "git_sha", "hostname"):
        assert field in event


def test_append_events_writes_jsonl(valid_envelope: Envelope, tmp_path: Path) -> None:
    events_path = tmp_path / "state/bench-events.jsonl"
    count = append_events(valid_envelope, run_id="r1", events_path=events_path)
    count += append_events(valid_envelope, run_id="r2", events_path=events_path)
    lines = events_path.read_text().splitlines()
    assert len(lines) == count == 2 * len(valid_envelope["results"])
    parsed = [json.loads(line) for line in lines]
    assert {e["run_id"] for e in parsed} == {"r1", "r2"}


def test_publish_appends_events(valid_envelope: Envelope, tmp_path: Path) -> None:
    events_path = tmp_path / "bench-events.jsonl"
    with (
        patch("mlx_benchmarks.publish.HfApi") as api_cls,
        patch("mlx_benchmarks.events.DEFAULT_EVENTS_PATH", events_path),
    ):
        path = publish(valid_envelope, token="dummy")
    api_cls.return_value.create_commit.assert_called_once()
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert len(lines) == len(valid_envelope["results"])
    # run_id joins the event back to its HF shard.
    assert all(e["run_id"] == Path(path).stem.removeprefix("run-canonical-") for e in lines)


def test_publish_dry_run_emits_nothing(valid_envelope: Envelope, tmp_path: Path) -> None:
    events_path = tmp_path / "bench-events.jsonl"
    with patch("mlx_benchmarks.events.DEFAULT_EVENTS_PATH", events_path):
        publish(valid_envelope, dry_run=True)
    assert not events_path.exists()


def test_replay_rebuilds_from_shards(valid_envelope: Envelope, tmp_path: Path) -> None:
    shard = tmp_path / "run-a.parquet"
    shard.write_bytes(rows_to_parquet(envelope_to_rows(valid_envelope)))
    events_path = tmp_path / "bench-events.jsonl"
    events_path.write_text("stale line\n")

    api = MagicMock()
    api.list_repo_files.return_value = [
        "data/run-a.parquet",
        "data/run-canonical-run-a.parquet",
        "README.md",
    ]
    api.hf_hub_download.return_value = str(shard)
    with patch("huggingface_hub.HfApi", return_value=api):
        total = replay(events_path=events_path)

    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert total == len(lines) == len(valid_envelope["results"])
    # Stale content replaced atomically, and original shards are not replayed twice.
    assert all(e["run_id"] == "run-a" for e in lines)
    api.hf_hub_download.assert_called_once()
    assert api.hf_hub_download.call_args.kwargs["filename"] == "data/run-canonical-run-a.parquet"
