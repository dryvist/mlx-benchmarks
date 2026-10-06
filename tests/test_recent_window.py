"""The rolling recent shard: window selection, row order, and the publish step that writes it."""

from __future__ import annotations

import datetime
import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mlx_benchmarks.dataset_schema import (
    PARQUET_ROW_SCHEMA,
    RECENT_SHARD_PATH,
    RECENT_WINDOW,
    recent_shard_paths,
    recent_shard_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = REPO_ROOT / "dataset-card" / "README.md"
SUFFIX = "abc1234-throughput-example-model-0a1b2c3d"


def _shard_path(stamp: str) -> str:
    return f"data/run-canonical-run-{stamp}-{SUFFIX}.parquet"


def _table(*stamps_and_values: tuple[str, float]) -> pa.Table:
    rows = [
        {
            "schema_version": "1",
            "timestamp": stamp,
            "suite": "throughput",
            "model": "example/model",
            "name": "request",
            "metric": "throughput",
            "value": value,
            "unit": "tokens_per_second",
        }
        for stamp, value in stamps_and_values
    ]
    return pa.Table.from_pylist(rows, schema=PARQUET_ROW_SCHEMA)


def _parquet(table: pa.Table) -> bytes:
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def _stamp(moment: datetime.datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H-%M-%S")


def test_window_ends_at_the_newest_shard_and_crosses_month_and_year_boundaries() -> None:
    newest_moment = datetime.datetime(2027, 1, 5)
    newest = _shard_path(_stamp(newest_moment))
    at_the_edge = _shard_path(_stamp(newest_moment - RECENT_WINDOW))
    just_outside = _shard_path(_stamp(newest_moment - RECENT_WINDOW - datetime.timedelta(seconds=1)))
    inside = _shard_path(_stamp(newest_moment - RECENT_WINDOW / 2))

    assert newest_moment.year != (newest_moment - RECENT_WINDOW / 2).year
    assert recent_shard_paths([newest, just_outside, at_the_edge, inside]) == [at_the_edge, inside, newest]


def test_window_follows_the_newest_shard_not_the_clock() -> None:
    old = [_shard_path("2020-03-01T00-00-00"), _shard_path("2020-03-02T00-00-00")]

    assert recent_shard_paths(old) == old


def test_window_leaves_out_shards_without_a_run_timestamp_and_nested_paths() -> None:
    kept = _shard_path("2026-09-09T12-00-00")
    ignored = [
        "data/run-canonical-run-rescue-older-evaluations.parquet",
        "data/run-canonical-train-00000-of-00001.parquet",
        f"data/run-{kept.removeprefix('data/run-canonical-run-')}",
        f"data/nested/{kept.removeprefix('data/')}",
        RECENT_SHARD_PATH,
        "README.md",
    ]

    assert recent_shard_paths([*ignored, kept]) == [kept]
    assert recent_shard_paths(ignored) == []


def test_recent_table_orders_runs_newest_first_and_keeps_published_order_within_a_run() -> None:
    older = _table(("2026-09-01T10:00:00Z", 1.0), ("2026-09-01T10:00:00Z", 2.0))
    newer = _table(("2026-09-09T10:00:00Z", 3.0), ("2026-09-09T10:00:00Z", 4.0))

    combined = recent_shard_table([older, newer])

    assert combined.schema.equals(PARQUET_ROW_SCHEMA, check_metadata=False)
    assert combined.column("value").to_pylist() == [3.0, 4.0, 1.0, 2.0]


def test_recent_table_pads_a_shard_written_before_a_column_existed() -> None:
    legacy_schema = pa.schema(
        [field for field in PARQUET_ROW_SCHEMA if not field.name.startswith("campaign_")]
    )
    legacy = pa.Table.from_pylist(
        [{"schema_version": "1", "timestamp": "2026-09-01T10:00:00Z", "value": 1.0}], schema=legacy_schema
    )

    combined = recent_shard_table([legacy, _table(("2026-09-09T10:00:00Z", 2.0))])

    assert combined.schema.equals(PARQUET_ROW_SCHEMA, check_metadata=False)
    assert combined.num_rows == 2
    assert combined.column("campaign_hardware_machine").to_pylist() == [None, None]


def test_recent_table_rejects_columns_outside_the_canonical_schema() -> None:
    stray = pa.table({"timestamp": ["2026-09-09T10:00:00Z"], "not_a_column": [1]})

    with pytest.raises(ValueError, match="not_a_column"):
        recent_shard_table([stray])


def test_recent_table_needs_at_least_one_shard() -> None:
    with pytest.raises(ValueError, match="no canonical shards"):
        recent_shard_table([])


class FakeHub:
    """In-memory stand-in for ``HfApi`` over one dataset repository of Parquet bytes."""

    def __init__(self, files: dict[str, bytes], workdir: Path) -> None:
        self.files = files
        self.workdir = workdir
        self.commits: list[dict[str, bytes]] = []

    def __call__(self, token: str | None = None) -> FakeHub:
        return self

    def list_repo_files(self, *, repo_id: str, repo_type: str) -> list[str]:
        return sorted(self.files)

    def hf_hub_download(self, *, repo_id: str, repo_type: str, filename: str) -> str:
        target = self.workdir / "downloads" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[filename])
        return str(target)

    def create_commit(
        self, *, repo_id: str, repo_type: str, operations: list[Any], commit_message: str
    ) -> None:
        added = {operation.path_in_repo: operation.path_or_fileobj for operation in operations}
        self.commits.append(added)
        self.files.update(added)

    def recent_values(self) -> list[float]:
        return pq.read_table(io.BytesIO(self.files[RECENT_SHARD_PATH])).column("value").to_pylist()


def _load_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "normalize_dataset_schema.py"
    spec = importlib.util.spec_from_file_location("normalize_dataset_schema", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def publish_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[ModuleType, FakeHub]:
    script = _load_script()
    hub = FakeHub({"README.md": b"previous card"}, tmp_path)
    monkeypatch.setattr(script, "HfApi", hub)
    monkeypatch.setattr(script, "DATASET_CARD_PATH", CARD_PATH)
    monkeypatch.setenv("HF_TOKEN", "unused")
    monkeypatch.delenv("MACHINE_LABELS_JSON", raising=False)
    return script, hub


def test_publish_step_writes_the_window_once_and_rolls_it_with_new_shards(
    publish_step: tuple[ModuleType, FakeHub], capsys: pytest.CaptureFixture[str]
) -> None:
    script, hub = publish_step
    hub.files[_shard_path("2026-08-01T10-00-00")] = _parquet(_table(("2026-08-01T10:00:00Z", 1.0)))
    hub.files[_shard_path("2026-09-01T10-00-00")] = _parquet(_table(("2026-09-01T10:00:00Z", 2.0)))
    hub.files[_shard_path("2026-09-09T10-00-00")] = _parquet(_table(("2026-09-09T10:00:00Z", 3.0)))
    hub.files["data/run-canonical-run-rescue-older-evaluations.parquet"] = _parquet(
        _table(("2026-04-01T10:00:00Z", 9.0))
    )

    script.normalize_dataset("example/dataset", apply=True)

    assert set(hub.commits[0]) == {RECENT_SHARD_PATH, "README.md"}
    assert hub.recent_values() == [3.0, 2.0]
    assert "recent_shards=2 recent_rows=2 recent_update=true" in capsys.readouterr().out

    script.normalize_dataset("example/dataset", apply=True)
    assert len(hub.commits) == 1, "an unchanged window and card must not produce another commit"

    hub.files[_shard_path("2026-10-15T10-00-00")] = _parquet(_table(("2026-10-15T10:00:00Z", 4.0)))
    script.normalize_dataset("example/dataset", apply=True)

    assert set(hub.commits[1]) == {RECENT_SHARD_PATH}, "rolling forward must not touch the card"
    assert hub.recent_values() == [4.0]


def test_publish_step_windows_a_shard_it_adds_in_the_same_commit(
    publish_step: tuple[ModuleType, FakeHub],
) -> None:
    script, hub = publish_step
    original = "data/run-2026-09-09T10-00-00-abc1234-throughput-example-model-0a1b2c3d.parquet"
    hub.files[original] = _parquet(_table(("2026-09-09T10:00:00Z", 5.0)))

    script.normalize_dataset("example/dataset", apply=True)

    assert _shard_path("2026-09-09T10-00-00") in hub.commits[0]
    assert hub.recent_values() == [5.0]
    assert original in hub.files


def test_publish_step_does_not_write_a_card_that_names_a_missing_window(
    publish_step: tuple[ModuleType, FakeHub],
) -> None:
    script, hub = publish_step
    hub.files["data/run-canonical-run-rescue-older-evaluations.parquet"] = _parquet(
        _table(("2026-04-01T10:00:00Z", 9.0))
    )

    with pytest.raises(ValueError, match="no canonical shards"):
        script.normalize_dataset("example/dataset", apply=True)

    assert hub.commits == []
