"""The dataset card's configurations name no calendar month and point at real publish outputs."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import DatasetCard

from mlx_benchmarks.dataset_schema import RECENT_SHARD_PATH, RECENT_WINDOW

CARD_PATH = Path(__file__).resolve().parents[1] / "dataset-card" / "README.md"
MONTH_AND_YEAR = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b"
)
YEAR_AND_MONTH = re.compile(r"\d{4}-\d{2}")
CANONICAL_GLOB = "data/run-canonical-*.parquet"


def _card() -> DatasetCard:
    return DatasetCard.load(CARD_PATH)


def _configs() -> list[dict[str, Any]]:
    return _card().data.to_dict()["configs"]


def _paths(config: dict[str, Any]) -> list[str]:
    return [entry["path"] for entry in config["data_files"]]


def test_default_config_reads_the_rolling_recent_file() -> None:
    defaults = [config for config in _configs() if config.get("default")]

    assert [config["config_name"] for config in defaults] == ["default"]
    assert _paths(defaults[0]) == [RECENT_SHARD_PATH]


def test_history_config_reads_every_canonical_shard() -> None:
    history = next(config for config in _configs() if config["config_name"] == "history")

    assert _paths(history) == [CANONICAL_GLOB]


def test_no_config_path_or_prose_names_a_calendar_month() -> None:
    for config in _configs():
        for path in _paths(config):
            assert not YEAR_AND_MONTH.search(path), f"{config['config_name']} path names a month: {path}"
    assert not MONTH_AND_YEAR.search(_card().text)


def test_card_states_the_window_the_publish_step_writes() -> None:
    assert f"{RECENT_WINDOW.days} days" in _card().text
    assert RECENT_SHARD_PATH in _card().text


def test_recent_file_is_outside_the_top_level_shard_glob() -> None:
    assert not PurePosixPath(RECENT_SHARD_PATH).match(CANONICAL_GLOB)
    assert "/" in RECENT_SHARD_PATH.removeprefix("data/")
