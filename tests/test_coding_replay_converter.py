"""End-to-end: coding-replay runner rows -> envelope -> passes schema validation."""

from __future__ import annotations

import pytest

from mlx_benchmarks.converters import get_converter
from mlx_benchmarks.converters.base import ConverterContext
from mlx_benchmarks.envelope import validate_envelope
from mlx_benchmarks.system import detect_system


def _ctx(**overrides: object) -> ConverterContext:
    defaults: dict = {
        "suite": "coding",
        "model": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        "git_sha": "deadbeef",
        "system": detect_system(),
    }
    defaults.update(overrides)
    return ConverterContext(**defaults)


def test_coding_replay_round_trip(coding_replay_sample: list[dict]) -> None:
    converter = get_converter("coding-replay")
    envelope = converter.build_envelope(coding_replay_sample, _ctx())
    validate_envelope(envelope)

    assert envelope["suite"] == "coding"
    assert envelope["timestamp"] == "2026-09-06T00:00:00Z"

    results = envelope["results"]
    assert all(r["name"] == "coding_replay" for r in results)
    metric_names = [r["metric"] for r in results]
    # 2 tasks -> 2 pass_at_1 rows + 1 aggregate pass_rate row.
    assert metric_names.count("pass_at_1") == 2
    assert metric_names.count("pass_rate") == 1


def test_coding_replay_pass_rate_is_the_headline_aggregate(coding_replay_sample: list[dict]) -> None:
    # Fixture: one passing task, one failing task -> pass_rate == 0.5.
    converter = get_converter("coding-replay")
    envelope = converter.build_envelope(coding_replay_sample, _ctx())
    validate_envelope(envelope)

    rate = next(r for r in envelope["results"] if r["metric"] == "pass_rate")
    assert rate["value"] == 0.5
    assert rate["unit"] == "ratio"
    assert rate["tags"]["n_tasks"] == "2"


def test_coding_replay_per_task_tags_and_duration(coding_replay_sample: list[dict]) -> None:
    converter = get_converter("coding-replay")
    envelope = converter.build_envelope(coding_replay_sample, _ctx())
    validate_envelope(envelope)

    passing = next(
        r
        for r in envelope["results"]
        if r["metric"] == "pass_at_1" and r["tags"]["task"] == "tofu-proxmox-1046"
    )
    assert passing["value"] == 1.0
    assert passing["tags"]["repo"] == "dryvist/tofu-proxmox"
    assert passing["tags"]["check"] == "none"
    assert passing["duration_seconds"] == 99.9

    failing = next(
        r
        for r in envelope["results"]
        if r["metric"] == "pass_at_1" and r["tags"]["task"] == "tofu-proxmox-1049"
    )
    assert failing["value"] == 0.0
    assert failing["tags"]["overlap"] == "0"


def test_coding_replay_extra_tags(coding_replay_sample: list[dict]) -> None:
    converter = get_converter("coding-replay")
    envelope = converter.build_envelope(coding_replay_sample, _ctx(extra_tags={"host": "mac-studio"}))
    validate_envelope(envelope)
    assert all(r["tags"]["host"] == "mac-studio" for r in envelope["results"])


def test_coding_replay_passes_through_run_context(coding_replay_sample: list[dict]) -> None:
    converter = get_converter("coding-replay")
    envelope = converter.build_envelope(
        coding_replay_sample, _ctx(env_class="isolated", serving={"stack": "mlx_lm.server"})
    )
    validate_envelope(envelope)
    assert envelope["env_class"] == "isolated"
    assert envelope["serving"] == {"stack": "mlx_lm.server"}


def test_coding_replay_empty_rows_is_an_error() -> None:
    converter = get_converter("coding-replay")
    with pytest.raises(ValueError, match="nothing to publish"):
        converter.build_envelope([], _ctx())


def test_coding_replay_accepts_a_single_dict_row() -> None:
    # cli.py only splits .jsonl input on newlines into a list; a lone dict
    # (e.g. a hand-built raw result) must still convert.
    converter = get_converter("coding-replay")
    row = {"task": "x-1", "repo": "o/x", "check": "none", "check_rc": 0, "overlap": 1, "pass": True}
    envelope = converter.build_envelope(row, _ctx())
    validate_envelope(envelope)
    assert len(envelope["results"]) == 2


def test_measurement_conditions_reach_the_parquet_without_a_publisher_tag(
    coding_replay_sample: list[dict],
) -> None:
    """The regression test for the whitelist leak.

    ``dedicated`` and ``reasoning_effort`` are recorded on every runner row, but
    converter tags are a whitelist — so both were dropped here and never reached
    the artifact. Shards published before this fix *looked* correct only because
    ``--tag dedicated=…`` was passed by hand, which lands via ``extra_tags``
    instead. So this test passes NO extra_tags: the row fields have to carry
    themselves the whole way to a parquet column.
    """
    import io

    import pyarrow.parquet as pq

    from mlx_benchmarks.publish import envelope_to_rows, rows_to_parquet

    rows = [dict(row, dedicated=True, reasoning_effort="xhigh") for row in coding_replay_sample]

    converter = get_converter("coding-replay")
    envelope = converter.build_envelope(rows, _ctx())
    validate_envelope(envelope)

    table = pq.read_table(io.BytesIO(rows_to_parquet(envelope_to_rows(envelope))))
    assert "tag_dedicated" in table.column_names
    assert "tag_reasoning_effort" in table.column_names
    assert set(table.column("tag_dedicated").to_pylist()) == {"True"}
    assert set(table.column("tag_reasoning_effort").to_pylist()) == {"xhigh"}
    # The runner's own --tag is a measurement condition too: it is how a
    # replication is told from the run it replicates.
    assert set(table.column("tag_run_tag").to_pylist()) == {"run1"}


def test_absent_conditions_publish_as_unstated_not_as_false(
    coding_replay_sample: list[dict],
) -> None:
    """Anti-vacuity: the fixture carries neither field, and the columns still
    appear. A missing value must not read as ``False``/``off`` — that is a claim
    about the run, and no measurement was made."""
    converter = get_converter("coding-replay")
    envelope = converter.build_envelope(coding_replay_sample, _ctx())
    validate_envelope(envelope)
    for result in envelope["results"]:
        assert result["tags"]["dedicated"] == "unstated"
        assert result["tags"]["reasoning_effort"] == "unstated"


def test_a_merged_file_reports_mixed_rather_than_the_first_rows_condition(
    coding_replay_sample: list[dict],
) -> None:
    """Concatenated runs are a real shape here (``cat run*.jsonl``). Labelling
    the aggregate from ``rows[0]`` would attach one arm's effort to another's
    result — a wrong condition is worse than an absent one."""
    rows = [
        dict(coding_replay_sample[0], reasoning_effort="high"),
        dict(coding_replay_sample[1], reasoning_effort="xhigh"),
    ]
    converter = get_converter("coding-replay")
    envelope = converter.build_envelope(rows, _ctx())
    validate_envelope(envelope)

    rate = next(r for r in envelope["results"] if r["metric"] == "pass_rate")
    assert rate["tags"]["reasoning_effort"] == "mixed"
    # Per-task rows keep their own verbatim value; only the aggregate is mixed.
    per_task = {
        r["tags"]["task"]: r["tags"]["reasoning_effort"]
        for r in envelope["results"]
        if r["metric"] == "pass_at_1"
    }
    assert per_task == {"tofu-proxmox-1046": "high", "tofu-proxmox-1049": "xhigh"}
