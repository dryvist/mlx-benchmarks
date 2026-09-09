"""coding-replay runner (harness/coding-replay/run.py) JSON Lines -> envelope v1.

One raw row per replayed task (see ``harness/coding-replay/run.py`` for the
row shape: task/repo/check identity, ``check_rc``, ``overlap``, ``pass``,
tokens, ttft, wall time). Each row becomes a per-task ``pass_at_1`` result
carrying task/repo/check as tags, plus one aggregate ``pass_rate`` result
across every row in the file — the suite's headline metric.

Tags are a WHITELIST, so a field the runner records reaches the published
parquet only if it is named here. The measurement-condition fields
(``dedicated``, ``reasoning_effort``, the runner's own ``tag``) were recorded
on every row and dropped at this boundary — visible in the raw JSON Lines,
absent from the artifact anyone reads. Passing the same value with the
publisher's ``--tag`` masked it, because that lands via ``ctx.extra_tags``
below. When adding a field to the runner, add it here too, or it goes nowhere.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

from mlx_benchmarks.converters.base import ConverterContext, apply_optional_fields
from mlx_benchmarks.envelope import Envelope, Result, System

log = logging.getLogger(__name__)


class CodingReplayConverter:
    kind = "coding-replay"

    def build_envelope(self, raw: list[dict[str, Any]] | dict[str, Any], ctx: ConverterContext) -> Envelope:
        rows = raw if isinstance(raw, list) else [raw]
        if not rows:
            raise ValueError("coding-replay raw results are empty — nothing to publish")

        timestamp = ctx.timestamp_override or _extract_timestamp(rows[0])
        system: System = ctx.system or {}  # type: ignore[assignment]

        results: list[Result] = [_task_result(row, ctx) for row in rows]
        results.append(_pass_rate_result(rows, ctx))

        envelope: Envelope = {
            "schema_version": "1",
            "timestamp": timestamp,
            "git_sha": ctx.git_sha,
            "trigger": ctx.trigger,
            "suite": ctx.suite,
            "model": ctx.model,
            "system": system,
            "results": results,
            "errors": [],
        }
        return apply_optional_fields(envelope, ctx)


def _task_result(row: dict[str, Any], ctx: ConverterContext) -> Result:
    result: Result = {
        "name": "coding_replay",
        "metric": "pass_at_1",
        "value": 1.0 if row.get("pass") else 0.0,
        "unit": "ratio",
        "tags": {
            "task": str(row.get("task", "")),
            "repo": str(row.get("repo", "")),
            "check": str(row.get("check", "")),
            "check_rc": str(row.get("check_rc", "")),
            "overlap": str(row.get("overlap", "")),
            # Measurement conditions. Defaults mirror the runner's own: an
            # absent value means the row predates the field, which is not the
            # same as the condition being false or known.
            "dedicated": str(row.get("dedicated", "unstated")),
            "reasoning_effort": str(row.get("reasoning_effort", "unstated")),
            "run_tag": str(row.get("tag", "")),
            **{k: str(v) for k, v in ctx.extra_tags.items()},
        },
    }
    wall = row.get("wall_s")
    if isinstance(wall, int | float):
        result["duration_seconds"] = float(wall)
    return result


def _pass_rate_result(rows: list[dict[str, Any]], ctx: ConverterContext) -> Result:
    passes = [1.0 if row.get("pass") else 0.0 for row in rows]
    return {
        "name": "coding_replay",
        "metric": "pass_rate",
        "value": round(sum(passes) / len(passes), 3),
        "unit": "ratio",
        "tags": {
            "n_tasks": str(len(rows)),
            # The headline row needs the conditions too — it is the row most
            # readers see, and a pass_rate without them invites comparison
            # across measurement classes.
            "dedicated": _run_level(rows, "dedicated"),
            "reasoning_effort": _run_level(rows, "reasoning_effort"),
            "run_tag": _run_level(rows, "tag", default=""),
            **{k: str(v) for k, v in ctx.extra_tags.items()},
        },
    }


def _run_level(rows: list[dict[str, Any]], key: str, default: str = "unstated") -> str:
    """A run-level condition, or ``mixed`` when the file merges disagreeing runs.

    Reading ``rows[0]`` would label a concatenated file with whichever run
    happened to be first, which is how a wrong condition becomes indistinguishable
    from a measured one.
    """
    values = {str(row.get(key, default)) for row in rows}
    return values.pop() if len(values) == 1 else "mixed"


def _extract_timestamp(row: dict[str, Any]) -> str:
    ts = row.get("timestamp")
    if isinstance(ts, str) and ts:
        return ts
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
