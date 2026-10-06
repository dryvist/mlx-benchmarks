"""Converter protocol: raw tool output -> envelope v1."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from mlx_benchmarks.envelope import CampaignDimensions, Envelope, Result, Serving, System


@dataclass(slots=True)
class ConverterContext:
    """Inputs a converter needs beyond the raw tool output itself."""

    suite: str
    model: str
    git_sha: str
    trigger: str = "local"
    pr_number: int | None = None
    # Run-provenance fields common to every suite; ``None`` -> omitted from the
    # envelope. env_class = machine load class (isolated / under-load),
    # concurrency = in-flight request count, serving = inference-server identity.
    env_class: str | None = None
    concurrency: int | None = None
    # reasoning_effort = the thinking level the run asked for, verbatim. Declared
    # by the caller because no harness can see a serving default or an agent
    # CLI's own config, and a guess is indistinguishable from a measurement.
    reasoning_effort: str | None = None
    serving: Serving | None = None
    # The campaign dimension groups and the reason code for each explicit null in
    # them. Declared by the caller (the CLI reads them from a file): a converter
    # sees one tool's output and cannot observe the host, build, or model facts.
    campaign_dimensions: CampaignDimensions | None = None
    dimension_null_reasons: dict[str, str] | None = None
    timestamp_override: str | None = None
    system: dict[str, Any] | None = None
    extra_tags: dict[str, str] = field(default_factory=dict)
    # Path to the raw results file on disk, when the converter is invoked via the
    # CLI. Lets converters locate sibling artefacts (e.g. lm-eval's
    # ``samples_*.jsonl`` files) for per-sample analysis like token counting.
    # Optional so library callers that already have the raw dict in memory can
    # keep working unchanged.
    source_path: Path | None = None


def apply_optional_fields(envelope: Envelope, ctx: ConverterContext) -> Envelope:
    """Copy optional run-context fields onto a built envelope, in place.

    Centralizes the ``pr_number`` / ``env_class`` / ``concurrency`` /
    ``reasoning_effort`` / ``serving`` / ``campaign_dimensions`` /
    ``dimension_null_reasons`` pass-through so every converter shares one omission
    rule: a field left unset on the context is absent from the envelope.
    Adding a run-context field here reaches every converter at once, which is why
    it belongs here rather than in each converter's own tag whitelist. Returns the
    same envelope for convenient ``return apply_optional_fields(...)`` use.
    """
    if ctx.pr_number is not None:
        envelope["pr_number"] = ctx.pr_number
    if ctx.env_class is not None:
        envelope["env_class"] = ctx.env_class
    if ctx.concurrency is not None:
        envelope["concurrency"] = ctx.concurrency
    if ctx.reasoning_effort is not None:
        envelope["reasoning_effort"] = ctx.reasoning_effort
    if ctx.serving is not None:
        envelope["serving"] = ctx.serving
    if ctx.campaign_dimensions is not None:
        envelope["campaign_dimensions"] = ctx.campaign_dimensions
    if ctx.dimension_null_reasons is not None:
        envelope["dimension_null_reasons"] = ctx.dimension_null_reasons
    return envelope


def simple_envelope(
    ctx: ConverterContext,
    results: list[Result],
    errors: list[str] | None = None,
    timestamp: str | None = None,
) -> Envelope:
    """Build an envelope around ``results`` for tools whose output carries no run metadata.

    The hardware-baseline converters (fio, nvbandwidth, mbw, gpu-burn) differ only
    in how they parse; the envelope around the rows is identical, so it lives here
    once. ``timestamp`` is the tool's own start time when it prints one.
    """
    now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    system: System = ctx.system or {}  # type: ignore[assignment]
    envelope: Envelope = {
        "schema_version": "1",
        "timestamp": ctx.timestamp_override or timestamp or now,
        "git_sha": ctx.git_sha,
        "trigger": ctx.trigger,
        "suite": ctx.suite,
        "model": ctx.model,
        "system": system,
        "results": results,
        "errors": errors or [],
    }
    return apply_optional_fields(envelope, ctx)


def thinking_level(value: Any) -> str:
    """The effort a cell ran at: verbatim for a graded run, on/off for a bool.

    Shared by every harness that sweeps thinking, because each one had its own
    copy and two of the three inverted the answer: ``"on" if value else "off"``
    returns ``"on"`` for the *string* ``"off"``, since a non-empty string is
    truthy. A graded cell therefore published the opposite of what it ran.

    A bool is still accepted — every row written before the harnesses recorded a
    level carries one, and those rows must keep publishing the same two values.
    """
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "unstated"


class Converter(Protocol):
    """Implementations turn a parsed raw result into a valid :class:`Envelope`."""

    kind: str

    def build_envelope(
        self, raw: dict[str, Any], ctx: ConverterContext
    ) -> Envelope:  # pragma: no cover - protocol
        ...
