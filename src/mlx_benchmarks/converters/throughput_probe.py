from __future__ import annotations

import datetime
import logging
from typing import Any

from mlx_benchmarks.converters.base import ConverterContext, apply_optional_fields
from mlx_benchmarks.envelope import Envelope, Result, System

log = logging.getLogger(__name__)

_METRICS: tuple[tuple[str, str, str, str], ...] = (
    ("sequential", "cumulative_tok_s", "throughput_total_toks_per_s", "tok/s"),
    ("sequential", "decode_tok_s", "throughput_output_toks_per_s", "tok/s"),
    ("sequential", "prefill_tok_s", "throughput_prompt_toks_per_s", "tok/s"),
    ("sequential", "ttft_s", "ttft_p50_ms", "ms"),
    ("sequential", "total_s", "request_duration_p50_s", "s"),
    ("concurrent", "aggregate_cumulative_tok_s", "throughput_aggregate_toks_per_s", "tok/s"),
    ("concurrent", "aggregate_decode_tok_s", "throughput_aggregate_output_toks_per_s", "tok/s"),
)


class ThroughputProbeConverter:
    kind = "throughput-probe"

    def build_envelope(self, raw: Any, ctx: ConverterContext) -> Envelope:
        timestamp = (
            ctx.timestamp_override
            or raw.get("started_utc")
            or datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        system: System = ctx.system or {}  # type: ignore[assignment]
        envelope: Envelope = {
            "schema_version": "1",
            "timestamp": timestamp,
            "git_sha": ctx.git_sha,
            "trigger": ctx.trigger,
            "suite": ctx.suite,
            "model": ctx.model,
            "system": system,
            "results": self._iter_results(raw, ctx),
            "errors": [],
        }
        for field in (
            "campaign",
            "cell_status",
            "context",
            "campaign_dimensions",
            "dimension_null_reasons",
        ):
            value = raw.get(field)
            if value is not None:
                envelope[field] = value
        return apply_optional_fields(envelope, ctx)

    def _iter_results(self, raw: Any, ctx: ConverterContext) -> list[Result]:
        if not isinstance(raw, dict) or not isinstance(raw.get("sequential"), dict):
            log.warning("throughput-probe output has no sequential summary")
            return []

        sequential = raw["sequential"]
        tags = {k: str(v) for k, v in ctx.extra_tags.items()}
        for key in ("n_ok", "n_err", "answered_rate", "truncated_rate"):
            if key in sequential:
                tags[key] = str(sequential[key])
        if isinstance(sequential.get("finish_reasons"), list):
            tags["finish_reasons"] = ",".join(map(str, sequential["finish_reasons"]))
        # ``thinking`` on its own is not readable: a run that passed no kwarg
        # records "off" while the model reasons at its own default (see
        # harness/throughput/run.py). ``think_kwarg_sent`` is what separates
        # "we asked for off" from "we asked for nothing", and ``think_value``
        # carries the graded level the boolean cannot hold — both were recorded
        # on every row and dropped here.
        for key in (
            "max_tokens",
            "thinking",
            "think_kwarg",
            "think_value",
            "think_kwarg_sent",
            "dedicated",
            "context_tokens_target",
        ):
            if key in raw:
                tags[key] = "unstated" if raw[key] is None else str(raw[key])
        # No dedicated schema field for this — `tags` is already the envelope's
        # free-form string bag, so speculative-decoding metadata folds in here
        # rather than widening schema.json's additionalProperties:false shape.
        spec = raw.get("speculative_decoding")
        if isinstance(spec, dict):
            for key in (
                "draft_model",
                "spec_type",
                "draft_acceptance_rate",
                "net_benefit",
                "baseline_cumulative_tok_s",
                "spec_cumulative_tok_s",
                "baseline_load_error",
            ):
                if key in spec and spec[key] is not None:
                    tags[f"spec_{key}"] = str(spec[key])
        sequential_runs = raw.get("sequential_runs")
        if isinstance(sequential_runs, list):
            tags["measured_repetitions"] = str(len(sequential_runs))
            prompt_tokens = {
                run["prompt_tokens"]
                for run in sequential_runs
                if isinstance(run, dict) and isinstance(run.get("prompt_tokens"), int)
            }
            if prompt_tokens:
                tags["context_tokens_actual"] = ",".join(map(str, sorted(prompt_tokens)))

        concurrent = raw.get("concurrent")
        concurrent_tags: dict[str, str] | None = None
        if isinstance(concurrent, dict):
            concurrent_tags = dict(tags)
            concurrent_tags["phase"] = "concurrent"
            for key in ("width", "n_ok", "n_err", "wall_s"):
                if concurrent.get(key) is not None:
                    concurrent_tags[key] = str(concurrent[key])
            if concurrent.get("width") is not None:
                concurrent_tags["concurrency"] = str(concurrent["width"])
            errors = concurrent.get("errors")
            if isinstance(errors, list) and errors:
                # A partial aggregate covers only the requests that returned.
                concurrent_tags["concurrent_errors"] = str(len(errors))

        results: list[Result] = []
        for phase, source_key, metric, unit in _METRICS:
            summary = raw.get(phase)
            if not isinstance(summary, dict):
                continue

            source = summary.get(source_key)
            if phase == "sequential":
                if not isinstance(source, dict) or not isinstance(source.get("median"), int | float):
                    continue
                value = float(source["median"])
                result_tags = dict(tags)
                for stat in ("min", "max"):
                    if isinstance(source.get(stat), int | float):
                        result_tags[f"{source_key}_{stat}"] = str(source[stat])
            else:
                if concurrent_tags is None or not isinstance(source, int | float):
                    continue
                value = float(source)
                result_tags = dict(concurrent_tags)

            result_tags["phase"] = phase
            if source_key == "ttft_s":
                value *= 1000.0
            results.append(
                {
                    "name": "throughput_probe",
                    "metric": metric,
                    "value": value,
                    "unit": unit,
                    "tags": result_tags,
                    "raw": raw,
                }
            )
        return results
