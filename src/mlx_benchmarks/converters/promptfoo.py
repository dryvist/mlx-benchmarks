"""Promptfoo JSON export -> envelope v1 score and assertion summaries."""

from __future__ import annotations

import datetime
import math
from collections.abc import Mapping
from typing import Any

from mlx_benchmarks.converters.base import ConverterContext, simple_envelope
from mlx_benchmarks.envelope import Envelope, Result


class PromptfooConverter:
    kind = "promptfoo"

    def build_envelope(self, raw: dict[str, Any], ctx: ConverterContext) -> Envelope:
        container = raw.get("results")
        if isinstance(container, Mapping):
            outputs = container.get("outputs") or container.get("results") or []
        else:
            outputs = raw.get("outputs") or []
        if not isinstance(outputs, list):
            outputs = []

        timestamp = raw.get("timestamp")
        if not isinstance(timestamp, str) and isinstance(container, Mapping):
            timestamp = container.get("timestamp")
        if not isinstance(timestamp, str) or not timestamp:
            timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        results: list[Result] = []
        scores: list[float] = []
        for index, output in enumerate(outputs):
            if not isinstance(output, Mapping):
                continue
            test_index = _index(output.get("testIdx"), index)
            prompt_index = output.get("promptIdx")
            base_tags = {**ctx.extra_tags, "promptfoo_test_idx": str(test_index)}
            if prompt_index is not None:
                base_tags["promptfoo_prompt_idx"] = str(prompt_index)

            score = _score(output)
            if score is not None:
                scores.append(score)
                test_tags = dict(base_tags)
                if isinstance(output.get("success"), bool):
                    test_tags["success"] = str(output["success"]).lower()
                results.append(
                    {
                        "name": f"test_{test_index}",
                        "metric": "score",
                        "value": score,
                        "unit": "ratio",
                        "tags": test_tags,
                    }
                )
            results.extend(_assertion_rows(output, test_index, base_tags))

        if scores:
            results.append(
                {
                    "name": "promptfoo_aggregate",
                    "metric": "score",
                    "value": sum(scores) / len(scores),
                    "unit": "ratio",
                    "tags": {**ctx.extra_tags, "test_count": str(len(scores))},
                }
            )
        return simple_envelope(ctx, results, timestamp=timestamp)


def _index(value: Any, fallback: int) -> int | str:
    return value if isinstance(value, int | str) else fallback


def _score(output: Mapping[str, Any]) -> float | None:
    score = output.get("score")
    if isinstance(score, int | float) and not isinstance(score, bool) and math.isfinite(score):
        return float(score)
    success = output.get("success")
    return float(success) if isinstance(success, bool) else None


def _assertion_rows(output: Mapping[str, Any], test_index: int | str, tags: dict[str, str]) -> list[Result]:
    grading = output.get("gradingResult")
    components = grading.get("componentResults") if isinstance(grading, Mapping) else None
    if not isinstance(components, list):
        return []

    rows: list[Result] = []
    for index, component in enumerate(components):
        if not isinstance(component, Mapping):
            continue
        passed = component.get("pass")
        score = component.get("score")
        if isinstance(passed, bool):
            value, metric = float(passed), "assertion_pass"
        elif isinstance(score, int | float) and not isinstance(score, bool) and math.isfinite(score):
            value, metric = float(score), "assertion_score"
        else:
            continue
        rows.append(
            {
                "name": f"test_{test_index}_assertion_{index}",
                "metric": metric,
                "value": value,
                "unit": "ratio",
                "tags": {**tags, "promptfoo_assertion_idx": str(index)},
            }
        )
    return rows
