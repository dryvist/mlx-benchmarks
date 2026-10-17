"""EvalScope aggregate report JSON -> envelope v1 score rows."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from mlx_benchmarks.converters.base import ConverterContext, simple_envelope
from mlx_benchmarks.envelope import Envelope, Result


class EvalScopeConverter:
    kind = "evalscope"

    def build_envelope(self, raw: dict[str, Any], ctx: ConverterContext) -> Envelope:
        execution = raw.get("execution_summary")
        if isinstance(execution, Mapping) and execution.get("incomplete") is True:
            raise ValueError("EvalScope report is incomplete")
        metrics = raw.get("metrics")
        if not isinstance(metrics, list):
            raise ValueError("EvalScope report contains no aggregate metrics")

        rows: list[Result] = []
        for index, metric in enumerate(metrics):
            if not isinstance(metric, Mapping):
                continue
            score = metric.get("score")
            if not isinstance(score, int | float) or isinstance(score, bool) or not math.isfinite(score):
                continue
            identity = metric.get("identity")
            key = identity.get("key") if isinstance(identity, Mapping) else None
            name = metric.get("legacy_name") or key or f"metric_{index}"
            rows.append(
                {
                    "name": str(raw.get("dataset_name") or ctx.suite),
                    "metric": str(name),
                    "value": float(score),
                    "unit": "ratio",
                    "tags": {"sample_count": str(metric.get("num", ""))},
                }
            )
        if not rows:
            raise ValueError("EvalScope report contains no finite aggregate scores")
        return simple_envelope(ctx, rows)
