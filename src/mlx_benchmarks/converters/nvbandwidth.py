"""nvbandwidth ``--format json`` -> envelope v1 converter (GPU copy baseline, suite ``nvbandwidth``)."""

from __future__ import annotations

import statistics
from typing import Any

from mlx_benchmarks.converters.base import ConverterContext, simple_envelope
from mlx_benchmarks.envelope import Envelope, Result


class NvbandwidthConverter:
    kind = "nvbandwidth"

    def build_envelope(self, raw: dict[str, Any], ctx: ConverterContext) -> Envelope:
        body = raw.get("nvbandwidth") or {}
        tags = {k: str(v) for k, v in ctx.extra_tags.items()}
        tags["nvbandwidth_version"] = str(body.get("version", "unknown"))
        results: list[Result] = []
        errors = [str(body["error"])] if body.get("error") else []
        for case in body.get("testcases", []):
            name = case.get("name", "unknown")
            # The matrix holds numbers as strings, with "N/A" where a pair was not measured.
            cells = [float(c) for row in case.get("bandwidth_matrix", []) for c in row if c != "N/A"]
            if case.get("status") != "Passed" or not cells:
                errors.append(f"{name}: {case.get('status', 'no data')} {case.get('error', '')}".strip())
                continue
            description = str(case.get("bandwidth_description", ""))
            is_latency = description.endswith("(ns)")  # latency tests share the matrix shape
            results.append(
                {
                    "name": name,
                    "metric": "latency" if is_latency else "bandwidth",
                    "value": statistics.fmean(cells),  # mean over measured pairs; one cell on a 1-GPU host
                    "unit": "ns" if is_latency else "GB/s",
                    "tags": {**tags, "pairs": str(len(cells)), "description": description},
                }
            )
        return simple_envelope(ctx, results, errors)
