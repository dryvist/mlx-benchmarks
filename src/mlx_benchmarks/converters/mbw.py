"""mbw text output -> envelope v1 converter (host RAM bandwidth baseline, suite ``mbw``).

Reads the per-method ``AVG`` lines, e.g.
``AVG<TAB>Method: MEMCPY<TAB>Elapsed: 0.15463<TAB>MiB: 1024.00000<TAB>Copy: 6622.538 MiB/s``.
Run mbw without ``-a`` or there is nothing to read.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from mlx_benchmarks.converters.base import ConverterContext, simple_envelope
from mlx_benchmarks.envelope import Envelope, Result

_AVG = re.compile(
    r"^AVG\s+Method:\s+(\w+)\s+Elapsed:\s+([\d.]+)\s+MiB:\s+([\d.]+)\s+Copy:\s+([\d.]+) MiB/s", re.M
)
_RUN = re.compile(r"^\d+\s+Method:\s+(\w+)", re.M)


class MbwConverter:
    kind = "mbw"

    def build_envelope(self, raw: dict[str, Any], ctx: ConverterContext) -> Envelope:
        text = str(raw.get("output", ""))  # the CLI wraps .txt/.log input as {"output": text}
        runs = Counter(_RUN.findall(text))
        results: list[Result] = [
            {
                "name": method.lower(),
                "metric": "copy_bandwidth_mib_per_s",
                "value": float(rate),
                "unit": "MiB/s",
                "tags": {
                    **{k: str(v) for k, v in ctx.extra_tags.items()},
                    "array_mib": str(round(float(mib))),
                    "runs": str(runs[method]),
                    "mean_elapsed_s": elapsed,
                },
            }
            for method, elapsed, mib, rate in _AVG.findall(text)
        ]
        errors = [] if results else ["mbw output has no AVG lines (was it run with -a?)"]
        return simple_envelope(ctx, results, errors)
