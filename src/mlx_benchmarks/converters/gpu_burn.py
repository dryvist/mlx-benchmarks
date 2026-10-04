"""gpu-burn stdout -> envelope v1 converter (sustained-compute baseline, suite ``gpu-burn``).

gpu_burn redraws one CR-terminated progress line, e.g.
``100.0%  proc'd: 3760 (4837 Gflop/s) - 3750 (4800 Gflop/s)   errors: 0 - 0   temps: 62 C - 60 C``
and ends with ``GPU 0: OK`` (or ``FAULTY``) per card. Gflop/s is per sampling interval, so the
run's median is reported; any single sample, the last included, is noise.
"""

from __future__ import annotations

import re
import statistics
from typing import Any

from mlx_benchmarks.converters.base import ConverterContext, simple_envelope
from mlx_benchmarks.envelope import Envelope, Result

_PROGRESS = re.compile(r"proc'd: (?P<proc>.*?)\s+errors: .*?\s+temps: (?P<temps>[^\r\n]*)")
_STATUS = re.compile(r"GPU (\d+): (OK|FAULTY)")
_AGGREGATE = {"gflops": (statistics.median, "Gflop/s"), "temp_max_c": (max, "C")}


def _row(gpu: int, metric: str, value: float, unit: str, tags: dict[str, str]) -> Result:
    return {"name": f"gpu{gpu}", "metric": metric, "value": value, "unit": unit, "tags": tags}


class GpuBurnConverter:
    kind = "gpu-burn"

    def build_envelope(self, raw: dict[str, Any], ctx: ConverterContext) -> Envelope:
        text = str(raw.get("output", ""))  # the CLI wraps .txt/.log input as {"output": text}
        samples: dict[tuple[int, str], list[float]] = {}
        for m in _PROGRESS.finditer(text):
            for gpu, g in enumerate(re.findall(r"\((\d+) Gflop/s\)", m["proc"])):
                if int(g) > 0:  # 0 = this GPU has not reported an interval yet
                    samples.setdefault((gpu, "gflops"), []).append(float(g))
            for gpu, c in enumerate(re.findall(r"(\d+) C|--", m["temps"])):  # "--" = no reading
                if c:
                    samples.setdefault((gpu, "temp_max_c"), []).append(float(c))
        tags = {k: str(v) for k, v in ctx.extra_tags.items()}
        results = [
            _row(
                gpu,
                metric,
                _AGGREGATE[metric][0](vals),
                _AGGREGATE[metric][1],
                {**tags, "samples": str(len(vals))},
            )
            for (gpu, metric), vals in sorted(samples.items())
        ]
        verdicts = _STATUS.findall(text)
        results += [
            _row(int(gpu), "burn_ok", float(status == "OK"), "bool", tags) for gpu, status in verdicts
        ]
        errors = [] if verdicts else ["gpu-burn printed no final per-GPU status (run interrupted?)"]
        return simple_envelope(ctx, results, errors)
