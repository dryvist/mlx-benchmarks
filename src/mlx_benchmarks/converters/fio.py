"""fio ``--output-format=json`` -> envelope v1 converter (storage baseline, suite ``fio``)."""

from __future__ import annotations

import datetime
from typing import Any

from mlx_benchmarks.converters.base import ConverterContext, simple_envelope
from mlx_benchmarks.envelope import Envelope, Result

# Job options kept as tags (option -> tag name). Paths (filename/directory) are left
# out on purpose: they describe the host, not the measurement.
_OPTS = {"rw": "rw", "readwrite": "rw", "bs": "bs", "blocksize": "bs"}
_OPTS |= {"iodepth": "iodepth", "numjobs": "numjobs", "ioengine": "ioengine"}


class FioConverter:
    kind = "fio"

    def build_envelope(self, raw: dict[str, Any], ctx: ConverterContext) -> Envelope:
        results: list[Result] = []
        for job in raw.get("jobs", []):
            opts = job.get("job options") or {}
            tags = {k: str(v) for k, v in ctx.extra_tags.items()}
            tags["fio_version"] = str(raw.get("fio version", "unknown"))
            tags |= {tag: str(opts[opt]) for opt, tag in _OPTS.items() if opt in opts}
            for ddir in ("read", "write"):
                stats = job.get(ddir) or {}
                if not stats.get("io_bytes"):
                    continue  # this direction moved no data
                values = {
                    f"{ddir}_bandwidth_mib_per_s": (stats["bw_bytes"] / 2**20, "MiB/s"),
                    f"{ddir}_iops": (stats["iops"], "iops"),
                }
                p99_ns = ((stats.get("clat_ns") or {}).get("percentile") or {}).get("99.000000")
                if p99_ns is not None:
                    values[f"{ddir}_clat_p99_ms"] = (p99_ns / 1e6, "ms")
                for metric, (value, unit) in values.items():
                    results.append(
                        {
                            "name": job.get("jobname", "fio"),
                            "metric": metric,
                            "value": float(value),
                            "unit": unit,
                            "duration_seconds": stats["runtime"] / 1000,  # fio reports ms
                            "tags": dict(tags),
                        }
                    )
        stamp = raw.get("timestamp")  # epoch seconds
        iso = None
        if isinstance(stamp, int | float):
            iso = datetime.datetime.fromtimestamp(stamp, datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        return simple_envelope(ctx, results, timestamp=iso)
