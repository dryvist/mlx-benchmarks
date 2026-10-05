"""Runtime detection of ``system`` envelope fields.

Replaces the hardcoded laptop-specific dict that previously shipped in
``scripts/publish_run.py``. The old behavior published wrong metadata for any
contributor not using a specific M4 Max.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from functools import lru_cache
from importlib import metadata
from typing import Any

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def detect_system() -> dict[str, Any]:
    """Build a ``system`` dict describing the benchmark target.

    A remote publisher can provide ``MLX_BENCH_SYSTEM_*`` declarations; when
    those are absent, the local machine remains the source of detected values.

    The schema-required fields ``os`` / ``chip`` / ``memory_gb`` are always
    populated — when a detector fails they fall back to ``"unknown"`` or
    ``0`` rather than being omitted, because the schema rejects envelopes
    missing these keys. Optional fields (``python_version``, ``kernel``,
    and the package versions below) are only added when actually detected.
    Consumers should treat everything except ``os`` / ``chip`` /
    ``memory_gb`` as best-effort metadata.

    Cached (``lru_cache``) since the hardware facts are fixed for a process, so
    the subprocess/``sysctl`` probes run once. Env-driven topology is resolved by
    the uncached :func:`_detect_topology`, so a caller setting ``MLX_BENCH_*``
    late (or a test) does not need a fresh interpreter to pick it up.
    """
    data: dict[str, Any] = {
        "os": os.environ.get("MLX_BENCH_SYSTEM_OS", "").strip() or _detect_os(),
        "chip": os.environ.get("MLX_BENCH_SYSTEM_CHIP", "").strip() or _detect_chip(),
        "memory_gb": _detect_memory_gb(),
        "kernel": _detect_kernel(),
        "python_version": platform.python_version(),
    }

    # hostname distinguishes machines the chip/memory pair cannot — e.g. a Mac
    # Studio and a MacBook Pro that are both "Apple M4 Max, 128 GB". Best-effort:
    # only recorded when the node name is resolvable.
    hostname = _detect_hostname()
    if hostname:
        data["hostname"] = hostname

    for pkg_name, envelope_key in (
        ("mlx", "mlx_version"),
        ("mlx-lm", "mlx_lm_version"),
        ("lm-eval", "lm_eval_version"),
        ("vllm", "vllm_mlx_version"),
    ):
        version = _package_version(pkg_name)
        if version:
            data[envelope_key] = version

    runner = os.environ.get("RUNNER_NAME") or os.environ.get("GITHUB_RUNNER_LABEL")
    if runner:
        data["runner"] = runner

    topology = _detect_topology()
    if topology:
        data["topology"] = topology

    data.update(_detect_accelerator())

    return data


def _detect_os() -> str:
    if sys.platform == "darwin":
        mac_ver = platform.mac_ver()[0]
        if mac_ver:
            return f"macOS {mac_ver}"
    return platform.platform()


def _detect_chip() -> str:
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=3
            ).strip()
            if out:
                return out
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    # Fallback for Linux/Windows: platform.processor() is often empty on macOS but useful elsewhere.
    return platform.processor() or platform.machine() or "unknown"


def _detect_memory_gb() -> int:
    declared = os.environ.get("MLX_BENCH_SYSTEM_MEMORY_GB", "").strip()
    if declared:
        try:
            memory_gb = int(declared)
        except ValueError:
            log.warning("MLX_BENCH_SYSTEM_MEMORY_GB=%r is not an integer; detecting locally", declared)
        else:
            if memory_gb >= 0:
                return memory_gb
            log.warning("MLX_BENCH_SYSTEM_MEMORY_GB=%r is negative; detecting locally", declared)

    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=3).strip()
            return round(int(out) / (1024**3))
        except (subprocess.SubprocessError, FileNotFoundError, ValueError):
            pass
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3))
    except ImportError:
        return 0


def _detect_topology() -> dict[str, Any] | None:
    """Cluster topology from ``MLX_BENCH_*`` env vars — env-driven, no probing.

    Read fresh on every call (unlike the lru_cached :func:`detect_system`) so a
    caller that sets the vars can pick them up without a cold interpreter.
    Returns ``None`` unless at least one topology var is set, keeping
    single-node runs topology-free.
    """
    topology: dict[str, Any] = {}

    world_size = os.environ.get("MLX_BENCH_WORLD_SIZE", "")
    if world_size.isdigit():
        topology["world_size"] = int(world_size)

    parallelism = os.environ.get("MLX_BENCH_PARALLELISM")
    if parallelism in ("pipeline", "tensor", "none"):
        topology["parallelism"] = parallelism

    interconnect = os.environ.get("MLX_BENCH_INTERCONNECT")
    if interconnect:
        topology["interconnect"] = interconnect

    nodes_raw = os.environ.get("MLX_BENCH_NODES")
    if nodes_raw:
        try:
            nodes = json.loads(nodes_raw)
        except json.JSONDecodeError:
            log.warning("MLX_BENCH_NODES is not valid JSON; ignoring")
        else:
            if isinstance(nodes, list):
                topology["nodes"] = nodes
            else:
                log.warning("MLX_BENCH_NODES must be a JSON array; ignoring")

    return topology or None


# Env var that declares each system.gpu / system.engine / top-level accelerator field.
_GPU_ENV = {
    "model": "MLX_BENCH_GPU_MODEL",
    "vram_gb": "MLX_BENCH_GPU_VRAM_GB",
    "driver": "MLX_BENCH_GPU_DRIVER",
    "cuda": "MLX_BENCH_GPU_CUDA",
}
_ENGINE_ENV = {"name": "MLX_BENCH_ENGINE_NAME", "version": "MLX_BENCH_ENGINE_VERSION"}
_HOST_ENV = {"power_limit_w": "MLX_BENCH_POWER_LIMIT_W", "container": "MLX_BENCH_CONTAINER"}
# Schema type is number: a non-numeric value is dropped with a warning, never published as a string.
_NUMERIC_KEYS = frozenset({"vram_gb", "power_limit_w"})


def _env_group(mapping: dict[str, str]) -> dict[str, Any]:
    group: dict[str, Any] = {}
    for key, var in mapping.items():
        raw = os.environ.get(var, "").strip()
        if not raw:
            continue
        if key not in _NUMERIC_KEYS:
            group[key] = raw
            continue
        try:
            group[key] = float(raw)
        except ValueError:
            log.warning("%s=%r is not a number; ignoring", var, raw)
    return group


def _detect_accelerator() -> dict[str, Any]:
    """NVIDIA host facts from ``MLX_BENCH_*`` env vars — declared, never probed.

    Same contract as :func:`_detect_topology`: the caller states what the run
    used, because the publisher may run on a different machine than the GPU host
    and probing ``nvidia-smi`` here would then record the wrong card.
    ``configs/LAYOUT.md`` carries the ``nvidia-smi`` snippet that fills these in.
    Returns only the ``gpu`` / ``engine`` / ``power_limit_w`` / ``container``
    keys that were set, so Apple Silicon runs stay free of them.
    """
    data: dict[str, Any] = {}
    for key, mapping in (("gpu", _GPU_ENV), ("engine", _ENGINE_ENV)):
        if group := _env_group(mapping):
            data[key] = group
    data.update(_env_group(_HOST_ENV))
    return data


def _detect_kernel() -> str:
    return platform.release() or "unknown"


def _detect_hostname() -> str | None:
    """Short host label (e.g. ``mac-studio``), stripped of any ``.local``/domain suffix."""
    node = platform.node().strip()
    return node.split(".", 1)[0] or None


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
