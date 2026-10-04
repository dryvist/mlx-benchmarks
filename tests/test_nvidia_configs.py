"""configs/nvidia/ runbooks stay wired to real converters and schema suites.

Nothing in-repo reads these TOML files at run time, so without this test a renamed
converter kind or suite would leave a runbook that fails only on the GPU host.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

from mlx_benchmarks.converters import get_converter
from mlx_benchmarks.envelope import load_schema

NVIDIA_DIR = Path(__file__).resolve().parents[1] / "configs" / "nvidia"
RUNBOOKS = sorted(NVIDIA_DIR.glob("*.toml"))
EXPECTED = {"throughput", "quality", "gpu-burn", "nvbandwidth", "mbw", "fio"}


def commands(node: Any) -> list[str]:
    """Every ``command`` string anywhere in a parsed runbook."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "command" and isinstance(value, str):
                found.append(value)
            else:
                found += commands(value)
    elif isinstance(node, list):
        for item in node:
            found += commands(item)
    return found


def test_every_runbook_is_present() -> None:
    assert {p.stem for p in RUNBOOKS} == EXPECTED


@pytest.mark.parametrize("path", RUNBOOKS, ids=lambda p: p.stem)
def test_publish_command_names_registered_kind_and_schema_suite(path: Path) -> None:
    text = "\n".join(commands(tomllib.loads(path.read_text())))
    kinds = re.findall(r"--kind\s+(\S+)", text)
    suites = re.findall(r"--suite\s+(\S+)", text)
    assert kinds, f"{path.name} has no publish command"
    for kind in kinds:
        get_converter(kind)  # raises ValueError for an unregistered kind
    enum = set(load_schema()["properties"]["suite"]["enum"])
    assert set(suites) <= enum


@pytest.mark.parametrize("stem", sorted(EXPECTED - {"throughput", "quality"}))
def test_hardware_baselines_publish_under_their_own_suite(stem: str) -> None:
    text = "\n".join(commands(tomllib.loads((NVIDIA_DIR / f"{stem}.toml").read_text())))
    assert f"--kind {stem} --suite {stem}" in text
    assert "--model hardware-baseline" in text


def test_throughput_publishes_the_documented_sweep_tags() -> None:
    text = "\n".join(commands(tomllib.loads((NVIDIA_DIR / "throughput.toml").read_text())))
    for key in ("prompt_tokens", "concurrency", "context_len"):
        assert f"--tag {key}=" in text
    tags = load_schema()["properties"]["results"]["items"]["properties"]["tags"]["properties"]
    assert {"prompt_tokens", "concurrency", "context_len"} <= set(tags)


def test_throughput_matrix_is_the_requested_sweep() -> None:
    matrix = tomllib.loads((NVIDIA_DIR / "throughput.toml").read_text())["matrix"]
    assert matrix["concurrency"] == [1, 4, 8]
    assert matrix["prompt_tokens"] == [8192, 65536, 131072]
    run = tomllib.loads((NVIDIA_DIR / "throughput.toml").read_text())["run"]["command"]
    assert "for P in 8192 65536 131072" in run
    assert "for C in 1 4 8" in run


def test_quality_runs_only_lm_eval_tasks_it_was_selected_for() -> None:
    config = tomllib.loads((NVIDIA_DIR / "quality.toml").read_text())
    run = config["run"]["command"]
    assert "--tasks leaderboard_mmlu_pro" in run
    # MMLU-Pro is multiple_choice: lm-eval scores it on completion endpoints only.
    assert "local-completions" in run
    assert "chat" not in run.replace("--apply_chat_template", "")
    # Saturated benchmarks must not creep back in.
    for saturated in ("gpqa", "ifeval", "aime", "livecodebench", "ruler", "needle"):
        assert saturated not in run.lower()
    assert config["deferred"], "non-lm-eval unsaturated benchmarks are listed, not silently dropped"


def test_runbooks_carry_no_real_hostnames_or_addresses() -> None:
    for path in RUNBOOKS:
        text = path.read_text()
        assert not re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text), f"{path.name} contains an IP address"
        assert "http://" not in text.replace("http://<gpu-host>", "")
