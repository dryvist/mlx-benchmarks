"""EvalScope runbooks keep selection, judging, and publishing declarative."""

from __future__ import annotations

import tomllib
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "evalscope"


def test_hle_runbook_uses_seeded_text_only_subject_sample_and_variable_judge() -> None:
    config = tomllib.loads((CONFIGS / "hle.toml").read_text())
    command = config["run"]["command"]

    assert "--datasets hle --dataset-hub huggingface" in command
    assert '"shuffle":true' in command and '"include_multi_modal":false' in command
    assert "--seed 1 --limit 25 --eval-batch-size 8" in command
    assert all(f"${name}" in command for name in ("JUDGE_BASE_URL", "JUDGE_API_KEY", "JUDGE_MODEL"))
    assert '"models":[' in command
    assert "uvx --from evalscope==1.13.0" in command
    assert "--kind evalscope --suite knowledge" in config["publish"]["command"]
    assert "http://" not in command and "https://" not in command


def test_aa_lcr_runbook_uses_seeded_long_context_sample_and_variable_judge() -> None:
    config = tomllib.loads((CONFIGS / "aa-lcr.toml").read_text())
    command = config["run"]["command"]

    assert "--datasets aa_lcr" in command
    assert "--seed 1 --limit 100 --eval-batch-size 8" in command
    assert all(f"${name}" in command for name in ("JUDGE_BASE_URL", "JUDGE_API_KEY", "JUDGE_MODEL"))
    assert "uvx --from evalscope==1.13.0" in command
    assert "--limit 5" in config["run"]["smoke"]["command"]
    assert "--kind evalscope --suite capability-comparison" in config["publish"]["command"]
    assert "http://" not in command and "https://" not in command
