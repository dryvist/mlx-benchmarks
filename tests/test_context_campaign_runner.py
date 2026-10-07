from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_runner() -> ModuleType:
    path = REPO_ROOT / "scripts" / "run-context-campaign.py"
    spec = importlib.util.spec_from_file_location("context_campaign", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def manifest() -> dict:
    return {
        "campaign_id": "context-20260828",
        "output_root": "/tmp/bench-runs",
        "defaults": {
            "targets": [32000, 64000],
            "configured_windows": [32768, 65536],
            "output_tokens": 512,
            "repeats": 4,
            "concurrency": 1,
        },
        "profiles": [
            {
                "model": "mlx-community/Qwen3.8-27B-4bit",
                "window_limit_tokens": 65536,
            }
        ],
    }


def test_expansion_is_deterministic_and_reserves_output_tokens() -> None:
    cells = runner.load_cells(manifest())
    assert len(cells) == 4
    assert [cell.status for cell in cells] == ["success", "not_applicable", "success", "success"]
    assert cells[0].profile_id == "Qwen3.8-27B-4bit"
    assert cells[0].cell_id == runner.load_cells(manifest())[0].cell_id
    assert len(cells[0].cell_id) == 16


def test_probe_and_dry_run_publish_keep_dimensions() -> None:
    cell = runner.load_cells(manifest())[0]
    probe = runner.probe_command(cell, Path("/tmp/out.json"))
    publish = runner.publisher_command(cell, Path("/tmp/out.json"))
    assert "--context-tokens" in probe and "0" in probe
    assert "--target-prompt-tokens" in probe and "32000" in probe
    assert "--max-tokens" in probe and "512" in probe
    assert "--expected-prompt-tokens" not in probe
    assert "--window-limit-tokens" in probe and "32768" in probe
    assert "throughput-probe" in publish and "--dry-run" in publish
    assert f"cell_id={cell.cell_id}" in publish
    assert "configured_window_tokens=32768" in publish
    assert "--published-metadata" in publish
    assert "/tmp/published-metadata.json" in publish


def test_invalid_manifest_is_rejected() -> None:
    bad = manifest()
    bad["defaults"]["output_tokens"] = 0
    with pytest.raises(ValueError, match="output_tokens"):
        runner.load_cells(bad)


def test_manifest_can_record_a_non_scored_capacity_outcome() -> None:
    data = manifest()
    data["profiles"][0]["outcome"] = {
        "status": "capacity_gated",
        "reason": "first nonzero swap interrupted the guarded probe",
        "target_tokens": 64000,
    }
    cells = runner.load_cells(data)
    assert [cell.status for cell in cells] == [
        "success",
        "not_applicable",
        "success",
        "capacity_gated",
    ]
    assert cells[3].reported_reason == "first nonzero swap interrupted the guarded probe"


def test_dry_run_does_not_create_external_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = manifest()
    data["output_root"] = str(tmp_path / "external")
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(data))
    assert runner.main([str(path), "--dry-run"]) == 0
    assert not (tmp_path / "external").exists()
    assert "planned:" in capsys.readouterr().out
