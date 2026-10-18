from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_run_index.py"


def _validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("validate_run_index", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(**overrides: object) -> dict[str, object]:
    return {
        "path": "data/run-2026-09-01T00-00-00-run.parquet",
        "status": "scored",
        "variant": "example",
        "comparison_group": "example",
        "context_band": "4k",
        "max_output_tokens": 512,
        "completion_state": "complete",
        "caveat": "",
        "machine": "Mac Studio M4 Max 128GB",
        "engine": "vLLM",
        "engine_version": "0.31.0",
        **overrides,
    }


def test_recent_shards_require_a_canonical_machine_label() -> None:
    with pytest.raises(ValueError, match="machine label"):
        _validator().validate_entries([_entry(machine="Mac Studio")])


def test_scored_llama_cpp_shards_use_release_tag_and_separate_commit() -> None:
    entry = _entry(
        engine="llama.cpp",
        engine_version="b11669",
        engine_build="b11669",
        engine_commit="620d2225c",
    )

    _validator().validate_entries([entry])

    with pytest.raises(ValueError, match="release tag"):
        _validator().validate_entries([{**entry, "engine_version": "0.7.0"}])


def test_uncaptured_llama_cpp_build_must_be_experimental_with_caveat() -> None:
    entry = _entry(
        status="experimental",
        engine="llama.cpp",
        engine_version=None,
        caveat="engine build not captured",
    )

    _validator().validate_entries([entry])

    with pytest.raises(ValueError, match="normalized engine or version"):
        _validator().validate_entries([{**entry, "status": "scored"}])
