"""Campaign recipes keep the contract the external Ansible consumer relies on.

Nothing in-repo reads these TOML files at run time, so without this test a renamed
converter kind, a placeholder the consumer does not substitute, or a recipe that lost
an axis would fail only when a campaign is dispatched on a target.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

from mlx_benchmarks.converters import get_converter
from mlx_benchmarks.envelope import load_schema

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
TARGET_INPUTS_DOC = REPO_ROOT / "docs" / "benchmark-campaign-target-inputs.md"

EXPECTED = {
    "llama-cpp/cross-card",
    "vllm/cross-card",
    "mlx/cross-card",
    "lm-eval/quick-intelligence",
    "lm-eval/gpqa-diamond",
    "evalscope/livecodebench",
}
ENGINES = {"llama_cpp", "vllm", "mlx_lm"}
AGENT_COUNTS = {1, 2, 4, 8}
LONG_CONTEXT_TOKENS = 131072
CONFIG_NAME = re.compile(r"^(llama-cpp|vllm|mlx|lm-eval|evalscope)/[a-z0-9-]+$")
# The only tokens the consumer's argv template substitutes.
PLACEHOLDERS = {
    "artifact_path",
    "benchmark_endpoint_root",
    "benchmark_endpoint",
    "model_id",
    "context_window_tokens",
    "context",
    "concurrency",
    "repetition",
    "output_tokens",
    "result_dir",
    "result_filename",
    "result_path",
    "output_path",
}
MODEL_PLACEHOLDERS = ("{{ model_id }}", "{{ artifact_path }}")
ENDPOINT_LITERAL = re.compile(r"https?://|\b\d{1,3}(?:\.\d{1,3}){3}\b")


def _load_recipes() -> dict[str, dict[str, Any]]:
    """Every structured campaign recipe, keyed by its path below ``configs/``."""
    recipes: dict[str, dict[str, Any]] = {}
    for path in sorted(CONFIGS.glob("*/*.toml")):
        data = tomllib.loads(path.read_text())
        if "config_name" in data:
            recipes[path.relative_to(CONFIGS).with_suffix("").as_posix()] = data
    return recipes


RECIPES = _load_recipes()
RUNS = [(name, run) for name, recipe in RECIPES.items() for run in recipe["run"]]
RUN_IDS = [f"{name}:{run['name']}" for name, run in RUNS]
CONVERTED_RUNS = [(name, run) for name, run in RUNS if "converter" in run]
CONVERTED_RUN_IDS = [f"{name}:{run['name']}" for name, run in CONVERTED_RUNS]


def test_declared_recipes_are_the_expected_set() -> None:
    assert set(RECIPES) == EXPECTED


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_config_name_matches_its_path_and_the_consumer_pattern(name: str) -> None:
    assert RECIPES[name]["config_name"] == name
    assert CONFIG_NAME.match(name)


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_recipe_declares_the_agent_and_long_context_axes(name: str) -> None:
    recipe = RECIPES[name]
    assert recipe["power_cap_w"] == "inventory"
    assert recipe["engines"]
    assert set(recipe["engines"]) <= ENGINES
    for axis in ("concurrency_list", "context_list"):
        values = recipe[axis]
        assert all(isinstance(value, int) and value > 0 for value in values)
        assert len(set(values)) == len(values)
    assert set(recipe["concurrency_list"]) >= AGENT_COUNTS
    assert max(recipe["context_list"]) >= LONG_CONTEXT_TOKENS


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_run_names_are_unique_within_a_recipe(name: str) -> None:
    names = [run["name"] for run in RECIPES[name]["run"]]
    assert names
    assert len(set(names)) == len(names)


@pytest.mark.parametrize(("name", "run"), RUNS, ids=RUN_IDS)
def test_run_declares_the_dispatch_fields(name: str, run: dict[str, Any]) -> None:
    assert run["artifact_id"] == "selected_model"
    assert run["run_on"] in {"benchmark_target", "controller"}
    assert set(run["dimensions"]) <= {"context_list", "concurrency_list"}
    assert isinstance(run["repetitions"], int) and run["repetitions"] >= 1
    assert isinstance(run["output_tokens"], int) and run["output_tokens"] > 0
    assert isinstance(run["executable"], str) and run["executable"]
    assert run["argv"] and all(isinstance(argument, str) for argument in run["argv"])


@pytest.mark.parametrize(("name", "run"), RUNS, ids=RUN_IDS)
def test_run_uses_only_placeholders_the_consumer_substitutes(name: str, run: dict[str, Any]) -> None:
    for argument in run["argv"]:
        for token in re.findall(r"\{\{.*?\}\}", argument):
            match = re.fullmatch(r"\{\{ ([a-z_]+) \}\}", token)
            assert match, f"{token!r} is not a single-space placeholder in {argument!r}"
            assert match.group(1) in PLACEHOLDERS, f"{token!r} is not substituted by the consumer"


@pytest.mark.parametrize(("name", "run"), RUNS, ids=RUN_IDS)
def test_run_takes_model_and_endpoint_from_the_survey(name: str, run: dict[str, Any]) -> None:
    """Identity comes from the registry placeholders, never from the recipe."""
    assert any(token in argument for argument in run["argv"] for token in MODEL_PLACEHOLDERS)
    assert not [argument for argument in run["argv"] if ENDPOINT_LITERAL.search(argument)]


@pytest.mark.parametrize(("name", "run"), RUNS, ids=RUN_IDS)
def test_axis_placeholders_are_declared_dimensions(name: str, run: dict[str, Any]) -> None:
    """A placeholder whose axis is not a dimension would render the unselected value."""
    argv = "\n".join(run["argv"])
    if "{{ context }}" in argv:
        assert "context_list" in run["dimensions"]
    if "{{ concurrency }}" in argv:
        assert "concurrency_list" in run["dimensions"]


@pytest.mark.parametrize(("name", "run"), CONVERTED_RUNS, ids=CONVERTED_RUN_IDS)
def test_converter_names_a_registered_kind_and_schema_suite(name: str, run: dict[str, Any]) -> None:
    converter = run["converter"]
    assert converter["executable"] == "mlx-bench-publish"
    assert converter["run_on"] == "controller"
    get_converter(converter["kind"])  # raises ValueError for an unregistered kind
    assert converter["suite"] in set(load_schema()["properties"]["suite"]["enum"])


def _argument_after(run: dict[str, Any], flag: str) -> str:
    argv = run["argv"]
    return argv[argv.index(flag) + 1]


def test_gpqa_diamond_runs_its_task_through_the_lm_eval_converter() -> None:
    (run,) = RECIPES["lm-eval/gpqa-diamond"]["run"]
    assert run["executable"] == "lm_eval"
    assert _argument_after(run, "--tasks") == "gpqa_diamond_cot_zeroshot"
    assert _argument_after(run, "--num_fewshot") == "0"
    assert "--apply_chat_template" in run["argv"]
    assert run["converter"]["kind"] == "lm-eval"


def test_livecodebench_runs_through_evalscope_without_a_docker_sandbox() -> None:
    (run,) = RECIPES["evalscope/livecodebench"]["run"]
    assert run["executable"] == "evalscope"
    assert _argument_after(run, "--datasets") == "live_code_bench"
    assert _argument_after(run, "--eval-type") == "openai_api"
    assert '"v6"' in _argument_after(run, "--dataset-args")
    assert "--sandbox" not in run["argv"]
    assert _argument_after(run, "--eval-batch-size") == "{{ concurrency }}"


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_recipe_is_listed_in_the_target_inputs_doc(name: str) -> None:
    assert f"`{name}`" in TARGET_INPUTS_DOC.read_text()
