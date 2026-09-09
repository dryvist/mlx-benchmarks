"""Unit tests for the promptstack runner's pure functions.

``harness/promptstack/run.py`` is a standalone PEP 723 script (not part of the
package), so it is loaded here via importlib. Its network dependency (httpx)
is imported lazily inside the request functions, keeping the module — and
these pure-function tests — importable in the plain dev environment.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = REPO_ROOT / "configs" / "promptstack" / "probes"
PROMPT_DIR = REPO_ROOT / "configs" / "promptstack" / "prompts"


def _load_runner() -> ModuleType:
    path = REPO_ROOT / "harness" / "promptstack" / "run.py"
    spec = importlib.util.spec_from_file_location("promptstack_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


# --- probe bank + prompt loading (also proves the shipped configs parse) ------


def test_all_probe_banks_load_and_have_a_bank_version() -> None:
    for probe_class in runner.PROBE_CLASSES:
        bank = runner.load_probe_bank(PROBE_DIR, probe_class)
        assert bank["probe_bank_version"] == "1"
        assert bank["probe_class"] == probe_class
        assert len(bank["tasks"]) >= 3


def test_every_surface_has_a_variant_and_current_prompt() -> None:
    for surface in ("hermes", "chat"):
        variant = runner.load_prompt(PROMPT_DIR, surface)
        current = runner.load_prompt(PROMPT_DIR, f"current-{surface}")
        assert "Ground truth before claims" in variant
        assert len(current) > 0
        assert variant != current


# --- score_reasoning ------------------------------------------------------------


def test_score_reasoning_matches_trailing_number() -> None:
    task = {"answer": "640"}
    assert runner.score_reasoning(task, "The answer is 640.") is True
    assert runner.score_reasoning(task, "640") is True
    assert runner.score_reasoning(task, "I think it's 641") is False
    assert runner.score_reasoning(task, "no numbers here") is False


def test_score_reasoning_handles_thousands_separator() -> None:
    assert runner.score_reasoning({"answer": "1200"}, "roughly 1,200 total") is True


# --- score_tool_call --------------------------------------------------------------


def test_score_tool_call_positive_valid() -> None:
    task = {"kind": "positive", "expected_tool": "get_weather", "expected_required": ["location"]}
    calls = [{"function": {"name": "get_weather", "arguments": '{"location": "Portland"}'}}]
    assert runner.score_tool_call(task, calls) is True


def test_score_tool_call_positive_wrong_tool() -> None:
    task = {"kind": "positive", "expected_tool": "get_weather", "expected_required": ["location"]}
    calls = [{"function": {"name": "search_docs", "arguments": '{"query": "x"}'}}]
    assert runner.score_tool_call(task, calls) is False


def test_score_tool_call_positive_missing_required_arg() -> None:
    task = {"kind": "positive", "expected_tool": "get_weather", "expected_required": ["location"]}
    calls = [{"function": {"name": "get_weather", "arguments": "{}"}}]
    assert runner.score_tool_call(task, calls) is False


def test_score_tool_call_positive_no_call_is_failure() -> None:
    task = {"kind": "positive", "expected_tool": "get_weather", "expected_required": ["location"]}
    assert runner.score_tool_call(task, None) is False


def test_score_tool_call_negative_no_fabrication_is_success() -> None:
    task = {"kind": "negative"}
    assert runner.score_tool_call(task, None) is True
    assert runner.score_tool_call(task, []) is True


def test_score_tool_call_negative_fabricated_call_is_failure() -> None:
    task = {"kind": "negative"}
    calls = [{"function": {"name": "get_weather", "arguments": '{"location": "somewhere"}'}}]
    assert runner.score_tool_call(task, calls) is False


# --- score_instruction ------------------------------------------------------------


def test_score_instruction_all_constraints_met() -> None:
    task = {
        "constraints": [
            {"type": "word_count", "n": 2},
            {"type": "lowercase_only"},
            {"type": "no_punctuation"},
        ]
    }
    met, total = runner.score_instruction(task, "hello world")
    assert (met, total) == (3, 3)


def test_score_instruction_partial_credit() -> None:
    task = {
        "constraints": [
            {"type": "word_count", "n": 2},
            {"type": "lowercase_only"},
        ]
    }
    met, total = runner.score_instruction(task, "Hello world")  # capitalized -> fails lowercase
    assert (met, total) == (1, 2)


def test_score_instruction_line_count_and_numbering() -> None:
    task = {
        "constraints": [
            {"type": "line_count", "n": 3},
            {"type": "no_numbering_prefix"},
        ]
    }
    met, total = runner.score_instruction(task, "apple\nbanana\ncherry")
    assert (met, total) == (2, 2)

    met, total = runner.score_instruction(task, "1. apple\n2. banana\n3. cherry")
    assert (met, total) == (1, 2)


def test_score_instruction_exact_match_ci() -> None:
    task = {"constraints": [{"type": "exact_match_ci", "values": ["yes", "no"]}]}
    assert runner.score_instruction(task, "Yes") == (1, 1)
    assert runner.score_instruction(task, "yes, definitely") == (0, 1)


# --- score_homelab_qa --------------------------------------------------------------


def test_score_homelab_qa_correct_and_no_fabrication() -> None:
    task = {"expected_facts": ["8080", "blue"], "forbidden_terms": ["9090", "green"]}
    correct, unsupported = runner.score_homelab_qa(task, "It listens on 8080 and is owned by team Blue.")
    assert (correct, unsupported) == (True, False)


def test_score_homelab_qa_missing_fact_is_incorrect() -> None:
    task = {"expected_facts": ["8080", "blue"], "forbidden_terms": []}
    correct, _ = runner.score_homelab_qa(task, "It is owned by team Blue.")
    assert correct is False


def test_score_homelab_qa_fabricated_term_flagged() -> None:
    task = {"expected_facts": ["8080", "blue"], "forbidden_terms": ["9090"]}
    correct, unsupported = runner.score_homelab_qa(task, "It's on 8080, team Blue, backup port 9090.")
    assert correct is True
    assert unsupported is True


# --- aggregation -------------------------------------------------------------------


def test_percentile_nearest_rank() -> None:
    values = [10, 20, 30, 40, 50]
    assert runner.percentile(values, 50) == 30
    assert runner.percentile([], 50) == 0.0


def test_summarize_probe_class_reasoning() -> None:
    records = [
        {"success": True, "latency_ms": 100, "prompt_tokens": 50, "completion_tokens": 5},
        {"success": False, "latency_ms": 200, "prompt_tokens": 55, "completion_tokens": 6},
    ]
    block = runner.summarize_probe_class("reasoning", records)
    assert block["n_tasks"] == 2
    assert block["task_success_rate"] == 0.5
    assert "valid_tool_call_rate" not in block
    assert "instruction_adherence_rate" not in block


def test_summarize_probe_class_tool_call_unsupported_claim_rate() -> None:
    records = [
        {"success": True, "kind": "positive", "latency_ms": 100},
        {"success": False, "kind": "negative", "latency_ms": 100},  # fabricated -> unsupported
        {"success": True, "kind": "negative", "latency_ms": 100},
    ]
    block = runner.summarize_probe_class("tool_call", records)
    assert block["valid_tool_call_rate"] == block["task_success_rate"]
    assert block["unsupported_claim_rate"] == 0.5  # 1 of 2 negative tasks fabricated


def test_summarize_probe_class_instruction_adherence() -> None:
    records = [
        {"success": True, "constraints_met": 3, "constraints_total": 3, "latency_ms": 100},
        {"success": False, "constraints_met": 1, "constraints_total": 2, "latency_ms": 100},
    ]
    block = runner.summarize_probe_class("instruction", records)
    assert block["instruction_adherence_rate"] == round((1.0 + 0.5) / 2, 4)


def test_summarize_probe_class_homelab_qa_unsupported_claim_rate() -> None:
    records = [
        {"success": True, "unsupported_claim": False, "latency_ms": 100},
        {"success": True, "unsupported_claim": True, "latency_ms": 100},
    ]
    block = runner.summarize_probe_class("homelab_qa", records)
    assert block["unsupported_claim_rate"] == 0.5


def test_cell_name() -> None:
    # on/off keep the exact names they have always produced — the name is a public
    # join key (--cells, published tags.cell), so renaming orphans published rows.
    assert runner.cell_name("base_plus_variant", "on") == "variant-base_plus_variant_think-on"
    assert runner.cell_name("current", "off") == "variant-current_think-off"
    # A graded level names its own cell instead of colliding with another's.
    assert runner.cell_name("current", "xhigh") == "variant-current_think-xhigh"


def test_thinking_body_kwargs_is_graded() -> None:
    assert runner.thinking_body_kwargs("enable_thinking", "on") == {
        "chat_template_kwargs": {"enable_thinking": True}
    }
    assert runner.thinking_body_kwargs("reasoning_effort", "off") == {"reasoning_effort": "low"}
    assert runner.thinking_body_kwargs("reasoning_effort", "max") == {"reasoning_effort": "max"}


def test_selected_filter() -> None:
    assert runner._selected("variant-current_think-on_reasoning", None) is True
    assert runner._selected("variant-current_think-on_reasoning", "current") is True
    assert runner._selected("variant-current_think-on_reasoning", "base_plus_variant") is False
