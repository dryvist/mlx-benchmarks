"""Unit tests for the factual runner's pure scoring functions.

``harness/factual/run.py`` is a standalone PEP 723 script (not part of the
package), so it is loaded here via importlib — the same pattern
``test_agentic_runner.py`` uses. Its network dependency (httpx) is imported
lazily inside the request functions, keeping these tests endpoint-free.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BANK_PATH = REPO_ROOT / "configs" / "factual" / "fixtures" / "homelab-digest.json"


def _load_runner() -> ModuleType:
    path = REPO_ROOT / "harness" / "factual" / "run.py"
    spec = importlib.util.spec_from_file_location("factual_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
BANK = json.loads(BANK_PATH.read_text())
CASES = {case["id"]: case for case in BANK["cases"]}


# --- number normalization -----------------------------------------------------


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("1,284", "1284"),
        ("1284", "1284"),
        ("1284.0", "1284"),
        ("07", "7"),
        ("0.80", "0.8"),
        ("not-a-number", "not-a-number"),
    ],
)
def test_normalize_number_canonicalizes_formatting(token: str, expected: str) -> None:
    assert runner.normalize_number(token) == expected


def test_number_set_is_blind_to_surrounding_punctuation() -> None:
    # An ISO timestamp and a dotted quad must decompose the same way on both
    # sides of the comparison, or formatting reads as fabrication.
    assert runner.number_set("2026-07-23T02:30:00Z") == {"2026", "7", "23", "2", "30", "0"}


# --- fabricated numbers (the metric the suite exists for) ---------------------


def test_evidence_numbers_are_grounded() -> None:
    assert runner.fabricated_numbers("Failures reached 1,284.", "count: 1284", "") == []


def test_digits_inside_identifiers_count_as_numbers_on_both_sides() -> None:
    # "web-02" yields the token 2. That is fine as long as the identifier is in
    # the evidence too — which is exactly why extraction must be symmetric.
    assert runner.fabricated_numbers("web-02 saw 1284 failures.", "host web-02 count 1284", "") == []
    assert runner.fabricated_numbers("web-02 saw 1284 failures.", "count 1284", "") == ["2"]


def test_invented_number_is_flagged() -> None:
    assert runner.fabricated_numbers("The rate rose to 37%.", "count: 1284", "") == ["37"]


def test_allowed_derived_covers_a_correct_total() -> None:
    evidence = "[{'count':1284},{'count':617},{'count':152}]"
    text = "Across 3 sources there were 2053 failures."
    assert runner.fabricated_numbers(text, evidence, "", ["2053", "3"]) == []
    # Without the declaration, the same derived figures are (correctly) unknown.
    assert runner.fabricated_numbers(text, evidence, "") == ["2053", "3"]


def test_prompt_numbers_are_grounded() -> None:
    assert runner.fabricated_numbers("Over the last 24 hours.", "", "the last 24 hours") == []


def test_date_reformatting_is_not_a_fabrication() -> None:
    # "July 23, 2026" against ISO evidence: a formatting difference, not an
    # invented figure. This is the regression the punctuation-blind regex buys.
    assert runner.fabricated_numbers("Ran July 23, 2026.", "last_run: 2026-07-23T02:30:00Z", "") == []


# --- tool-syntax leakage ------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Sure. [Tool call: run_splunk_query(query='...')]",
        '<tool_call>{"name": "read_file"}</tool_call>',
        "I'll use functions.get_system_status({})",
        "<|channel|>analysis",
        "<|tool_call_begin|>",
    ],
)
def test_leaked_tool_syntax_detected(text: str) -> None:
    assert runner.leaked_tool_syntax(text) is True


def test_clean_prose_does_not_leak() -> None:
    assert runner.leaked_tool_syntax("I ran a Splunk query and found 1284 failures.") is False


# --- required facts -----------------------------------------------------------


def test_missing_facts_accepts_reformatted_numbers() -> None:
    assert runner.missing_facts("Host WEB-02 hit 1,284 failures.", ["1284", "web-02"]) == []


def test_missing_facts_reports_absentees() -> None:
    assert runner.missing_facts("Something happened.", ["1284", "web-02"]) == ["1284", "web-02"]


# --- whole-case scoring against the shipped bank ------------------------------


def test_ideal_answer_passes() -> None:
    case = CASES["digest-001-ssh-failures"]
    result = runner.score_case(case, "Host web-02 was worst hit with 1284 failed logins.")
    assert result["success"] is True
    assert result["fabricated_numbers"] == []


def test_transposed_digits_fail_the_index_case() -> None:
    case = CASES["digest-003-index-inventory"]
    result = runner.score_case(case, "The os index holds 9120434 events; network retains 30 days.")
    assert result["success"] is False
    assert result["forbidden_hits"] == ["9120434"]


def test_invented_retention_fails_the_abstention_case() -> None:
    case = CASES["digest-004-absent-field"]
    result = runner.score_case(case, "linux_secure has 88012 events with 30 day retention.")
    assert result["success"] is False
    assert result["forbidden_hits"] == ["30 day"]


def test_correct_abstention_passes() -> None:
    case = CASES["digest-004-absent-field"]
    result = runner.score_case(
        case, "linux_secure holds 88012 events; the tool result did not include a retention period."
    )
    assert result["success"] is True


def test_leak_alone_fails_an_otherwise_correct_answer() -> None:
    case = CASES["digest-002-queue-backlog"]
    result = runner.score_case(
        case, "[Tool call: get_system_status] Queue depth is 417, error rate 0.8%, status degraded."
    )
    assert result["missing_facts"] == []
    assert result["tool_syntax_leak"] is True
    assert result["success"] is False


# --- bank integrity -----------------------------------------------------------


def test_every_case_declares_the_fields_the_scorer_reads() -> None:
    for case in BANK["cases"]:
        assert case["required_facts"], f"{case['id']} has no required facts"
        assert case["evidence"] and case["prompt"]
        # A forbidden phrase that appears in the evidence would fail a model for
        # quoting its own input — the fixture would be scoring the wrong thing.
        haystack = (case["evidence"] + case["prompt"]).lower()
        for forbidden in case["forbidden_facts"]:
            assert forbidden.lower() not in haystack, (
                f"{case['id']}: forbidden fact {forbidden!r} appears in its own evidence"
            )


def test_required_facts_are_themselves_grounded() -> None:
    # Every required fact must be derivable from the evidence — otherwise the
    # fixture asks the model to state something it was never given.
    for case in BANK["cases"]:
        haystack = (case["evidence"] + case["prompt"]).lower()
        for fact in case["required_facts"]:
            assert fact.lower() in haystack, f"{case['id']}: required fact {fact!r} absent from evidence"


# --- aggregation --------------------------------------------------------------


def test_summarize_cell_rates() -> None:
    records = [
        {
            "success": True,
            "missing_facts": [],
            "fabricated_numbers": [],
            "forbidden_hits": [],
            "tool_syntax_leak": False,
            "error": None,
            "latency_ms": 100.0,
            "completion_tokens": 40,
        },
        {
            "success": False,
            "missing_facts": ["1284"],
            "fabricated_numbers": ["37"],
            "forbidden_hits": [],
            "tool_syntax_leak": False,
            "error": None,
            "latency_ms": 300.0,
            "completion_tokens": 60,
        },
    ]
    block = runner.summarize_cell(records, wall_seconds=1.0)
    assert block["n_responses"] == 2
    assert block["grounded_accuracy"] == 0.5
    assert block["fact_recall_rate"] == 0.5
    assert block["fabricated_number_rate"] == 0.5
    assert block["tool_syntax_leak_rate"] == 0.0
    assert block["latency_p50_ms"] == 100.0


def test_summarize_cell_handles_no_records() -> None:
    block = runner.summarize_cell([], wall_seconds=0.0)
    assert block["n_responses"] == 0
    assert block["grounded_accuracy"] == 0.0
    assert block["latency_p50_ms"] == 0.0


def test_cell_name_keeps_on_off_and_grades_anything_else() -> None:
    # The name is a public join key (--cells, published tags.cell): renaming the
    # two existing levels would orphan every row already in the dataset.
    assert runner.cell_name("on") == "factual_think-on"
    assert runner.cell_name("off") == "factual_think-off"
    assert runner.cell_name("xhigh") == "factual_think-xhigh"


def test_thinking_body_kwargs_is_graded() -> None:
    """`on`/`off` send the bodies they always have; a graded level goes verbatim,
    so `high` and `xhigh` reach the model as themselves instead of collapsing."""
    assert runner.thinking_body_kwargs("enable_thinking", "on") == {
        "chat_template_kwargs": {"enable_thinking": True}
    }
    assert runner.thinking_body_kwargs("enable_thinking", "off") == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert runner.thinking_body_kwargs("reasoning_effort", "on") == {"reasoning_effort": "high"}
    assert runner.thinking_body_kwargs("reasoning_effort", "xhigh") == {"reasoning_effort": "xhigh"}
