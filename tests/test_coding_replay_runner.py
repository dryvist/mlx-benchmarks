"""Unit tests for the coding-replay runner's pure functions.

``harness/coding-replay/run.py`` is a standalone PEP 723 script (not part of
the package), so it is loaded here via importlib — the same pattern
``test_agentic_runner.py`` and ``test_factual_runner.py`` use. It has no
network or subprocess dependency at import time, so no live endpoint, served
model, or git checkout is needed to exercise these functions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import time
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "harness" / "coding-replay" / "run.py"
    spec = importlib.util.spec_from_file_location("coding_replay_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


# --- task_name / build_prompt / physical_model ---------------------------------


def test_task_name() -> None:
    assert runner.task_name("dryvist/tofu-proxmox", 1046) == "tofu-proxmox-1046"
    assert runner.task_name("dryvist/ansible-proxmox-ai", 655) == "ansible-proxmox-ai-655"


def test_build_prompt_includes_title_and_body() -> None:
    prompt = runner.build_prompt("fix: ignore plan files", "adds tfplan* to .gitignore")
    assert "fix: ignore plan files" in prompt
    assert "adds tfplan* to .gitignore" in prompt
    assert "Do not open a PR or commit" in prompt


def test_build_prompt_tolerates_empty_body() -> None:
    prompt = runner.build_prompt("title only", "")
    assert prompt.endswith("title only\n\n")


def test_row_carries_model_id_as_the_physical_id() -> None:
    """The published model id must be the served model, not the agent reference.

    ``model`` is the agent-CLI reference and carries a provider prefix; the
    publisher's extractor reads ``model_id`` from a JSON Lines run. Two arms
    served behind different provider names must still compare as one model id,
    and a row without ``model_id`` publishes as model="unknown" silently — the
    extractor falls back rather than raising, and the documented publish
    command passes no ``--model``. Cross-checked against the extractor itself
    in ``test_cli.py``.
    """
    agent_ref = "kimi/mlx-community/Kimi-Linear-48B-A3B-Instruct-6bit"
    assert runner.physical_model(agent_ref) == "mlx-community/Kimi-Linear-48B-A3B-Instruct-6bit"


def _parse(*extra: str) -> argparse.Namespace:
    """Parse a minimal valid argv, plus whatever the caller adds."""
    argv = [
        "--tasks-json",
        "t.json",
        "--clone-map-json",
        "c.json",
        "--work-dir",
        "wd",
        "--model",
        "m",
        "--tag",
        "tg",
        "--output",
        "o.jsonl",
        *extra,
    ]
    return runner.build_parser().parse_args(argv)


def test_dedicated_defaults_to_false() -> None:
    """An undeclared measurement environment must record as NOT dedicated.

    This is the anti-vacuity half of the pair below: it proves the recorded
    value is carried by the default rather than by ``--dedicated`` happening to
    be passed on every real invocation. The field exists because the row's other
    condition flags (``rate_limited``, ``slot_opened``, ``agent_launched``) only
    catch contention that MANIFESTED — a shared endpoint that happens not to
    trip a 429 is still not a dedicated measurement, and no per-row symptom can
    tell the difference. Defaulting true would silently relabel every historical
    shared run as clean.
    """
    assert _parse().dedicated is False


def test_dedicated_is_settable_both_ways() -> None:
    assert _parse("--dedicated").dedicated is True
    assert _parse("--no-dedicated").dedicated is False


def test_physical_model_strips_router_prefix() -> None:
    assert runner.physical_model("mlx/mlx-community/Qwen3.8-27B-4bit") == "mlx-community/Qwen3.8-27B-4bit"
    assert runner.physical_model("mlx-community/Qwen3.8-27B-4bit") == "mlx-community/Qwen3.8-27B-4bit"


def test_physical_model_strips_any_provider_not_only_mlx() -> None:
    """A provider named anything but ``mlx`` must still be stripped.

    Matching the literal ``mlx/`` left the prefix on for every other provider
    name, so the readiness probe asked the endpoint for an id it does not
    serve. That 404s on every poll and the runner aborts reporting a serving
    fault, which is the one diagnosis that is certainly wrong.
    """
    assert (
        runner.physical_model("kimi/mlx-community/Kimi-Linear-48B-A3B-Instruct-6bit")
        == "mlx-community/Kimi-Linear-48B-A3B-Instruct-6bit"
    )
    assert runner.physical_model("anything/org/name") == "org/name"


def test_physical_model_leaves_a_bare_two_segment_id_alone() -> None:
    """Anti-vacuity: the rule must not eat the org from a bare physical id.

    ``mlx-community/X`` is already physical. Stripping its first segment would
    send a bare model name to the endpoint and fail exactly like the bug this
    replaced, so the two cases must stay distinguishable.
    """
    assert runner.physical_model("mlx-community/Qwen3.6-35B-A3B-4bit") == "mlx-community/Qwen3.6-35B-A3B-4bit"
    assert runner.physical_model("lmstudio-community/Qwen3-Coder-Next-MLX-4bit") == (
        "lmstudio-community/Qwen3-Coder-Next-MLX-4bit"
    )
    # Single-segment physical id behind the historical prefix still works.
    assert runner.physical_model("mlx/some-single-segment-model") == "some-single-segment-model"


# --- parse_changed_files / overlap_files ----------------------------------------


def test_parse_changed_files_handles_ordinary_and_renamed_entries() -> None:
    porcelain = " M modules/proxmox-stack/locals.tf\n?? new-file.tf\nR  old.tf -> new.tf\n"
    assert runner.parse_changed_files(porcelain) == [
        "modules/proxmox-stack/locals.tf",
        "new-file.tf",
        "new.tf",
    ]


def test_parse_changed_files_empty() -> None:
    assert runner.parse_changed_files("") == []
    assert runner.parse_changed_files("\n\n") == []


def test_overlap_files_intersects_preserving_changed_order() -> None:
    changed = ["b.tf", "a.tf", "unrelated.tf"]
    pr_files = ["a.tf", "b.tf"]
    assert runner.overlap_files(changed, pr_files) == ["b.tf", "a.tf"]


def test_overlap_files_no_overlap() -> None:
    assert runner.overlap_files(["x.tf"], ["a.tf", "b.tf"]) == []


# --- passed: the pass@1 definition ----------------------------------------------


def test_passed_requires_both_clean_check_and_real_overlap() -> None:
    assert runner.passed(0, 1) is True


def test_passed_is_false_on_touched_nothing_but_exited_clean() -> None:
    # The exact false-green this suite exists to catch: check_rc == 0 (a
    # `check: none` task, or a check that happens to pass on an untouched
    # tree) with zero files overlapping the real PR must NOT score a pass.
    assert runner.passed(0, 0) is False


def test_passed_is_false_on_failing_check_even_with_overlap() -> None:
    assert runner.passed(1, 3) is False


def test_passed_is_false_on_unrecognized_check() -> None:
    assert runner.passed(2, 1) is False


# --- check_steps -----------------------------------------------------------------


def test_check_steps_none_is_a_no_op_pass() -> None:
    assert runner.check_steps("none") == []


def test_check_steps_known_checks() -> None:
    assert runner.check_steps("markdownlint") == [["markdownlint-cli2", "**/*.md"]]
    assert runner.check_steps("tofu-validate") == [
        ["tofu", "init", "-backend=false"],
        ["tofu", "validate"],
    ]
    assert runner.check_steps("ansible-lint") == [["ansible-lint", "--offline"]]
    assert runner.check_steps("nix-eval") == [["nix", "flake", "check"]]
    assert runner.check_steps("json-valid") == [["jq", ".", "deployment.json.example"]]


def test_check_steps_bats_prefix_carries_the_path() -> None:
    assert runner.check_steps("bats:tests/shell/test_foo.bats") == [
        ["nix", "shell", "nixpkgs#bats", "-c", "bats", "tests/shell/test_foo.bats"]
    ]


def test_check_steps_unknown_returns_none() -> None:
    assert runner.check_steps("shell-tests") is None
    assert runner.check_steps("") is None


# --- rate-limit / event parsing / token + ttft aggregation ----------------------


def test_is_rate_limited() -> None:
    assert runner.is_rate_limited('{"type":"error","statusCode":429}') is True
    assert runner.is_rate_limited('{"type":"text","part":{}}') is False


def test_parse_events_skips_malformed_lines() -> None:
    raw = '{"type":"text"}\nnot json\n{"type":"step_finish"}\n\n'
    events = runner.parse_events(raw)
    assert events == [{"type": "text"}, {"type": "step_finish"}]


def test_aggregate_tokens_empty() -> None:
    assert runner.aggregate_tokens([]) == {"input": None, "output": None, "cache_read": None, "steps": 0}


def test_aggregate_tokens_sums_step_finish_events() -> None:
    events = [
        {"type": "step_finish", "part": {"tokens": {"input": 100, "output": 10, "cache": {"read": 500}}}},
        {"type": "step_finish", "part": {"tokens": {"input": 50, "output": 5, "cache": {"read": 200}}}},
        {"type": "text", "part": {}},
    ]
    assert runner.aggregate_tokens(events) == {"input": 150, "output": 15, "cache_read": 700, "steps": 2}


def test_first_text_start_ms_returns_first_match() -> None:
    events = [
        {"type": "step_finish"},
        {"type": "text", "part": {"time": {"start": 1234567.0}}},
        {"type": "text", "part": {"time": {"start": 9999999.0}}},
    ]
    assert runner.first_text_start_ms(events) == 1234567.0


def test_first_text_start_ms_none_when_absent() -> None:
    assert runner.first_text_start_ms([{"type": "step_finish"}]) is None


def test_ttft_seconds() -> None:
    # first_text_start_ms is epoch-milliseconds; request_start is epoch-seconds.
    assert runner.ttft_seconds(1237000.0, 1000.0) == 237.0
    assert runner.ttft_seconds(None, 1000.0) is None


# --- serialization round trip for the config the runner reads ------------------


def test_bundled_tasks_json_is_jsonl_of_12_tasks() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "coding-replay" / "tasks.json"
    # A single JSON array, not JSON Lines: the repo's check-json pre-commit hook
    # validates every .json file as one document, and JSONL under a .json name
    # fails it. Renaming to .jsonl would have made the check stop applying
    # instead of pass.
    tasks = json.loads(path.read_text())
    assert len(tasks) == 12
    for task in tasks:
        assert {"repo", "pr", "base", "title", "files", "check"} <= task.keys()
        assert runner.check_steps(task["check"]) is not None


# --- agent launch detection ----------------------------------------------------


def test_agent_launched_false_when_no_step_events_and_instant_exit() -> None:
    # Measured against the real CLI: a bare physical model id with no provider
    # prefix exits 1 in 0.8s having emitted only an error event. Scored, that is
    # indistinguishable from a genuine failure — pass False, overlap 0 — so a
    # whole run of them reads as a verdict on the model.
    events = [{"type": "error", "error": {"name": "UnknownError"}}]
    assert runner.agent_launched(events, 0.8) is False


def test_agent_launched_false_on_empty_transcript() -> None:
    assert runner.agent_launched([], 0.5) is False


def test_agent_launched_true_on_a_real_transcript() -> None:
    events = [
        {"type": "step_start"},
        {"type": "text", "part": {"time": {"start": 1000}}},
        {"type": "step_finish", "part": {"tokens": {"input": 1, "output": 1}}},
    ]
    assert runner.agent_launched(events, 12.3) is True


def test_agent_launched_true_for_a_slow_run_that_did_start() -> None:
    # A long run that produced steps counts as launched even with no text part.
    assert runner.agent_launched([{"type": "step_start"}], 900.0) is True


# --- timeout stdout normalisation ---------------------------------------------


def test_decode_timeout_stdout_decodes_bytes() -> None:
    # subprocess.TimeoutExpired.stdout carries RAW BYTES even when the call set
    # text=True — the decoding wrapper never runs on the timeout path. A capped
    # run is the common outcome for a slow local model, so this is the hot path,
    # not an edge case.
    assert runner.decode_timeout_stdout(b'{"type":"text"}\n') == '{"type":"text"}\n'


def test_decode_timeout_stdout_passes_str_through() -> None:
    assert runner.decode_timeout_stdout('{"type":"text"}') == '{"type":"text"}'


def test_decode_timeout_stdout_handles_none() -> None:
    assert runner.decode_timeout_stdout(None) == ""


def test_decode_timeout_stdout_survives_invalid_utf8() -> None:
    # A run killed mid-write can leave a partial multi-byte sequence; losing the
    # whole capped task to a UnicodeDecodeError would be worse than one mojibake
    # character in one event line.
    assert runner.decode_timeout_stdout(b"ok\xff") == "ok�"


def test_timed_out_events_still_parse_after_decoding() -> None:
    # End to end for the bug: bytes off a timeout, through the decoder, into the
    # event parser that scoring depends on.
    raw = b'{"type":"text","part":{"time":{"start":1000}}}\n{"type":"step_finish"}\n'
    events = runner.parse_events(runner.decode_timeout_stdout(raw))
    assert len(events) == 2
    assert runner.first_text_start_ms(events) == 1000


# --- slot wait: bounded by wall clock, not by an attempt count -----------------


class _FakeResp:
    status = 200

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def test_wait_for_slot_returns_true_on_a_200(monkeypatch: Any) -> None:
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *a, **k: _FakeResp())
    assert runner.wait_for_slot("http://x/v1", "m", deadline_s=5.0) is True


def test_wait_for_slot_gives_up_within_its_deadline(monkeypatch: Any) -> None:
    # The regression this guards: the old form bounded the loop by an attempt
    # COUNT (60) while each attempt's cost came from a separate 120s request
    # timeout, so the real ceiling was ~2 hours and no caller could see it. A
    # wall-clock deadline is the only bound that holds when a request hangs.
    def boom(*_a: object, **_k: object) -> None:
        raise runner.urllib.error.URLError("refused")

    monkeypatch.setattr(runner.urllib.request, "urlopen", boom)
    started = time.monotonic()
    assert runner.wait_for_slot("http://x/v1", "m", deadline_s=0.3, interval_s=0.05) is False
    assert time.monotonic() - started < 5.0


def test_wait_for_slot_never_lets_one_request_outlive_the_budget(monkeypatch: Any) -> None:
    # A single request must not outlive the whole budget — that is precisely how
    # a 120s timeout under a 60-attempt loop turned into hours of silent stall.
    seen: list[float] = []

    def capture(*_a: object, **kwargs: Any) -> None:
        seen.append(float(kwargs["timeout"]))
        raise runner.urllib.error.URLError("refused")

    monkeypatch.setattr(runner.urllib.request, "urlopen", capture)
    runner.wait_for_slot("http://x/v1", "m", deadline_s=0.2, interval_s=0.05, request_timeout_s=30.0)
    assert seen, "no request was attempted"
    assert max(seen) <= 0.2


# --- sandbox isolation ---------------------------------------------------------


def _git(*args: str) -> None:
    subprocess.run(["git", *args], check=True, capture_output=True)


def _seed_repo(path: Path) -> str:
    """A real one-commit repo; returns the commit sha."""
    path.mkdir(parents=True)
    _git("-C", str(path), "init", "-q")
    _git("-C", str(path), "config", "user.email", "t@example.test")
    _git("-C", str(path), "config", "user.name", "t")
    (path / "AGENTS.md").write_text("# seed\n")
    _git("-C", str(path), "add", "AGENTS.md")
    # No signing in a throwaway fixture: -c disables it for THIS call only and
    # never writes commit.gpgsign into any config.
    _git("-C", str(path), "-c", "commit.gpgsign=false", "commit", "-q", "-m", "seed")
    out = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


def test_prepare_checkout_gives_the_task_a_real_git_dir_not_a_worktree_pointer(tmp_path: Path) -> None:
    # THE regression this guards. A git worktree's `.git` is a FILE holding
    # "gitdir: <source>/.git/worktrees/<name>". Any CLI resolving its project
    # root through git follows that back to the source repo and edits there —
    # measured, including a completed edit of a tracked file, while the scored
    # tree stayed empty. A real .git DIRECTORY terminates that resolution inside
    # the sandbox, which is why the sandbox must be a clone.
    source = tmp_path / "source"
    sha = _seed_repo(source)
    dest = tmp_path / "task"
    runner.prepare_checkout(source, sha, dest)

    assert (dest / ".git").is_dir(), ".git must be a real directory, not a worktree pointer file"
    assert (dest / "AGENTS.md").read_text() == "# seed\n"


def test_source_fingerprint_notices_a_write_to_the_source_clone(tmp_path: Path) -> None:
    # The containment canary. Isolation has already failed silently once, so the
    # runner compares this either side of every agent run instead of trusting
    # that the sandbox held.
    source = tmp_path / "source"
    _seed_repo(source)
    before = runner.source_fingerprint(source)
    (source / "AGENTS.md").write_text("# tampered\n")
    assert runner.source_fingerprint(source) != before


def test_run_agent_repoints_PWD_at_the_sandbox(tmp_path: Path, monkeypatch: Any) -> None:
    # THE bug that made this suite write to real repositories. subprocess sets
    # the child's working directory but leaves the inherited PWD naming the
    # caller's directory, and a Node/Bun CLI resolves its project from
    # process.env.PWD, not process.cwd(). Measured: cwd and the git root were
    # both correct and both ignored; the agent edited the source repo.
    seen: dict[str, Any] = {}

    def fake_run(argv: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)

        class P:
            returncode = 0
            stdout = ""

        return P()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setenv("PWD", "/somewhere/else")
    monkeypatch.setenv("DIRENV_DIR", "-/somewhere/else")
    monkeypatch.setenv("DIRENV_FILE", "/somewhere/else/.envrc")

    runner.run_agent("do a thing", tmp_path, "mlx/m", ["opencode", "run"], 10)

    assert seen["cwd"] == tmp_path
    assert seen["env"]["PWD"] == str(tmp_path), "PWD must name the sandbox, not the caller's directory"
    assert not [k for k in seen["env"] if k.startswith("DIRENV_")], "direnv vars re-pin the old project"


def test_is_rate_limited_separates_contention_from_a_bad_config() -> None:
    # Both produce "exit 1, no step events". Only this distinguishes them, and
    # the abort message picks its advice from it — telling someone to check
    # their provider prefix while another consumer holds the slot sends them
    # after the wrong thing.
    contention = '{"type":"error","error":{"data":{"statusCode":429}}}'
    bad_config = '{"type":"error","error":{"name":"UnknownError"}}'
    assert runner.is_rate_limited(contention) is True
    assert runner.is_rate_limited(bad_config) is False
    assert runner.agent_launched(runner.parse_events(contention), 7.9) is False
    assert runner.agent_launched(runner.parse_events(bad_config), 0.8) is False


def test_source_fingerprint_is_stable_when_only_the_task_clone_changes(tmp_path: Path) -> None:
    # Anti-vacuity: the canary must not fire on the sandbox doing its job, or it
    # would abort every run and prove nothing.
    source = tmp_path / "source"
    sha = _seed_repo(source)
    dest = tmp_path / "task"
    runner.prepare_checkout(source, sha, dest)
    before = runner.source_fingerprint(source)
    (dest / "AGENTS.md").write_text("# edited in the sandbox\n")
    assert runner.source_fingerprint(source) == before


def test_reasoning_effort_defaults_to_unstated() -> None:
    """An undeclared reasoning effort must record as literally "unstated".

    The anti-vacuity half: a run built without ``--reasoning-effort`` still
    carries the field, so the assertion is held by the default rather than by
    the flag happening to be passed. The default is deliberately not a
    plausible value such as "default" or "medium" — once a guess is in the row
    it is indistinguishable from a measurement, and every run recorded before
    this field existed genuinely has an unknown effort.
    """
    assert _parse().reasoning_effort == "unstated"


def test_reasoning_effort_is_recorded_verbatim() -> None:
    """Effort is part of an arm's identity, so it is stored exactly as declared.

    No normalisation and no allowed-value list: harnesses name these
    differently (high/xhigh/max, on/off, a numeric budget), and silently
    rewriting one into another would merge two arms that are not the same arm.
    """
    for value in ("low", "medium", "high", "xhigh", "max", "off"):
        assert _parse("--reasoning-effort", value).reasoning_effort == value
