#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""coding-replay — replay merged PRs through an agentic CLI, score pass@1.

Per task: make an isolated clone of the target repo at the PR's base commit,
prompt the configured agentic CLI (default: ``opencode run --pure --auto
--agent build --format json``) with the PR title + body and an instruction to
implement and stop (no commit, no PR), then score the run:

    pass@1 = check_rc == 0 AND overlap > 0

Exit code 0 alone is not a pass — a run that touches none of the real PR's
changed files and exits cleanly (e.g. a ``check: none`` task) must score a
failure. ``overlap`` is the count of changed files that intersect the real
PR's file list; ``check`` is a named repo check (``markdownlint``,
``tofu-validate``, ``ansible-lint``, ``nix-eval``, ``bats:<path>``,
``json-valid``, ``none``) run in that clone after the CLI exits.

The local serving gate refuses rather than queues (one slot per model), so a
readiness probe waits for a real 200 completion before each task starts, and a
run whose stdout carries an HTTP 429 re-waits and retries (``--rate-limit-attempts``,
default 4). The probe proves a slot was free at probe time only — on a shared
endpoint another consumer can take it before the agent's first call, so a 429
after a clean probe is contention, not misconfiguration.

Run (never against a busy Studio without asking)::

    uv run harness/coding-replay/run.py \\
        --tasks-json configs/coding-replay/tasks.json \\
        --clone-map-json clone-map.json \\
        --work-dir /tmp/coding-replay-wt \\
        --base-url http://127.0.0.1:11434/v1 \\
        --model mlx/mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \\
        --tag run1 \\
        --output out.jsonl

The agent is PINNED in the default ``--agent-cmd``. An agentic CLI picks a
default primary agent from the runner's own configuration, and an
exploration-oriented one reads and greps without ever editing — which scores
as a model that cannot do the task. Measured on one model and one task: the
machine's default agent made 5 grep / 4 read / 1 glob / 2 bash calls and
changed nothing; ``--agent build`` made 6 edit calls and changed a file. Any
replacement ``--agent-cmd`` must pin an editing agent or the run measures the
operator's config rather than the model.

``--base-url`` is used ONLY for the readiness probe. The agentic CLI resolves
its own endpoint, so it must be configured separately or it will fail to reach
the model at all — for opencode that means ``OPENCODE_CONFIG`` pointing at a
config that declares the provider, and a ``--model`` of the form
``<provider>/<physical id>``. A bare physical id has no provider and the CLI
exits immediately.

``clone-map.json`` maps each task's ``repo`` (``owner/name``) to the local
path of its clone, e.g.
``{"dryvist/tofu-proxmox": "/Users/me/git/.../tofu-proxmox"}`` — not
committed; environment-specific.

Output is one raw-results JSON Lines file (one row per task), appended to as
each task finishes; publish it with ``mlx-bench-publish out.jsonl --kind
coding-replay --suite coding``.

Scoring, prompt construction, check selection, and result parsing live in
importable pure functions so ``tests/test_coding_replay_runner.py`` can
exercise them without a live endpoint, a served model, or a git checkout.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Pure functions (unit-tested via tests/test_coding_replay_runner.py)
# ---------------------------------------------------------------------------


def task_name(repo: str, pr: int) -> str:
    """``dryvist/tofu-proxmox`` + 1046 -> ``tofu-proxmox-1046``."""
    return f"{repo.rsplit('/', 1)[-1]}-{pr}"


def build_prompt(title: str, body: str) -> str:
    return f"Implement this change in the repository, then stop. Do not open a PR or commit.\n\nTitle: {title}\n\n{body or ''}"


def physical_model(model: str) -> str:
    """Strip an agent-CLI provider prefix for the readiness-probe body.

    The agent CLI addresses a model as ``<provider>/<physical id>``; the served
    endpoint wants the physical id alone. A physical id is itself normally
    ``<org>/<name>``, so a prefixed ref carries two slashes and a bare one
    carries at most one — which is what distinguishes them without knowing the
    provider's name.

    Matching on the literal ``mlx/`` instead was wrong in a way that cost a
    600 s run: a provider named anything else left the prefix on, the probe
    asked the endpoint for a model id it does not serve, got a 404 on every
    poll, and the runner aborted blaming the endpoint ("check that the model is
    loaded"). The configuration fault was reported as a serving fault.

    The bare ``mlx/<name>`` case is kept for single-segment physical ids.
    """
    _, sep, rest = model.partition("/")
    if sep and "/" in rest:
        return rest
    return model.removeprefix("mlx/")


def parse_changed_files(porcelain: str) -> list[str]:
    """Parse ``git status --porcelain`` output into a list of changed paths."""
    files: list[str] = []
    for line in porcelain.splitlines():
        if not line.strip():
            continue
        path = line[3:]
        if " -> " in path:  # rename: "old -> new"
            path = path.split(" -> ", 1)[1]
        files.append(path)
    return files


def overlap_files(changed: Sequence[str], pr_files: Sequence[str]) -> list[str]:
    """Changed files that intersect the real PR's file list, order preserved."""
    pr_set = set(pr_files)
    return [f for f in changed if f in pr_set]


def passed(check_rc: int | None, overlap: int) -> bool:
    """pass@1: a clean check AND at least one real-PR file actually touched.

    A clean exit / clean check alone is not enough — a run that touches
    nothing still exits 0 on a ``check: none`` task, and would otherwise
    score a false pass.
    """
    return check_rc == 0 and overlap > 0


def agent_launched(events: Sequence[Mapping[str, Any]], wall_s: float) -> bool:
    """Whether the agentic CLI actually started and produced a transcript.

    A CLI that cannot reach its model exits within a second having emitted no
    events (or one error event) — and that is INDISTINGUISHABLE from a genuine
    failure once scored: `pass` is False either way, `overlap` is 0 either way.
    A whole run in that state reads as a damning verdict on the model when the
    real fault is configuration. Measured: a bare physical model id with no
    provider produced exit 1, zero steps, wall 0.8s, and scored as a legitimate
    task failure.

    A real run always emits at least one ``step_start``/``step_finish``.
    """
    if wall_s < 2.0 and not any(e.get("type", "").startswith("step") for e in events):
        return False
    return any(e.get("type", "").startswith("step") for e in events)


def check_steps(check: str) -> list[list[str]] | None:
    """Ordered shell argv steps for a named repo check; ``None`` if unrecognized.

    Every step must exit 0 for the check to pass; the first non-zero exit
    short-circuits the rest.
    """
    if check == "none":
        return []
    if check == "markdownlint":
        return [["markdownlint-cli2", "**/*.md"]]
    if check == "tofu-validate":
        return [["tofu", "init", "-backend=false"], ["tofu", "validate"]]
    if check == "ansible-lint":
        return [["ansible-lint", "--offline"]]
    if check == "nix-eval":
        return [["nix", "flake", "check"]]
    if check == "json-valid":
        return [["jq", ".", "deployment.json.example"]]
    if check.startswith("bats:"):
        return [["nix", "shell", "nixpkgs#bats", "-c", "bats", check.removeprefix("bats:")]]
    return None


def is_rate_limited(raw_stdout: str) -> bool:
    """Whether a headless run's JSON stdout carries an HTTP 429 status."""
    return '"statusCode":429' in raw_stdout


def parse_events(raw_stdout: str) -> list[dict[str, Any]]:
    """Parse one JSON object per line, skipping any line that fails to parse."""
    events: list[dict[str, Any]] = []
    for line in raw_stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def aggregate_tokens(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum token counts across every ``step_finish`` event."""
    steps = [e["part"]["tokens"] for e in events if e.get("type") == "step_finish" and "part" in e]
    if not steps:
        return {"input": None, "output": None, "cache_read": None, "steps": 0}
    return {
        "input": sum(t.get("input") or 0 for t in steps),
        "output": sum(t.get("output") or 0 for t in steps),
        "cache_read": sum((t.get("cache") or {}).get("read") or 0 for t in steps),
        "steps": len(steps),
    }


def first_text_start_ms(events: Iterable[Mapping[str, Any]]) -> float | None:
    """Epoch-ms timestamp of the first streamed text part, or ``None``."""
    for e in events:
        if e.get("type") == "text":
            start = ((e.get("part") or {}).get("time") or {}).get("start")
            if start is not None:
                return float(start)
    return None


def ttft_seconds(first_text_start_ms_value: float | None, request_start_epoch_s: float) -> float | None:
    if first_text_start_ms_value is None:
        return None
    return round(first_text_start_ms_value / 1000 - request_start_epoch_s, 1)


# ---------------------------------------------------------------------------
# Orchestration (network + subprocess; exercised against fixtures, not live)
# ---------------------------------------------------------------------------


def wait_for_slot(
    base_url: str,
    model: str,
    deadline_s: float = 600.0,
    interval_s: float = 5.0,
    request_timeout_s: float = 30.0,
) -> bool:
    """Poll the endpoint for a real 200 completion; the gate refuses rather than queues.

    Bounded by WALL CLOCK, not by an attempt count. The earlier form counted 60
    attempts against a 120 s per-request timeout, so a hung endpoint blocked for
    up to ~2 hours per call — and `run_task` calls this inside a two-attempt
    retry, making ~4 hours per task. Measured: a run stalled at 1 of 12 tasks
    after 128 minutes with no output, which is exactly this path. An attempt
    count cannot bound anything when each attempt's cost is set elsewhere.

    The probe asks for one token, so 30 s is already generous for it; the long
    wait belongs in the deadline, where it is visible and configurable, not
    hidden in a per-request timeout multiplied by a loop.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": "OK"}], "max_tokens": 1}
    ).encode()
    deadline = time.monotonic() + deadline_s
    announced = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=min(request_timeout_s, remaining)) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            pass
        # Say something the first time the slot is busy. A silent wait is
        # indistinguishable from a wedged run for however long the deadline is.
        if not announced:
            print(
                f"waiting up to {deadline_s:.0f}s for a free slot on {model}...",
                file=sys.stderr,
            )
            announced = True
        time.sleep(min(interval_s, max(0.0, deadline - time.monotonic())))


def decode_timeout_stdout(raw: bytes | str | None) -> str:
    """Normalise ``TimeoutExpired.stdout`` to text.

    ``subprocess.TimeoutExpired.stdout`` carries RAW BYTES even when the call
    set ``text=True`` — the decoding wrapper never runs on the timeout path.
    This matters here rather than being a formality: a capped run is the common
    outcome for a slow local model, so the timeout branch is the hot path, and
    handing bytes to the event parser downstream would lose every task that ran
    out of clock.
    """
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace")
    return raw


def run_agent(
    prompt: str, checkout: Path, model: str, agent_cmd: Sequence[str], timeout_s: int
) -> tuple[int, str]:
    """Run the agentic CLI headless in ``checkout``; returns (exit_code, stdout).

    ``cwd`` ALONE DOES NOT CONFINE THE AGENT, and this is the whole reason the
    suite was writing to real repositories. ``subprocess`` sets the child's
    working directory but leaves the INHERITED ``PWD`` untouched, and a
    Node/Bun-based CLI commonly resolves its project directory from
    ``process.env.PWD`` rather than ``process.cwd()``. Measured 2026-09-06: with
    ``cwd`` set to the sandbox and ``PWD`` still naming the source clone, every
    file the agent read and edited was in the SOURCE — under a git worktree and
    again under a real clone whose ``rev-parse --show-toplevel`` was the sandbox.

    ``direnv`` variables pin the old directory the same way, so they go too;
    leaving them lets the CLI's shell re-enter the source project's environment.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("DIRENV_")}
    env["PWD"] = str(checkout)
    env.pop("OLDPWD", None)
    argv = [*agent_cmd, "-m", model, prompt]
    try:
        proc = subprocess.run(
            argv,
            cwd=checkout,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            stdin=subprocess.DEVNULL,
        )
        return proc.returncode, proc.stdout
    except subprocess.TimeoutExpired as exc:
        return 124, decode_timeout_stdout(exc.stdout)


def run_check(check: str, cwd: Path) -> int:
    steps = check_steps(check)
    if steps is None:
        return 2
    for step in steps:
        rc = subprocess.run(step, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        if rc != 0:
            return rc
    return 0


def prepare_checkout(clone: Path, base: str, dest: Path) -> None:
    """Give the task its own real clone. NEVER a git worktree.

    A git worktree's ``.git`` is a FILE holding ``gitdir: <source>/.git/worktrees/<name>``.
    An agentic CLI that resolves its own project root through git follows that
    pointer back to the SOURCE clone, then reads and EDITS there — outside the
    tree the score is computed from.

    Measured 2026-09-06 on the first real run: with ``cwd`` set to the worktree,
    every path the agent touched was still rooted at the source clone — the glob
    root, three reads, and a completed edit of a tracked file. ``git status`` in
    the worktree stayed empty, so the task scored zero while the real repository
    was modified. Passing ``cwd`` to the subprocess does not constrain a CLI that
    resolves its root itself.

    A clone has a real ``.git`` DIRECTORY, so git-based root resolution
    terminates inside the sandbox. On its own that did NOT stop the leak — the
    actual cause was an inherited ``PWD``; see ``run_agent``. Both matter, and
    ``source_touched`` proves it held on every run rather than assuming it.
    """
    subprocess.run(["git", "-C", str(clone), "fetch", "-q", "origin", base], check=False)
    subprocess.run(["rm", "-rf", str(dest)], check=False)
    subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", str(clone), str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", "--detach", base], check=True)


def source_fingerprint(clone: Path) -> str:
    """Working-tree state of the SOURCE clone, as the containment canary.

    Compared either side of the agent run. Isolation is a property that has now
    failed silently once, and the only honest way to claim it holds is to check
    it every time rather than trust the sandbox construction.
    """
    proc = subprocess.run(
        ["git", "-C", str(clone), "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return proc.stdout


def run_task(
    task: dict[str, Any],
    clone_map: Mapping[str, str],
    work_dir: Path,
    model: str,
    tag: str,
    agent_cmd: Sequence[str],
    base_url: str,
    task_timeout_s: int,
    slot_wait_s: float = 600.0,
    rate_limit_attempts: int = 4,
    dedicated: bool = False,
    reasoning_effort: str = "unstated",
) -> dict[str, Any]:
    repo, pr, base = task["repo"], task["pr"], task["base"]
    name = task_name(repo, pr)
    clone = Path(clone_map[repo])
    checkout = work_dir / f"{name}-{tag}"
    prepare_checkout(clone, base, checkout)
    prompt = build_prompt(task["title"], task.get("body", ""))
    phys = physical_model(model)
    source_before = source_fingerprint(clone)

    start = end = time.time()
    rc, stdout = -1, ""
    slot_opened = False
    # A single retry is not enough on a shared endpoint. `wait_for_slot` proves a
    # slot was free at probe time, but another consumer can take it in the gap
    # before the agent's first call — so a 429 here is CONTENTION, not a verdict
    # and not a misconfiguration. Measured: a 12-task run lost its third task to
    # exactly this, 7.9s and zero steps, with two attempts exhausted.
    for attempt in range(rate_limit_attempts):
        if not wait_for_slot(base_url, phys, deadline_s=slot_wait_s):
            break
        slot_opened = True
        start = time.time()
        rc, stdout = run_agent(prompt, checkout, model, agent_cmd, task_timeout_s)
        end = time.time()
        if is_rate_limited(stdout) and attempt + 1 < rate_limit_attempts:
            subprocess.run(["git", "-C", str(checkout), "checkout", "-q", "--", "."], check=False)
            print(
                f"{name}: rate-limited, re-waiting for a slot (attempt {attempt + 1}/{rate_limit_attempts})",
                file=sys.stderr,
            )
            continue
        break

    events = parse_events(stdout)
    # Keep the agent's raw transcript. A scored zero is otherwise unexplainable
    # after the fact: `pass` false with `overlap` 0 looks identical whether the
    # model reasoned for the whole cap without reaching an edit, or emitted one
    # error and quit. Reconstructing that needed a separate probe run against a
    # live model, which is a poor substitute for having kept the evidence.
    transcript = work_dir / f"{name}-{tag}.transcript.jsonl"
    transcript.write_text(stdout)

    status = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    changed = parse_changed_files(status.stdout)
    overlap = overlap_files(changed, task.get("files", []))
    check_rc = run_check(task["check"], checkout)

    return {
        "model": model,
        # The physical id, under the key the publisher's model extractor reads
        # for a JSON Lines run. Without it every coding-replay row publishes as
        # model="unknown" — silently, because the extractor falls back rather
        # than raising, and the documented publish command passes no --model.
        # `model` above stays the agent-CLI reference (provider prefix and all)
        # for provenance; only this one is the served model, so two arms behind
        # different provider names still compare as the same model id.
        "model_id": physical_model(model),
        "tag": tag,
        "task": name,
        "repo": repo,
        "pr": pr,
        "exit_code": rc,
        "check": task["check"],
        "check_rc": check_rc,
        "overlap": len(overlap),
        "overlap_files": overlap,
        "changed": changed,
        "pass": passed(check_rc, len(overlap)),
        "agent_launched": agent_launched(events, round(end - start, 1)),
        # Whether a serving slot ever opened. False means the agent was never
        # given the chance to run, so the zero below is about the endpoint, not
        # the model — same false-negative class as `agent_launched`.
        "slot_opened": slot_opened,
        # Whether the FINAL attempt died to a 429. With agent_launched false this
        # says contention, not configuration — the two need different fixes.
        "rate_limited": is_rate_limited(stdout),
        # Was this arm the ONLY consumer of its endpoint? Declared by the caller,
        # never inferred: the harness cannot see what else is on the box. The
        # fields above catch contention that MANIFESTED — a shared endpoint that
        # happens not to trip a 429 still is not a dedicated measurement, and
        # nothing in a row can tell the difference. Defaults false, because an
        # unstated measurement environment is an untrusted one. Compare within a
        # class; never across one.
        "dedicated": dedicated,
        # At what reasoning effort did this arm run? Declared by the caller,
        # never inferred: effort is set in the agent CLI's own configuration or
        # by the serving default, and neither is visible from here.
        #
        # An arm is weights PLUS quant PLUS effort PLUS serving config. The same
        # weights at two efforts are two arms, and comparing them as one model
        # is the same error as comparing across a `dedicated` boundary. Public
        # leaderboards already treat effort as part of a model's identity
        # ("27B (xhigh)"); a row without it cannot be placed against them or
        # against our own history.
        #
        # Defaults to "unstated" rather than to a plausible value such as
        # "default" or "medium". A guess here is indistinguishable from a
        # measurement, and an unstated condition must stay visibly unstated.
        "reasoning_effort": reasoning_effort,
        # Containment: did the agent modify the SOURCE clone? True means the
        # sandbox leaked and both the result and the repository are suspect.
        "source_touched": source_fingerprint(clone) != source_before,
        "transcript": str(transcript),
        "tokens": aggregate_tokens(events),
        "ttft_s": ttft_seconds(first_text_start_ms(events), start),
        "wall_s": round(end - start, 1),
        "timestamp": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--tasks-json", type=Path, required=True, help="task list: a JSON array of task objects")
    ap.add_argument(
        "--clone-map-json", type=Path, required=True, help="JSON file mapping repo -> local clone path"
    )
    ap.add_argument("--work-dir", type=Path, required=True, help="scratch directory for per-task clones")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="run tag recorded on every task row")
    ap.add_argument("--filter", default=".", help="regex over 'repo#pr', matched with re.search")
    ap.add_argument("--task-timeout-s", type=int, default=900)
    ap.add_argument(
        "--rate-limit-attempts",
        type=int,
        default=4,
        help="times to re-wait for a slot and retry after an HTTP 429. A shared "
        "endpoint can hand the slot to another consumer between the readiness "
        "probe and the agent's first call",
    )
    ap.add_argument(
        "--slot-wait-s",
        type=float,
        default=600.0,
        help="wall-clock ceiling on waiting for a free serving slot, per attempt. "
        "Bounds the whole wait; a busy endpoint aborts the run instead of stalling",
    )
    ap.add_argument(
        "--agent-cmd",
        # `--agent build` is load-bearing, not decoration. Without it the CLI
        # uses whatever primary agent the RUNNER'S OWN config makes default,
        # which on a customised machine is not a stock editing agent — measured
        # side by side on one model and one task, the default agent made 5 grep
        # / 4 read / 1 glob / 2 bash calls and changed no files, while
        # `--agent build` made 6 edit calls and changed one. Same model, same
        # prompt, same config: the benchmark was scoring the operator's agent,
        # not the model.
        default="opencode run --pure --auto --agent build --format json",
        help="agentic CLI invocation, split on whitespace; '-m <model> <prompt>' is appended. "
        "Must pin an editing agent, or results measure the runner's local agent config",
    )
    ap.add_argument(
        "--reasoning-effort",
        default="unstated",
        help="declare the reasoning effort this arm ran at (e.g. low, medium, high, "
        "xhigh, max, off). Cannot be detected — effort lives in the agent CLI's "
        "config or the serving default, neither visible to the harness — so it is "
        "declared, and defaults to 'unstated' because a plausible guess is "
        "indistinguishable from a measurement. The same weights at two efforts are "
        "two arms; compare within one.",
    )
    ap.add_argument("--output", type=Path, required=True, help="JSON Lines file; one row appended per task")
    ap.add_argument(
        "--dedicated",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="declare that this arm was the ONLY consumer of its endpoint. Cannot "
        "be detected — the harness cannot see what else is on the box — so it is "
        "declared, and defaults to false because an unstated measurement "
        "environment is an untrusted one. Runs are comparable only within one "
        "value of this flag",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = json.loads(args.tasks_json.read_text())
    clone_map = json.loads(args.clone_map_json.read_text())
    args.work_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(args.filter)
    agent_cmd = args.agent_cmd.split()

    with args.output.open("a") as out:
        for task in tasks:
            label = f"{task['repo']}#{task['pr']}"
            if not pattern.search(label):
                continue
            if task["repo"] not in clone_map:
                print(f"no clone mapped for {task['repo']}; skipping {label}", file=sys.stderr)
                continue
            row = run_task(
                task,
                clone_map,
                args.work_dir,
                args.model,
                args.tag,
                agent_cmd,
                args.base_url,
                args.task_timeout_s,
                args.slot_wait_s,
                args.rate_limit_attempts,
                args.dedicated,
                args.reasoning_effort,
            )
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(
                f"{row['task']}: pass={row['pass']} overlap={row['overlap']} check_rc={row['check_rc']}",
                file=sys.stderr,
            )
            if row["source_touched"]:
                # Stop at once: this is a containment breach, not a result. The
                # source clone is a real repository the operator works in, and
                # every further task would keep writing to it while scoring zero.
                print(
                    f"ABORT: {row['task']} modified the SOURCE clone at "
                    f"{clone_map[task['repo']]}. The agent escaped its sandbox, so "
                    "this row is not a model result and the repository needs "
                    "inspecting — check `git status` there and revert what the run "
                    "wrote. No further tasks attempted.",
                    file=sys.stderr,
                )
                return 3
            if not row["slot_opened"]:
                # Same reasoning as the agent_launched abort below: no slot
                # means no attempt, and every remaining task would score an
                # identical false zero against a model that was never asked.
                print(
                    f"ABORT: no serving slot opened for {row['task']} within "
                    f"{args.slot_wait_s:.0f}s. This is an endpoint fault, not a "
                    "model result — check that the model is loaded and that no "
                    "other run holds the single local slot. Raise --slot-wait-s "
                    "only if the endpoint is known to be slow to admit, never to "
                    "paper over a wedged one. No further tasks attempted.",
                    file=sys.stderr,
                )
                return 2
            if not row["agent_launched"]:
                # Abort rather than grind through the remaining tasks. Every
                # one would score a false failure, and a full sheet of zeros
                # reads as a verdict on the model instead of a broken setup.
                #
                # Name the RIGHT cause. A 429 and a bad provider prefix both
                # produce "exit 1, no step events", but one is a busy endpoint
                # and the other is a typo — telling someone to check their
                # config while another consumer holds the slot sends them after
                # the wrong thing entirely.
                if row["rate_limited"]:
                    print(
                        f"ABORT: {row['task']} was rate-limited on every one of "
                        f"{args.rate_limit_attempts} attempts. This is CONTENTION, "
                        "not configuration and not a model result — another consumer "
                        "holds the single serving slot. Re-run when the endpoint is "
                        "quiet, or raise --rate-limit-attempts. No further tasks "
                        "attempted.",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"ABORT: the agentic CLI did not run for {row['task']} "
                        f"(exit {row['exit_code']}, {row['wall_s']}s, no step events). "
                        "This is a configuration fault, not a model result — the CLI "
                        "resolves its own endpoint, so check its config and that "
                        "--model carries a provider prefix. No further tasks attempted.",
                        file=sys.stderr,
                    )
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
