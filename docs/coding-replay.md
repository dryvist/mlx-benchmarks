# coding-replay — real-PR replay suite

Runs a small merged PR back through an agentic CLI against one served model
and scores whether the model actually landed the change, rather than whether
it produced *some* text and exited 0.

Runner: [`harness/coding-replay/run.py`](../harness/coding-replay/run.py).
Task list: [`configs/coding-replay/tasks.json`](../configs/coding-replay/tasks.json)
(a JSON array — one real merged PR per entry: repo, PR number, base commit,
title/body, changed files, and a named repo check). It is a single JSON
document rather than JSON Lines because the `check-json` pre-commit hook
validates every `.json` file as one document; a `.jsonl` name would have made
that check stop applying rather than pass. Converter:
[`../src/mlx_benchmarks/converters/coding_replay.py`](../src/mlx_benchmarks/converters/coding_replay.py),
`--kind coding-replay`.

## What "pass" means

```text
pass@1 = check_rc == 0 AND overlap > 0
```

`check_rc` is the exit code of the task's named repo check
(`markdownlint`, `tofu-validate`, `ansible-lint`, `nix-eval`, `bats:<path>`,
`json-valid`, `none`) run in the task's clone after the CLI exits. `overlap`
is the count of changed files that intersect the real PR's file list.

A clean exit or a clean check is not by itself evidence of anything: a
`check: none` task, or a check that happens to pass on an untouched tree,
scores `check_rc == 0` whether or not the model changed a single file. Only
the overlap with the real PR's changed files proves the model touched the
right code — `passed()` in the runner is the single source of truth for this
and is unit-tested against exactly that false-green case.

## Resolution of the current task set

Measured over six clean runs — two models, three runs each, isolated
single-consumer endpoints. Nine of the twelve tasks pass in every run, one
fails in every run, and only two ever vary:

| Task | passes / 6 |
| --- | --- |
| tofu-proxmox#1060 | 0 |
| ansible-proxmox-ai#641 | 1 |
| nix-darwin#2330 | 5 |
| the other nine | 6 each |

So `pass_rate` on this task set carries roughly two tasks of signal. Two
causes, both in the task list rather than the runner:

- Ten of the twelve real PRs touch exactly one file, so `overlap > 0` reduces
  to "found the one file". 64 of 72 rows score `overlap` exactly 1, and a
  fractional score could not exceed 1.0 either.
- The two tasks that do vary fail by reaching `--task-timeout-s`, not by
  producing wrong code. Halving endpoint throughput lowers the score with no
  change in the model — measured directly: the same model and task set scored
  10/12 with one timeout when sharing its endpoint with a second concurrent
  run, against 10-11/12 with one timeout across three sole-consumer runs.

Until the task list carries multi-file PRs and checks that exercise behaviour
rather than lint, read `pass_rate` here as saturated. The per-row `wall_s`,
`tokens.steps` and `tokens.output` still separate models cleanly: across the
runs above, one arm solved the same tasks at a 127 s median with 6.2 steps
where the other took 238 s and 18.2 steps.

## Envelope shape

One `pass_at_1` result per task (tags: `task`, `repo`, `check`, `check_rc`,
`overlap`), plus one aggregate `pass_rate` result across every task in the
run — the suite's headline number. `suite` is `coding` (shared with the
`lm-eval`/humaneval-mbpp coding suite in [`RUNBOOK.md`](RUNBOOK.md); this
suite does not feed a `RANKINGS.md` row of its own).

## Running it

Per task: an isolated `--shared` clone of the target repo at the PR's base
commit, a prompt of the PR title + body plus an instruction to implement and
stop (no commit, no PR), the configured agentic CLI headless with a timeout,
then scoring.

> **`cwd` does not confine an agentic CLI. `PWD` does.** Measured 2026-09-06:
> with `cwd` set to the task sandbox, the CLI read and *edited* files in the
> **source** clone — a completed edit of a tracked file — while `git status` in
> the sandbox stayed empty and the task scored zero. `subprocess` sets the
> child's working directory but leaves the inherited `PWD` alone, and a
> Node/Bun CLI commonly resolves its project directory from `process.env.PWD`
> rather than `process.cwd()`. It reproduced under a git worktree *and* under a
> real clone whose `git rev-parse --show-toplevel` returned the sandbox — cwd
> and the git root were both correct and both ignored.
>
> The runner now passes an explicit environment with `PWD` set to the sandbox
> and every `DIRENV_*` variable stripped (they pin the old project directory the
> same way). Verified both directions: the agent's edit lands in the sandbox,
> and `source_touched` stays false.
>
> Every row still carries `source_touched`, sampled either side of the run, and
> the runner **aborts** on a true one. Isolation is a property that failed
> silently once; it is checked every run rather than assumed. Any `pass@1`
> produced before this fix is invalid — changes landed outside the scored tree,
> so `overlap` was 0 and `pass` False regardless of model.

The sandbox is also a `--shared` clone rather than a git worktree: a worktree's
`.git` is a file pointing back at the source repository, so git-based root
resolution walks home. Necessary, but it was not what was leaking. The local serving gate refuses rather than queues (one slot per
model), so the runner polls for a real 200 completion before each task and
retries once if a run's stdout carries an HTTP 429.

That poll is bounded by wall clock (`--slot-wait-s`, default 600). If no slot
opens inside it the run **aborts** rather than scoring the task: a model that
was never asked would otherwise record a zero indistinguishable from a real
failure. Raise the flag only for an endpoint known to be slow to admit — never
to ride out a wedged one.

```sh
uv run harness/coding-replay/run.py \
  --tasks-json configs/coding-replay/tasks.json \
  --clone-map-json clone-map.json \
  --work-dir /tmp/coding-replay-wt \
  --base-url http://127.0.0.1:11434/v1 \
  --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
  --tag run1 \
  --output out.jsonl

.venv/bin/mlx-bench-publish out.jsonl --kind coding-replay --suite coding --dry-run
```

`clone-map.json` maps each task's `repo` (`owner/name`) to the local path of
its clone — environment-specific, not committed.
