# configs/ layout

One TOML file per `(upstream-tool, suite)` pair. These are **runbooks**: they
record the task list and tool-native options for a suite. No file here is read
by in-repo code — they document *what to run* so a run is reproducible.

## Layout (as shipped)

```text
configs/
├── LAYOUT.md                 # this file
├── lm-eval/
│   ├── reasoning.toml        # arc_challenge_chat (quick) / gsm8k (canonical)
│   ├── coding.toml           # humaneval, mbpp
│   ├── math-hard.toml        # minerva_math500
│   └── qwen3-tasks/          # optional <think>-stripping overlay (see below)
├── vllm/
│   └── benchmark_serving.toml # vllm throughput cross-check; no local install
├── agentic/
│   └── tool-calling.toml     # in-repo runner: harness/agentic/run.py
├── llama-cpp/
│   └── throughput.toml       # in-repo runner: harness/throughput/run.py,
│                             # pointed at an llm-4080 llama.cpp/Vulkan guest;
│                             # draft-model A/B sweep for speculative decoding
├── promptstack/
│   ├── promptstack.toml      # in-repo runner: harness/promptstack/run.py
│   ├── probes/               # frozen probe banks, one JSON per probe class
│   └── prompts/              # composed system prompts under test
├── factual/
│   ├── grounded-summary.toml # in-repo runner: harness/factual/run.py
│   └── fixtures/             # evidence bundles with known-correct answers
├── coding-replay/
│   └── tasks.json            # in-repo runner: harness/coding-replay/run.py
└── shootout/
    └── candidates.toml       # agent-brain slate: model ids, measured weights,
                              # fit budget, and why each rejection was rejected
```

Three files here break the "(tool, suite) runbook" rule, deliberately:
`promptstack/probes/`, `factual/fixtures/`, and `coding-replay/tasks.json` are
**data read by their runners at run time**, not documentation — freezing them
beside their config is what makes a score reproducible across runs. Each
`coding-replay` task pins a real merged PR (repo, PR number, base commit,
title/body, changed files, and a named repo check) so a replay always starts
from the same base and scores against the same file set; the repo-to-local-
clone-path map a run needs is environment-specific and passed via
`--clone-map-json`, never committed here. `shootout/candidates.toml` is a
slate, not a suite; it records what will be run and what was excluded, so the
next sweep does not re-litigate the same rejections.

## Where the run command lives (single source of truth)

The canonical way to run these suites is the thin `uvx` wrappers in the serving
stack (nix-ai `modules/mlx/packages.nix`), **not** a script in this repo:

- `mlx-eval <tasks…>` — lm-eval against the live vllm-mlx server. It owns the
  connection args: `base_url`, `max_length=32768`, `num_concurrent`
  (`MLX_EVAL_CONCURRENT`, **default 1** — the wrapper sets `:-1` because
  production serving is intentionally serial while upstream concurrency is
  qualified; the coding suite raises it explicitly), `--apply_chat_template`.
  Do **not** re-specify those as authoritative here — the `[model_args]` blocks
  below mirror them only so the runbook reads standalone.
- `mlx-bench` / `mlx-bench-raw` — vllm-mlx / raw `mlx_lm.benchmark` throughput.
  **These are not interchangeable, and `mlx-bench` is conditional.** The serving
  stack gates it (and `mlx-bench-engine`) behind `vllm-mlx` being an enabled
  backend, commented there as "preserved for future requalification; absent from
  deployed hosts while the backend is disabled". Where that backend is disabled,
  `mlx-bench` is deliberately not installed — it is not a packaging oversight to
  report.

  The consequence for this repo is structural rather than incidental: the
  running-server throughput path in the RUNBOOK depends on a backend a host may
  have retired, while the load path (`mlx-bench-raw`, trap 4 — server must be
  down) does not. On such a host there is no non-destructive throughput route,
  so a throughput row cannot be produced without either requalifying that
  backend or adding a path that measures the serving engine actually in use.
  Confirm which backend a host runs before planning a throughput suite.
- `mlx-wait` — health-gate the server before a run.

This repo owns only the step *after* a run: convert the tool's JSON to envelope
v1 and publish it (`mlx-bench-publish`). See the top-level
[README](../README.md) → "Run + publish a benchmark".

## qwen3-tasks overlay (the coding default)

`configs/lm-eval/qwen3-tasks/` provides `humaneval`/`mbpp` variants whose
custom filter strips `<think>…</think>` content AND extracts the fenced
Python code block from a chat-style answer. This overlay is the **default**
for the coding suite (see `coding.toml`), not an optional extra: chat-served
Instruct models answer in prose + markdown, and the plain `humaneval`/`mbpp`
extractors grab only the echoed prompt — measured 2026-07-08, a 30B Instruct
model scored humaneval pass@1 = 0.0 / mbpp 0.128 under the plain tasks purely
as an extraction artifact. Reserve the plain tasks for completion-style
(non-chat) endpoints that emit bare code.

## TOML shape

Keep configs declarative and tool-native. The runner injects per-invocation
values (`model`, output paths); the converter maps the tool's JSON to
[`schema.json`](../schema.json).

## Local vs cloud execution

**Default: local models only** — they share the vllm-mlx backend via llama-swap
and run sequentially (one model resident at a time on the MacBook; the Studio
keeps a resident pair). Cloud comparison models go through the Bifrost gateway
(`http://localhost:30080/v1/chat/completions`) and only belong in a sweep when
`cloud`/`full` is explicitly requested. Always verify model names against the
live catalog first:
`curl -s http://localhost:30080/v1/models | grep -o '"id":"[^"]*"'`.

## Adding a new config

1. Identify which upstream tool covers the measurement.
2. Add a TOML under the matching subdirectory; keep options tool-native — a
   wrapper shim is a signal the wrong tool is being used.
3. Smoke it against one model, confirm the envelope validates against
   `schema.json`, publish the Parquet to the HF dataset.
4. Open a PR adding a row to the README upstream-tools table if new.
