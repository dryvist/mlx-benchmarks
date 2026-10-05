# configs/ layout

One TOML file per `(upstream-tool, suite)` pair. Most files are documentation
runbooks that record the task list and tool-native options. The four
`cross-card` / `quick-intelligence` campaign files below are structured recipes
for the external Ansible consumer. They are not executable in this repository:
the TOML reader/dispatcher lives separately and uses `community.general.from_toml`
to parse a selected allow-listed file, validate survey values against inventory
and the model registry, then dispatch its argv entries.

## Layout (as shipped)

```text
configs/
├── LAYOUT.md                 # this file
├── lm-eval/
│   ├── reasoning.toml        # arc_challenge_chat (quick) / gsm8k (canonical)
│   ├── coding.toml           # humaneval, mbpp
│   ├── math-hard.toml        # minerva_math500
│   ├── quick-intelligence.toml # registry-selected arc_challenge_chat quick pass
│   └── qwen3-tasks/          # optional <think>-stripping overlay (see below)
├── vllm/
│   ├── benchmark_serving.toml # vllm throughput cross-check; no local install
│   └── cross-card.toml       # declarative vllm bench serve matrix
├── agentic/
│   └── tool-calling.toml     # in-repo runner: harness/agentic/run.py
├── llama-cpp/
│   ├── throughput.toml       # historical OpenAI-compatible serving recipe;
│                             # draft-model A/B sweep for speculative decoding
│   └── cross-card.toml       # llama-bench and batched-bench diagnostics + probe envelope
├── mlx/
│   └── cross-card.toml       # mlx_lm.benchmark diagnostics + throughput-probe envelope
├── nvidia/                   # NVIDIA/CUDA campaign; see "NVIDIA hosts" below
│   ├── throughput.toml       # `vllm bench serve`, 1/4/8 concurrent x 8k/64k/128k
│   │                         # prompts -> existing --kind vllm converter
│   ├── quality.toml          # lm-eval, non-saturated tasks only -> --kind lm-eval
│   ├── gpu-burn.toml         # sustained compute baseline -> --kind gpu-burn
│   ├── nvbandwidth.toml      # GPU copy bandwidth baseline -> --kind nvbandwidth
│   ├── mbw.toml              # host RAM bandwidth baseline -> --kind mbw
│   └── fio.toml              # storage baseline -> --kind fio
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

The existing MLX suites use the thin `uvx` wrappers in the serving stack
(`modules/mlx/packages.nix`), **not** a script in this repo:

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

The parameterized campaign TOMLs are inputs for a separate Ansible consumer;
its inventory, endpoint, cache, and power contract is documented in
[`docs/benchmark-campaign-target-inputs.md`](../docs/benchmark-campaign-target-inputs.md).

The repo owns conversion after a supported run: `vllm bench serve` output uses
the existing `--kind vllm` converter; the OpenAI-compatible
`harness/throughput/run.py` output uses `--kind throughput-probe`; and lm-eval
JSON uses `--kind lm-eval`. `llama-bench -o json` and
`mlx_lm.benchmark` produce raw stdout diagnostics only; no existing converter
accepts either format. The campaign recipes do not claim those native outputs
as envelope data. The llama.cpp native rows use the source procedure's 512 and
8192 prompt sizes; the 32768-token depth row declares a `requires_capability`
check for `llama-bench -d 32768`. Before that row dispatches, the external
consumer must prove the selected target's `llama-bench` supports `-d` and the
authoritative model/target context window is at least 41088 tokens (32768
depth + 8192 prompt + 128 output); otherwise it marks the row N/A or rejects
it without dispatch. The batched diagnostic is limited to SMALL/MEDIUM models
by the external consumer. See the top-level
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

Keep configs declarative and tool-native. Campaign entries use `[[run]]` tables
with `executable`, `argv`, `dimensions`, `repetitions`, and the symbolic
`artifact_id = "selected_model"`. The consumer supplies the selected registry
artifact, endpoint, repetition number, and output paths. `run_on` is either
`benchmark_target` (native tools and the vLLM client) or `controller` (tools
that require the pinned benchmark checkout). Converter metadata also names its
controller execution location. Argv contains no shell commands. Only runs with
an existing compatible converter declare converter metadata. Raw stdout
diagnostics set `capture_stdout = true` and omit converter metadata; the
consumer captures and fetches them as supplemental files.

## NVIDIA hosts (`configs/nvidia/`)

Runbooks for a vLLM serving host with an NVIDIA GPU. They follow the llama-cpp
pattern: this repo never starts the server, the runbook names the exact upstream
command, and a converter turns that tool's output into an envelope. Only the four
hardware baselines needed new converters (each under 50 lines of Python); throughput
and quality reuse `--kind vllm` and `--kind lm-eval`.

The system block (`system.gpu`, `system.engine`, `system.power_limit_w`,
`system.container`) is **declared, not probed**: `detect_system()` reads it from
`MLX_BENCH_*` environment variables, like cluster topology, because the publisher may
run on a different machine than the GPU. Export these on the GPU host and publish from
it, so `os` / `chip` / `memory_gb` describe the same machine:

```sh
export MLX_BENCH_GPU_MODEL="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
export MLX_BENCH_GPU_VRAM_GB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | awk '{printf "%.1f", $1 / 1024}')"
export MLX_BENCH_GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
export MLX_BENCH_GPU_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -n1)"
export MLX_BENCH_POWER_LIMIT_W="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -n1)"
export MLX_BENCH_ENGINE_NAME=vllm
export MLX_BENCH_ENGINE_VERSION="$(vllm --version)"
export MLX_BENCH_CONTAINER="<image:tag>"   # only when the engine ran in a container
```

A variable that is unset, blank, or (for the two numeric ones) not a number is left out
of the envelope rather than recorded wrong.

Conventions the runbooks rely on:

- **Sweep tags.** Throughput rows carry `--tag prompt_tokens=…`, `--tag concurrency=…`
  and `--tag context_len=…` (all strings; `context_len` is the server's configured window,
  never the prompt length). `schema.json` documents them.
- **Hardware baselines have no model.** Publish them with `--model hardware-baseline`.
  `.txt` / `.log` output (mbw, gpu-burn) is passed straight to the CLI, which wraps it as
  `{"output": text}`.
- **Relaxed verdict gate.** NVIDIA-campaign model rows carry `--tag campaign=nvidia
  --tag verdict_gate=relaxed`; see [`docs/verdict-policy.md`](../docs/verdict-policy.md).

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
