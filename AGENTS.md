---
skill-groups: [core, ai]
---
# mlx-benchmarks — AI Agent Documentation

Agent-facing notes for AI coding sessions in this repo. For the
human-facing overview, install instructions, and contribution guide see
[`README.md`](README.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), and
[`docs/architecture.md`](docs/architecture.md).

## Project overview

Benchmark harness for MLX-quantized and locally-hosted LLMs on Apple Silicon.
Orchestration configs and the envelope v1 schema live here; results publish
to the [`JacobPEvans/mlx-benchmarks`](https://huggingface.co/datasets/JacobPEvans/mlx-benchmarks)
HF dataset, visualized at the
[`mlx-benchmarks-viewer`](https://huggingface.co/spaces/JacobPEvans/mlx-benchmarks-viewer)
HF Space.

## Agent skills

Repeatable procedures live as skills under `.claude/skills/` so any agent can
follow them without reverse-engineering the docs. Read the one matching your
task first; each `SKILL.md` is a thin router — it points at the canonical doc
and flags the traps, it does not duplicate them.

| Skill | Use when | Path |
| --- | --- | --- |
| `run-benchmark` | Running any suite or judging if a model counts | [`.claude/skills/run-benchmark/SKILL.md`](.claude/skills/run-benchmark/SKILL.md) |
| `publish-results` | Publishing a shard or updating `RANKINGS.md` | [`.claude/skills/publish-results/SKILL.md`](.claude/skills/publish-results/SKILL.md) |
| `agentic-suite` | Driving the agentic tool-calling benchmark specifically | [`.claude/skills/agentic-suite/SKILL.md`](.claude/skills/agentic-suite/SKILL.md) |

## Repository shape (short)

```text
src/mlx_benchmarks/    Python package (envelope, publish, converters, CLI)
scripts/               validate_schema.py (schema + TOML config validator)
configs/               TOML runbook per (tool, suite) pair — see configs/LAYOUT.md
schema.json            Envelope v1 authoritative contract
examples/              Canonical valid + invalid envelope fixtures
tests/                 Pytest suite with fixtures
space/                 Gradio viewer (deployed to HF Space)
docs/                  architecture.md, schema.md, journal/ (session notes)
.github/workflows/     ci-gate (test + lint + scan + dry-run-publish +
                       schema-validate via paths-filter), release-please
                       (wraps JacobPEvans/.github reusable), deploy-space
                       (CodeQL is via GitHub's default setup, not a workflow)
```

## Key conventions (non-negotiable)

- **Envelope contract**: every published result validates against
  `schema.json` inside `publish()`. Do not bypass with `validate=False` in
  real runs. Breaking changes require a `$id` bump + `schema_version`
  increment.
- **Unique filenames**: `data/run-<ts>-<git_sha>-<suite>-<slug>.parquet`.
  Never overwrite. Use `target_path(envelope)` to compute.
- **System detection**: always use `detect_system()`. Never hardcode system
  metadata.
- **Publisher runs from the venv**: `.venv/bin/mlx-bench-publish` and the
  quality gates run from `.venv/bin/*` (not `uv run` / `uvx`). The evaluation
  tools themselves (lm-eval, vllm `benchmark_serving`) are NOT dependencies of
  this repo — run them via the serving stack's `mlx-eval` / `mlx-bench`
  wrappers; this repo only parses their JSON output.
- **Conventional commits**: `release-please` consumes them. `feat:` minor,
  `fix:` patch. Never manually edit `CHANGELOG.md`.
- **Pre-commit must pass**: `.venv/bin/pre-commit run --all-files`. CI
  re-runs ruff + ruff-format + pyright + pytest. Zero tolerance for `# noqa`
  suppressions — fix the underlying issue.

## Common tasks

```bash
# Quality gates (run all before committing)
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/pyright src/mlx_benchmarks
.venv/bin/pytest tests space/tests
.venv/bin/python scripts/validate_schema.py

# Publish a run (dry-run first!)
.venv/bin/mlx-bench-publish run-output/<...>/results_*.json \
  --kind lm-eval --suite reasoning --dry-run
.venv/bin/mlx-bench-publish run-output/<...>/results_*.json \
  --kind lm-eval --suite reasoning

# Run an lm-eval smoke (lm-eval is external — via the mlx-eval wrapper or your
# own install; this repo only publishes its output). num_concurrent>1 is the
# biggest speedup for long suites.
lm_eval --model local-chat-completions \
  --model_args "base_url=http://localhost:11434/v1/chat/completions,model=$MODEL,num_concurrent=4,max_length=32768,timeout=3600" \
  --tasks gsm8k --limit 10 \
  --apply_chat_template --fewshot_as_multiturn --log_samples \
  --output_path ./run-output
```

## Environment requirements

- macOS on Apple Silicon (inference); CI runs publisher on ubuntu-latest.
- `vllm-mlx` (via `llama-swap`) on `http://localhost:11434/v1`. The
  `base_url` for lm-eval must include the full `/v1/chat/completions`
  path — not just `/v1`.
- Python 3.13+.
- `HF_TOKEN` with write scope on the dataset namespace (for publish) and
  on the space namespace (for deploy).

## Benchmarking a model (the playbook)

The full any-model-any-host procedure is [`docs/RUNBOOK.md`](docs/RUNBOOK.md) —
read it before running anything. The essentials an agent must not get wrong:

- **Required suite set for "fully benchmarked":** `throughput`, `coding`,
  `math-hard`, `reasoning`, `tool-calling`. A model is not complete in
  [`RANKINGS.md`](RANKINGS.md) until all five have a published shard (or one
  suite is decisive enough to disqualify a role).
- **Host constraints:**
  - MacBook `llama-swap` caps at `concurrencyLimit=4` (nix-ai#1190; was 2) →
    **`MLX_EVAL_CONCURRENT` must equal the deployed cap** or lm-eval 0.4.11 crashes (`Session is closed`) on a 429 burst.
  - Studio `jevans-ms` (128 GB, wired ceiling ~118 GB) serves production on
    `127.0.0.1:11434` IPv4 plain HTTP — **always `curl -s4 127.0.0.1`** (caddy
    holds the same port on IPv6/TLS).
  - **One actor per host.** Never edit `llama-swap` config or restart serving
    while a bench is in flight.
  - A **managed window** (solo `vllm-mlx serve`) takes production Hermes
    offline: `launchctl bootout gui/501/dev.vllm-mlx.server` → serve → restore
    with `launchctl bootstrap gui/501 …dev.vllm-mlx.server.plist`. **Notify the
    user before, restore after.**
- **Traps checklist:** coding needs the qwen3 overlay; read `math_verify` not
  `exact_match`; lm-eval needs `--tasks a,b` (positional selects zero); `mlx-bench`
  loads the model directly (server must be DOWN); run both agentic thinking
  tracks; check sampling parity when a bench winner misbehaves live; match the
  serving parser to the family (`hermes` for general Qwen3, not `qwen3_coder`);
  `--timeout 3600` for agent brains; `--gpu-memory-utilization ≤0.85`. Full
  detail: [`docs/benchmark-traps.md`](docs/benchmark-traps.md#traps-checklist).
- **Publish flow + token:** the ambient `HF_TOKEN` is **read-only**. Publishing
  needs the write token, `HF_WRITE_TOKEN`:
  `HF_TOKEN="$HF_WRITE_TOKEN" .venv/bin/mlx-bench-publish <json>
  --kind <lm-eval|agentic|factual|promptstack|vllm> --suite <suite> --hostname <host>`. Dry-run
  first. Dataset: `JacobPEvans/mlx-benchmarks`.
- **Ranking duty:** after every publish, update the model's row in
  [`RANKINGS.md`](RANKINGS.md) in the **same PR**, pulling the numbers back from
  the dataset (loop in that file). The page must never drift from the dataset.

## Verdict policy (hard rules — [docs/verdict-policy.md](docs/verdict-policy.md))

Results drift between runs, so verdicts are earned, not declared. Non-negotiable
for agents:

- **Never write "X is the best/worst model"** anywhere — docs, PR bodies, serving
  comments. Check maturity first and write **"X leads/lags as of N runs"**, naming
  the environment class once past a gate. A dismissal gates *this cycle's* action
  ("not the brain this week"), never a permanent judgment.
- **Never publish or score a single unreplicated run.** Every benchmark is a
  **consecutive pair** (identical config, back-to-back); if the two diverge past
  the threshold (default relative Δ >15% on the primary metric, tunable) **discard
  both halves** — a divergent pair does not count toward maturity.
- **Both environment classes required** for a final verdict: **isolated** (clean
  room, managed window only if it can't co-reside) and **under-load** (production
  live, no window). A big isolated-vs-under-load gap is itself a finding.
- **Maturity gate:** ≥4 validated runs ≥5 days apart in each class before a
  verdict moves from provisional to final. Record host + concurrent load per run.
  The NVIDIA campaign (`configs/nvidia/`) relaxes this gate only, and labels its rows
  `verdict_gate=relaxed` — see [verdict-policy](docs/verdict-policy.md#nvidia-campaign-relaxed-gate-1).

## Gotchas learned the hard way

- **Model names**: verify against the live catalog
  (`curl http://localhost:11434/v1/models`). Don't trust docs.
- **Never discard completed runs**: publish with `tags.caveat=...` and file
  an issue rather than throwing away benchmark time.
- **Coding benchmarks execute model-generated code** — see
  [`SECURITY.md`](SECURITY.md) before running outside a sandbox.
