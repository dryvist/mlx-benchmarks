# Agent-brain shootout — procedure

Which single model, running **alone** on the Studio, is the most accurate brain
for an autonomous agent that makes MCP tool calls, reads the results, and writes
factual digests? This composes the [`agentic`](agentic.md) and `factual` suites
and ranks them with `mlx-bench-shootout`. Slate, measured weights, rejections:
[`../configs/shootout/candidates.toml`](../configs/shootout/candidates.toml).

> Output is **PROVISIONAL** — one sweep cannot mature a verdict
> ([`verdict-policy.md`](verdict-policy.md)). It gates *this cycle's* choice,
> and the "leads/lags as of N runs" wording rule applies to every artifact.

## The three criteria, and how each is scored

Priority order is fixed: a model that cannot call tools reliably is not a brain
no matter how fast or how honest it is.

| # | Criterion | Metric | Where it comes from |
| --- | --- | --- | --- |
| 1 | Tool-call fidelity | `valid_tool_call_rate` at the gate cell, then multi-turn survival | `agentic` suite |
| 2 | Factual accuracy | `grounded_accuracy`, then `fabricated_number_rate` | `factual` suite |
| 3 | Latency | `request_latency_p50_ms` at the gate cell (TTFT reported alongside) | `agentic` suite |

Every number is computed by code from the raw results. No criterion needs a
human to read a transcript, and none uses a model as a judge.

### 1. Tool-call fidelity — programmatic pass/fail per call

The agentic runner sends a **22-tool registry** of realistic MCP-shaped schemas
(Splunk, filesystem, shell, memory, wiki, Slack, cron, web fetch) plus
near-duplicate distractors that force a choice between similar names. Validity
rules and failure taxonomy: [`agentic.md`](agentic.md).

One consequence answers a question the shootout raises: a model that emits
`[Tool call: ...]` as prose produces no `tool_calls` and scores `no_tool_call`.
**Leaking raw tool syntax into prose is therefore already a scored failure**,
and the 20-round
multi-turn track is what exposes it — stock 4-bit quants degrade into that
fallback around round 5 while looking perfect on single-shot.

Multi-turn survival is the share of rounds completed before the first
degradation, taken as the **worst** of the thinking-on and thinking-off tracks:
a brain that only holds together in one thinking mode is a brain with a footgun.

### 2. Factual accuracy — fixtures with a known answer

The `factual` runner ([`../harness/factual/run.py`](../harness/factual/run.py))
hands the model a tool result whose contents are known and asks for the digest a
human would read. Bank:
[`../configs/factual/fixtures/homelab-digest.json`](../configs/factual/fixtures/homelab-digest.json).
Four deterministic checks per response:

- **fact recall** — every `required_facts` entry appears (case-insensitive, or
  as a normalized number, so `1,284` satisfies `1284`).
- **fabricated numbers** — the metric the suite exists for. Every numeric token
  in the response must appear in the evidence, in the prompt, or in the case's
  declared `allowed_derived` list (a correct total, a row count). **Any residue
  is a fabrication.** Extraction is punctuation-blind and runs identically over
  evidence and response, so reformatting (`July 23, 2026` for an ISO timestamp)
  never reads as invention.
- **forbidden claims** — the plausible-but-wrong answer for that case:
  transposed digits, an inverted status, a retention figure the evidence lacks.
- **tool-syntax leak** — raw call syntax in prose that should carry none.

A response passes only on all four; `grounded_accuracy` is the share that does.
One case (`digest-004-absent-field`) has **no correct value to state** — a model
that invents one fails, one that says so passes. That abstention is what a
digest-writing agent needs.

### 3. Latency

Gate-cell `request_latency_p50_ms` and `first_token_p50_ms` from the agentic
suite, measured at concurrency 1 — the shootout is a single-user question.

### How the ranking combines them

`mlx-bench-shootout` sorts **lexicographically by the priority above**, never by
a weighted sum (which would smuggle in importance numbers nobody agreed). Rates
quantize to **0.10** first — the verdict policy's own Gate 2 divergence
threshold for a bounded-[0,1] metric — so two rates closer than that tie and the
next criterion decides. Latency, where a small difference is real and
repeatable, is the final tiebreak. A model missing either suite is not ranked.

## Memory ceiling

nix-darwin's `mac-studio` config sets `maxLocalLlmGb = 100` — wired limit
102400 MiB (100 GiB), a 28 GiB OS reserve, and a 99 GiB cap beneath it. The
binding constraint is tighter: the 2026-07-09 session measured the Metal
allocation wall at **~92 GB** at `--gpu-memory-utilization 0.80`.

The slate caps **weights at 75 GB**. The largest candidate,
`MiniMax-M2-REAP-139B-A10B-mxfp4`, measures **73.9 GB** — **~18 GB** under that
wall for KV, framework, and activation scratch. A 64K KV cache costs ~1.6 GB on
a `qwen3_next`-class hybrid and several times that on a dense 70B, so the
headroom is sized for the dense worst case. Every candidate fits alone.

Two things this budget is **not**:

- It is not the deployed budget. The Studio runs **single-model mode** — one
  resident plus a small always-available swap tier — so the shootout's
  one-model-alone assumption matches how it serves today.
- It is not the concurrency-4 budget behind the 2026-07-09 negative result on
  the 50–90 GB tier. That wall was weights + **four** concurrent 20K KV caches.
  Every 40B+ model here is served single-slot, so the shootout runs at
  concurrency 1 and a 70 GB model is not disqualified by it.

## Running the sweep

### Before you start

1. **Take a maintenance window.** This takes production serving offline.
   Notify first — [`RUNBOOK.md` Step 3 Option B](RUNBOOK.md#step-3--serve-the-model)
   is the canonical bootout sequence, including the watcher agents that would
   otherwise relaunch serving mid-window.
2. **One actor per host, and never during a cluster drill.** Confirm no other
   bench is in flight. The two-Mac cluster drill is a separate operator
   activity on this same hardware. Its link watcher **quiesces normal serving**
   (sweeping away agents, model server included) and **re-pins
   `iogpu.wired_limit_mb`** while a rank is serving — moving the very ceiling
   every fit decision below derives from. The two must never overlap, in either
   direction.
3. **Pre-fetch weights** into `/Volumes/HuggingFace` (`HF_HOME`) *before* the
   window — `hf download <id>` per candidate. ~500 GB downloaded inside the
   window is window wasted.
4. **Check served names against `/v1/models`** once serving is up, never a
   filename.

### Per candidate

Serve one model at a time — the memory budget assumes exactly one resident.

```sh
MODEL=<id from configs/shootout/candidates.toml>
SLUG=$(basename "$MODEL")

vllm-mlx serve "$MODEL" \
  --port 11434 \
  --tool-call-parser <parser> \
  --reasoning-parser <parser> \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 1 \
  --timeout 3600

# 1. Agentic — criteria 1 and 3. conc1 only: this is a single-user question.
uv run harness/agentic/run.py \
  --base-url http://127.0.0.1:11434/v1 --api-key-env OPENAI_API_KEY \
  --model "$MODEL" --concurrency 1 \
  --output "run-output/agentic_${SLUG}.json"

# 2. Factual — criterion 2.
uv run harness/factual/run.py \
  --base-url http://127.0.0.1:11434/v1 --api-key-env OPENAI_API_KEY \
  --model "$MODEL" \
  --output "run-output/factual_${SLUG}.json"
```

Parser flags: [parser map](benchmark-traps.md#parser-map). Two bite on this
slate. **gpt-oss needs `--reasoning-parser gpt_oss`** or harmony channel markers
leak into `message.content`, plus `--thinking-kwarg reasoning_effort` on both
runners. **`qwen3_next` models need the paged KV cache off** — paged-block
reconstruction fails on every multi-turn request.

Then rank:

```sh
mlx-bench-shootout run-output/ --gate conc1_think-on_ctx-large
```

### Wall-clock budget

Per candidate, at concurrency 1:

| Step | Time |
| --- | --- |
| Load + warm | 2–10 min (a 70 GB model from a cold cache is the slow end) |
| Agentic, conc1 only (8 cells × 10 repeats + 2×20 multi-turn rounds) | ~60–90 min |
| Factual (5 cases × 5 repeats × 2 thinking modes) | ~10–20 min |
| **Subtotal** | **~1.5–2 h** |

For 11 candidates: **~17–22 h of serving time per pass.** The verdict policy
requires a **validated consecutive pair** (each suite run twice, both discarded
if they diverge), so a protocol-valid sweep is **~35–45 h** — one run of the
four needed before any verdict stops being provisional.

Two ways to cut that without breaking the protocol:

- **Screen first, then replicate.** Run one unreplicated pass over all 11 to
  find the top ~4, then run validated pairs on those only. The screening pass is
  directional and must not be published or scored as a verdict.
- **Smoke before committing.** `--cells conc1_think-on_ctx-large_stream
  --repeats 3` (agentic) and `--thinking off --repeats 1` (factual) take minutes
  per model and catch a wrong parser, a bad thinking kwarg, or a model that will
  not load — before you spend two hours on it.

### Afterwards

1. **Restore production first**, before any analysis. The sweep never edits nix
   config, so restore is a restart, not a reconfigure — the posture comes back
   from what nix already declares. Kill the solo server, then bring the agents
   back:

   ```sh
   pkill -f 'vllm-mlx serve' || true          # the sweep's solo server
   for a in vllm-mlx.server vllm-mlx.warmup \
            mlx-night.watcher mlx-night.rank mlx-night.prefetch; do
     launchctl bootstrap gui/501 ~/Library/LaunchAgents/dev.$a.plist || true
   done

   # The observable: a real completion from the restored resident.
   curl -s4 http://127.0.0.1:11434/v1/chat/completions \
     -H 'content-type: application/json' -d '{
       "model":"mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
       "messages":[{"role":"user","content":"reply with the word ready"}],
       "max_tokens":8}' | jq -r '.choices[0].message.content'
   ```

   That must return generated text. A process being up is not the check — the
   Hermes agent depends on this brain continuously, so inference answering is
   the only thing that proves the window is closed. Also confirm `/v1/models`
   lists exactly two ids (single-model mode): the resident Coder-30B and the
   `Qwen3.5-9B-MLX-4bit` swap tier. A candidate still listed, a third id, or a
   non-answering endpoint means it is not. If config was somehow touched,
   `darwin-rebuild switch` restores it from nix; never hand-edit to recover.

   **The incumbent is restored whether or not a challenger won.** Adoption is a
   separate, deliberate change, never a side effect of benchmarking.

2. **Publish every run.** Never discard completed benchmark time — publish with
   `--tag caveat=<reason>` and file an issue instead.

   ```sh
   HF_TOKEN="$HF_WRITE_TOKEN" \
     .venv/bin/mlx-bench-publish run-output/factual_<slug>.json \
     --kind factual --suite grounded-summary --hostname jevans-ms
   ```

3. **Update [`../RANKINGS.md`](../RANKINGS.md)** in the same PR as the publish,
   keeping the maturity count and the provisional wording honest.

## Adopting a winner

A shootout result is an input to a decision, not the decision. The Studio is in
single-model mode, so adoption means repointing `programs.mlx.singleModel` and
the role set in nix-darwin's `lib/hosts/mac-studio.nix`, after the model's serve
args are validated into nix-ai's `modules/mlx/catalog-data.nix`. Physical model
ids belong in that catalog and in this repo's candidate list — **never in a nix
module option or option example**, which a CI check rejects.

Two traps this fabric has already hit: **sampling parity** — the 2026-07-08
winner passed every cell and then degenerated into repetition loops under
production defaults, so bench numbers are only valid at the sampling settings
that produced them; and **environment class** — an isolated result says nothing
about behavior under concurrent load, and the verdict policy requires both.
