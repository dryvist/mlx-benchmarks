# Model rankings

A single-pager ranking every model benchmarked in this repo, seeded from the
[`JacobPEvans/mlx-benchmarks`](https://huggingface.co/datasets/JacobPEvans/mlx-benchmarks)
HF dataset plus the per-run notes in [`docs/journal/`](docs/journal/). It is a
**snapshot**, not a live query — regenerate it with the loop in
[Keeping this page current](#keeping-this-page-current) after every publish.

A model is only "fully benchmarked" once it has the required suites filled —
**throughput**, **coding**, **math-hard**, **reasoning**, **agentic
tool-calling** — in **both** environment classes. See
[`docs/RUNBOOK.md`](docs/RUNBOOK.md) for the procedure that produces each column.

> **Every verdict below is PROVISIONAL.** Per the
> [verdict policy](docs/verdict-policy.md), no model is permanently dismissed or
> crowned "best" until it has **≥4 runs ≥5 days apart**, each a **validated
> consecutive pair**, in **both the isolated and under-load environment
> classes**. The Maturity column counts **protocol-valid runs** (a validated
> pair in one env class) toward the 4 needed. No historical shard was collected
> under that protocol, so **every model currently sits at `1/4`** — one
> pre-protocol run of four.

## How to read the columns

- **Size GB** — approximate resident weight footprint of the quant, not the
  file size. Nominal; the capacity math in the RUNBOOK is what gates fit.
- **Throughput tok/s** — batched tok/s from the `throughput` suite (vllm
  `benchmark_serving`) at listed concurrency; not comparable to agentic
  tok/s. Headline is now cumulative, not decode-only (see
  [docs/schema.md](docs/schema.md)); cells here predate that policy.
- **math_verify** — `minerva_math500`, the `math_verify` metric (read this, not
  `exact_match`, which is prose-depressed on chat-served models).
- **Agentic** — `valid_tool_call_rate` at the pass gate cell
  (concurrency 4, thinking ON, large context), then the multi-turn
  `first_degraded_round` with **thinking ON** (`clean` = ran all 20 rounds).
  Multi-turn degradation, not single-shot validity, is the decisive signal.
- **Maturity** — `N/4`: [protocol-valid runs](docs/verdict-policy.md)
  (validated pair, one env class, ≥5 days apart). All rows provisional.
- **Role** — the provisional verdict ("leads/lags as of N runs"): what this model
  is good for *this cycle*, not a permanent judgment.

All numbers below are **isolated-class** (or single-run legacy) except the
footnote-³ smokes (**under-load**); final verdicts need both classes.

## Agent-brain leaderboard (tool-calling, `jevans-ms`, 2026-07-08)

The decisive comparison: eight candidates for the resident tool-calling brain,
all run through the identical 22-tool agentic grid on the Studio. Ranked by
agentic fitness, then throughput. Single-stream tok/s is the agentic
`conc1-large` effective rate from the selection run.

| Prov. rank | Model | Size GB | Maturity | Agentic valid% (conc4/on/large) | Degraded round (thinking ON) | conc1-large tok/s | Role (as of N runs) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **Qwen3.6-35B-A3B-OptiQ-4bit** | ~19.5 | 1/4 | 100% | **clean (20/20)** | 7.4 | Leads this cycle — resident brain; thinking ON + rep-penalty guardrail |
| 2 | Qwen3-Next-80B-A3B-Thinking-4bit | ~45 | 1/4 | 100% | round 17 | 12.0 | Runner-up; long-transcript pick, higher tok/s but heavier |
| 3 | Qwen3.6-35B-A3B-4bit (stock) | ~19.5 | 1/4 | 100% | clean (20/20) | 4.1 | Clean but ~half the leader's speed |
| 4 | Qwen3.6-35B-A3B-MLX-8bit (lmstudio) | ~35 | 1/4 | 100% | round 19 | 7.2 | Near-clean; 8-bit weight cost for one late slip |
| 5 | Qwen3.6-35B-A3B-8bit (mlx) | ~35 | 1/4 | 100% | round 6 | 7.2 | Degrades early despite 8-bit — quant recipe beats bit width |
| 6 | GLM-4.7-Flash-4bit | ~18 | 1/4 | 100% single-shot | round 1 (tool-dead) | 15.1 | Fastest here; lags as a brain this cycle |
| 7 | Qwen3-Coder-30B-A3B-Instruct-4bit | ~17 | 1/4 | 0% / 67% | round 1 | — | Coding sidecar this cycle; malformed calls under agentic load |
| 8 | gpt-oss-120b-MXFP4-Q8 | ~63 | 1/4 | 0% | round 1 | 2.0 | Lags as a tool-calling brain this cycle |

Date count alone matures nothing: both stay `1/4`.

**Production addendum (winner):** OptiQ-4bit must be served with thinking ON
and a repetition-penalty guardrail (`repetition_penalty ~1.05`, `temp 0.6–0.7`).
With production defaults (`temperature=None`, `repetition_penalty=None`) the
4-bit quant degenerates into repetition loops (same sentence 100+ times, ~37
duplicate tool calls/turn) even though the bench cell passed — see the
[sampling-parity trap](docs/benchmark-traps.md#trap-6-sampling-parity) and the
[2026-07-08 journal](docs/journal/2026-07-08-agentic-brain-selection.md). This
isolated-vs-under-load gap is the worked example behind
[verdict-policy Gate 3](docs/verdict-policy.md#gate-3--both-environment-classes):
the isolated pass and the under-load failure are both required before a verdict.

**80B addendum (2026-07-13):** the 80B's round-17 slip is token-budget
truncation (`finish_reason: length`), not tool-format. At an 8192 budget
it runs 20/20 with `repetition_penalty 1.05` (18/20 without), a guardrail its
deep-brain alias now carries (JacobPEvans/ansible-proxmox-apps#891). Single unreplicated
runs — directional only; maturity stays 1/4.

## Full catalog

Every model with at least one published metric. Blank = not yet run for that
suite. Throughput is the best published output tok/s (concurrency in
parentheses). `math_verify` is `minerva_math500`. Agentic is the pass-gate
`valid%` / `first_degraded_round` (thinking ON) where a `tool-calling` sweep
exists.

| Model | Size GB | Maturity | Throughput tok/s | math_verify | Agentic (valid% / deg round) | Role (as of N runs) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.6-35B-A3B-OptiQ-4bit | ~19.5 | 1/4 | | | 100% / clean | Leads as agent brain this cycle |
| Qwen3-Next-80B-A3B-Thinking-4bit | ~45 | 1/4 | 25.1 | 0.08 | 100% / r17 | Agent brain runner-up; weak on math_verify |
| Qwen3-Next-80B-A3B-Instruct-4bit | ~45 | 1/4 | 28.2 | 0.34 | | Strong all-rounder MoE |
| Qwen3.6-35B-A3B-4bit | ~19.5 | 1/4 | 24.9 / 28.4 (c2) † | | 100% / clean | Clean agent brain, slower |
| Qwen3.6-35B-A3B-8bit | ~35 | 1/4 | | | 100% / r6 | Early multi-turn degrade |
| Qwen3.6-35B-A3B-MLX-8bit (lmstudio) | ~35 | 1/4 | | | 100% / r19 | Near-clean agent brain |
| Qwen3-Coder-30B-A3B-Instruct-4bit | ~17 | 1/4 | 136.7 (c4) | 0.47 | 0–67% / r1 | Coding sidecar; throughput + math leader this cycle |
| Qwen3-Coder-30B-A3B-Instruct-8bit | ~32 | 1/4 | 41.2 | 0.37 | | Coding sidecar, 8-bit |
| gpt-oss-120b-MXFP4-Q8 | ~63 | 1/4 | 44.4 (c4) | | 0% / r1 | High-throughput generalist; lags as a tool brain |
| gpt-oss-120b-4bit | ~63 | 1/4 | 44.9 | 0.42 | | Generalist; strong math_verify |
| GLM-4.7-Flash-4bit | ~18 | 1/4 | | | 100%¹ / r1 | Fast, tool-dead multi-turn |
| Devstral-2-123B-Instruct-2512-4bit | ~63 | 1/4 | 2.5 | 0.42 | | Large coder; very slow decode |
| Devstral-Small-2-24B-Instruct-2512-4bit | ~13 | 1/4 | | 0.37 | | Small coder |
| Qwen3.5-122B-A10B-4bit | ~63 | 1/4 | 24.6 | 0.08 | | Legacy flagship MoE |
| Qwen3.5-35B-A3B-4bit | ~19.5 | 1/4 | 32.9 | | | Legacy A3B workhorse |
| Qwen3.5-27B-4bit | ~15 | 1/4 | 22.9 | | | Legacy dense mid |
| Qwen3.5-9B-MLX-4bit | ~5 | 1/4 | 68.5 | | 100%³ | Small, fast |
| DeepSeek-R1-0528-Qwen3-8B-4bit | ~5 | 1/4 | 58.7 | | | Small reasoning distill |
| Qwen3-4B-Instruct-2507-4bit | ~2.5 | 1/4 | | | 80%³ | Resident `judge` alias on jevans-mbp; smoke only |
| Seed-OSS-36B-Instruct-4bit | ~19 | 1/4 | 18.6 | | | Mid generalist |
| gemma-4-31b-it-4bit | ~17 | 1/4 | 18.4 | | | Dense generalist |
| gemma-4-e4b-it-4bit | ~3 | 1/4 | 59.9 | | | Tiny, fast |
| GLM-4.5-Air-4bit | ~60 | 1/4 | | 0.08 | | Legacy MoE |
| Qwopus3.5-122B-A10B-…-abliterated-4bit | ~69 | 1/4 | 52.8² | | 1.0 (c1) / OOM (c4) | Fast single-stream; OOMs conc4 + **abliterated** — not adopted |
| Hermes-4-70B-MLX-4bit | ~37 | 1/4 | 11.8² | | 0.875 (c1) / OOM | Dense 70B; needs thinking, OOMs on concurrency AND long history — not a viable brain here |

¹ GLM-4.7-Flash passes the single-shot pass-gate cell (100% valid) but the
multi-turn track collapses at round 1 — the exact case the degradation track
exists to expose. Single-shot validity alone is not a passing agentic verdict.

² Single-stream (concurrency 1) agentic decode rate, **not** the batched
`throughput` suite — the two are not comparable. These two rows come from the
2026-07-09 flagship isolated-window session
([journal](docs/journal/2026-07-09-flagship-isolated-window.md)).

³ jevans-mbp quick smoke, 2026-08-23 (under-load, conc 1): single unreplicated
runs, reduced matrix — not pass-gate comparable.
[Journal](docs/journal/2026-08-23-jevans-mbp-quick-smokes.md).

† 2026-09-09, ISOLATED class, `dedicated=true`, on a private loopback endpoint
nothing routes to. Aggregate decode rate at concurrency 1 and 2, three runs each,
256 max output tokens over a 2048-token prompt. Both cells pair-validated: the
widest gap between consecutive runs is 0.65% at concurrency 1 and 0.39% at
concurrency 2, and concurrency 2 is +14.1% on aggregate cumulative throughput
with zero errors.

The clean room is what makes these comparable, and it was proven rather than
assumed: the gate logged 12 × 502 and zero 200s across the measurement interval,
so it was demonstrably unable to reach any backend while the runs were in flight.
The same grid on a contended host swung 60.8% and 35.5% between consecutive runs —
so an un-quiesced throughput figure measures the contention, not the model.

Do not compare these against the `²` rows: those are single-stream agentic decode,
a different metric. Also note that the suite's cumulative tok/s counts prompt plus
completion over wall time against a large prompt cache, so it partly tracks cache
hit rate; the decode figure above is the model-speed one.

Cloud baselines: see the [quick-smokes journal](docs/journal/2026-08-23-jevans-mbp-quick-smokes.md).

## Flagship investigation (2026-07-09) — the 50–90 GB tier does not fit here

An 8-hour isolated-window sweep for a 50–90 GB "flagship" brain to maximize the
128 GB Studio reached a firm negative: **no available 60–70 GB model is a viable
brain for the concurrent, long-history agentic workload on this hardware.** The
two independent walls, both measured:

- **Weights vs concurrent KV cache.** A ~70 GB-weight model + four concurrent
  20K-token KV caches exceeds the ~92 GB Metal allocation limit at
  `gpu-memory-utilization 0.80` — Qwopus-122B-A10B-4bit OOMs the instant the
  grid hits concurrency 4.
- **Dense = slow + KV-heavy.** The only 70B that leaves weight headroom
  (Hermes-4-70B-4bit, dense, 37 GB) decodes at ~12 tok/s and its per-token KV
  cache is so large it OOMs at concurrency 4 (peak 102.8 GB) *and* at
  concurrency 1 once a 20-round history accumulates.

The 128 GB Studio serving the concurrent fleet is structurally best matched by a
**~20–45 GB MoE** (low active params for speed, small weights for KV headroom) —
which the resident `Qwen3.6-35B-A3B-OptiQ-4bit` already is. A 70 GB flagship only
pays off for a **dedicated low-concurrency Hermes endpoint** (not shared with the
cron fleet) or once **RDMA MacBook→Studio** adds memory. The non-abliterated
122B-A10B community quants (`OptiQ-2bit`, `Text-mxfp4`) additionally **fail to
load** in vllm-mlx 0.4.0 — they ship `vision_tower` weights the strict loader
rejects; only the abliterated Qwopus repackaging loads.

## Keeping this page current

This page is a snapshot of the dataset. After you publish a new shard
(`mlx-bench-publish …`), refresh the affected row here in the same PR. The
publish→edit loop:

1. **Publish** the run (see [`docs/RUNBOOK.md`](docs/RUNBOOK.md) → "Publish").
2. **Pull the new numbers back** from the dataset so the page reflects what was
   actually stored, not what you think you ran:

   ```sh
   .venv/bin/python - <<'PY'
   from huggingface_hub import HfApi, hf_hub_download
   import pyarrow.parquet as pq
   api = HfApi()
   files = [f for f in api.list_repo_files("JacobPEvans/mlx-benchmarks",
                                           repo_type="dataset") if f.endswith(".parquet")]
   rows = []
   for f in files:
       rows += pq.read_table(hf_hub_download("JacobPEvans/mlx-benchmarks", f,
                                             repo_type="dataset")).to_pylist()
   # filter rows to your model/suite and read metric/value/tag_* columns
   PY
   ```

3. **Edit the table** row: fill the suite column, bump the **Maturity** count if
   this is a new date ≥5 days from the last (and a validated pair — a divergent
   pair is discarded, not counted), and re-word the provisional verdict.
4. **Commit** in the same PR as the publish, so the ranking never drifts from
   the dataset.

A provisional verdict only becomes **final** once the model clears all three
gates of the [verdict policy](docs/verdict-policy.md) — ≥4 runs ≥5 days apart,
each a validated consecutive pair, in **both** the isolated and under-load
environment classes. Until then keep the "leads/lags as of N runs" wording; a
verdict gates this cycle's actions, not a permanent judgment.
