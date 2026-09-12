# Model catalog and flagship investigation

Moved out of [RANKINGS.md](../RANKINGS.md) to keep that page under the
repository file-size limit. Same data, same update rules.

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
| NVIDIA-Nemotron-3-Super-120B-A12B-4bit | ~63 | 1/4 | 225.3 (c1) ‡ | | | Fastest cumulative rate; will not serve c2 |
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

‡ 2026-09-09/10, ISOLATED class, `dedicated=true`, on a private loopback
endpoint nothing routes to, host `jevans-ms`. Cell is the mean aggregate
cumulative rate at
**concurrency 1 only**: 224.24 and 226.26 tok/s, 0.90% apart, pair-validated.
Sequential per-request decode was 23.03-23.40 tok/s and TTFT 47.6-48.5s across six
repetitions.

**Concurrency 2 is a recorded loss, not a missing measurement.** Two of three runs
failed with swap engaged against 63 GB of weights; the surviving run (162.77
aggregate) is unpaired and cannot be validated. The wired-memory ceiling was not
raised to chase it — it is fixed by standing ruling — so the honest finding is that
this model does not serve two concurrent requests on this host at that ceiling.

Read the headline against its own definition: cumulative tok/s counts prompt plus
completion, so a long prefill inflates it relative to a decode-only figure. This
model's *decode* is unremarkable; its prefill is what puts it at the top of the
column. The published rows keep the two separable —
`throughput_aggregate_output_toks_per_s` is 3.33 at this cell while
`throughput_total_toks_per_s` is 225.75, and conflating them is easy and wrong.

Two runs from this campaign are retained but excluded: a duplicate benchmark
driver was started against the same model, so both processes recorded
`dedicated=true` while sharing one GPU and per-request decode fell about 7×. They
carry `annotation_env_class=contended` and are not part of any cell above.
`dedicated` is a claim the runner cannot verify — treat it as an assertion about
method, not a measurement.

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
