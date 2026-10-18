# Verdict policy — when a model may be called best or worst

Benchmark numbers on this fabric drift over time — serving-stack state,
concurrent load, quant/toolchain updates, and sampling defaults all shift
results between runs. So **no model may ever be permanently dismissed or crowned
"best" from a single run.** A verdict is earned only after a model clears three
composing gates. Until then every verdict is **PROVISIONAL** and gates *actions*
("don't serve it as the brain this week"), never a permanent judgment.

`RUNBOOK.md`, `RANKINGS.md`, and `AGENTS.md` all defer to this file.

## The three gates

### Gate 1 — temporal maturity (≥4 runs, ≥5 days apart)

A model needs **at least 4 separate benchmark runs with 5 or more days between
each** before its verdict can move from provisional to final. Two runs less than
5 days apart count as one. The gap is the point: it samples across the random
environmental drift that a single run cannot see.

- Re-bench cadence: **≥5 days** between counted runs.
- Each run covers the **same required suite set** (throughput + coding +
  math-hard + reasoning + tool-calling — see [`RUNBOOK.md`](RUNBOOK.md)).
- Record the **environmental context** of every run (host, serving config,
  concurrent load) so variance is explainable rather than mysterious.

### Gate 2 — consecutive replication (validated pairs only)

Every benchmark is run **at least twice, back-to-back, with identical config**,
and the two results are compared **before publishing**:

- **Similar pair → validated.** Publish both (or their mean); the run counts
  toward Gate 1 maturity.
- **Divergent pair → discard entirely.** A large gap between two identical-config
  runs means the environment was unstable, so **both halves are invalid** and the
  run does **not** count toward maturity. Do not cherry-pick the better half.

**Default divergence threshold:** relative Δ > **15%** on the suite's primary
metric between the two runs — or, for a bounded-[0,1] rate metric, absolute
Δ > **0.10**. The threshold is **tunable**; record the value used. Primary
metrics: throughput → output tok/s; coding → pass@1; math-hard → `math_verify`;
reasoning → accuracy; tool-calling → `valid_tool_call_rate` (and
`first_degraded_round`).

A single unreplicated run is never published or scored.

### Gate 3 — both environment classes

Every model is benchmarked in **two** classes, recorded as separate result
dimensions. A divergence between them is itself a first-class finding.

- **ISOLATED:** Clean room; only minimal-impact processes run. Record what ran.
  Use a managed window only if the model cannot co-reside
  ([Step 3](RUNBOOK.md#step-3--serve-the-model)).
- **UNDER-LOAD:** Measure with background load present. Do not use a managed
  window.

The under-load class needs no managed window, which drastically reduces
production downtime: managed windows are only for isolated-class runs of models
that cannot co-reside. Record per run: **environment class** + **what was
concurrently running** (a `llama-swap` `/running` snapshot and load average).

**Worked example — why the two classes exist.** Qwen3.6-35B-A3B-OptiQ-4bit
benchmarked **isolated** at 100% valid tool calls with zero multi-turn
degradation (clean through 20 rounds). The **same** model under concurrent
production load degenerated into repetition loops (the same sentence 100+ times,
~37 duplicate tool calls per turn) and needed a `repetition_penalty` guardrail to
serve. The isolated number alone would have crowned it unconditionally; the
under-load number is what caught the production failure mode. (The under-load
figure here is a production observation, pending a formal under-load bench.)

## How the gates compose

A model is **fully benchmarked with a final verdict** only when, **for each
environment class (isolated *and* under-load)**, it has **≥4 dated runs ≥5 days
apart**, where **each run is a validated consecutive pair** across the **full
required suite set**. Missing any of those → the verdict stays PROVISIONAL.

```text
final verdict ⇐ isolated: ≥4 runs (≥5 days apart), each a validated pair,
                 full suite set
          AND under-load: ≥4 runs (≥5 days apart), each a validated pair,
                          full suite set
```

## Counting maturity in RANKINGS

`RANKINGS.md` shows a maturity column as `N/4`, counting **protocol-valid runs**
toward the 4 needed: a run only counts once it is a validated pair (Gate 2)
recorded in a named environment class (Gate 3), ≥5 days from the previous counted
run.

**Crediting rule (why `1/4`, not `0/4`).** A model that has been benchmarked at
all — even by a single pre-protocol shard that was never a validated pair or
env-class-tagged — is credited **one** run: its column reads `1/4`, not `0/4`.
`0/4` is reserved for a model with no benchmark data whatsoever. Only a
**protocol-valid** run (validated pair, named env class, ≥5 days after the last
counted run) increments the count past that first credit. So:

- No data → `0/4`.
- Any historical/pre-protocol data → `1/4` (one credited run), PROVISIONAL.
- Each subsequent protocol-valid run → `2/4`, `3/4`, `4/4` → verdict final.

Because every current shard predates the protocol, **every benchmarked model sits
at `1/4` today**, even ones with shards on several dates — date count alone does
not mature a verdict.

## NVIDIA campaign: relaxed Gate 1

The NVIDIA/CUDA campaign ([`configs/nvidia/`](../configs/nvidia/)) relaxes
**Gate 1 only**. A model row from this campaign may be reported without ≥4
runs ≥5 days apart.
The relaxation is labelled, never silent:

- The runbooks publish every campaign model row with
  `--tag campaign=nvidia --tag verdict_gate=relaxed`. The label travels with
  the data into the dataset.
- In [`RANKINGS.md`](../RANKINGS.md), such a row reads `relaxed (NVIDIA)` in
  the Maturity column instead of `N/4`. Its Role keeps the leads/lags-by-run
  wording. A relaxed row is never a final verdict.
- Hardware baselines (gpu-burn, nvbandwidth, mbw, fio) have no model under
  test and make no verdict. They carry `campaign=nvidia` but no gate label.

This section amends Gate 1 only; Gates 2 and 3 are unchanged here.

## 2026-10-18 overnight recommendation record

This record is provisional pending operator confirmation. Raw rows are in the
[Hugging Face dataset](https://huggingface.co/datasets/JacobPEvans/mlx-benchmarks).
The hardware labels below are the canonical dataset labels.

- **RTX PRO 6000 Max-Q 96GB:** Qwen3.8 27B NVFP4 with vLLM 0.31.0,
  FP8 KV, and MTP. Use 64k × 16 for everyday and 256k × 5 for long context.
  Use reasoning OFF/LOW by default.
- **Mac Studio M4 Max 128GB:** Flash-Next IQ3_S (ISTA-DASLab GSQ-RCO).
  Use llama.cpp Metal v0.7.0 and prompts up to about 30k.
- **RTX 4080 SUPER 16GB:** Qwen3.8 27B UD-IQ3_XXS with llama.cpp.
  Use 1 × 131072 as the fallback.

Nemotron 3.5 Bolt on the RTX PRO 6000 Max-Q is not in this provisional
recommendation set for this cycle: repository QA was comparable, and its XHIGH
coding run did not finish within the run window.

### Repository QA

The deterministic set has 60 questions and a standard error of about 6
percentage points. Scores are reported counts. XHIGH helps Qwen3.8 27B on the
20 multi-step questions only.

| Candidate | Context | OFF | XHIGH |
| --- | --- | ---: | ---: |
| Qwen3.8 27B NVFP4, RTX PRO 6000 Max-Q 96GB | 30k | 35 | 42 |
| Qwen3.8 27B NVFP4, RTX PRO 6000 Max-Q 96GB | 120k | 32 | 39 |
| Flash-Next UD-Q2, RTX PRO 6000 Max-Q 96GB | 30k | 40 | 40 |
| Flash-Next IQ3_S, Mac Studio M4 Max 128GB | 30k | 42 | 42 |
| Qwen3.8 27B UD-IQ3_XXS, RTX 4080 SUPER 16GB | 30k | 41 | 41 |
| Nemotron 3.5 Bolt, RTX PRO 6000 Max-Q 96GB | 30k | 40 | 40 |
| Nemotron 3.5 Bolt, RTX PRO 6000 Max-Q 96GB | 120k | 41 | 40 |

Flash-Next IQ3_S did not finish the Studio 120k run after 3 h 19 min and 7
timeouts.

### Coding and agentic results

BigCodeBench-Hard v0.1.4 used its official evaluator. Its controls failed all
three deliberately wrong outputs and passed all three canonical solutions. No
model pair differed significantly on shared items.

| Model and hardware | Score | Output tokens per item |
| --- | ---: | ---: |
| Qwen3.8 27B NVFP4, RTX PRO 6000 Max-Q 96GB, OFF | 52/148 (35.1%) | 685 |
| Qwen3.8 27B NVFP4, RTX PRO 6000 Max-Q 96GB, LOW | 55/148 (37.2%) | 2,182 |
| Qwen3.8 27B NVFP4, RTX PRO 6000 Max-Q 96GB, XHIGH | 39/107 (36.5%) | 16,339 |
| Flash-Next UD-Q2, RTX PRO 6000 Max-Q 96GB | 26/71 (36.6%) | 29,505 |
| Flash-Next IQ3_S, Mac Studio M4 Max 128GB | 42/123 (34.1%) | 18,317 |
| Qwen3.8 27B UD-IQ3_XXS, RTX 4080 SUPER 16GB | 5/17 (29%) | 30,881 |

For Harbor, Qwen3.8 27B NVFP4 solved 14/18 tasks in minutes per task.
Flash-Next UD-Q2 solved 2/2 before the run stopped; each task took 1.6–2 hours.
Nemotron 3.5 Bolt solved 0/3, with two timeouts.

Nemotron 3.5 Bolt speed used 1024 input / 2048 output. At c1, output / total
throughput was 66.9 / 102.3 tok/s, TTFT p50/p99 was 15,999.7 / 16,213.3 ms,
TPOT p50/p99 was 7.11 / 7.11 ms, and failed requests were 0.

The c4 cell completed 11/12 requests: output / total throughput was
71.7 / 109.5 tok/s, TTFT was 56,523.8 / 59,420.3 ms, TPOT p50/p99 was
26.51 / 26.55 ms, and one request failed with `ServerDisconnectedError`.

### Benchmark lessons

- Repository QA and BigCodeBench-Hard do not separate candidates at current
  sample sizes. Grow the repository-QA reasoning subset to at least 150 items.
- BigCodeBench-Hard scoring from EvalScope is invalid; use the official
  evaluator.
- Output limits are experiment settings. Use a 32k minimum where the slot
  permits; cap output at slot size minus prompt size minus 512 tokens, and
  record truncation.
- The benchmark-run template uses the multi-architecture client image
  `vllm/vllm-openai-cpu:v0.31.0-arm64`
  ([ansible-proxmox-ai #1596](https://github.com/dryvist/ansible-proxmox-ai/pull/1596)).
- Harbor and IFBench separated candidates; shared-item BigCodeBench-Hard
  comparisons did not.

Configurations pruned for this comparison: 27B UD-Q4_K_M with llama.cpp on the
RTX PRO 6000 Max-Q; Flash-Next IQ3_S with llama.cpp on the RTX PRO 6000 Max-Q;
Flash-Next NVFP4 with vLLM on the RTX PRO 6000 Max-Q because it does not fit;
27B Q4 with llama.cpp and 27B MLX 8-bit on the Mac Studio; and Flash-Next at
120k on the Mac Studio.

## Language rule

Before writing that a model is the best or worst anywhere — docs, PR bodies,
serving comments — check its maturity. Never write "X is the best/worst model."
Write **"X leads/lags as of N runs"** and, once past a gate, name the class
("isolated" vs "under-load"). A dismissal is "not serving it as the brain this
cycle," not "X is bad."
