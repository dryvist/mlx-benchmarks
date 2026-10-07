# Model-class notes — agentic tool-calling on Apple Silicon (MLX)

Durable, per-model-class quirks that decide whether a model can drive a
many-tool agent (structured tool calls, thinking/reasoning parsing,
concurrency) on vllm-mlx. Model-currency facts below were refreshed
2026-10-07; dated links inline. Per-run findings belong in
[`docs/journal/`](./journal/) — this file holds only what stays true across
runs.

## Model currency and target fit (2026-10-07)

`OLD` means a newer architecture generation in the same family has a
documented runnable MLX path. It is not an Artificial Analysis score label.
Candidate facts are machine-readable in
[`configs/shootout/candidates.toml`](../configs/shootout/candidates.toml);
this table records the successor and fit evidence for affected models.

| Model | Arch gen | Date | Weights / params | MLX quants | Evidence |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8-Flash-Next | Qwen4 experimental architecture, `qwen4_exp` | 2026-08-26 | Weights available, license `other`; 176B per QwenLM component counts (125B transformer + 51B n-gram); HF metadata reports 180B; 6B active | 2-bit mixed-precision MLX-OptiQ: 80.8 GB on disk; SSD streaming, card reports 4.6 GB peak | Current Qwen architecture generation with a [documented coding run](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-OptiQ-2bit); use `mlx-optiq>=0.5.14` and `optiq serve`. The official [release notes](https://github.com/QwenLM/Qwen3.8-Flash-Next) describe it as an early preview of the architecture used in Qwen4. |
| **OLD** Qwen3-Next-80B-A3B | Qwen3 generation, hybrid Gated DeltaNet + gated attention MoE, `qwen3_next` | 2025-09-11 | Yes, Apache-2.0; 80B / 3B active | MLX 4-bit Instruct 44.8 GB; Thinking 4-bit ~45 GB | Superseded by [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), tagged `qwen4_exp`; its [2-bit MLX-OptiQ artifact](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-OptiQ-2bit) is 80.8 GB on disk, requires `mlx-optiq>=0.5.14` and SSD streaming, and documents a coding run. The card reports a 4.6 GB peak working set. Release date: [official Qwen3.8 release record](https://github.com/QwenLM/Qwen3.8/blob/main/README.md); runnable quant sources: [Instruct 4-bit](https://huggingface.co/mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit), [Thinking 4-bit](https://huggingface.co/mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit). |
| **OLD** Qwen3-30B-A3B-Instruct-2507 | Qwen3 MoE, `qwen3_moe` | 2025-07-30 | Yes, Apache-2.0; 30.5B / 3.3B active | MLX 4-bit 17.2 GB; MLX 8-bit 32.4 GB | Superseded by runnable Qwen4-generation Qwen3.8-Flash-Next; [official Qwen3 release record](https://github.com/QwenLM/Qwen3/blob/main/README.md), [official model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507), [4-bit artifact](https://huggingface.co/mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit), [8-bit artifact](https://huggingface.co/mlx-community/Qwen3-30B-A3B-Instruct-2507-8bit), and Qwen3.8 links above. |
| **OLD; PROTECTED: recorded resident** Qwen3-Coder-30B-A3B | Qwen3 MoE, `qwen3_moe` | 2025-07-22 | Yes, Apache-2.0; 30.5B / 3.3B active | MLX 4-bit 17.2 GB; MLX 8-bit 32.4 GB | Superseded by runnable Qwen4-generation Qwen3.8-Flash-Next; current residency UNVERIFIED and route retirement lead-owned. [Qwen3-Coder release](https://qwenlm.github.io/blog/qwen3-coder/), [official model card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct), [4-bit artifact](https://huggingface.co/mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit), [8-bit artifact](https://huggingface.co/mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit), and Qwen3.8 links above. |
| **OLD** Qwen3-Coder-Next-80B-A3B | Qwen3 generation, Qwen3-Next hybrid MoE, `qwen3_next` | 2026-02-02 | Yes, Apache-2.0; 80B / 3B active | Candidate MLX 4-bit 44.8 GB | Superseded by runnable Qwen4-generation Qwen3.8-Flash-Next; [official Qwen3-Coder-Next release](https://qwen.ai/blog?id=qwen3-coder-next), [HF model card](https://huggingface.co/Qwen/Qwen3-Coder-Next), [MLX artifact](https://huggingface.co/mlx-community/Qwen3-Coder-Next-4bit), and Qwen3.8 links above. |
| **OLD** Qwen3.5-27B | Qwen3.5 dense hybrid Gated DeltaNet + gated attention, `qwen3_5` | 2026-02-24 | Yes, Apache-2.0; 27B | MLX 4-bit 16.1 GB | Superseded by runnable Qwen4-generation Qwen3.8-Flash-Next; [official release record](https://github.com/QwenLM/Qwen3.8/blob/main/README.md), [official model card](https://huggingface.co/Qwen/Qwen3.5-27B), [MLX artifact](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit), and Qwen3.8 links above. |
| **OLD** Qwen3.6-27B | Qwen3.6 dense hybrid Gated DeltaNet + gated attention, `qwen3_5` | 2026-04-22 | Yes, Apache-2.0; 27B | MLX 4-bit 16.1 GB | Superseded by runnable Qwen4-generation Qwen3.8-Flash-Next; [official release record](https://github.com/QwenLM/Qwen3.8/blob/main/README.md), [official model card](https://huggingface.co/Qwen/Qwen3.6-27B), [MLX artifact](https://huggingface.co/mlx-community/Qwen3.6-27B-4bit), and Qwen3.8 links above. |
| **OLD; PROTECTED: served family** Qwen3.6-35B-A3B | Qwen3.6 MoE hybrid Gated DeltaNet + gated attention, `qwen3_5_moe` | 2026-04-16 | Yes, Apache-2.0; 35B / 3B active | MLX 4-bit 20.4 GB; OptiQ 4-bit 24.7 GB; MLX 8-bit 37.7 GB | Superseded by runnable Qwen4-generation Qwen3.8-Flash-Next above. The served family remains protected; route retirement is lead-owned. Its 80.8 GB artifact exceeds this shootout's 75 GB weight ceiling. [Official release record](https://github.com/QwenLM/Qwen3.8/blob/main/README.md), [official model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [4-bit artifact](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit), [OptiQ artifact](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit), and [8-bit artifact](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-8bit). |
| **OLD** Qwen3.5-122B-A10B | Qwen3.5 MoE hybrid Gated DeltaNet + gated attention, `qwen3_5_moe` | 2026-02-24 | Yes, Apache-2.0; 122B / 10B active | MLX 4-bit 69.6 GB | Superseded by runnable Qwen4-generation Qwen3.8-Flash-Next; [official release record](https://github.com/QwenLM/Qwen3.8/blob/main/README.md), [official model card](https://huggingface.co/Qwen/Qwen3.5-122B-A10B), [MLX artifact](https://huggingface.co/mlx-community/Qwen3.5-122B-A10B-4bit), and Qwen3.8 links above. |
| **OLD** Qwen3.5-35B-A3B | Qwen3.5 MoE hybrid Gated DeltaNet + gated attention, `qwen3_5_moe` | 2026-02-24 | Yes, Apache-2.0; 35B / 3B active | MLX 4-bit 20.4 GB | Superseded by runnable Qwen4-generation Qwen3.8-Flash-Next; [official release record](https://github.com/QwenLM/Qwen3.8/blob/main/README.md), [official model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B), [MLX artifact](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit), and Qwen3.8 links above. |
| **OLD; PROTECTED: served** Qwen3.5-9B | Qwen3.5 dense hybrid Gated DeltaNet + gated attention, `qwen3_5` | 2026-03-02 | Yes, Apache-2.0; 9B | MLX 4-bit artifact: 5.95 GB per HF card | Superseded by runnable Qwen4-generation Qwen3.8-Flash-Next; the catalog retains a 5.2 GB sizing value, distinct from the artifact size. Route retirement is lead-owned. [Official release record](https://github.com/QwenLM/Qwen3.8/blob/main/README.md), [official model card](https://huggingface.co/Qwen/Qwen3.5-9B), [MLX artifact](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit), and Qwen3.8 links above. |
| **OLD** Qwen3-4B-Instruct-2507 | Qwen3 dense Transformer, `qwen3` | 2025-08-06 | Yes, Apache-2.0; 4.0B | MLX 4-bit 2.28 GB | Superseded by runnable Qwen4-generation Qwen3.8-Flash-Next; [official Qwen3 release record](https://github.com/QwenLM/Qwen3/blob/main/README.md), [official model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [MLX artifact](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit), and Qwen3.8 links above. |
| Ling-3.0-Flash | Ling 3.0 / Bailing hybrid KDA + MLA | 2026-08-02 | Yes, MIT; 124B / 5.1B active | MLX 4-bit: 70.03 GB | Runnable same-target successor; the [Apple-silicon validation](https://huggingface.co/TensorFold/Ling-3.0-flash-MLX-4bit) documents successful model loading and text generation. |
| **OLD** Ling-2.6-Flash | Ling 2.6 generation, Bailing hybrid MoE | 2026-04-23 | Yes, MIT; 104B / 7.4B active | Candidate DWQ 4-bit 58.6 GB | Superseded by runnable Ling-3.0-Flash above; [official Ling-2.6 release](https://developer.ant-ling.com/en/blogs/ling-2.6-flash-release), [official model card](https://huggingface.co/inclusionAI/Ling-2.6-flash), [first card commit](https://huggingface.co/inclusionAI/Ling-2.6-flash/commit/93d187d078ce712b2c07a0a272f3466c238724b5), and [Ling-3.0 MLX artifact](https://huggingface.co/TensorFold/Ling-3.0-flash-MLX-4bit). |
| MiniMax-M2 | MiniMax M2 / `minimax_m2` | 2025-10-27 | Weights available, license `other`; 230B / 10B active | REAP-pruned MLX MXFP4: 139B / 10B active, 73.9 GB | Retain for this target; [MiniMax-M3 MLX 4-bit](https://huggingface.co/mlx-community/MiniMax-M3-4bit) is 241 GB and does not fit the 100 GiB device limit. |
| MiniMax-M2.7 | MiniMax M2 / `minimax_m2` | 2026-03-18 | Weights available, license `other`; 229B / active count UNVERIFIED | MLX quant size UNVERIFIED | Retain for this target; MiniMax-M3 MLX 4-bit is 241 GB. Release source: [MiniMax](https://www.minimax.io/news/minimax-m27-en). |
| MiniMax-M3 | MiniMax M3 sparse attention (MSA) | 2026-06-01 | Weights available, license `other`; about 428B / 23B active (HF metadata: 427B) | MLX 4-bit: 241 GB; 3-bit: 186 GB; target-fit quant UNVERIFIED | Newer model generation, but no fitting MLX artifact found; [official card](https://huggingface.co/MiniMaxAI/MiniMax-M3), [4-bit MLX card](https://huggingface.co/mlx-community/MiniMax-M3-4bit), and [3-bit MLX card](https://huggingface.co/pipenetwork/MiniMax-M3-MLX-3bit). |
| GLM-5.3-Flash | GLM 5.3 / `glm5_next` | 2026-08-26 | Yes, MIT; card: 320B / 18B active (HF metadata: 321B) | NVIDIA NVFP4: about 190.4 GiB; fitting MLX quant UNVERIFIED | Newer model generation, but no fitting MLX artifact found; [official release record](https://github.com/zai-org/GLM-V/blob/main/README.md) and [official card](https://huggingface.co/zai-org/GLM-5.3-Flash). |

### GLM-4.7-Flash — NOT PROVEN OLD for this MLX target

- Architecture: GLM 4.7, `glm4_moe_lite`; release: 2026-01-19.
- Open weights: yes, MIT; parameters: 30B total, 3.6B active.
- Runnable MLX quant: candidate 8-bit, 31.8 GB.
- GLM-5.3 has no verified fitting MLX quant. Its NVIDIA NVFP4 is 190.4 GiB.

The Qwen3.8-Flash-Next artifact is runnable through SSD streaming but is an
extreme 2-bit quant. Its weight files exceed the shootout's 75 GB entry limit;
the artifact facts establish architecture availability and target runnability,
not a benchmark-quality result. Any remaining quant-size gap is marked
`UNVERIFIED` rather than inferred from scores.

Serving-flag quick reference (vllm-mlx):

- **OLD** Qwen3 / Qwen3.6 dense + MoE
  - Tool-call parser: `hermes`
  - Reasoning parser: `qwen3`
  - Thinking control: `chat_template_kwargs.enable_thinking`
- **OLD** Qwen3-Coder / Coder-Next
  - Tool-call parser: `qwen3_coder`
  - Reasoning parser: `qwen3`
  - Thinking control: `enable_thinking`
- **OLD** Qwen3-Next (hybrid attention)
  - Tool-call parser: `hermes`
  - Reasoning parser: `qwen3`
  - Thinking control: `enable_thinking`; never spec-decode/MTP
- GLM-4.7-Flash
  - Tool-call parser: `glm47`
  - Reasoning parser: `glm45`
  - Thinking control: thinking on by default
- gpt-oss (harmony)
  - Tool-call parser: `harmony`
  - Reasoning parser: `gpt_oss`
  - Thinking control: `chat_template_kwargs.reasoning_effort`
- DeepSeek-V4-Flash
  - Tool-call parser: `deepseek` + `--enable-auto-tool-choice`
  - Reasoning parser: `deepseek_r1`
  - Thinking control: native `<think>`
- MiniMax-M2.7: `minimax` tool-call parser; native reasoning and thinking.
- Hermes-4
  - Tool-call parser: `hermes`
  - Reasoning parser: —
  - Thinking control: —

vllm-mlx's reasoning guide documents `qwen3` and `deepseek_r1` explicitly; the
wider tool-parser set comes from the server flags
([vllm-mlx docs](https://github.com/waybarrios/vllm-mlx/blob/main/docs/guides/reasoning.md)).

## OLD — Qwen3.x MoE (Qwen3.6-35B-A3B, Qwen3-30B-A3B-Instruct-2507, Qwen3.6-27B)

- **Quantization is the dominant tool-calling variable.** Stock
  `mlx-community` uniform quants degrade **multi-turn** tool calling: 4-bit
  falls back to plain-text `[Tool call: ...]` pseudo-calls at ~round 5, 8-bit
  at ~round 13, while GGUF Q4_K_XL and cloud fp ran 70/70 clean
  ([mlx-lm #1011](https://github.com/ml-explore/mlx-lm/issues/1011), opened
  2026-03-16). Stock 4-bit KL-divergence is ~3× worse than DWQ-4bit
  ([smcleod.net, Apr 2026](https://smcleod.net/2026/04/measuring-model-quantisation-quality-with-kl-divergence/)).
  Use OptiQ/DWQ mixed-precision quants for agent brains; benchmark multi-turn
  (the `agentic` suite's degradation track exists exactly for this — a
  single-shot test cannot see it). A DWQ/OptiQ 4-bit 35B-A3B is the proven
  resident brain: 100% valid tool calls and a clean 20-round track
  ([2026-07-08 selection run](./journal/2026-07-08-agentic-brain-selection.md)).
- **Thinking must stay ON for the agent brain.** The same OptiQ-4bit that ran
  0/20 multi-turn degradation with thinking ON degraded at round 6 with thinking
  OFF — the mixed-precision quant does not carry multi-turn tool calling without
  the reasoning pass. Serve this class with `enable_thinking:true`, not just a
  good quant.
- **Never put the `qwen3_coder` tool-call parser on a general (non-Coder)
  Qwen3.6.** It produces empty `function.name` "repairs" on every call; the
  `hermes` tool-call parser (with `--reasoning-parser qwen3`) drives the general
  MoE clean, and is what the shipped serving uses for this brain.
- **Sampling parity is a serving-side variable, not a bench one.** A quant that
  passes the agentic bench (client default `temperature=1.0`) can still
  degenerate in production if requests arrive with `temperature=None` +
  `repetition_penalty=None` — the 4-bit weights fall into repetition loops (the
  same sentence 100+ times, ~37 duplicate tool calls per turn). Serve this class
  with a guardrail: `repetition_penalty ~1.05` and `temperature 0.6–0.7` in
  thinking mode. When a bench winner misbehaves live, check the sampling delta
  before re-benchmarking (2026-07-08).
- Tool-call format is unstable with `enable_thinking=false` in
  token-in/token-out rollout across vLLM 0.18–0.20; 9B and older Qwen3 are
  unaffected ([verl #6223](https://github.com/verl-project/verl/issues/6223)).
- When the model emits the XML `tool_call` **inside `<think>`**,
  `--reasoning-parser qwen3` consumes everything before `</think>` and the
  response surfaces with empty `tool_calls`
  ([vllm #39056](https://github.com/vllm-project/vllm/issues/39056)).
- `--reasoning-parser qwen3` + `enable_thinking:false` mis-routes streamed
  tokens into `delta.reasoning` instead of `delta.content` in some vLLM
  versions; non-streaming is unaffected
  ([vllm #40816](https://github.com/vllm-project/vllm/issues/40816)).
- Speculative decoding skips/drops tokens with Qwen3 — leave it off
  ([mlx-lm #846](https://github.com/ml-explore/mlx-lm/issues/846)).
- Batch inference for Qwen3.6-27B is broken in vLLM 0.19.1
  ([vllm #40621](https://github.com/vllm-project/vllm/issues/40621), 2026-04-22).
- A3B economics: 3B active parameters keep per-token KV small and tok/s high —
  this class is the concurrency workhorse; give it a real KV budget
  (`--cache-memory-mb` well above the 3 GB default class) before blaming the
  weights for slow decode.

## OLD — Qwen3-Next (hybrid SDPA + Gated-DeltaNet linear attention)

- The early-2026 "crashes when two requests batch" reputation
  (conv_state shape errors) no longer holds: vllm-metal lists the family as
  supported ("Hybrid SDPA + GDN linear") for load + continuous batching
  ([vllm-metal supported models](https://docs.vllm.ai/projects/vllm-metal/en/latest/supported_models/)).
  Re-verify empirically per release, but do not carry the crash claim forward
  without a current reproduction.
- **Automatic prefix caching stays unsupported** for the family — you keep
  batching but lose prefix-cache TTFT reuse across many-tool loops (same
  source).
- MTP/speculative decoding remains buggy for the hybrid DeltaNet
  ([Rapid-MLX #477](https://github.com/raullenchai/Rapid-MLX/issues/477)) —
  run without it.
- Linear attention gives the smallest KV growth of any class here — the
  long-transcript pick when reasoning depth beats raw tok/s.

## OLD — Qwen3-Coder (30B-A3B, Coder-Next)

- Parser is `qwen3_coder`, not `qwen`; with the global
  `--enable-auto-tool-choice`, a registered model missing its parser exits at
  argparse — every swap-in 500s ("upstream command exited prematurely").
- Coder-tuned: fine as a coding sidecar, weak as an autonomous agent brain —
  observed malformed tool calls under sustained agentic load where the
  same-size instruct MoE ran clean (0–67% valid tool calls in the 2026-07-08
  selection run vs 100% for the instruct MoE). Poor-output reports on Coder-Next
  builds ([llama.cpp #19305](https://github.com/ggml-org/llama.cpp/issues/19305)).

## gpt-oss (120b / 20b — harmony format)

- Tool calls and reasoning ride harmony *channels*, not XML. Without
  `--reasoning-parser gpt_oss`, channel markers (`analysis` ...
  `assistantfinal`) leak verbatim into streamed `message.content`. Parser
  pairing `--tool-call-parser harmony --reasoning-parser gpt_oss` coexists
  fine on vllm-mlx ≥ 0.4.0.
- **Agent-loop leak:** in 5+ tool-call runs it sometimes emits tool calls on
  the `analysis` channel (should be `commentary`), leaking raw
  `<|channel|>analysis to=functions...<|call|>` into content and ending the
  run early — pair with a parser/auto-recovery that tolerates it
  ([LangChain forum](https://forum.langchain.com/t/harmony-response-format-sometimes-outputted-when-using-gpt-oss-120b-as-an-agent/2554)).
  This realized in the 2026-07-08 selection run: 0% valid structured tool calls
  at ~2 tok/s — keep gpt-oss as an on-demand reasoning model, not the resident
  tool-calling brain.
- vllm-mlx 0.4.0's paged KV cache is incompatible with gpt-oss's alternating
  sliding-window attention (`[broadcast_shapes]` failures) — paged cache and
  prefix caching go off for this model only.
- Thinking control is `reasoning_effort` (low/medium/high), not
  `enable_thinking`.

## GLM-4.7-Flash

- The most complete parser story of the local-agent class:
  `--tool-call-parser glm47 --reasoning-parser glm45`, and the only
  co-resident-size candidate with working automatic prefix cache on
  vllm-metal ([supported models](https://docs.vllm.ai/projects/vllm-metal/en/latest/supported_models/),
  [guide](https://aicybr.com/blog/glm-4-7-flash-complete-guide)).
- Thinking is on by default; purpose-built for local agentic/coding use
  (30B MoE, ~3.6B active, 200K context).
- **Speed does not equal agentic fitness.** In the 2026-07-08 selection run it
  was the fastest candidate (15.1 tok/s) yet went tool-dead from round 1 of the
  multi-turn track — unfit as an autonomous agent brain despite the best parser
  story of the class. Fine as a fast single-shot/coding sidecar, not for
  multi-turn tool driving.

## MiniMax-M2.7

- SOTA open agentic scores (SWE-Pro 56.22, Terminal-Bench-2 57.0) but 10B
  active — noticeably slower than any A3B on M-series.
- Known failure mode: emits a *textual simulation* of a tool call instead of
  a structured one under some conditions — `--tool-call-parser minimax` plus
  plain-text auto-recovery is required for unattended loops
  ([arXiv 2605.08761](https://arxiv.org/pdf/2605.08761)).

## DeepSeek-V4-Flash

- Native DSML tool grammar; serve with `--tool-call-parser deepseek
  --enable-auto-tool-choice --reasoning-parser deepseek_r1`; `<think>` →
  `reasoning_content` splitting is native.
- Only ~2-bit builds fit 128 GB (≈90 GB) — cited as a strong native
  tool-caller even at 2-bit, but 2-bit reasoning quality is a real risk
  versus 4-bit A3B alternatives; solo-resident only
  ([HF discussion](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/discussions/16)).

## Streaming pitfalls (cross-class)

- Truncated streams manufacture "model" bugs: a client or proxy that kills a
  slow stream mid-tool-call yields partial JSON that surfaces as *invalid
  tool call / empty `function.name`* — indistinguishable from a weights
  problem in the app log. Rule out transport (client stream timeouts,
  router/proxy per-attempt timeouts) before blaming the model class.
- **Server-side disconnect guards are a hidden killer.** A vllm-mlx server whose
  disconnect guard was 300s aborted a legitimate 6320-token agentic generation
  at 301.2s (`ABORTING orphaned request ...`), surfacing as "invalid tool call"
  with empty content. Guards exist to reap genuinely-orphaned work, not to cap
  legitimate long generations. Order the timeout chain so each outer layer
  outlives the inner and the **client is the sole decider**: server
  `--timeout` (e.g. 3600s) > router per-attempt timeout > client stream
  read/stale timeout. If the server or router timeout is the shortest, it will
  reap real work mid-tool-call.
- Streaming last-chunk `tool_calls` with empty `type` broke strict clients
  ([vllm #38603](https://github.com/vllm-project/vllm/issues/38603));
  `--tool-call-parser hermes` + streaming returned raw text instead of parsed
  calls ([vllm #31871](https://github.com/vllm-project/vllm/issues/31871)).
  Benchmark both streaming and non-streaming — they fail differently.

## Serving stacks (state of play, mid-2026)

- **vllm-mlx v0.4.0 (2026-06-28):** Continuous batching (4.3× at 16-way),
  paged/system KV, tool + reasoning parsers — the default choice.
- **Rapid-MLX v0.10.3 (2026-07-07):** 17 tool parsers + plain-text tool-call
  auto-recovery (mitigates mlx-lm #1011) — worth evaluating.
- **llama.cpp (Metal):** GGUF Q4_K_XL is the known-good multi-turn tool-calling
  fallback; MLX still ~20–40 % faster on Apple Silicon.
- **mlx-lm server:** Basic batching; spec-decode buggy (mlx-lm #846) — not for
  concurrency.

Sources: [vllm-mlx releases](https://github.com/waybarrios/vllm-mlx/releases),
[Rapid-MLX](https://github.com/raullenchai/Rapid-MLX). vllm-mlx 0.4.0 also
adds `MLX_BUFFER_CACHE_LIMIT` (configurable Metal buffer cache) and a
multi-slot system-KV cache with hit-ratio counters.
