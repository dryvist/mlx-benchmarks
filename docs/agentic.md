# agentic suite — many-tool tool-call reliability

The `agentic` suite measures whether a served model produces **valid
structured tool calls** when carrying a production-sized tool registry under
production-shaped load. Small toy registries hide the failure mode that broke
live agent sessions; this suite ships a 22-tool registry (Splunk trio,
filesystem, shell, memory, wiki, Slack, cron, web fetch, plus near-duplicate
distractors) with realistic JSON-schema parameter definitions — the registry
itself is the load under test.

Runner: [`harness/agentic/run.py`](../harness/agentic/run.py) — a standalone
PEP 723 script (`uv run` resolves its only dependency, httpx). It targets any
OpenAI-compatible `/v1` endpoint and needs nothing server-side, so results are
portable across llama-swap, vllm-mlx, and router deployments.

## What it measures

**Single-shot matrix** — every cell sends requests that *should* produce a
tool call and classifies each response:

| Dimension | Values | Why |
| --- | --- | --- |
| thinking | on / off | reasoning changes tool-call formatting paths |
| concurrency | 1 / 4 | saturation failures only appear under parallel load |
| context | small (~1K) / large (~20K synthetic tool-result history) | truncation/degradation shows at real context sizes |
| transport | streaming / non-streaming | the production failure was stream truncation mid-tool-call |

Per cell: `valid_tool_call_rate` (name non-empty and in the registry,
arguments parse as JSON with all required keys, `finish_reason == tool_calls`),
a failure taxonomy (`no_tool_call` / `empty_function_name` / `bad_json_args` /
`unknown_tool` / `http_error` / `timeout` / `stream_truncated`), latency
p50/p95, effective tok/s, and reasoning-content presence. Streaming cells
assemble `tool_calls` from deltas exactly the way a real client does.

**Multi-turn degradation track** (mlx-lm #1011) — stock 4-bit Qwen3.x quants
degrade to plain-text `[Tool call: ...]` fallback around round 5 (8-bit ~13);
single-shot cases cannot catch this. The runner drives 20 scripted rounds of
(ask → tool call → synthesized tool result) with full accumulated history and
reports `first_degraded_round` (never degraded → clean through 20), at
thinking on and off. This is the primary quant discriminator.

## Running

Against llama-swap / vllm-mlx directly (default local serving port):

```bash
uv run harness/agentic/run.py \
  --base-url http://localhost:11434/v1 \
  --api-key-env OPENAI_API_KEY \
  --model mlx-community/Qwen3.6-35B-A3B-4bit
```

Against a router (LiteLLM / gateway), only the base URL and key env change:

```bash
uv run harness/agentic/run.py \
  --base-url http://localhost:4000/v1 \
  --api-key-env LITELLM_API_KEY \
  --model qwen3.6-35b
```

Notes:

- `--api-key-env` names the **environment variable** holding the key — the
  runner never accepts a literal key on the command line.
- Harmony models (gpt-oss): add `--thinking-kwarg reasoning_effort`. Default
  is `enable_thinking` via `chat_template_kwargs` (Qwen3.x / GLM).
- `--thinking` takes a comma list of levels, sent verbatim. `on` and `off` keep
  their per-family mapping (`reasoning_effort` high/low, `enable_thinking`
  true/false) and their existing cell names, so published rows stay comparable.
  Any other value — `low`, `high`, `xhigh`, `max` — is passed straight through
  and names its own cell, e.g. `conc1_think-xhigh_ctx-small_nostream`.
  **Effort is part of the arm**, so a sweep across levels is a sweep across
  arms, not a variation within one.
- The full matrix (16 cells × 10 repeats + 2×20 multi-turn rounds) takes hours
  on slow models. Smoke first: `--cells conc4_think-on_ctx-large_stream --repeats 3`
  (`--cells` is a comma-separated substring filter; include `multiturn` to
  keep that track).

The runbook with copy-paste commands lives at
[`configs/agentic/tool-calling.toml`](../configs/agentic/tool-calling.toml).

## Pass gate

Mirrors the serving goal — at **concurrency ≥ 4, thinking ON, large context**
(the `conc4_think-on_ctx-large_*` cells, both transports):

- `valid_tool_call_rate == 1.0`
- zero `empty_function_name` failures
- effective throughput ≥ 15 tok/s

A model that passes single-shot but degrades in the multi-turn track is not
fit for agent serving; treat `first_degraded_round` as a hard signal when
comparing quantizations.

## Publishing

The runner writes one raw-results JSON per (model, run). Convert + publish via
the standard flow:

```bash
mlx-bench-publish run-output/agentic_<model>.json \
  --kind agentic --suite tool-calling --hostname jevans-ms --dry-run
```

Each matrix cell becomes metric rows named `tool_calling`
(`valid_tool_call_rate`, `finish_reason_tool_calls_rate`,
`reasoning_present_rate`, `request_latency_p50_ms`, `request_latency_p95_ms`)
with the sweep dimensions and failure counts as string tags; the multi-turn
track maps to `first_degraded_round` (unit `round`, value `0` +
`tags.degraded="false"` when clean).
