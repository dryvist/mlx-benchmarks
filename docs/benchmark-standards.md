# Benchmark standards

Use these standards for comparable benchmark rows. Store raw results in the
[Hugging Face dataset](https://huggingface.co/datasets/JacobPEvans/mlx-benchmarks).
This page records the method and current MTP depth comparison.

## Standards

1. **Engine releases.** Use the latest stable release shown by the upstream
   GitHub `releases/latest` page. Ignore prereleases. Record the engine, version,
   build or commit, and release date on each row. Keep rows from older versions;
   do not replace them.

2. **Fixed-length client.** Use the native grid command
   `vllm bench sweep serve --bench-params <json>` with
   `--ignore-eos --dataset-name random`. Run the fixed-length client from
   `vllm/vllm-openai-cpu:v0.31.0-arm64`. Give every cell a unique seed. For Mac
   MLX servers, pass EOS-token `logit_bias` through `--extra-body`. Do not use
   custom clients or loops.

3. **Concurrency.** Test c = 1, 4, 8, 12, and 16 wherever every request fits the
   KV pool. Also test the ceiling and ceiling + 1, where the ceiling is the
   smaller of the engine's floored capacity at that context and the profile's
   `max_num_seqs` or slot count. Read capacity from the engine: vLLM's startup
   line for maximum concurrency at the requested token length, or llama.cpp's
   `/props` slot count and per-slot `n_ctx`. Skip a standard concurrency only
   when it does not fit, and record the reason.

4. **Load shapes.** Report batched load with `--request-rate inf` and staggered
   Poisson load with `--request-rate R --burstiness 1`, at the same maximum
   concurrency. Include a staggered agentic pair in every throughput block:
   4,096 input and 1,024 output tokens, plus at least 32,768 input tokens.

5. **Context.** Include true 128k rows and 256k rows when the model card permits
   them. Never exceed the model card's maximum context.

6. **MTP.** Run matched MTP-on and MTP-off arms for each supported model and
   engine. The Qwen3.8-27B vLLM comparison at 64k × 8 FP8 KV, 1,024/2,048
   tokens, c8 measured 774 output tok/s with two MTP draft tokens and 66% draft
   acceptance; MTP-3 and MTP-4 are being measured. Record depth and the ratio
   of `vllm:spec_decode_num_accepted_tokens_total` to
   `vllm:spec_decode_num_draft_tokens_total` on every MTP row.

7. **Intelligence.** Use quick suites with room below saturation. Drop any
   suite where a compared model scores 90% or higher. Use IFBench through
   EvalScope for the current quick suite; drop saturated MMLU-Pro generative
   CoT, ARC, and GSM8K. Run the OFF, LOWEST, and HIGHEST reasoning arms. Record
   each arm as `--reasoning-effort off|low|xhigh`; use
   `enable_thinking: false` for OFF, `reasoning_effort: "low"` for LOWEST, and
   `reasoning_effort: "xhigh"` for HIGHEST. Record `n` and standard error.
   Claim a difference only when it exceeds about two standard errors.

8. **Judge-scored suites.** Use the router's judge route and its standard
   zero-data-retention fallback rungs. Use an external judge. Benchmark tooling
   must not read provider keys.

9. **Runtime budget.** Declare it before each run: 1–2 h by day or up to 8 h
   overnight. Stop when the budget expires.

10. **Isolation.** Run one serving unit per GPU host during a cell. Prove that
    one unit is active, record the listening port's owner PID, and verify its
    start time follows the converge. Run no foreign clients. On Mac, stop
    resident model workers before the cell and restore them after it.

11. **Environment.** Record ambient temperature, fan duty, power limit,
    dedicated status, and host and guest uptime at block start on every row.
    Set GPU fan duty to 100%. On Mac, use the highest built-in power mode;
    macOS does not expose fan duty. Publish uptime as `host_uptime_s` and
    `guest_uptime_s`. If an uptime read fails, record `null` with its reason.
    Use these machine labels: `RTX PRO 6000 Max-Q 96GB`,
    `RTX 4080 SUPER 16GB`, and `Mac Studio M4 Max 128GB`.

12. **Models.** Use trusted sources only: nvidia, unsloth, bartowski,
    ISTA-DASLab, mlx-community, lmstudio-community, the official releaser, or an
    organization with high likes and downloads. Test current-generation models.
    Keep one registry entry for a superseded model and mark it
    `OLD — DONT BOTHER WITH THIS MODEL ANYMORE`. Comment out duplicate entries
    and point them to that registry entry.

13. **Pruning.** Stop testing a model or configuration when a close alternative
    clearly beats it on both speed and intelligence. Record why testing stopped.

## Related docs

- [Runbook](RUNBOOK.md) — run and publish a benchmark.
- [Verdict policy](verdict-policy.md) — validate pairs and compare models.
- [Context campaign](context-campaign.md) — measure context sensitivity.
- [Dataset](https://huggingface.co/datasets/JacobPEvans/mlx-benchmarks) — raw results.
