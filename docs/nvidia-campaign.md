# NVIDIA campaign

## Runbooks

[`configs/nvidia/`](../configs/nvidia/) contains runbooks for an NVIDIA GPU host serving vLLM:

- Serving throughput: 1/4/8 concurrent requests at 8k/64k/128k prompt sizes via `vllm bench serve`.
- Quality: lm-eval tasks that are not already saturated.
- Hardware baselines: gpu-burn, nvbandwidth, mbw, and fio.

## Envelope fields

The envelope supports optional `system.gpu`, `system.engine`,
`system.power_limit_w`, and `system.container` fields, plus four hardware
baseline suites. Legacy envelopes remain valid; see [`schema.md`](schema.md).

## Verdict gate

The campaign relaxes the default ≥4 runs / ≥5 days gate (Gate 1 in
[`verdict-policy.md`](verdict-policy.md)). Published rows carry
`campaign=nvidia` and `verdict_gate=relaxed` tags and appear in
[`RANKINGS.md`](../RANKINGS.md) as `relaxed (NVIDIA)`. They remain
“leads/lags as of N runs,” not final verdicts. The other verdict gates remain
unchanged.
