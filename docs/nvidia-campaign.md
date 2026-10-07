# NVIDIA campaign

## Runbooks

[`configs/nvidia/`](../configs/nvidia/) holds NVIDIA GPU runbooks for vLLM:

- Serving throughput: closed-loop concurrency at 8k/64k/128k prompt sizes with
  `vllm bench serve`.
- Quality: lm-eval suites with measurable headroom at current scores.
- Hardware baselines: gpu-burn, nvbandwidth, mbw, and fio.

## Power limit is a result dimension

Since #299, every public result row records the enforced `power_limit_w` value
from `system.power_limit_w`. Treat each setting as a separate run arm. The
NVIDIA result dataset contains measurements and run-defining variables such as
the power limit; operational telemetry stays in the internal stack.

The current Qwen3.8-27B-NVFP4, vLLM 0.30.0, `medium-a` measurements used a
96 GB RTX PRO 6000 Max-Q with a 196,608-token model limit, eight sequences, and
a 973,859-token KV pool. At the 300 W factory limit, sustained load reached
92 °C. At 290 W, temperature stayed at or below 91 °C. Across four samples per
cell and 30 cells, 290 W reduced throughput by 1.0–2.4% and increased time to
first token by 0–4% versus 300 W; saturation points did not change. At 1k input,
2,048 output tokens, and concurrency 8, output throughput measured 431.5 tok/s
at 300 W and 424.6 tok/s at 290 W. An earlier 275 W test cost 3–4% throughput;
275 W is a temporary cap while ventilation improves.

## Concurrency and open-loop request rate

The closed-loop sweep sets a maximum number of in-flight requests. For prompts
up to 32k tokens, throughput peaked at concurrency 8. Raising concurrency
beyond 8 added queueing: at 1k input, concurrency 16 kept throughput flat while
median time to first token rose from 0.6 s to 19.7 s. At 131k input, concurrency
4 was best at 57.5 output tok/s at 290 W; concurrency 8 and 16 were slower. At
194k input, throughput stayed near 7.4 tok/s from concurrency 2 with 256 output
tokens. Concurrency 16 halved throughput and raised time to first token to
9 minutes. The 131k, concurrency-8 result varied by about ±10% between runs and
needs more samples. A 194,560-token prompt plus 2,048 output tokens exceeds the
196,608 model limit after chat-template tokens and correctly returns HTTP 400.

Use the measured admission caps as a starting point for this model and profile:
8 concurrent requests through 32k context, 4 at 131k, and 2 at 194k or more.
Re-measure after changing the model, context window, engine, quantization, or
power limit.

The open-loop series uses Poisson arrivals, 512 output tokens, and a 290 W
limit. For 1k input, rates of 0.25 and 0.5 requests/s stayed unsaturated. A
rate of 1 request/s saturated at about 400 output tok/s with 5 s median time to
first token. Rates of 2–4 requests/s added no throughput and raised the median
to 10 s. The sustainable rate was about
0.75 requests/s. At 8k input, the sustainable range was 0.5–1 request/s; 1
request/s gave about 10 s median time to first token. At 32k input, the model
was saturated even at 0.25 requests/s: it produced 77 output tok/s against 128
offered, with 61 s median time to first token. Prefill took about 3.8 s of GPU
time, and sustainable admission was about one 32k request every 6–8 s.

## Device comparison and selection policy

For an apples-to-apples device comparison, match the model, quantization,
context, and request mix. One measured single-user comparison used the same
Qwen3.8-27B model: MLX 4-bit on an M4 Max Mac Studio decoded at 27.3–27.4 tok/s;
vLLM NVFP4 on the RTX PRO 6000 was about 2.3× faster. This is a measured
same-model, one-user cross-stack result, not a matched-quant comparison. Keep
that limit visible when citing the ratio.

The Mac Studio's 128 GB unified memory supports serial runs of larger models
that fit there. The RTX measurements characterize shared concurrent-request
capacity and context-dependent saturation. Use those workloads to describe each
device's role.

Give intelligence and speed equal weight when selecting a serving candidate.
Choose intelligence suites where current scores leave measurable headroom, and
repeat short suites to check score stability. The short ARC-Challenge recipe
uses two repetitions. Rank candidates using both speed and intelligence
results.

## Envelope fields

The envelope supports optional `system.gpu`, `system.engine`,
`system.power_limit_w`, and `system.container` fields, plus four hardware
baseline suites. The publisher flattens `system.power_limit_w` into the public
`power_limit_w` result column. Legacy envelopes remain valid; see
[`schema.md`](schema.md).

## Verdict gate

The campaign relaxes the default ≥4 runs / ≥5 days gate (Gate 1 in
[`verdict-policy.md`](verdict-policy.md)). Published rows carry
`campaign=nvidia` and `verdict_gate=relaxed` tags and appear in
[`RANKINGS.md`](../RANKINGS.md) as `relaxed (NVIDIA)`. They remain
“leads/lags as of N runs,” not final verdicts. The other verdict gates remain
unchanged.
