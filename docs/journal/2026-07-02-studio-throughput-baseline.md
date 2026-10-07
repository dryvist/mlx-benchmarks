# Mac Studio — First Serving-Throughput Baseline

**Date:** 2026-07-02 (published 2026-07-07)

**Tracking:** [#86](https://github.com/JacobPEvans/mlx-benchmarks/issues/86)

First `throughput` sweep from the Mac Studio (the always-on serving host),
cross-checking the resident model pair with vllm's `benchmark_serving` against
the live vllm-mlx endpoint. These are the "before" numbers for serving-side
tuning — the goal remains fastest-accurate local inference.

## Setup

| Component | Value |
| --- | --- |
| Machine | Mac Studio, Apple M4 Max, 128 GB unified memory (the serving host) |
| Inference backend | vllm-mlx via llama-swap on `http://127.0.0.1:11434` |
| Tool | vllm `benchmark_serving.py`, `--backend openai-chat` |
| Dataset | `random`, input/output len 256/256 |
| Prompts | 100 (30 for the c1 gpt-oss run) |
| Concurrency sweep | 1 / 2 / 4 (`--max-concurrency`) |

## Results (aggregate output throughput)

| Model | Concurrency | Output tok/s | TTFT p50 (ms) | TPOT p50 (ms) |
| --- | --- | --- | --- | --- |
| Qwen3-Coder-30B-A3B-Instruct-4bit | 2 | 119.3 | 346 | 12.0 |
| Qwen3-Coder-30B-A3B-Instruct-4bit | 4 | 136.7 | 574 | 30.5 |
| gpt-oss-120b-MXFP4-Q8 | 1 | 24.8 | 736 | 35.9 |
| gpt-oss-120b-MXFP4-Q8 | 2 | 34.0 | 1153 | 48.6 |
| gpt-oss-120b-MXFP4-Q8 | 4 | 44.4 | 1221 | 84.7 |

A sixth run (Qwen3-Coder, warm-up-inclusive) was also published with
`caveat=warmup-run` — its abnormally low request throughput reflects model-load
time folded into the measurement window. Kept, not discarded.

**Reading:** Qwen3-Coder-30B (4-bit, 3 B active) clears the >100 tok/s community
reference comfortably. gpt-oss-120b (MXFP4-Q8, 5.1 B active) single-stream lands
at ~55–65% of the community reference (40–50 tok/s on M4 Max 128 GB) — tuning
headroom, not a stack problem.

## Publishing

The results ran on the Studio but were published from the MacBook Pro once a
write-scoped `HF_TOKEN` was available. The public projection omits the
publisher's host identifier; `os` and `kernel` describe the publishing
machine. Each shard is tagged with the run's `concurrency=N`. Command shape:

```sh
mlx-bench-publish <benchmark_serving>.json \
  --kind vllm --suite throughput \
  --model <model_id> --published-metadata <metadata.json> \
  --tag concurrency=<N> --timestamp <run-time-UTC>
```
