# NVIDIA hosts (`configs/nvidia/`)

Runbooks for a vLLM serving host with an NVIDIA GPU. They follow the llama-cpp
pattern: this repo never starts the server, the runbook names the exact upstream
command, and a converter turns that tool's output into an envelope. Only the four
hardware baselines needed new converters; throughput and quality reuse
`--kind vllm` and `--kind lm-eval`.

The system block (`system.gpu`, `system.engine`, `system.power_limit_w`,
`system.container`) is **declared, not probed**: `detect_system()` reads it from
`MLX_BENCH_*` environment variables, like cluster topology, because the publisher
may run on a different machine than the GPU. Export these on the GPU host and
publish from it, so `os` / `chip` / `memory_gb` describe the same machine:

`system.power_limit_w` is flattened into `power_limit_w` on each public result
row. Keep it with the measurement so each enforced power setting remains a
separate comparison dimension; operational telemetry stays internal.

```sh
export MLX_BENCH_GPU_MODEL="$(
  nvidia-smi --query-gpu=name --format=csv,noheader | head -n1
)"
export MLX_BENCH_GPU_VRAM_GB="$(
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits |
    head -n1 | awk '{printf "%.1f", $1 / 1024}'
)"
export MLX_BENCH_GPU_DRIVER="$(
  nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1
)"
export MLX_BENCH_GPU_CUDA="$(
  nvidia-smi |
    sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -n1
)"
export MLX_BENCH_POWER_LIMIT_W="$(
  nvidia-smi --query-gpu=enforced.power.limit --format=csv,noheader,nounits |
    head -n1
)"
export MLX_BENCH_ENGINE_NAME=vllm
export MLX_BENCH_ENGINE_VERSION="$(vllm --version)"
# Set only when the engine ran in a container.
export MLX_BENCH_CONTAINER="<image:tag>"
```

A variable that is unset, blank, or (for the two numeric ones) not a number is
left out of the envelope rather than recorded wrong.

Conventions the runbooks rely on:

- **Sweep tags.** Throughput rows carry `--tag prompt_tokens=…`,
  `--tag concurrency=…`, and `--tag context_len=…` (all strings; `context_len`
  is the server's configured window, never the prompt length). `schema.json`
  documents them.
- **Hardware baselines have no model.** Publish them with `--model hardware-baseline`.
  `.txt` / `.log` output (mbw, gpu-burn) is passed straight to the CLI, which
  wraps it as `{"output": text}`.
- **Relaxed verdict gate.** NVIDIA-campaign model rows carry
  `--tag campaign=nvidia --tag verdict_gate=relaxed`; see
  [`docs/verdict-policy.md`](../../docs/verdict-policy.md).
