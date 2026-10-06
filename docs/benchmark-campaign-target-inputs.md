# Benchmark campaign target inputs

The parameterized campaign TOMLs are read by a separate Ansible consumer that
uses `community.general.from_toml` and argv-based command dispatch. This
repository has no TOML-to-argv runner. The consumer supplies its checkout path;
`config_name` selects one allow-listed recipe.

The consumer validates these survey values at runtime:

- `machine`: inventory alias for the selected target.
- `engine`, `model_size`, `concurrency_list`, and `context_list`: recipe axes
  resolved against the model registry and the selected inventory.
- `benchmark_endpoint_root`: HTTPS endpoint origin for the target.
- `benchmark_cache_path`: writable cache path on that target.
- `power_cap_w`: inventory-derived cap checked against target telemetry.

The consumer gathers target system facts from inventory and records them in the
result envelope. For MLX, it accepts a zero power cap when no power limit
applies and requires the registered artifact to exist under the supplied cache
path. Recipes contain no target identity, endpoint value, model ID, artifact
location, or numeric cap. The `power_cap_w = "inventory"` recipe value is a
validation instruction; the consumer records an inventory-derived numeric cap
when present and omits the envelope field when the target has no cap.

Before dispatch, the consumer checks each prompt-plus-output reservation
against the authoritative artifact and engine context limit, and each
concurrency against the selected serving profile cap. It fails closed if a
limit is unavailable or exceeded. For throughput-probe rows, it resolves
`context_window_tokens` from the selected model registry record and passes it
as `--window-limit-tokens`; an absent registry value fails before dispatch.
When conversion runs on the controller, it sets `MLX_BENCH_SYSTEM_OS`,
`MLX_BENCH_SYSTEM_CHIP`, and `MLX_BENCH_SYSTEM_MEMORY_GB` from the selected
target's inventory facts so the envelope describes the benchmark target.

## Recipes

| `config_name` | Engines | Measures |
| --- | --- | --- |
| `llama-cpp/cross-card` | `llama_cpp` | llama.cpp native diagnostics and serving throughput |
| `vllm/cross-card` | `vllm` | `vllm bench serve` throughput at two output sizes |
| `mlx/cross-card` | `mlx_lm` | `mlx_lm.benchmark` diagnostics and serving throughput |
| `lm-eval/quick-intelligence` | `llama_cpp`, `vllm`, `mlx_lm` | ARC-Challenge chat quick pass |
| `lm-eval/gpqa-diamond` | `llama_cpp`, `vllm`, `mlx_lm` | GPQA-Diamond chain-of-thought, zero-shot |
| `evalscope/livecodebench` | `llama_cpp`, `vllm` | LiveCodeBench v6 through EvalScope's Docker sandbox |

Every recipe declares 1, 2, 4, and 8 concurrent agents in `concurrency_list` and a
`context_list` that reaches 131072 tokens. The target, the model class (`small`,
`medium`, or `large`), and the engine are survey values resolved against the
inventory and the model registry, so one recipe covers every class on every target
that serves one of its engines.

`lm-eval/gpqa-diamond` reads a gated dataset: the benchmark target needs a Hugging
Face read credential in its own environment. `evalscope/livecodebench` needs
`evalscope[sandbox]` and a running Docker engine on the benchmark target; the recipe
declares no converter, so its raw reports stay in the target results directory.
`tests/test_campaign_recipes.py` pins this list, the axes, and the placeholders the
consumer substitutes.
