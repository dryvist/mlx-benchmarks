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
