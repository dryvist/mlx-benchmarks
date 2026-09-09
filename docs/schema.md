# Envelope schema (v1)

Canonical JSON Schema: [`schema.json`](../schema.json). This file is a prose
walk-through. When the two disagree, `schema.json` wins — please open a PR.

## Required top-level fields

| Field | Type | Notes |
| --- | --- | --- |
| `schema_version` | `"1"` | Bump only on breaking changes. |
| `timestamp` | ISO 8601 UTC | `YYYY-MM-DDTHH:MM:SSZ`; use start-of-run, not end. |
| `git_sha` | 7–64 hex | SHA of **this repo** at run time, not the model. |
| `trigger` | `local \| schedule \| pr \| workflow_dispatch` | How the run was kicked off. |
| `suite` | enum | Must be in the closed set below. |
| `model` | string | HF model ID (e.g. `mlx-community/Qwen3.5-9B-MLX-4bit`). |
| `system` | object | See below. `os`, `chip`, `memory_gb` required. |
| `results` | array | Per-measurement rows. Empty array is not invalid but `publish()` refuses it at serialization time. |

Closed suite set: `throughput`, `ttft`, `tool-calling`, `code-accuracy`,
`framework-eval`, `capability-comparison`, `coding`, `reasoning`,
`knowledge`, `evalplus`, `math-hard`. Adding a suite means editing
`schema.json` and filing a schema update PR.

## Optional top-level fields

| Field | Type | Added when |
| --- | --- | --- |
| `pr_number` | integer \| null | `trigger == "pr"` |
| `env_class` | `isolated \| under-load` | Machine load class during the run (verdict-policy gate 3). |
| `concurrency` | integer (≥1) | The run drove more than a token's worth of parallelism (in-flight request count). |
| `reasoning_effort` | string, free-form | `--reasoning-effort` declared the thinking level the run asked for. See below. |
| `serving` | object | Inference-server identity: `stack` / `endpoint_port` / `served_model` (all optional). |
| `model_revision` | string | Model provides HF revision or commit SHA. |
| `quantization` | string | Runtime reports it (e.g. `mlx-4bit`, `mxfp4`). |
| `skipped` | boolean | Suite intentionally skipped (CI without hardware). |
| `seed` | integer | Seeded generation. |
| `gen_kwargs` | object | `max_gen_toks` / `temperature` / `top_p` / `top_k`. |
| `memory_snapshots` | array | Future work — RSS / swap per phase. |
| `errors` | array of string | Non-fatal warnings recorded during the run. |
| `campaign` | object | Immutable `id`, `cell_id`, and serving `profile` for a coordinated campaign. |
| `cell_status` | enum | Only `success` rows may be scored; all other listed states preserve a non-scored outcome. |
| `context` | object | Context dimensions: model/catalog/proxy/worker maxima when known, selected window, requested and actual prompt, and output reservation. |
| `readiness` | object | First request, excluded from warmed scoring. Stores initial residency and discarded timings. |

### `reasoning_effort`

An arm is weights *plus* quant *plus* effort *plus* serving config, so two
efforts are two arms and a row without one cannot be placed against another.

Stored verbatim, and deliberately not an enum: each model family names its own
levels, so rewriting `on` into `high` merges arms that are not the same arm.

Declared by the caller, never inferred. Effort lives in the agent CLI's config
or the serving default, and a harness can see neither. Where nothing was
declared the value is `unstated` — **absent is not `off`**, and a plausible
default here would be indistinguishable from a measurement.

## `system` object

Required: `os`, `chip`, `memory_gb`.

Optional (all populated automatically by `detect_system()`):
`hostname` (short host label — distinguishes machines with identical
chip/memory, e.g. a Mac Studio vs a MacBook Pro), `python_version`,
`mlx_version`, `mlx_lm_version`, `lm_eval_version`, `kernel`,
`runner` (for GitHub Actions), `vllm_mlx_version`.

`topology` (object, multi-node runs only): `world_size`, `parallelism`
(`pipeline` / `tensor` / `none`), `interconnect` (e.g. `tb5-rdma`), and
`nodes[]` (`hostname` / `chip` / `memory_gb` per node). Populated from
`MLX_BENCH_WORLD_SIZE` / `MLX_BENCH_PARALLELISM` / `MLX_BENCH_INTERCONNECT` /
`MLX_BENCH_NODES` (JSON array); absent for single-node runs.

## `results[]` items

| Field | Type | Notes |
| --- | --- | --- |
| `name` | string (required) | Task or measurement ID (`gsm8k_cot_zeroshot`, `tok_per_sec_512`). |
| `metric` | string (required) | Display metric name (`exact_match_flexible`, `pass_at_1`, `throughput`). |
| `value` | number (required) | The measurement. |
| `unit` | string (required) | `ratio`, `tok/s`, `ms`, etc. |
| `duration_seconds` | number | Wall-clock for this measurement (first-class replacement for `tags.total_eval_time_s`). |
| `prompt_tokens_per_second` | number | Aggregate prefill throughput (prompt tokens / `duration_seconds`). Supporting detail. |
| `decode_tokens_per_second` | number | Aggregate decode-only throughput (completion tokens / `duration_seconds`). Supporting detail — see below. |
| `total_tokens_per_second` | number | **Headline throughput metric.** Cumulative (prompt + completion) tokens / `duration_seconds`. |
| `first_token_latency_ms` | number | Time to first token, when measurable (streaming-aware harnesses). |
| `peak_rss_mb` | number | Peak RSS observed during this result, when available at per-result granularity. |
| `tags` | object\[string\] | Free-form string key-value metadata. |
| `raw` | any | Original untransformed tool output (optional archive). |

### Headline throughput metric: `total_tokens_per_second`

As of 2026-07-27, `total_tokens_per_second` — cumulative (prompt + completion)
tokens divided by wall-clock duration — is the **primary** throughput number
to report and compare, not `decode_tokens_per_second`. A decode-only rate
hides prefill-engine improvements entirely, even though a faster prefill is a
real, felt latency win for any caller sending a non-trivial prompt: two models
with identical decode speed but a 4-6x prefill gap are not equivalent in
practice, and a decode-only headline reports them as if they were.
`decode_tokens_per_second` and `prompt_tokens_per_second` are kept as
supporting detail (useful for root-causing *why* the cumulative number moved)
— this is purely a description/policy change, no schema fields were added,
removed, or renamed, so older published rows remain fully valid and
comparable.

## Validation

Every envelope is validated inside `mlx_benchmarks.publish.publish()` —
bypass only by passing `validate=False` (strongly discouraged). The
validator collects *all* errors before raising, so a single run-through
surfaces everything wrong instead of one-at-a-time iteration.

Locally:

```python
from mlx_benchmarks.envelope import validate_envelope

# Raises EnvelopeValidationError with every problem
validate_envelope(my_envelope)
```

## Versioning

- Adding an optional field: non-breaking, no version bump.
- Adding an enum value to `suite`: non-breaking (downstream just ignores
  unknown suites in current viewers), but please file as a `feat:` PR.
- Removing or renaming a field, changing a type, tightening validation,
  changing the filename pattern: breaking. Bump `schema_version` to `"2"`
  and update `$id` accordingly.

Current `$id`: `https://github.com/JacobPEvans/mlx-benchmarks/schema/v1.json`.
