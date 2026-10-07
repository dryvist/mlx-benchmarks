# Envelope schema (v1)

Canonical JSON Schema: [`schema.json`](../schema.json). This file is a prose
walk-through. When the two disagree, `schema.json` wins — please open a PR.

Upstream Hub source URLs and the read date are recorded in
[`huggingface-metadata-sources.md`](huggingface-metadata-sources.md).

## Published score rows

The `definitions.published_result` schema is the contract for one dataset
score. It includes model identity and shape resolved from the Hub API, the
registered benchmark task, the HF `.eval_results` fields, and run variables.
The exact field names are written as top-level Parquet columns. Composite values
such as `dtype`, array-valued `base_model`, `hardware`, and
`dimension_null_reasons` use JSON encoding in their own columns; `tags` and
`architectures` are Parquet lists.
Every required field is present. A null requires a matching entry in
`dimension_null_reasons`; accepted codes are `not_applicable` and
`na_backfill`. `na_backfill` is valid only before the
`2026-10-07T04:15:00Z` cutover.

The raw benchmark record supplies `model_revision` as the 40-character Hub
commit SHA captured by the runner. The publisher reads model metadata at that
SHA and rejects evaluation benchmark IDs without a registered `eval.yaml`
task. The task's framework, config, and split must match the Hub definition.
Performance-only rows set all dataset fields to null with the explicit
`not_applicable` reason and do not emit an HF evaluation sidecar.

The one-time historical migration is exposed as
`mlx-bench-publish-backfill`. It defaults to dry-run, reads every Parquet shard,
and adds these columns to every historical row. Hub metadata is looked up only
when the row has a recorded model SHA. Missing historical facts are stored as
null with `na_backfill`; applying the planned rewrite requires the lead's
approval.

## Required top-level fields

| Field | Type | Notes |
| --- | --- | --- |
| `schema_version` | `"1"` | Bump only on breaking changes. |
| `timestamp` | ISO 8601 UTC | `YYYY-MM-DDTHH:MM:SSZ`; use start-of-run, not end. |
| `git_sha` | 7–64 hex or `null` | SHA of **this repo** at run time, not the model; use `null` only when the runtime code revision was not captured. |
| `trigger` | `local \| schedule \| pr \| workflow_dispatch` | How the run was kicked off. |
| `suite` | enum | Must be in the closed set below. |
| `model` | string | HF model ID (e.g. `mlx-community/Qwen3.5-9B-MLX-4bit`). |
| `system` | object | See below. `os`, `chip`, `memory_gb` keys required; explicit `null` preserves blocked or unmeasured runs. |
| `results` | array | Per-measurement rows. Empty array is not invalid but `publish()` refuses it at serialization time. |

Closed suite set: `throughput`, `ttft`, `tool-calling`, `code-accuracy`,
`framework-eval`, `capability-comparison`, `coding`, `reasoning`,
`knowledge`, `evalplus`, `math-hard`, `promptstack`, `grounded-summary`, plus the
four hardware baselines `gpu-burn`, `nvbandwidth`, `mbw` and `fio` (no model under
test: publish them with `--model hardware-baseline`). Adding a suite means editing
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
| `campaign_dimensions` | object | Optional typed fields in nine campaign groups; each leaf accepts explicit `null`. |
| `dimension_null_reasons` | object of string | Optional field-path-to-reason-code map for null values in `campaign_dimensions`. |
| `readiness` | object | First request, excluded from warmed scoring. Stores initial residency and discarded timings. |

### `campaign_dimensions`

The object is optional so existing v1 envelopes remain valid. Its optional
groups contain the dimension fields defined for each group in `schema.json`; every
leaf is optional and accepts an explicit `null` when a measurement is
unavailable or inapplicable. The publisher flattens each leaf into one stable
`campaign_<group>_<field>` Parquet column. Array-valued build architecture,
build-flag objects, and throttle-reason fields use a `_json` column suffix.
Older canonical shards are normalized with nulls for all new columns, so every
file has the same schema.

`dimension_null_reasons` maps paths such as `run.kv_cache_dtype` to a concise
reason code. Use a reason for each explicit null in newly enriched campaign
envelopes; a missing property remains distinct from an observed-but-unknown
value.

Converters keep both objects. `mlx-bench-publish --campaign-dimensions PATH`
reads a JSON file with a `campaign_dimensions` object and an optional
`dimension_null_reasons` object and copies them onto the envelope for every
`--kind`; `throughput-probe` also keeps them when its result file carries them.

- **hardware:** machine; accelerator model, memory, and bandwidth; host CPU, RAM, and speed;
  PCIe generation/width; power cap; UPS/circuit; chassis/container.
- **software:** OS; kernel; driver; accelerator runtime; engine/version/commit;
  backend; GPU architectures; build flags; flash attention.
- **model:** family/ID/HF repo/revision; total/active parameters; architecture;
  total/active experts; quantization; bits per weight; file size; license;
  native context; single-file SHA-256.
- **run:** allocated context; KV dtype; prompt/depth/output tokens; concurrency;
  batch/ubatch; parallel slots; prefix cache; speculative/MTP; draft model;
  temperature; thinking; chat template; seed; repeats; warm/cold state.
- **speed:** TTFT p50/p90/p99; prefill/decode/aggregate/total throughput; TPOT;
  ITL p50/p99; MTP acceptance; request success rate.
- **resource:** accelerator/host memory peak; CPU offload; average/max GPU utilization;
  accelerator/system/SoC/component power; energy per token; tokens per watt;
  temperature; GPU/CPU clocks; throttling; fan.
- **quality:** benchmark/version/subset/sample count; score; standard error and
  confidence interval; judge model; contamination note.
- **provenance:** UTC timestamp; run ID; config name/SHA; operator/agent; raw
  output path; schema version.
- **cost:** kWh and local electricity cost per million output tokens; median
  rental and API cost per million.

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
`hostname` (removed from the public Parquet projection), `python_version`,
`mlx_version`, `mlx_lm_version`, `lm_eval_version`, `kernel`,
`runner` (for GitHub Actions), `vllm_mlx_version`.

NVIDIA / CUDA hosts add four optional fields (absent on Apple Silicon runs):

| Field | Type | Notes |
| --- | --- | --- |
| `gpu` | object | `model` (driver-reported name), `vram_gb` (GiB, fractional allowed), `driver`, `cuda` (highest driver-supported version). All optional. |
| `engine` | object | `name` and `version` of the inference engine (e.g. `vllm` / `0.30.0`). Complements `serving.stack`, which names the endpoint. |
| `power_limit_w` | number (≥0) | Enforced board power limit in watts; two limits are two arms. |
| `container` | string | Image reference the engine ran in; absent for bare-metal runs. |

These are **declared, not probed**: `detect_system()` reads them from
`MLX_BENCH_GPU_MODEL`, `MLX_BENCH_GPU_VRAM_GB`, `MLX_BENCH_GPU_DRIVER`,
`MLX_BENCH_GPU_CUDA`, `MLX_BENCH_ENGINE_NAME`, `MLX_BENCH_ENGINE_VERSION`,
`MLX_BENCH_POWER_LIMIT_W` and `MLX_BENCH_CONTAINER`, because the publisher may run on a
different machine than the GPU. A variable that is unset, blank, or non-numeric where a
number is required is omitted. The `nvidia-smi` snippet that fills them is in
[`configs/LAYOUT.md`](../configs/LAYOUT.md#nvidia-hosts-configsnvidia). In the Parquet
shard `gpu` and `system_engine` ride as JSON-string columns (like `topology`); `power_limit_w`
and `container` are plain columns.

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
| `tags` | object\[string\] | Free-form string key-value metadata; numeric values are strings. Sweep keys are listed below. |
| `raw` | any | Original untransformed tool output (optional archive). |

Sweep metadata may include:

- `prompt_tokens`: tokens in each request prompt.
- `concurrency`: in-flight request count, matching the top-level field.
- `context_len`: configured server window capacity in tokens, not prompt length.

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

Every envelope and published score row is validated inside
`mlx_benchmarks.publish.publish()`. Validation cannot be bypassed. The
validators collect *all* errors before raising, so a single run-through
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
