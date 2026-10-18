---
pretty_name: "MLX Benchmarks"
license: apache-2.0
language:
  - en
tags:
  - benchmark
  - evaluation
  - llm
  - apple-silicon
  - nvidia
  - throughput
  - latency
size_categories:
  - "n<10K"
configs:
  - config_name: default
    default: true
    data_files:
      - split: train
        path: "data/recent/latest.parquet"
  - config_name: history
    data_files:
      - split: train
        path: "data/run-canonical-*.parquet"
---

# MLX Benchmarks

Structured benchmark results for local hardware and hosted-provider language
model evaluations. Rows include both local-device runs and provider results;
recorded host and accelerator fields may be absent for provider rows. Results
are published as immutable Parquet shards by the
[mlx-benchmarks project](https://github.com/dryvist/mlx-benchmarks).

The default configuration opens `data/recent/latest.parquet`: the canonical
shards published within 30 days of the newest one, newest run first. The
publish workflow regenerates it from the canonical shards; it adds no rows of
its own. Select `history` to browse every canonical shard, including older
hosted-provider evaluations. Optional result and system fields use nulls when a
run did not record them. Historical tag columns remain available individually,
the tags_json column retains arbitrary tags, and extra_json retains historical
columns outside the stable schema.

Original Parquet shards remain in the repository at their original paths. The
normalized shards preserve their rows and add null-valued columns where an
original shard did not contain a field.

The public [run index](https://huggingface.co/datasets/JacobPEvans/mlx-benchmarks/blob/main/metadata/run-index-v1.json)
records evidence status and run-specific context for canonical shards. A
`scored` shard is eligible for scored comparisons; `experimental` shards are
retained for exploration but are not part of scored comparisons; `invalid`
shards failed the evidence checks described by their index caveats; and
`recovered` shards contain recovered results with their limitations recorded.
Invalid shards remain available in `history`, but are excluded from `default`
and hidden from the viewer.

## Model context limits

Max recommended context is the longest documented model limit. YaRN-extended context
is shown separately and requires engine support; runtime or memory can
impose a smaller limit. Native values come from `max_position_embeddings` in
linked configs or model cards. For conversions and quantizations, the base card
supplies limits when the derived card has none.

| Model | Native context | YaRN extension | Max rec. context | Context sources |
| --- | ---: | ---: | ---: | --- |
| [nvidia/Qwen3.8-27B-NVFP4][nvfp4-card] | 262,144 | Not documented | 262,144 | [Card: up to 262K][nvfp4-card]; [config][nvfp4-config] |
| [unsloth/Qwen3.8-27B-GGUF][unsloth-27b-card] | 262,144 | 1,000,000 | 1,000,000 | [card][unsloth-27b-card]; [config][unsloth-27b-config] |
| [mlx-community/Qwen3.8-27B-8bit][mlx-27b-card] | 262,144 | 1,000,000 | 1,000,000 | [base card][qwen-27b-card]; [config][mlx-27b-config] |
| [unsloth/Qwen3.8-Flash-Next-GGUF][unsloth-flash-card] | 262,144 | 1,000,000 | 1,000,000 | [card][unsloth-flash-card]; [config][flash-config] |
| [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF][ista-flash-card] | 262,144 | 1,000,000 | 1,000,000 | [Base card][flash-card]; [base config][flash-config] |
| [nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF][nemotron-card] | 262,144 | Not documented | 262,144 | [card][nemotron-card] |

[nvfp4-card]: https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4
[nvfp4-config]: https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/resolve/main/config.json
[unsloth-27b-card]: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
[unsloth-27b-config]: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/config.json
[mlx-27b-card]: https://huggingface.co/mlx-community/Qwen3.8-27B-8bit
[mlx-27b-config]: https://huggingface.co/mlx-community/Qwen3.8-27B-8bit/resolve/main/config.json
[qwen-27b-card]: https://huggingface.co/Qwen/Qwen3.8-27B
[unsloth-flash-card]: https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF
[ista-flash-card]: https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
[flash-card]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
[flash-config]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/config.json
[nemotron-card]: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF

## Quickstart

In the dataset SQL console, start with the measurements available for each
task and unit:

    SELECT
      suite,
      COALESCE(name, metric_name) AS task,
      COALESCE(metric, metric_metric) AS metric,
      COALESCE(unit, metric_unit) AS unit,
      COUNT(COALESCE(value, metric_value)) AS measured_rows
    FROM train
    WHERE COALESCE(value, metric_value) IS NOT NULL
    GROUP BY
      suite,
      COALESCE(name, metric_name),
      COALESCE(metric, metric_metric),
      COALESCE(unit, metric_unit)
    ORDER BY suite, task, metric, unit;

Inspect individual throughput runs with their recorded workload:

    SELECT
      timestamp,
      COALESCE(name, metric_name) AS task,
      COALESCE(metric, metric_metric) AS metric,
      model, quantization, chip, engine,
      concurrency AS agents, tag_context_tokens_actual AS context_tokens,
      tag_prompt_tokens AS prompt_tokens, tag_max_gen_toks AS output_budget_tokens,
      tag_truncated_rate,
      COALESCE(value, metric_value) AS value,
      COALESCE(unit, metric_unit) AS unit
    FROM train
    WHERE suite = 'throughput' AND COALESCE(value, metric_value) IS NOT NULL
    ORDER BY timestamp DESC
    LIMIT 50;

Keep the unit and workload fields with each result when comparing rows. A
null context, prompt, output budget, or completion field means that detail was
not recorded for that run; it is not a zero or a value shared with other runs.

## Result fields

The normalized default split uses the flat result columns **name**, **metric**,
**value**, and **unit**. In original legacy shards, the equivalent nested result
fields may appear as **metric_name**, **metric_metric**, **metric_value**, and
**metric_unit**. When combining downloaded current and legacy shards, coalesce
each pair and keep the corresponding unit:

    COALESCE(name, metric_name) AS name,
    COALESCE(metric, metric_metric) AS metric,
    COALESCE(value, metric_value) AS value,
    COALESCE(unit, metric_unit) AS unit

Optional system and workload columns include **machine label**, **chip**, **gpu**,
**engine**, **quantization**, **concurrency**, and projected **tag_*** fields
such as **tag_context_tokens_actual**, **tag_prompt_tokens**, and
**tag_max_gen_toks**. These fields are null when the run did not record or use
that dimension. **tags_json** retains arbitrary tags, and **extra_json** retains
historical fields outside the stable schema.

Detailed campaign envelopes add stable `campaign_<group>_<field>` columns for
hardware, software, model, run, speed, resource, quality, provenance, and cost
dimensions. `campaign_dimension_null_reasons_json` retains the reason codes for
explicit unknown or inapplicable values. Historical shards receive nulls in
new columns unless their run-index entry supplies a per-shard projection.

Cache dimensions describe the host memory limit, prefix caching, KV and weight
offload settings, cache workload shape, GPU KV capacity, cache hit rates,
occupancy, eviction delay, preemptions, host RAM peak, and first versus later
round TTFT. Memory settings use GiB or MiB as named; hit rates are fractions and
KV occupancy is a percentage. Fields that were unavailable or inapplicable are
null with a corresponding reason code.

Canonical rows use hardware labels for machine identity. Original shards remain
at their original paths as historical source data.
