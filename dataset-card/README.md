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
these columns during schema normalization.

Canonical rows use hardware labels for machine identity. Original shards remain
at their original paths as historical source data.
