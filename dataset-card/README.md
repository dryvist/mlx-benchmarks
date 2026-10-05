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
    data_files:
      - split: train
        path: "data/run-canonical-*.parquet"
---

# MLX Benchmarks

Structured benchmark results for locally hosted language models. Results are
published as immutable Parquet shards by the
[mlx-benchmarks project](https://github.com/dryvist/mlx-benchmarks).

The default configuration reads schema-normalized shards. Optional result and
system fields use nulls when a run did not record them. Historical tag columns
remain available individually, the tags_json column retains arbitrary tags,
and extra_json retains historical columns outside the stable schema.

Original Parquet shards remain in the repository at their original paths. The
normalized shards used by the default configuration preserve their rows and add
null-valued columns where an original shard did not contain a field.
