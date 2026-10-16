# Serving knob catalog

[`knob-catalog.yaml`](../knob-catalog.yaml) records Wave 2/3 workload cases and
engine settings. Each entry has a stable `case_id`, the matching engine value,
the settings to record in `campaign_dimensions.run.knobs`, and per-knob metadata
(`name`, `flag_env`, `values_seen`, `default`, and `source_url`). A `null`
default means the cited source did not document one. The shared section uses
`engine: "*"` for workload cases that apply to any engine.

Store the selected `case_id` in `campaign_dimensions.run.knob_case`, the
canonical engine name in `campaign_dimensions.software.engine`, and the
observed settings in `campaign_dimensions.run.knobs`. The `deploy-space`
workflow uploads the catalog to the dataset repo root when the catalog changes
on `main`, and during its scheduled or manually dispatched runs.

## Find catalog cases absent from the published dataset

DuckDB's community YAML extension reads the catalog and its `hf://` support
reads the public dataset shards:

```sql
INSTALL yaml FROM community;
LOAD yaml;
WITH catalog AS (
  SELECT section, engine, case_id, knobs
  FROM read_yaml('hf://datasets/JacobPEvans/mlx-benchmarks/knob-catalog.yaml')
),
tested AS (
  SELECT lower(campaign_software_engine) AS engine,
         campaign_run_knob_case AS case_id,
         count(*) AS runs
  FROM read_parquet('hf://datasets/JacobPEvans/mlx-benchmarks/data/*.parquet')
  GROUP BY 1, 2
)
SELECT c.section, c.engine, c.case_id, c.knobs, coalesce(t.runs, 0) AS runs
FROM catalog c
LEFT JOIN tested t
  ON t.case_id = c.case_id
 AND (c.engine = '*' OR lower(c.engine) = t.engine)
WHERE coalesce(t.runs, 0) = 0
ORDER BY 1, 2, 3;
```

This reports missing case IDs. It does not establish that every individual knob
value in a case was measured; inspect `knob_catalog[].values_seen` and the
published `knobs` map for value-level coverage.

Reference: [DuckDB YAML extension](https://duckdb.org/community_extensions/extensions/yaml),
[Hugging Face DuckDB access](https://huggingface.co/docs/hub/main/datasets-duckdb).
