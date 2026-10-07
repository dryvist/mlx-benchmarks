---
name: publish-results
description: Use when you have a raw suite results JSON to turn into a published dataset shard (convert, dry-run, publish with the write token), or when RANKINGS.md needs updating after a publish. Covers the --kind/--suite selection and the mandatory same-PR RANKINGS rule.
---

# Publish results

Turns a raw suite JSON into a validated Parquet shard on the
`JacobPEvans/mlx-benchmarks` dataset. Canonical flow:
[RUNBOOK Step 5](../../../docs/RUNBOOK.md#step-5--publish-each-shard).

## Steps

1. **Dry-run first — always.** Validates against `schema.json` and plans the
   shard path; no network:

   ```bash
   .venv/bin/mlx-bench-publish <results.json> --kind <lm-eval|agentic|promptstack|throughput-probe|llamacpp-server|vllm> --suite <suite> --dry-run
   ```

   `--kind` selects the converter; `--suite` must be in the schema enum. The
   envelope always validates inside `publish()` — never pass `--no-validate` on
   a real run.

2. **Publish with the write token.** The ambient `HF_TOKEN` is read-only;
   publishing needs the write token, `HF_WRITE_TOKEN`:

   ```bash
   HF_TOKEN="$HF_WRITE_TOKEN" \
     .venv/bin/mlx-bench-publish <results.json> --kind <...> --suite <...> --hostname <host>
   ```

   Optional run-context flags — `--env-class`, `--concurrency`, `--serving-*` —
   are documented in [`docs/schema.md`](../../../docs/schema.md).

3. **Ranking duty (same PR).** After every publish, update the model's row in
   [`RANKINGS.md`](../../../RANKINGS.md), pulling numbers back from the dataset
   (loop in [Keeping this page current](../../../RANKINGS.md#keeping-this-page-current)).
   The page must never drift from the dataset.

## Never

Never discard a completed run — publish it with `tags.caveat=...` and file an
issue instead of throwing away benchmark time.
