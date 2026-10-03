# FAQ & Troubleshooting

Common questions and operational issues when working with mlx-benchmarks.

## Envelope / Schema

### Q: Why do I get `additionalProperties: false` validation errors?

**A:** The envelope schema is strict—it rejects unknown fields. Check your
envelope object for typos in field names. The authoritative fields are in
[`schema.json`](../schema.json) and documented in [`docs/schema.md`](schema.md).

### Q: What fields are required vs. optional in v1?

**A:** Required: `schema_version`, `timestamp`, `git_sha`, `trigger`, `suite`,
`model`, `system` (with `os`, `chip`, `memory_gb`), and `results` (non-empty array).

Optional: `pr_number`, `model_revision`, `quantization`, `seed`, `gen_kwargs`, `errors`, etc.
The publisher auto-populates optional fields like `python_version` and `mlx_version`
via `detect_system()`. See [`docs/schema.md`](schema.md) for the full table.

### Q: When do I bump `schema_version`?

**A:** Only on breaking changes: removing/renaming a field, changing a type, or
tightening validation. Adding optional fields does not require a bump. See
[`docs/schema.md`](schema.md) for the versioning policy.

## Inference / Performance

### Q: My benchmark runs very slowly. Is that expected?

**A:** Slow runs usually indicate memory pressure, suboptimal config, or a process
leak. Diagnostic order:

1. **Memory pressure:** Compare `footprint -p $(pgrep -f "vllm-mlx")` to the
   expected footprint for your model size.
2. **Config:** Check `--cache-memory-mb`, `--enable-prefix-cache`, and quantization
   settings against your model.
3. **Accumulated allocations:** See [`docs/quick-reset.md`](quick-reset.md) to
   reclaim memory without rebooting.

### Q: I get 404 or connection errors when running lm-eval. What is wrong?

**A:** The lm-eval configs expect `http://localhost:11434/v1/chat/completions`
(note the `/chat/completions` suffix). A common mistake is omitting that suffix.

Verify the endpoint is reachable: `curl http://localhost:11434/v1/models`

## Publishing

### Q: I published results and want to re-publish. What happens?

**A:** Filenames are content-addressed: `run-<timestamp>-<git_sha>-<suite>-<model_slug>.parquet`.
Re-publishing the same envelope overwrites the existing shard — no duplicates.
To correct a wrong model name, change the envelope so it produces a different filename.

### Q: HF upload fails with 401. What token scope do I need?

**A:** The publisher reads `HF_TOKEN`, which must have **write** scope on the
target dataset.

### Q: I published successfully but the viewer does not show my result. Why?

**A:** The Space viewer caches the dataset. Wait a few minutes, then refresh.
Confirm the shard landed by checking the HF dataset repo directly.

## CI

### Q: Pip-audit or OSV scanner flags a dependency I want to suppress. How?

**A:** Prefer fixing it — bump the dependency so the advisory clears. If it
genuinely can't be fixed yet, this repo inherits the org-wide ignore policy from
`dryvist/.github/osv-scanner.toml`. For a repo-specific suppression, create a
local `osv-scanner.toml` at the repo root (it takes precedence over the org
default):

```toml
[[IgnoredVulns]]
id = "GHSA-xxxx-xxxx-xxxx"
ignoreUntil = 2026-12-31
reason = "<justification, linking a tracking issue>"
```

`ignoreUntil` (an unquoted TOML date literal) forces the entry back through
review when the date passes.

---

Still stuck? Check [`CLAUDE.md`](../CLAUDE.md) for architecture notes,
[`CONTRIBUTING.md`](../CONTRIBUTING.md) for the dev workflow, or
[open an issue](https://github.com/JacobPEvans/mlx-benchmarks/issues).
