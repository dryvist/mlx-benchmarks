# Worked example — benchmarking on the MacBook

A complete, copy-pasteable run on the **MacBook workstation** for a model that
already has a `llama-swap` slot (no managed window needed). Follows
[`../docs/RUNBOOK.md`](../docs/RUNBOOK.md). Substitute your own model id; the
commands are otherwise real.

Scenario: benchmark `mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit` (~19.5 GB, fits
the workstation) for reasoning, math, and an agentic smoke.

```sh
MODEL="mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"
SLUG="qwen36-35b-optiq-4bit"
```

## 1. Confirm the served name and that the endpoint is up

```sh
curl -s http://localhost:11434/v1/models | grep -o '"id":"[^"]*"'
```

The model must appear in that list (it is a `llama-swap` slot). If it does not,
add it to the swap config out-of-band — do not edit the config mid-run.

## 2. Reasoning — quick indicative pass (~10 min)

The MacBook's `llama-swap` caps at `concurrencyLimit=2`, so **pin the eval
concurrency to 2** or lm-eval 0.4.11 crashes on a 429 burst
([trap](../docs/RUNBOOK.md#environments)).

```sh
MLX_EVAL_CONCURRENT=2 mlx-eval arc_challenge_chat \
  --limit 15 --output_path ./run-output/$SLUG
```

Publish (dry-run, then real — note the write token from `.env`):

```sh
.venv/bin/mlx-bench-publish ./run-output/$SLUG/results_*.json \
  --kind lm-eval --suite reasoning --dry-run

HF_TOKEN="$HF_WRITE_TOKEN" \
  .venv/bin/mlx-bench-publish ./run-output/$SLUG/results_*.json \
  --kind lm-eval --suite reasoning
```

## 3. Math — `minerva_math500` (~45 min)

```sh
MLX_EVAL_CONCURRENT=2 mlx-eval minerva_math500 --output_path ./run-output/$SLUG
```

Read the **`math_verify`** metric in the output, not `exact_match`. Publish with
`--suite math-hard`.

## 4. Agentic — smoke the pass-gate cell first (~minutes)

Before committing to the full grid, smoke the goal cell:

```sh
uv run harness/agentic/run.py \
  --base-url http://localhost:11434/v1 \
  --api-key-env OPENAI_API_KEY \
  --model "$MODEL" \
  --cells conc4_think-on_ctx-large_stream --repeats 3 \
  --output run-output/agentic_${SLUG}_smoke.json
```

If the smoke passes (100% valid, no `empty_function_name`), run the full grid —
both thinking tracks — then publish `--kind agentic --suite tool-calling`.

## 5. Update the ranking

Refresh the model's row in [`../RANKINGS.md`](../RANKINGS.md) with the numbers
pulled back from the dataset, in the same PR as the publish.

---

Everything ran against the workstation's own `llama-swap`; production serving on
the Studio was never touched.
