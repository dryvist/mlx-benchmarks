# Worked example — benchmarking on a Mac Studio

A complete run on the **Mac Studio** for a model that is **not** in the
`llama-swap` config and needs a **managed window** — production Hermes serving
goes offline for the duration. Follows
[`../docs/RUNBOOK.md`](../docs/RUNBOOK.md). Substitute your own model id.

Scenario: benchmark `mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit` (~45 GB)
solo, full agentic grid, then restore serving.

```sh
MODEL="mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit"
SLUG="qwen3-next-80b-thinking-4bit"
```

## 0. Notify, then open the managed window

Production Hermes is about to go down. **Tell the user first.** Confirm no bench
run is already in flight (one actor per host):

```sh
pgrep -fl 'vllm-mlx|mlx-eval|mlx-bench|run.py' || echo "host idle — safe"
```

Stop the production serving LaunchAgent:

```sh
launchctl bootout gui/501/dev.vllm-mlx.server
```

## 1. Serve the model solo

Pick parser flags from the [parser map](../docs/benchmark-traps.md#parser-map). Qwen3-Next
is a hybrid-attention family: `hermes` tool parser, `qwen3` reasoning parser,
**no** speculative decoding/MTP, no prefix cache. HF auth may be unset on the
Studio; a model that needs downloading reads `HF_TOKEN` (cache lives on
`/Volumes/HuggingFace`).

```sh
vllm-mlx serve "$MODEL" \
  --port 11434 \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.80 \
  --timeout 3600
```

Wait for readiness, then confirm over **IPv4** (caddy holds the same port on
IPv6/TLS — always `-4`/`127.0.0.1`):

```sh
curl -s4 http://127.0.0.1:11434/v1/models | grep -o '"id":"[^"]*"'
```

## 2. Run the full agentic grid (~30 min)

```sh
uv run harness/agentic/run.py \
  --base-url http://127.0.0.1:11434/v1 \
  --api-key-env OPENAI_API_KEY \
  --model "$MODEL" \
  --output ~/bench-runs/agentic_${SLUG}.json
```

The runner drives both thinking tracks and the two 20-round multi-turn tracks.
Judge at `conc4 / thinking-on / large-ctx` and record `first_degraded_round` for
each track.

## 3. Restore production serving

As soon as the run's output JSON is written, bring Hermes back:

```sh
launchctl bootstrap gui/501 ~/Library/LaunchAgents/dev.vllm-mlx.server.plist
curl -s4 http://127.0.0.1:11434/v1/models >/dev/null && echo "serving restored"
```

Publishing does not need the model resident, so do it after restoring — keep the
window as short as the run itself.

## 4. Publish and rank

The public dataset omits host identifiers:

```sh
.venv/bin/mlx-bench-publish ~/bench-runs/agentic_${SLUG}.json \
  --kind agentic --suite tool-calling \
  --published-metadata ~/bench-runs/published-metadata.json --dry-run

HF_TOKEN="$HF_WRITE_TOKEN" \
  .venv/bin/mlx-bench-publish ~/bench-runs/agentic_${SLUG}.json \
  --kind agentic --suite tool-calling \
  --published-metadata ~/bench-runs/published-metadata.json
```

Then update the model's row in [`../RANKINGS.md`](../RANKINGS.md) in the same PR.

---

For a chained multi-suite Studio run, wrap each suite in `nohup` with a
`===== $(date -u) START/DONE $MODEL =====` log marker and monitor those lines —
see [trap 10](../docs/benchmark-traps.md#trap-10-run-hygiene).
