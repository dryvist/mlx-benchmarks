# 2026-08-23 — jevans-mbp quick smokes: judge 4B + small 9B

Single-run smoke session on `jevans-mbp`, **under-load** class (`mlx_lm.server`
via llama-swap, conc cap 1). All shards published with
`caveat=quick-smoke-*` tags; maturity unchanged, no verdict claims.

## Results

| Model | ARC reasoning (`arc_challenge_chat_qwen3`, limit 15) | Agentic valid (conc1/think-on/small/stream ×5) |
| --- | --- | --- |
| Qwen3.5-9B-MLX-4bit | 93.3% | 100% (zero errors) |
| Qwen3-4B-Instruct-2507-4bit | 86.7% | 80% |

Agentic cell is the reduced matrix (no multi-turn track, repeats 5) — not
pass-gate comparable.

## Overlay fix rode along

First round scored flat 0.0 on both models with visibly correct answers in the
logged samples ("The best answer is C"). Root cause: stock `arc_challenge_chat`
relies on partial-assistant prefilling (`gen_prefix`) that chat-completions
cannot do; `remove_whitespace` never recovers the letter. The fix existed as an
unmerged commit since 2026-07-24 (`fix/arc-chat-gen-prefix`, `4f22fc2`);
cherry-picked into the results PR so the published shards' git SHA carries the
task definition they were scored against.

## Session notes

- A pi-coding-agent (screenpipe) held the 9B's single concurrency slot during
  early probes (HTTP 429); it cleared before the scored runs — zero recorded
  request errors in any kept shard.
- Publish needed the fine-grained write token (`HF_WRITE_TOKEN`); the plain
  `HF_TOKEN` is read-only and 403s.

## Cloud baselines (moved from RANKINGS.md, size budget)

`reasoning` suite (`arc`/`gsm8k`, limit 100) reference points, not local
candidates: `gemini-2.5-flash`, `openrouter/auto`, `openai/gpt-4.1-mini`.
