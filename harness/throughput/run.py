#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["httpx>=0.28.1"]
# ///
"""Measured throughput probe for an OpenAI-compatible endpoint.

Streams so time-to-first-token separates prefill from decode:
  prefill tok/s    = prompt_tokens / ttft
  decode  tok/s    = (completion_tokens - 1) / (total - ttft)
  cumulative tok/s = (prompt_tokens + completion_tokens) / total   <- HEADLINE

``cumulative_tok_s`` is the primary, consumer-facing number this probe
reports: decode-only throughput hides prefill-engine improvements entirely,
even though a faster prefill is a real, felt latency win for anyone sending
non-trivial prompts. Two models with identical decode speed but a 4-6x
prefill gap are *not* equivalent in practice, and a decode-only headline
metric reports them as if they were. ``prefill_tok_s`` and ``decode_tok_s``
are kept as supporting detail — useful for root-causing *why* the cumulative
number moved — but neither is the figure to lead with.

Run 1 of every sequence is a discarded readiness probe.  It records the first
request after the caller's declared model state (cold, resident, or unknown)
but is never folded into the warmed throughput statistics. Reports median +
min/max over the measured runs.

Output is one raw-results JSON. Publish it through the envelope pipeline with
``mlx-bench-publish --kind throughput-probe --suite throughput``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROMPT = (
    "You are reviewing a production incident. Write a clear, structured "
    "postmortem covering: what happened, the root cause, the blast radius, "
    "the fix that was applied, and three concrete follow-up actions. "
    "The incident: a two-node compute cluster failed to re-form after a "
    "network link was taken down for maintenance, and a resource guard "
    "latched, requiring a reboot. Be specific and thorough."
)


def prompt_for_context(context_tokens: int, variant: int = 0) -> str:
    if context_tokens < 1:
        return PROMPT
    return (
        "Read the following operational notes and then state their shared theme.\n\n"
        # The variant must appear in every repeated unit, not merely as a
        # suffix: llama-swap caches prompt prefixes, and a suffix would turn a
        # nominally cold long-context replicate into an almost-total cache hit.
        + (f"benchmark context variant-{variant} evidence " * context_tokens)
        + "\n\nWhat is the shared theme?"
    )


# Order matters: this is also the order fields are reported in, and the
# first entry is the headline metric.
_SUMMARY_KEYS = (
    "cumulative_tok_s",
    "decode_tok_s",
    "prefill_tok_s",
    "ttft_s",
    "total_s",
    # Numeric, so they aggregate like the rates. finish_reason is deliberately
    # NOT here — it is a string and would break median/min/max; it is reported
    # separately in summarize().
    "answer_chars",
    "reasoning_chars",
)


def cumulative_tok_s(
    prompt_tokens: int | None, completion_tokens: int | None, total_s: float
) -> float | None:
    """Headline throughput: (prompt + completion) tokens / wall-clock seconds.

    This is the consumer-visible measure — it counts prefill (prompt
    processing) and decode (completion generation) toward the same number,
    because both are real time a caller waits on. Returns ``None`` when the
    inputs can't produce a meaningful rate (missing token counts or
    non-positive duration).
    """
    if not prompt_tokens and not completion_tokens:
        return None
    if total_s <= 0:
        return None
    return round(((prompt_tokens or 0) + (completion_tokens or 0)) / total_s, 2)


async def one(
    client, url, model, prompt, max_tokens, think_kwarg, think_val, request_timeout_s
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if think_kwarg:
        body["chat_template_kwargs"] = {think_kwarg: think_val}

    t0 = time.perf_counter()
    try:
        return await _stream(client, url, body, t0, request_timeout_s)
    except Exception as e:  # server restart / disconnect mid-run
        return {"error": f"{type(e).__name__}: {e}"}


async def _stream(client, url, body, t0, request_timeout_s) -> dict[str, Any]:
    ttft = None
    usage = None
    ntok = 0
    # Separate answer from reasoning. Counting them together makes a model that
    # emits pure reasoning and zero answer produce a perfect throughput row —
    # the exact failure a throughput table cannot otherwise see. finish_reason
    # is captured for the same reason: a truncated stream and a completed one
    # are indistinguishable from the rates alone.
    answer_chars = 0
    reasoning_chars = 0
    finish_reason = None
    async with client.stream("POST", url, json=body, timeout=request_timeout_s) as r:
        status = r.status_code
        if status != 200:
            txt = await r.aread()
            return {"error": f"HTTP {status}: {txt[:300].decode(errors='replace')}"}
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]
                d = ch.get("delta") or {}
                content = d.get("content") or ""
                reasoning = (d.get("reasoning") or "") + (d.get("reasoning_content") or "")
                if content or reasoning:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    ntok += 1
                answer_chars += len(content)
                reasoning_chars += len(reasoning)
    total = time.perf_counter() - t0
    if ttft is None:
        return {"error": "no tokens streamed", "total_s": total, "usage": usage}
    ptok = (usage or {}).get("prompt_tokens")
    ctok = (usage or {}).get("completion_tokens") or ntok
    return {
        "ttft_s": round(ttft, 3),
        "total_s": round(total, 3),
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "cumulative_tok_s": cumulative_tok_s(ptok, ctok, total),
        "prefill_tok_s": round(ptok / ttft, 2) if ptok else None,
        "decode_tok_s": round((ctok - 1) / (total - ttft), 2) if ctok > 1 else None,
        "answer_chars": answer_chars,
        "reasoning_chars": reasoning_chars,
        "finish_reason": finish_reason,
    }


async def one_retry(
    client, url, model, prompt, max_tokens, think_kwarg, think_val, request_timeout_s, attempts=6, label=""
) -> dict[str, Any]:
    """Retry around a shared, actively-churned endpoint (429 / worker restart).

    A failed attempt is never scored; only a clean streamed run is returned.
    """
    # Never None: callers test `"error" in result`, and returning None there
    # raises TypeError instead of reporting the failure it was meant to carry.
    last: dict[str, Any] = {"error": f"no attempt made (attempts={attempts})"}
    for i in range(attempts):
        r = await one(client, url, model, prompt, max_tokens, think_kwarg, think_val, request_timeout_s)
        if "error" not in r:
            if i:
                r["retries"] = i
            return r
        last = r
        print(f"  {label}retry {i + 1}/{attempts}: {r['error'][:120]}", file=sys.stderr, flush=True)
        await asyncio.sleep(20)
    return last


def summarize(runs) -> dict[str, Any]:
    """Aggregate a sequence of per-run dicts into median/min/max per key.

    ``_SUMMARY_KEYS`` is ordered headline-first: ``cumulative_tok_s`` (the
    number to report) before the decode/prefill breakdown (why it moved).
    """
    ok = [r for r in runs if "error" not in r]
    if not ok:
        return {"n_ok": 0, "errors": [r.get("error") for r in runs]}
    # Annotated because this dict is deliberately heterogeneous: counts (int),
    # rates (nested dict), rates-as-fractions (float) and finish_reasons (list).
    # Without it the type is inferred as dict[str, int] from the first two keys
    # and every later assignment is flagged.
    out: dict[str, object] = {"n_ok": len(ok), "n_err": len(runs) - len(ok)}
    for k in _SUMMARY_KEYS:
        vals = [r[k] for r in ok if r.get(k) is not None]
        if vals:
            out[k] = {
                "median": round(statistics.median(vals), 2),
                "min": round(min(vals), 2),
                "max": round(max(vals), 2),
            }
    # The stuck-model guard. A model that emits only reasoning and never an
    # answer scores a perfect row on every rate above, because reasoning deltas
    # count as tokens exactly like answer deltas. answered_rate is the metric
    # that catches it: 1.0 means every run produced answer text, 0.0 means the
    # throughput figures describe a model that answered nothing.
    out["answered_rate"] = round(sum(1 for r in ok if (r.get("answer_chars") or 0) > 0) / len(ok), 3)
    stops = [r.get("finish_reason") for r in ok]
    out["finish_reasons"] = sorted({s for s in stops if s})
    # length = the generation hit max_tokens. On a thinking model that usually
    # means the answer was cut off, or never reached, rather than that the model
    # is slow. Treat any rate measured alongside it as suspect.
    out["truncated_rate"] = round(sum(1 for s in stops if s == "length") / len(ok), 3)
    errs = [r["error"] for r in runs if "error" in r]
    if errs:
        out["errors"] = errs
    return out


def context_validation(
    run: dict[str, Any],
    expected_prompt_tokens: int | None,
    tolerance_tokens: int,
    window_limit_tokens: int | None,
    max_tokens: int,
) -> str | None:
    actual = run.get("prompt_tokens")
    if not isinstance(actual, int):
        return "server did not report prompt_tokens"
    if expected_prompt_tokens is not None and abs(actual - expected_prompt_tokens) > tolerance_tokens:
        return f"actual prompt_tokens={actual} outside {expected_prompt_tokens}±{tolerance_tokens}"
    if window_limit_tokens is not None and actual + max_tokens > window_limit_tokens:
        return (
            f"prompt_tokens + output reservation ({actual}+{max_tokens}) exceeds window {window_limit_tokens}"
        )
    return None


def calibrated_repetitions(
    first_prompt_tokens: int, second_prompt_tokens: int, target_prompt_tokens: int
) -> int:
    tokens_per_repetition = second_prompt_tokens - first_prompt_tokens
    fixed_tokens = first_prompt_tokens - tokens_per_repetition
    if tokens_per_repetition < 1:
        raise ValueError("calibration did not increase prompt_tokens per repeated unit")
    repetitions = round((target_prompt_tokens - fixed_tokens) / tokens_per_repetition)
    if repetitions < 1:
        raise ValueError("target prompt tokens are below the fixed prompt overhead")
    return repetitions


async def main():
    import httpx

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument(
        "--context-tokens",
        type=int,
        default=0,
        help="synthetic long-context target; actual prompt tokens come from server usage",
    )
    ap.add_argument(
        "--expected-prompt-tokens",
        type=int,
        help="required actual server prompt-token count for a comparable campaign cell",
    )
    ap.add_argument(
        "--target-prompt-tokens",
        type=int,
        help="calibrate repeated context units from server usage to this actual prompt-token target",
    )
    ap.add_argument(
        "--prompt-tolerance-tokens",
        type=int,
        default=64,
        help="allowed absolute difference from --expected-prompt-tokens (default: 64)",
    )
    ap.add_argument(
        "--window-limit-tokens",
        type=int,
        help="configured total context admission limit; validates prompt plus output reservation",
    )
    ap.add_argument("--campaign-id", help="immutable campaign identifier")
    ap.add_argument("--cell-id", help="immutable campaign cell identifier")
    ap.add_argument("--profile", help="serving profile, for example base or mtp")
    ap.add_argument(
        "--initial-model-state",
        choices=("cold", "resident", "unknown"),
        default="unknown",
        help="model residency before the discarded readiness request (default: unknown)",
    )
    ap.add_argument("--repeats", type=int, default=4, help="measured runs (plus 1 warm-up)")
    ap.add_argument("--concurrency", type=int, default=4, help="parallel probe width")
    ap.add_argument(
        "--request-timeout-s",
        type=float,
        default=1800.0,
        help="per-request stream timeout in seconds; raise for intentionally serialized long-context probes",
    )
    ap.add_argument("--think-kwarg", default=None)
    ap.add_argument("--think", default="off", choices=["on", "off"])
    # Escape hatch from the on/off binary. Reasoning-effort vocabularies differ
    # per model family and are NOT interchangeable: gpt-oss takes low/medium/
    # high, while Qwen3.8 takes xhigh (its default)/medium/low and raises a
    # template exception on anything else — so the on->"high" mapping below is
    # silently invalid for that family. Pass the literal the model documents.
    ap.add_argument(
        "--think-value",
        default=None,
        help="Literal value for --think-kwarg, overriding the on/off mapping "
        "(e.g. 'low', 'medium', 'xhigh'). Use for models whose effort "
        "vocabulary is not low/high.",
    )
    ap.add_argument("--skip-concurrent", action="store_true")
    ap.add_argument(
        "--dedicated",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="declare that this arm was the ONLY consumer of its endpoint. Cannot "
        "be detected — the harness cannot see what else is on the box — so it is "
        "declared, and defaults to false because an unstated measurement "
        "environment is an untrusted one. Runs are comparable only within one "
        "value of this flag",
    )
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    url = a.base_url.rstrip("/") + "/chat/completions"
    if a.context_tokens < 0 or a.prompt_tolerance_tokens < 0:
        ap.error("--context-tokens must be >= 0")
    if a.expected_prompt_tokens is not None and a.expected_prompt_tokens < 0:
        ap.error("--expected-prompt-tokens must be >= 0")
    if a.target_prompt_tokens is not None and a.target_prompt_tokens < 1:
        ap.error("--target-prompt-tokens must be >= 1")
    if a.window_limit_tokens is not None and a.window_limit_tokens < 1:
        ap.error("--window-limit-tokens must be >= 1")
    if a.request_timeout_s <= 0:
        ap.error("--request-timeout-s must be > 0")
    think_val = a.think == "on"
    if a.think_kwarg == "reasoning_effort":
        think_val = "high" if a.think == "on" else "low"
    if a.think_value is not None:
        think_val = a.think_value

    context = {"output_reservation_tokens": a.max_tokens}
    if a.window_limit_tokens is not None:
        context["configured_window_tokens"] = a.window_limit_tokens
    expected_prompt_tokens = a.target_prompt_tokens or a.expected_prompt_tokens
    if expected_prompt_tokens is not None:
        context["requested_prompt_tokens"] = expected_prompt_tokens
    res = {
        "model": a.model,
        "base_url": a.base_url,
        "max_tokens": a.max_tokens,
        "context_tokens_target": a.context_tokens,
        "context_cache_busting": a.context_tokens > 0 or a.target_prompt_tokens is not None,
        "cell_status": "success",
        "context": context,
        "thinking": a.think,
        "think_kwarg": a.think_kwarg,
        # The value actually sent, not just the on/off switch. Without this the
        # recorded regime is wrong in the most misleading direction: a run that
        # passes no kwarg at all records thinking="off" while the model reasons
        # at ITS OWN default, which for Qwen3.8 is reasoning_effort=xhigh. A
        # measurement of maximum-effort reasoning was labelled "off".
        "think_value": think_val if a.think_kwarg else None,
        "think_kwarg_sent": bool(a.think_kwarg),
        # Declared, never inferred: the harness cannot see what else is on the
        # box. A throughput number from a shared endpoint measures the sharing,
        # not the model, so runs compare only within one value of this.
        "dedicated": a.dedicated,
        "readiness": {"initial_model_state": a.initial_model_state},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if a.campaign_id and a.cell_id and a.profile:
        res["campaign"] = {"id": a.campaign_id, "cell_id": a.cell_id, "profile": a.profile}

    async with httpx.AsyncClient() as client:
        context_repetitions = a.context_tokens
        if a.target_prompt_tokens is not None:
            calibration_runs = []
            for repetitions in (1, 2):
                calibration = await one_retry(
                    client,
                    url,
                    a.model,
                    prompt_for_context(repetitions, variant=-(repetitions + 1)),
                    1,
                    a.think_kwarg,
                    think_val,
                    a.request_timeout_s,
                    label=f"calibration[{repetitions}] ",
                )
                calibration_runs.append(calibration)
            res["context_calibration"] = calibration_runs
            try:
                first, second = (run["prompt_tokens"] for run in calibration_runs)
                context_repetitions = calibrated_repetitions(first, second, a.target_prompt_tokens)
            except (KeyError, TypeError, ValueError) as exc:
                res["cell_status"] = "aborted"
                res["aborted"] = f"context calibration failed: {exc}"
                Path(a.output).write_text(json.dumps(res, indent=2))
                return 1
            res["context_tokens_target"] = context_repetitions
            res["context"]["synthetic_repetitions"] = context_repetitions
        print("warm-up run (discarded)...", file=sys.stderr, flush=True)
        warm = await one_retry(
            client,
            url,
            a.model,
            prompt_for_context(context_repetitions, variant=0),
            a.max_tokens,
            a.think_kwarg,
            think_val,
            a.request_timeout_s,
            label="warmup ",
        )
        res["warmup"] = warm
        res["readiness"]["first_request"] = warm
        print(f"  warmup: {warm}", file=sys.stderr, flush=True)
        if "error" in warm:
            res["aborted"] = "warm-up failed; refusing to record scores"
            print(json.dumps(res, indent=2))
            Path(a.output).write_text(json.dumps(res, indent=2))
            return 1

        warm_context_error = context_validation(
            warm, expected_prompt_tokens, a.prompt_tolerance_tokens, a.window_limit_tokens, a.max_tokens
        )
        if warm_context_error:
            res["cell_status"] = "aborted"
            res["aborted"] = warm_context_error
            print(json.dumps(res, indent=2))
            Path(a.output).write_text(json.dumps(res, indent=2))
            return 1

        seq = []
        for i in range(a.repeats):
            r = await one_retry(
                client,
                url,
                a.model,
                prompt_for_context(context_repetitions, variant=i + 1),
                a.max_tokens,
                a.think_kwarg,
                think_val,
                a.request_timeout_s,
                label=f"seq[{i}] ",
            )
            print(f"  seq[{i}]: {r}", file=sys.stderr, flush=True)
            error = context_validation(
                r, expected_prompt_tokens, a.prompt_tolerance_tokens, a.window_limit_tokens, a.max_tokens
            )
            if error:
                r["error"] = error
            seq.append(r)
        res["sequential_runs"] = seq
        res["sequential"] = summarize(seq)
        actuals = [r["prompt_tokens"] for r in seq if isinstance(r.get("prompt_tokens"), int)]
        if actuals and len(set(actuals)) == 1:
            res["context"]["actual_prompt_tokens"] = actuals[0]
        elif actuals:
            res["cell_status"] = "aborted"
            res["aborted"] = f"prompt token counts varied across repeats: {sorted(set(actuals))}"

        if res.get("aborted"):
            res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            Path(a.output).write_text(json.dumps(res, indent=2))
            return 1

        if not a.skip_concurrent:
            print(f"concurrent probe x{a.concurrency}...", file=sys.stderr, flush=True)
            t0 = time.perf_counter()
            conc = await asyncio.gather(
                *[
                    one(
                        client,
                        url,
                        a.model,
                        prompt_for_context(context_repetitions, variant=a.repeats + 1 + i),
                        a.max_tokens,
                        a.think_kwarg,
                        think_val,
                        a.request_timeout_s,
                    )
                    for i in range(a.concurrency)
                ]
            )
            wall = time.perf_counter() - t0
            ok = [c for c in conc if "error" not in c]
            out_toks = sum(c.get("completion_tokens") or 0 for c in ok)
            in_toks = sum(c.get("prompt_tokens") or 0 for c in ok)
            res["concurrent_runs"] = conc
            res["concurrent"] = {
                "width": a.concurrency,
                "wall_s": round(wall, 2),
                "n_ok": len(ok),
                "n_err": len(conc) - len(ok),
                # Headline first: aggregate cumulative throughput across the
                # concurrent batch, then the decode-only figure as detail.
                "aggregate_cumulative_tok_s": round((in_toks + out_toks) / wall, 2) if wall else None,
                "aggregate_decode_tok_s": round(out_toks / wall, 2) if wall else None,
                "errors": [c["error"] for c in conc if "error" in c],
            }
            print(f"  concurrent: {res['concurrent']}", file=sys.stderr, flush=True)

    res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(a.output).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k in ("model", "sequential", "concurrent")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
