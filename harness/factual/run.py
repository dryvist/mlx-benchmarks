#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["httpx>=0.28.1"]
# ///
"""factual — grounded-summary accuracy benchmark.

The agentic suite asks whether a model can *call* a tool. This one asks what it
does with what the tool returned: given a tool result whose contents are known,
write the digest a human would read, and get scored on whether every figure in
that digest actually came from the data.

Three deterministic scores per case, no model-graded judgment anywhere:

``fact_recall``
    Every string in the fixture's ``required_facts`` appears in the response.
``fabricated_numbers``
    Numbers in the response that appear neither in the evidence, nor in the
    prompt, nor in the fixture's ``allowed_derived`` list (sums and counts a
    correct summary may legitimately compute). Any residue is a fabrication.
    This is the metric the suite exists for.
``tool_syntax_leak``
    Raw tool-call syntax (``[Tool call: ...]``, ``<tool_call>``, harmony channel
    markers) surfacing in prose that should contain none — the mlx-lm #1011
    degradation mode seen from the reader's side.

Run (never against a busy Studio without asking)::

    uv run harness/factual/run.py --base-url http://127.0.0.1:11434/v1 \\
        --model mlx-community/Qwen3.6-35B-A3B-4bit \\
        --fixtures configs/factual/fixtures/homelab-digest.json

Output is one raw-results JSON; publish it with
``mlx-bench-publish <out.json> --kind factual --suite grounded-summary``.

Scoring lives in importable pure functions (``fabricated_numbers``,
``score_case``, ...) so ``tests/test_factual_runner.py`` exercises them
without a live endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import re
import statistics
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# Maximal digit run with optional comma grouping and decimal part. Deliberately
# blind to surrounding punctuation: "2026-07-23T02:30" yields 2026/07/23/02/30
# and "203.0.113.44" yields 203.0/113.44. Being blind is the point — the SAME
# extraction runs over the evidence and over the response, so a token can only
# be called fabricated when it is absent from the evidence under identical
# rules. A stricter regex would flag "July 23" against an ISO-dated evidence
# bundle, which is a formatting difference, not a fabrication.
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

# Raw tool-call syntax that must never reach prose. Sources: mlx-lm #1011
# plain-text fallback, the OpenAI/Qwen/harmony wire formats, and the
# DeepSeek-style sentinel.
_LEAK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\[\s*tool\s*call\s*:", re.IGNORECASE),
    re.compile(r"</?tool_calls?>", re.IGNORECASE),
    re.compile(r"</?function_calls?>", re.IGNORECASE),
    re.compile(r"<\|tool[_▁]?call", re.IGNORECASE),
    re.compile(r"<\|(?:channel|start|constrain)\|>"),
    re.compile(r"\bfunctions\.[a-z_]+\s*\(", re.IGNORECASE),
)


# ---------------------------------------------------------------------------
# Pure functions (unit-tested via tests/test_factual_runner.py)
# ---------------------------------------------------------------------------


def normalize_number(token: str) -> str:
    """Canonicalize a numeric token so formatting differences never read as fabrication.

    ``"1,284"``, ``"1284"`` and ``"1284.0"`` all collapse to ``"1284"``; ``"07"``
    collapses to ``"7"`` so a zero-padded evidence field matches an unpadded
    response. Returns the token unchanged when it will not parse.
    """
    try:
        value = float(token.replace(",", ""))
    except ValueError:
        return token
    return str(int(value)) if value.is_integer() else str(value)


def number_set(text: str) -> set[str]:
    """All normalized numeric tokens appearing anywhere in ``text``."""
    return {normalize_number(m) for m in _NUMBER_RE.findall(text)}


def fabricated_numbers(
    response_text: str, evidence: str, prompt: str, allowed_derived: Iterable[str] = ()
) -> list[str]:
    """Numbers in the response that are not traceable to the inputs.

    Grounded sources, in order of how a correct summary gets a figure: the
    evidence bundle, the question itself, and the fixture's declared
    ``allowed_derived`` values (a total, a row count — figures a faithful
    summary computes rather than copies). Anything else was invented.
    """
    grounded = number_set(evidence) | number_set(prompt)
    for value in allowed_derived:
        grounded |= number_set(str(value))
    return sorted(number_set(response_text) - grounded)


def leaked_tool_syntax(response_text: str) -> bool:
    """True when raw tool-call syntax surfaced in prose."""
    return any(pattern.search(response_text) for pattern in _LEAK_PATTERNS)


def missing_facts(response_text: str, required_facts: Sequence[str]) -> list[str]:
    """Required facts absent from the response, compared case-insensitively.

    Numeric facts are compared through the normalized number set as well, so a
    response writing ``1,284`` satisfies a required fact of ``1284``.
    """
    lowered = response_text.lower()
    numbers = number_set(response_text)
    return [
        fact
        for fact in required_facts
        if fact.lower() not in lowered and normalize_number(fact) not in numbers
    ]


def score_case(case: Mapping[str, Any], response_text: str) -> dict[str, Any]:
    """Score one response against one fixture. Every check is deterministic."""
    absent = missing_facts(response_text, case.get("required_facts") or [])
    invented = fabricated_numbers(
        response_text,
        case.get("evidence", ""),
        case.get("prompt", ""),
        case.get("allowed_derived") or [],
    )
    lowered = response_text.lower()
    forbidden_hits = [f for f in case.get("forbidden_facts") or [] if f.lower() in lowered]
    leaked = leaked_tool_syntax(response_text)
    return {
        "case_id": case.get("id", ""),
        "missing_facts": absent,
        "fabricated_numbers": invented,
        "forbidden_hits": forbidden_hits,
        "tool_syntax_leak": leaked,
        # A case passes only on all four counts: it said everything it had to,
        # invented nothing, asserted nothing known-false, and stayed in prose.
        "success": not absent and not invented and not forbidden_hits and not leaked,
    }


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile; single sorted pass, no numpy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1))))
    return ordered[rank]


def summarize_cell(records: Sequence[Mapping[str, Any]], wall_seconds: float) -> dict[str, Any]:
    """Aggregate per-response records into the per-cell metric block."""
    n = len(records)
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]
    completion = [r["completion_tokens"] for r in records if r.get("completion_tokens") is not None]

    def rate(key: str, *, truthy: bool = True) -> float:
        """Share of records whose ``key`` is (non-)empty."""
        if not n:
            return 0.0
        return sum(1 for r in records if bool(r.get(key)) is truthy) / n

    return {
        "n_responses": n,
        "wall_seconds": round(wall_seconds, 3),
        "grounded_accuracy": rate("success"),
        "fact_recall_rate": rate("missing_facts", truthy=False),
        # The headline honesty metric: share of responses carrying >=1 number
        # that came from nowhere. Lower is better, so the shootout ranker
        # inverts it rather than sorting on it directly.
        "fabricated_number_rate": rate("fabricated_numbers"),
        "forbidden_claim_rate": rate("forbidden_hits"),
        "tool_syntax_leak_rate": rate("tool_syntax_leak"),
        "error_rate": rate("error"),
        "latency_p50_ms": round(percentile(latencies, 50), 1),
        "latency_p95_ms": round(percentile(latencies, 95), 1),
        "tokens_completion_avg": round(statistics.mean(completion), 1) if completion else None,
        "responses": list(records),
    }


def cell_name(thinking: str) -> str:
    """Cell identity, and a public join key (``--cells``, published ``tags.cell``),
    so ``on``/``off`` keep the exact names they have always produced."""
    return f"factual_think-{thinking}"


def thinking_body_kwargs(kwarg: str, level: str) -> dict[str, Any]:
    """``on``/``off`` send exactly the bodies they always have; any other level is
    passed through verbatim, so a graded sweep reaches the model as itself."""
    if kwarg == "reasoning_effort":
        # ponytail: harmony models can't fully disable reasoning; low is the off-analog
        if level in ("on", "off"):
            return {"reasoning_effort": "high" if level == "on" else "low"}
        return {"reasoning_effort": level}
    if level in ("on", "off"):
        return {"chat_template_kwargs": {kwarg: level == "on"}}
    return {"chat_template_kwargs": {kwarg: level}}


# ---------------------------------------------------------------------------
# Endpoint I/O (httpx imported lazily — the module stays importable in the
# test env, where only the pure functions above are exercised)
# ---------------------------------------------------------------------------


async def one_request(
    client: Any,
    args: argparse.Namespace,
    system_prompt: str,
    case: Mapping[str, Any],
    thinking: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{case['evidence']}\n\n{case['prompt']}"},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        **thinking_body_kwargs(args.thinking_kwarg, thinking),
    }
    if args.repetition_penalty is not None:
        body["repetition_penalty"] = args.repetition_penalty

    record: dict[str, Any] = {
        "content": "",
        "latency_ms": None,
        "completion_tokens": None,
        "http_status": None,
        "error": None,
    }
    start = time.monotonic()
    try:
        response = await client.post("chat/completions", json=body)
        record["http_status"] = response.status_code
        response.raise_for_status()
        data = response.json()
        message = (data.get("choices") or [{}])[0].get("message") or {}
        record["content"] = message.get("content") or ""
        record["completion_tokens"] = (data.get("usage") or {}).get("completion_tokens")
    except Exception as exc:  # broad on purpose: any network/timeout error is a scored failure
        # Never let a transport error fall through to the scorers: an empty
        # response would score as "no fabricated numbers" and would inflate the
        # honesty metric of a model that simply failed to answer.
        record["error"] = type(exc).__name__
    record["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
    return record


async def run_cell(
    client: Any,
    args: argparse.Namespace,
    bank: Mapping[str, Any],
    thinking: str,
) -> dict[str, Any]:
    system_prompt = bank.get("system_prompt", "")
    cases = bank["cases"]

    # Untimed warm-up, excluded from every statistic — the first request after a
    # thinking switch pays prompt-cache-miss cost that would skew this cell.
    await one_request(client, args, system_prompt, cases[0], thinking)

    records: list[dict[str, Any]] = []
    start = time.monotonic()
    for case in cases:
        for _ in range(args.repeats):
            req = await one_request(client, args, system_prompt, case, thinking)
            if req["error"] or req["http_status"] != 200:
                record = {
                    "case_id": case["id"],
                    "missing_facts": list(case.get("required_facts") or []),
                    "fabricated_numbers": [],
                    "forbidden_hits": [],
                    "tool_syntax_leak": False,
                    "success": False,
                }
            else:
                record = score_case(case, req["content"])
            record |= {
                "latency_ms": req["latency_ms"],
                "completion_tokens": req["completion_tokens"],
                "http_status": req["http_status"],
                "error": req["error"],
            }
            records.append(record)
    wall = time.monotonic() - start
    return {"name": cell_name(thinking), "thinking": thinking, **summarize_cell(records, wall)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grounded-summary factual accuracy benchmark")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible /v1 base URL")
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="NAME of the env var holding the API key (never pass a literal key)",
    )
    parser.add_argument("--model", required=True, help="Model id as served by the endpoint")
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("configs/factual/fixtures/homelab-digest.json"),
        help="Fixture bank JSON",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Responses per case")
    parser.add_argument(
        "--thinking",
        default="on,off",
        help="Comma list of effort levels, sent verbatim. 'on'/'off' keep their "
        "per-family mapping and cell names; any other value is passed through",
    )
    parser.add_argument(
        "--thinking-kwarg",
        default="enable_thinking",
        help="chat_template_kwargs bool name, or 'reasoning_effort' for harmony models",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Send repetition_penalty in the request body (omitted if unset)",
    )
    parser.add_argument("--timeout", type=float, default=600, help="Per-request timeout, seconds")
    parser.add_argument("--output", type=Path, help="Results JSON path (default: factual_results_<ts>.json)")
    return parser


async def _run(args: argparse.Namespace, timestamp: str) -> dict[str, Any]:
    import httpx

    bank = json.loads(args.fixtures.read_text())
    # Verbatim, never coerced: `== "on"` turned every unrecognised level into
    # False, so a graded sweep silently ran one cell twice under one name.
    thinking_modes = [m.strip() for m in args.thinking.split(",") if m.strip()]

    headers = {}
    api_key = os.environ.get(args.api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # httpx resolves a relative request path against base_url only when base_url
    # ends with "/" — otherwise it drops base_url's last path segment (the "/v1"
    # prefix). Keep the trailing slash and post a relative path below.
    base_url = args.base_url if args.base_url.endswith("/") else args.base_url + "/"
    cells: list[dict[str, Any]] = []
    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=args.timeout) as client:
        for thinking in thinking_modes:
            print(f"cell {cell_name(thinking)} ...", file=sys.stderr)
            cells.append(await run_cell(client, args, bank, thinking))

    return {
        "benchmark": "factual",
        "model": args.model,
        "timestamp": timestamp,
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "fixtures": str(args.fixtures),
            "fixture_bank_version": str(bank.get("fixture_bank_version", "")),
            "n_cases": len(bank["cases"]),
            "repeats": args.repeats,
            "thinking": list(thinking_modes),
            "thinking_kwarg": args.thinking_kwarg,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
            "timeout": args.timeout,
        },
        "cells": cells,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    ts = timestamp.replace(":", "").replace("-", "")
    out: Path = args.output or Path(f"factual_results_{ts}.json")
    results = asyncio.run(_run(args, timestamp))
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
