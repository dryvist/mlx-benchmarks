#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["httpx>=0.28.1"]
# ///
"""promptstack — system prompt as the independent variable.

Every other suite in this repo holds the prompt fixed and varies the model.
promptstack holds the model fixed and varies the *system prompt*:
``base_plus_variant`` vs. a surface's ``current`` prompt, scored across four
deterministic probe classes (reasoning, tool_call, instruction, homelab_qa).
This is what decides whether the shared base prompt + surface variant gets
adopted for a given surface (see ``docs/promptstack.md`` for the adoption
rule).

Run (never against a busy Studio without asking)::

    uv run harness/promptstack/run.py --base-url http://localhost:11434/v1 \\
        --model mlx-community/Qwen3.6-35B-A3B-4bit \\
        --prompt-set configs/promptstack/prompts/ \\
        --probe-bank configs/promptstack/probes/ \\
        --surface hermes

Output is one raw-results JSON; publish it with
``mlx-bench-publish <out.json> --kind promptstack --suite promptstack``.

Scoring lives in importable pure functions (``score_reasoning``,
``score_tool_call``, ``score_instruction``, ``score_homelab_qa``, ...) so
``tests/test_promptstack_runner.py`` can exercise them without a live
endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import re
import statistics
import string
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROBE_CLASSES: tuple[str, ...] = ("reasoning", "tool_call", "instruction", "homelab_qa")

# ---------------------------------------------------------------------------
# Probe bank loading
# ---------------------------------------------------------------------------


def load_probe_bank(probe_dir: Path, probe_class: str) -> dict[str, Any]:
    """Load one probe class's frozen task bank from ``<probe_dir>/<probe_class>.json``."""
    return json.loads((probe_dir / f"{probe_class}.json").read_text())


def load_prompt(prompt_set_dir: Path, name: str) -> str:
    """Load a composed system-prompt text file (``<name>.txt``)."""
    return (prompt_set_dir / f"{name}.txt").read_text()


# ---------------------------------------------------------------------------
# Deterministic scoring — one pure function per probe class
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def score_reasoning(task: Mapping[str, Any], response_text: str) -> bool:
    """Numeric-verify: last number in the response must equal the task's answer."""
    matches = _NUMBER_RE.findall(response_text)
    if not matches:
        return False
    got = matches[-1].replace(",", "")
    want = str(task["answer"]).replace(",", "")
    try:
        return float(got) == float(want)
    except ValueError:
        return got == want


def score_tool_call(task: Mapping[str, Any], tool_calls: Sequence[Mapping[str, Any]] | None) -> bool:
    """Positive tasks: a valid, correctly-named, fully-argued call. Negative tasks: no fabricated call."""
    calls = tool_calls or []
    if task["kind"] == "negative":
        # Correct behavior is to refuse or ask — never fabricate a call.
        return len(calls) == 0
    if not calls:
        return False
    call = calls[0]
    fn = call.get("function") or {}
    if fn.get("name") != task["expected_tool"]:
        return False
    try:
        args = json.loads(fn.get("arguments") or "")
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(args, dict):
        return False
    return all(key in args and args[key] is not None for key in task["expected_required"])


def _check_constraint(constraint: Mapping[str, Any], text: str) -> bool:
    ctype = constraint["type"]
    stripped = text.strip()
    if ctype == "word_count":
        return len(stripped.split()) == constraint["n"]
    if ctype == "lowercase_only":
        return stripped == stripped.lower()
    if ctype == "no_punctuation":
        return not any(ch in string.punctuation for ch in stripped)
    if ctype == "line_count":
        lines = [line for line in stripped.splitlines() if line.strip()]
        return len(lines) == constraint["n"]
    if ctype == "no_numbering_prefix":
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        return not any(re.match(r"^(\d+[.)]|[-*•])\s", line) for line in lines)
    if ctype == "exact_match_ci":
        return stripped.lower() in {v.lower() for v in constraint["values"]}
    raise ValueError(f"unknown constraint type: {ctype!r}")


def score_instruction(task: Mapping[str, Any], response_text: str) -> tuple[int, int]:
    """Return (constraints_met, total_constraints)."""
    constraints = task["constraints"]
    met = sum(1 for c in constraints if _check_constraint(c, response_text))
    return met, len(constraints)


def score_homelab_qa(task: Mapping[str, Any], response_text: str) -> tuple[bool, bool]:
    """Return (correct_and_cited, unsupported_claim_present)."""
    lowered = response_text.lower()
    correct = all(fact.lower() in lowered for fact in task["expected_facts"])
    unsupported = any(term.lower() in lowered for term in task.get("forbidden_terms", []))
    return correct, unsupported


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile; single sorted pass, no numpy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1))))
    return ordered[rank]


def summarize_probe_class(probe_class: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-request records for one probe class into its metric block."""
    n = len(records)
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]
    prompt_toks = [r["prompt_tokens"] for r in records if r.get("prompt_tokens") is not None]
    completion_toks = [r["completion_tokens"] for r in records if r.get("completion_tokens") is not None]

    block: dict[str, Any] = {
        "probe_class": probe_class,
        "n_tasks": n,
        "task_success_rate": (sum(1 for r in records if r["success"]) / n) if n else 0.0,
        "tokens_prompt_avg": round(statistics.mean(prompt_toks), 1) if prompt_toks else None,
        "tokens_completion_avg": round(statistics.mean(completion_toks), 1) if completion_toks else None,
        "latency_p50_ms": round(percentile(latencies, 50), 1),
        "latency_p95_ms": round(percentile(latencies, 95), 1),
    }
    if probe_class == "tool_call":
        block["valid_tool_call_rate"] = block["task_success_rate"]
        neg = [r for r in records if r.get("kind") == "negative"]
        if neg:
            block["unsupported_claim_rate"] = sum(1 for r in neg if not r["success"]) / len(neg)
    if probe_class == "instruction":
        adherence = [
            r["constraints_met"] / r["constraints_total"] for r in records if r.get("constraints_total")
        ]
        block["instruction_adherence_rate"] = round(statistics.mean(adherence), 4) if adherence else 0.0
    if probe_class == "homelab_qa":
        block["unsupported_claim_rate"] = (
            sum(1 for r in records if r.get("unsupported_claim")) / n if n else 0.0
        )
    return block


def cell_name(prompt_variant: str, thinking: str) -> str:
    """Cell identity, and a public join key (``--cells``, published ``tags.cell``),
    so ``on``/``off`` keep the exact names they have always produced."""
    return f"variant-{prompt_variant}_think-{thinking}"


# ---------------------------------------------------------------------------
# Endpoint I/O (httpx imported lazily — the module stays importable in the
# test env, where only the pure functions above are exercised)
# ---------------------------------------------------------------------------


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


async def one_request(
    client: Any,
    args: argparse.Namespace,
    system_prompt: str,
    user_prompt: str,
    thinking: str,
    tools: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    body: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        **thinking_body_kwargs(args.thinking_kwarg, thinking),
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    record: dict[str, Any] = {
        "content": "",
        "tool_calls": None,
        "latency_ms": None,
        "prompt_tokens": None,
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
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        record["content"] = message.get("content") or ""
        record["tool_calls"] = message.get("tool_calls")
        usage = data.get("usage") or {}
        record["prompt_tokens"] = usage.get("prompt_tokens")
        record["completion_tokens"] = usage.get("completion_tokens")
    except Exception as exc:  # broad on purpose: any network/timeout error classifies as a scored failure
        record["error"] = str(exc)
    record["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
    return record


async def run_probe_class(
    client: Any,
    args: argparse.Namespace,
    probe_dir: Path,
    probe_class: str,
    system_prompt: str,
    thinking: str,
) -> dict[str, Any]:
    bank = load_probe_bank(probe_dir, probe_class)
    tools = bank.get("tools")
    records: list[dict[str, Any]] = []
    for task in bank["tasks"]:
        for _ in range(args.repeats):
            if probe_class == "homelab_qa":
                user_prompt = f"{task['bundle']}\n\n{task['prompt']}"
            else:
                user_prompt = task["prompt"]
            req = await one_request(client, args, system_prompt, user_prompt, thinking, tools)
            record: dict[str, Any] = {
                "task_id": task["id"],
                "latency_ms": req["latency_ms"],
                "prompt_tokens": req["prompt_tokens"],
                "completion_tokens": req["completion_tokens"],
            }
            if req.get("error") or req.get("http_status") != 200:
                # A network/timeout error or non-200 response is a scored
                # failure — never let it fall through to a scorer that reads
                # None as "no output" and mistakes that for a passing probe
                # (e.g. score_tool_call treats tool_calls=None as an empty
                # call list, which a negative tool-call task would count as
                # a correct refusal).
                record["success"] = False
                if probe_class == "tool_call":
                    record["kind"] = task["kind"]
            elif probe_class == "reasoning":
                record["success"] = score_reasoning(task, req["content"])
            elif probe_class == "tool_call":
                record["kind"] = task["kind"]
                record["success"] = score_tool_call(task, req["tool_calls"])
            elif probe_class == "instruction":
                met, total = score_instruction(task, req["content"])
                record["constraints_met"] = met
                record["constraints_total"] = total
                record["success"] = met == total
            elif probe_class == "homelab_qa":
                correct, unsupported = score_homelab_qa(task, req["content"])
                record["success"] = correct
                record["unsupported_claim"] = unsupported
            records.append(record)
    return summarize_probe_class(probe_class, records)


async def run_cell(
    client: Any,
    args: argparse.Namespace,
    probe_dir: Path,
    prompt_variant: str,
    system_prompt: str,
    thinking: str,
) -> dict[str, Any]:
    probe_results = [
        await run_probe_class(client, args, probe_dir, probe_class, system_prompt, thinking)
        for probe_class in PROBE_CLASSES
        if _selected(f"{cell_name(prompt_variant, thinking)}_{probe_class}", args.cells)
    ]
    return {
        "name": cell_name(prompt_variant, thinking),
        "prompt_variant": prompt_variant,
        "thinking": thinking,
        "probe_results": probe_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="promptstack — system prompt as the independent variable")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible /v1 base URL")
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="NAME of the env var holding the API key (never pass a literal key)",
    )
    parser.add_argument("--model", required=True, help="Model id as served by the endpoint")
    parser.add_argument("--prompt-set", type=Path, required=True, help="Dir of <surface>.txt prompt files")
    parser.add_argument("--probe-bank", type=Path, required=True, help="Dir of <probe_class>.json banks")
    parser.add_argument("--surface", required=True, help="Prompt-set base name, e.g. 'hermes'")
    parser.add_argument(
        "--cells", help="Comma-separated substrings; only cell_probe-class combos matching one run"
    )
    parser.add_argument("--repeats", type=int, default=10, help="Repeats per probe task")
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
    parser.add_argument("--timeout", type=float, default=120, help="Per-request timeout, seconds")
    parser.add_argument(
        "--output", type=Path, help="Results JSON path (default: promptstack_results_<ts>.json)"
    )
    return parser


def _selected(name: str, cells_filter: str | None) -> bool:
    if not cells_filter:
        return True
    return any(part.strip() and part.strip() in name for part in cells_filter.split(","))


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    import httpx

    # Verbatim, never coerced: `== "on"` turned every unrecognised level into
    # False, so a graded sweep silently ran one cell twice under one name.
    thinking_modes = [m.strip() for m in args.thinking.split(",") if m.strip()]
    variants = {
        "base_plus_variant": load_prompt(args.prompt_set, args.surface),
        "current": load_prompt(args.prompt_set, f"current-{args.surface}"),
    }

    headers = {}
    api_key = os.environ.get(args.api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # httpx resolves a relative request path against base_url only when base_url
    # ends with "/" — otherwise it drops base_url's last path segment (e.g. a
    # "/v1" prefix). Keep the trailing slash here and post a relative path below.
    base_url = args.base_url if args.base_url.endswith("/") else args.base_url + "/"
    cells: list[dict[str, Any]] = []
    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=args.timeout) as client:
        for prompt_variant, system_prompt in variants.items():
            for thinking in thinking_modes:
                name = cell_name(prompt_variant, thinking)
                print(f"cell {name} ...", file=sys.stderr)
                cells.append(
                    await run_cell(client, args, args.probe_bank, prompt_variant, system_prompt, thinking)
                )

    return {
        "benchmark": "promptstack",
        "model": args.model,
        "surface": args.surface,
        "timestamp": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "surface": args.surface,
            "repeats": args.repeats,
            "thinking": list(thinking_modes),
            "thinking_kwarg": args.thinking_kwarg,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "probe_bank_version": "1",
        },
        "cells": cells,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = asyncio.run(_run(args))
    ts = results["timestamp"].replace(":", "").replace("-", "")
    out: Path = args.output or Path(f"promptstack_results_{ts}.json")
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
