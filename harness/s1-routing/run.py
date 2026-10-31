#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["httpx>=0.28.1"]
# ///
"""System 1 routing classifier benchmark.

Scoring functions are importable without a live endpoint. Per-row output is
safe to publish only because it carries ids, not task text.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import hashlib
import json
import math
import os
import re
import time
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

LABELS = ("bulk", "bulk-complex", "standard", "complex", "plan", "deep")
CONFIG = Path(__file__).resolve().parents[2] / "configs/s1-routing/s1-routing.toml"


def parse_systemone(
    data: Mapping[str, Any], cost: Mapping[str, Any] | None = None
) -> tuple[str | None, float | None]:
    """Extract a valid choice and its optional confidence; with ``cost``, pick the cheapest expected label."""
    try:
        route = data["answers"]["route"]
        choice = route["choice"]
        confidence = route.get("answer_confidence")
        probabilities = route.get("probabilities")
        if cost and isinstance(probabilities, Mapping) and set(probabilities) <= set(LABELS):
            choice = cheapest_label(probabilities, cost)
        return (choice, confidence) if choice in LABELS else (None, None)
    except (KeyError, TypeError):
        return None, None


def parse_chat(content: str) -> str | None:
    """Read constrained JSON, then the first standalone label in plain text."""
    try:
        value = json.loads(content)
        label = (value.get("label") or value.get("route")) if isinstance(value, dict) else None
        if label in LABELS:
            return label
    except (ValueError, TypeError):
        pass
    match = re.search(r"(?<![\w-])(bulk-complex|bulk|standard|complex|plan|deep)(?![\w-])", content, re.I)
    return match.group().lower() if match else None


def redact_url(url: str) -> str:
    """Keep only scheme and explicit port, never a host or credentials."""
    parsed = urlsplit(url)
    return f"{parsed.scheme}://REDACTED" + (f":{parsed.port}" if parsed.port else "")


def row_cost(pred: str, truth: str, cost: Mapping[str, Any]) -> float:
    """Misroute cost of predicting ``pred`` for a row whose true tier is ``truth``."""
    ranks = cost["ranks"]
    difference = ranks[pred] - ranks[truth]
    if difference < 0:
        return -difference * cost["under"]
    if difference > 0:
        return difference * cost["over"]
    return cost["within_top"] if pred != truth and ranks[truth] == 3 else 0.0


def cheapest_label(weights: Mapping[str, float], cost: Mapping[str, Any]) -> str:
    """The label with the lowest expected cost under the given label weights."""
    total = sum(weights.values()) or 1.0
    return min(LABELS, key=lambda pred: sum(w / total * row_cost(pred, t, cost) for t, w in weights.items()))


def knn_predict(
    vector: Sequence[float],
    examples: Sequence[tuple[Sequence[float], str]],
    k: int,
    cost: Mapping[str, Any] | None = None,
) -> tuple[str | None, float | None]:
    """Vote by inverse cosine distance; identical vectors dominate the vote."""

    def distance(other: Sequence[float]) -> float:
        if len(vector) != len(other) or not vector:
            raise ValueError("incompatible embeddings")
        norm = math.sqrt(sum(x * x for x in vector) * sum(x * x for x in other))
        if not norm:
            raise ValueError("zero embedding")
        return max(0.0, 1 - sum(x * y for x, y in zip(vector, other, strict=True)) / norm)

    nearest = sorted((distance(v), label) for v, label in examples)[:k]
    if not nearest:
        return None, None
    weights: dict[str, float] = dict.fromkeys(LABELS, 0.0)
    for dist, label in nearest:
        weights[label] += 1 / max(dist, 1e-9)
    winner = cheapest_label(weights, cost) if cost else max(LABELS, key=lambda label: weights[label])
    return winner, weights[winner] / sum(weights.values())


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank-style percentile matching the factual runner."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round(pct / 100 * (len(ordered) - 1))]


def summarize(records: Sequence[Mapping[str, Any]], cost: Mapping[str, Any]) -> dict[str, Any]:
    """Score invalid predictions as errors, using standard only for cost."""
    matrix = {label: dict.fromkeys((*LABELS, "invalid"), 0) for label in LABELS}
    total_cost = 0.0
    severe = 0
    ranks = cost["ranks"]
    for row in records:
        truth = row["label"]
        pred = row["pred"] if not row["invalid"] and row["pred"] in LABELS else None
        matrix[truth][pred or "invalid"] += 1
        effective = pred or "standard"
        total_cost += row_cost(effective, truth, cost)
        severe += ranks[truth] == 3 and ranks[effective] == 0
    f1 = []
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = sum(matrix[label].values()) - tp
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    n = len(records)
    latencies = [row["latency_ms"] for row in records]
    return {
        "count": n,
        "accuracy": sum(matrix[label][label] for label in LABELS) / n if n else 0.0,
        "macro_f1": sum(f1) / len(LABELS),
        "confusion_matrix": matrix,
        "invalid_rate": sum(matrix[label]["invalid"] for label in LABELS) / n if n else 0.0,
        "mean_cost": total_cost / n if n else 0.0,
        "severe_count": severe,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
    }


async def request(
    client: Any, args: argparse.Namespace, text: str, prompt: Mapping[str, Any]
) -> tuple[Any, float]:
    """Make one call; failures produce an invalid prediction without logging data."""
    start = time.monotonic()
    result = None
    try:
        if args.kind == "const":
            result = (args.label, None)
        elif args.kind == "systemone":
            body = {"state": text, "questions": {"route": {"type": "choice", **prompt}}}
            response = await client.post(args.url, json=body)
            response.raise_for_status()
            result = parse_systemone(response.json(), args.decision_cost)
        elif args.kind == "chat":
            # Servers that ignore response_format still need the answer shape spelled out.
            system = (
                prompt["instructions"]
                + "\n"
                + "\n".join(f"{k}: {v}" for k, v in prompt["criteria"].items())
                + '\nReply with only {"label": "<one of: '
                + ", ".join(LABELS)
                + '>"} and nothing else.'
            )
            body = {
                "model": args.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
                "temperature": 0,
                "max_tokens": 64,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "route",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"label": {"type": "string", "enum": list(LABELS)}},
                            "required": ["label"],
                            "additionalProperties": False,
                        },
                    },
                },
            }
            response = await client.post("chat/completions", json=body)
            response.raise_for_status()
            result = (parse_chat(response.json()["choices"][0]["message"]["content"]), None)
        else:
            response = await client.post("embeddings", json={"model": args.model, "input": text})
            response.raise_for_status()
            result = response.json()["data"][0]["embedding"]
    except Exception:
        pass
    return result, round((time.monotonic() - start) * 1000, 1)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    import httpx

    config = tomllib.loads(CONFIG.read_text())
    prompt = config["prompts"][args.prompt]
    args.decision_cost = config["cost"] if args.decision == "cost" else None
    raw = args.dataset.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if any(
        row["label"] not in LABELS
        or row["quality"] not in ("gold", "silver")
        or row["split"] not in ("dev", "test")
        for row in rows
    ):
        raise ValueError("invalid dataset label, quality or split")
    selected = [
        r for r in rows if r["split"] == args.split and (args.quality == "all" or r["quality"] == "gold")
    ]
    if not selected:
        raise ValueError("no rows selected")
    headers = {}
    if args.bearer_env and os.environ.get(args.bearer_env):
        headers["Authorization"] = "Bearer " + os.environ[args.bearer_env]
    base = args.url if args.kind == "systemone" else (args.url or "")
    async with httpx.AsyncClient(
        base_url=base.rstrip("/") + "/" if base else "", headers=headers, timeout=args.timeout
    ) as client:
        for i in range(args.warmup):
            await request(client, args, selected[i % len(selected)]["text"], prompt)
        semaphore = asyncio.Semaphore(args.concurrency)

        async def call(row: Mapping[str, Any]) -> tuple[Any, float]:
            async with semaphore:
                return await request(client, args, row["text"], prompt)

        dev = [r for r in rows if r["split"] == "dev"] if args.kind == "embed-knn" else []
        dev_results = await asyncio.gather(*(call(row) for row in dev))
        results = await asyncio.gather(*(call(row) for row in selected))
    records = []
    for row, (result, latency) in zip(selected, results, strict=True):
        pred, confidence = None, None
        if args.kind == "embed-knn" and result is not None:
            examples = [
                (value, item["label"])
                for item, (value, _) in zip(dev, dev_results, strict=True)
                if value is not None and item["id"] != row["id"]
            ]
            with contextlib.suppress(ValueError, TypeError):
                pred, confidence = knn_predict(result, examples, config.get("k", 7), args.decision_cost)
        elif args.kind != "embed-knn" and result is not None:
            pred, confidence = result
        records.append(
            {
                "id": row["id"],
                "label": row["label"],
                "pred": pred,
                "confidence": confidence,
                "latency_ms": latency,
                "invalid": pred not in LABELS,
            }
        )
    gold = [record for row, record in zip(selected, records, strict=True) if row["quality"] == "gold"]
    return {
        "metadata": {
            "kind": args.kind,
            "model": args.model,
            "url": redact_url(args.url) if args.url else None,
            "prompt": args.prompt,
            "split": args.split,
            "quality": args.quality,
            "decision": args.decision,
            "dataset_sha256": hashlib.sha256(raw).hexdigest(),
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        },
        "metrics": {"all": summarize(records, config["cost"]), "gold": summarize(gold, config["cost"])},
        "rows": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="System 1 routing benchmark")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--kind", choices=("systemone", "chat", "embed-knn", "const"), required=True)
    parser.add_argument("--url", help="Endpoint URL or OpenAI-compatible base URL")
    parser.add_argument("--model")
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--bearer-env", help="Name of environment variable containing bearer token")
    parser.add_argument("--split", choices=("test", "dev"), default="test")
    parser.add_argument("--quality", choices=("gold", "all"), default="all")
    parser.add_argument("--prompt", default="plugin")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--decision",
        choices=("argmax", "cost"),
        default="argmax",
        help="cost: pick the label with the lowest expected misroute cost when label probabilities exist",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.concurrency < 1 or args.warmup < 0 or args.timeout <= 0:
        parser.error("concurrency and timeout must be positive; warmup must be nonnegative")
    if args.kind != "const" and not args.url:
        parser.error("--url is required for this adapter")
    if args.kind in ("chat", "embed-knn") and not args.model:
        parser.error("--model is required for this adapter")
    if args.kind == "const" and not args.label:
        parser.error("--label is required for const")
    if args.prompt not in tomllib.loads(CONFIG.read_text())["prompts"]:
        parser.error("unknown prompt variant")
    args.out.write_text(json.dumps(asyncio.run(run(args)), indent=2) + "\n")


if __name__ == "__main__":
    main()
