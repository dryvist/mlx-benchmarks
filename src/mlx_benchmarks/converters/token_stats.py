"""Shared token counting helpers for sample-level evaluation data."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, cast

log = logging.getLogger(__name__)


class TokenizerLike(Protocol):
    def encode(self, text: str) -> Any:  # pragma: no cover - protocol
        ...


TokenizerLoader = Callable[[str], TokenizerLike | None]


def default_tokenizer_loader(model: str) -> TokenizerLike | None:
    """Load a Hugging Face tokenizer, returning None when it is unavailable."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        log.debug("tokenizers package not available; sample token fields will be unset")
        return None
    try:
        return cast(TokenizerLike, Tokenizer.from_pretrained(model))
    except Exception as exc:
        log.warning("could not load tokenizer for %s: %s", model, exc)
        return None


def count_tokens(tokenizer: TokenizerLike, text: str) -> int:
    if not text:
        return 0
    encoded = tokenizer.encode(text)
    ids = getattr(encoded, "ids", None)
    if ids is not None:
        return len(ids)
    if isinstance(encoded, list):
        return len(encoded)
    return 0


def summarize_token_counts(counts: Sequence[int]) -> tuple[float, float, int]:
    """Return mean, median, and nearest-rank p90 for a non-empty count set."""
    if not counts:
        raise ValueError("cannot summarize an empty token count set")
    ordered = sorted(counts)
    middle = len(ordered) // 2
    median = float(ordered[middle]) if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    p90 = ordered[math.ceil(0.9 * len(ordered)) - 1]
    return sum(ordered) / len(ordered), median, p90


def api_output_tokens(sample: Mapping[str, Any]) -> int | None:
    """Return an API-reported completion count when the sample has one."""
    model_output = sample.get("model_output")
    candidates = [
        model_output.get("usage") if isinstance(model_output, Mapping) else None,
        sample.get("usage"),
    ]
    for usage in candidates:
        if not isinstance(usage, Mapping):
            continue
        for key in ("output_tokens", "completion_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
    return None


def api_reasoning_tokens(sample: Mapping[str, Any]) -> int | None:
    """Return an API-reported reasoning count when the sample has one."""
    model_output = sample.get("model_output")
    candidates = [
        model_output.get("usage") if isinstance(model_output, Mapping) else None,
        sample.get("usage"),
    ]
    for usage in candidates:
        if not isinstance(usage, Mapping):
            continue
        value = usage.get("reasoning_tokens")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        for key in ("completion_tokens_details", "output_tokens_details"):
            details = usage.get(key)
            value = details.get("reasoning_tokens") if isinstance(details, Mapping) else None
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
    return None


def finish_reason(sample: Mapping[str, Any]) -> str | None:
    model_output = sample.get("model_output")
    choices = model_output.get("choices") if isinstance(model_output, Mapping) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        reason = choices[0].get("finish_reason")
        if isinstance(reason, str):
            return reason
    reason = sample.get("finish_reason")
    return reason if isinstance(reason, str) else None
