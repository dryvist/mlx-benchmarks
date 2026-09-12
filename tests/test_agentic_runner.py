"""Unit tests for the agentic runner's pure functions.

``harness/agentic/run.py`` is a standalone PEP 723 script (not part of the
package), so it is loaded here via importlib. Its network dependency (httpx)
is imported lazily inside the request functions, keeping the module — and
these pure-function tests — importable in the plain dev environment.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import httpx
import pytest


def _load_runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "harness" / "agentic" / "run.py"
    spec = importlib.util.spec_from_file_location("agentic_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()

REQUIRED = runner.required_params(runner.TOOLS)


def _call(name: str, arguments: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


# --- registry -----------------------------------------------------------------


def test_registry_has_22_unique_tools() -> None:
    names = [t["function"]["name"] for t in runner.TOOLS]
    assert len(names) == 22
    assert len(set(names)) == 22
    assert REQUIRED["run_splunk_query"] == ["query"]
    assert REQUIRED["get_splunk_indexes"] == []
    assert REQUIRED["read_file_chunk"] == ["path", "offset", "length"]


def test_registry_schemas_are_objects_with_properties() -> None:
    for tool in runner.TOOLS:
        params = tool["function"]["parameters"]
        assert params["type"] == "object"
        assert params["additionalProperties"] is False


# --- classify_message ----------------------------------------------------------


def test_classify_valid() -> None:
    message = _call("run_splunk_query", json.dumps({"query": "search index=main"}))
    assert runner.classify_message(message, "tool_calls", REQUIRED) == "valid"


def test_classify_no_tool_call() -> None:
    assert runner.classify_message({"role": "assistant", "content": "hi"}, "stop", REQUIRED) == "no_tool_call"
    assert runner.classify_message(None, None, REQUIRED) == "no_tool_call"


def test_classify_plain_text_fallback_is_no_tool_call() -> None:
    # mlx-lm #1011 degradation shape: prose pseudo-call, no structured tool_calls.
    message = {"role": "assistant", "content": "[Tool call: run_splunk_query(query=...)]"}
    assert runner.classify_message(message, "stop", REQUIRED) == "no_tool_call"


def test_classify_empty_function_name() -> None:
    message = _call("", json.dumps({"query": "x"}))
    assert runner.classify_message(message, "tool_calls", REQUIRED) == "empty_function_name"
    message = _call("   ", json.dumps({"query": "x"}))
    assert runner.classify_message(message, "tool_calls", REQUIRED) == "empty_function_name"


def test_classify_unknown_tool() -> None:
    message = _call("splunk_query", json.dumps({"query": "x"}))
    assert runner.classify_message(message, "tool_calls", REQUIRED) == "unknown_tool"


def test_classify_bad_json_args() -> None:
    truncated = _call("run_splunk_query", '{"query": "search index=')
    assert runner.classify_message(truncated, "tool_calls", REQUIRED) == "bad_json_args"
    missing_required = _call("run_splunk_query", json.dumps({"earliest_time": "-24h"}))
    assert runner.classify_message(missing_required, "tool_calls", REQUIRED) == "bad_json_args"
    non_object = _call("run_splunk_query", json.dumps(["search"]))
    assert runner.classify_message(non_object, "tool_calls", REQUIRED) == "bad_json_args"


def test_classify_stream_truncated() -> None:
    # Structurally fine call, but finish_reason says the stream was cut off.
    message = _call("run_splunk_query", json.dumps({"query": "search index=main"}))
    assert runner.classify_message(message, "length", REQUIRED) == "stream_truncated"
    assert runner.classify_message(message, None, REQUIRED) == "stream_truncated"


# --- assemble_stream -----------------------------------------------------------


def _chunk(delta: dict, finish_reason: str | None = None) -> dict:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


def test_assemble_stream_reassembles_split_tool_call() -> None:
    events = [
        _chunk({"role": "assistant", "reasoning_content": "Let me "}),
        _chunk({"reasoning_content": "search Splunk."}),
        _chunk(
            {
                "tool_calls": [
                    {"index": 0, "id": "call_9", "function": {"name": "run_splunk", "arguments": ""}}
                ]
            }
        ),
        _chunk({"tool_calls": [{"index": 0, "function": {"name": "_query", "arguments": '{"que'}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ry": "search index=main"}'}}]}),
        _chunk({}, finish_reason="tool_calls"),
        {"choices": [], "usage": {"completion_tokens": 42, "prompt_tokens": 9000}},
    ]
    message, finish_reason, usage = runner.assemble_stream(events)

    assert finish_reason == "tool_calls"
    assert usage == {"completion_tokens": 42, "prompt_tokens": 9000}
    assert message["reasoning_content"] == "Let me search Splunk."
    calls = message["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["id"] == "call_9"
    assert calls[0]["function"]["name"] == "run_splunk_query"
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "search index=main"}
    # The assembled message classifies as valid end to end.
    assert runner.classify_message(message, finish_reason, REQUIRED) == "valid"


def test_assemble_stream_parallel_calls_ordered_by_index() -> None:
    events = [
        _chunk(
            {"tool_calls": [{"index": 1, "id": "b", "function": {"name": "read_file", "arguments": "{}"}}]}
        ),
        _chunk(
            {
                "tool_calls": [
                    {"index": 0, "id": "a", "function": {"name": "list_directory", "arguments": "{}"}}
                ]
            }
        ),
        _chunk({}, finish_reason="tool_calls"),
    ]
    message, _, _ = runner.assemble_stream(events)
    assert [c["id"] for c in message["tool_calls"]] == ["a", "b"]


def test_assemble_stream_content_only() -> None:
    events = [_chunk({"content": "Hello "}), _chunk({"content": "world"}, finish_reason="stop")]
    message, finish_reason, usage = runner.assemble_stream(events)
    assert message["content"] == "Hello world"
    assert "tool_calls" not in message
    assert finish_reason == "stop"
    assert usage is None


# --- context builder / aggregation ----------------------------------------------


def test_build_history_reaches_target_size() -> None:
    history = runner.build_history(20000)
    total = sum(runner.approx_tokens(json.dumps(m)) for m in history)
    assert total >= 20000
    roles = {m["role"] for m in history}
    assert roles == {"user", "assistant", "tool"}
    # Tool results reference the assistant tool_call ids that precede them.
    ids = [c["id"] for m in history if m.get("tool_calls") for c in m["tool_calls"]]
    tool_refs = [m["tool_call_id"] for m in history if m["role"] == "tool"]
    assert tool_refs == ids


def test_percentile_nearest_rank() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert runner.percentile(values, 50) == 30.0
    assert runner.percentile(values, 95) == 50.0
    assert runner.percentile([], 50) == 0.0
    assert runner.percentile([7.5], 95) == 7.5


def test_summarize_cell_counts_and_rates() -> None:
    records = [
        {
            "outcome": "valid",
            "finish_reason": "tool_calls",
            "reasoning_present": True,
            "latency_ms": 1000.0,
            "first_token_ms": 100.0,
            "completion_tokens": 50,
            "http_status": 200,
        },
        {
            "outcome": "empty_function_name",
            "finish_reason": "tool_calls",
            "reasoning_present": True,
            "latency_ms": 2000.0,
            "first_token_ms": 200.0,
            "completion_tokens": 40,
            "http_status": 200,
        },
        {
            "outcome": "http_error",
            "finish_reason": None,
            "reasoning_present": False,
            "latency_ms": 10.0,
            "first_token_ms": None,
            "completion_tokens": None,
            "http_status": 503,
        },
    ]
    cell = runner.summarize_cell(records, wall_seconds=3.0)
    assert cell["n_requests"] == 3
    assert cell["valid_tool_call_rate"] == 1 / 3
    assert cell["finish_reason_tool_calls_rate"] == 2 / 3
    assert cell["failures"]["empty_function_name"] == 1
    assert cell["failures"]["http_error"] == 1
    assert cell["failures"]["timeout"] == 0
    assert cell["http_5xx"] == 1
    assert cell["http_429"] == 0
    assert cell["latency_p50_ms"] == 1000.0
    assert cell["aggregate_tokens_per_second"] == 30.0  # 90 tokens / 3 s
    assert cell["effective_tokens_per_second"] == 35.0  # mean(50/1, 40/2)


def test_cell_name_and_thinking_kwargs() -> None:
    """`on`/`off` must keep the exact names and bodies they have always produced.

    `cell_name` is a public join key — `--cells` filters on it, the recovery JSONL
    is keyed by it, and it publishes as `tags.cell`. Renaming either level would
    orphan every row already in the dataset.
    """
    assert runner.cell_name(4, "on", "large", True) == "conc4_think-on_ctx-large_stream"
    assert runner.cell_name(1, "off", "small", False) == "conc1_think-off_ctx-small_nostream"
    assert runner.thinking_body_kwargs("enable_thinking", "on") == {
        "chat_template_kwargs": {"enable_thinking": True}
    }
    assert runner.thinking_body_kwargs("enable_thinking", "off") == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert runner.thinking_body_kwargs("reasoning_effort", "on") == {"reasoning_effort": "high"}
    assert runner.thinking_body_kwargs("reasoning_effort", "off") == {"reasoning_effort": "low"}


def test_a_graded_level_is_sent_verbatim_and_names_its_own_cell() -> None:
    """`high` and `xhigh` are two arms. Before this they both parsed to False,
    producing one cell name twice and one request body — no error, no signal."""
    assert runner.cell_name(1, "xhigh", "small", False) == "conc1_think-xhigh_ctx-small_nostream"
    assert runner.cell_name(1, "high", "small", False) == "conc1_think-high_ctx-small_nostream"
    assert runner.thinking_body_kwargs("reasoning_effort", "xhigh") == {"reasoning_effort": "xhigh"}
    assert runner.thinking_body_kwargs("enable_thinking", "medium") == {
        "chat_template_kwargs": {"enable_thinking": "medium"}
    }


def test_a_graded_sweep_produces_one_cell_per_level(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The end-to-end shape of the trap: a two-level sweep must run two cells."""
    seen: list[str] = []

    async def fake_run_cell(client, args, concurrency, thinking, context, stream, required, history):
        seen.append(thinking)
        return {"name": runner.cell_name(concurrency, thinking, context, stream), "n_requests": 1}

    async def fake_run_multiturn(client, args, thinking, required):
        return {"thinking": thinking, "rounds": [], "first_degraded_round": None}

    monkeypatch.setattr(runner, "run_cell", fake_run_cell)
    monkeypatch.setattr(runner, "run_multiturn", fake_run_multiturn)

    args = runner.build_parser().parse_args(
        [
            "--base-url",
            "http://localhost:1/v1",
            "--model",
            "m",
            "--thinking",
            "high,xhigh",
            "--concurrency",
            "1",
            "--context",
            "small",
            "--stream",
            "nostream",
        ]
    )
    out = asyncio.run(runner._run(args, "2026-09-09T00:00:00Z", tmp_path / "partial.jsonl"))

    assert seen == ["high", "xhigh"]
    assert [cell["name"] for cell in out["cells"]] == [
        "conc1_think-high_ctx-small_nostream",
        "conc1_think-xhigh_ctx-small_nostream",
    ]
    # The config block records the levels asked for, not a pair of booleans.
    assert out["config"]["thinking"] == ["high", "xhigh"]


# --- runner robustness: warm-up, malformed body, incremental persistence -------
#
# These drive the real request path through httpx.MockTransport (no live
# endpoint): the fake transport returns canned or deliberately-malformed bodies
# so one_request / run_cell exercise their actual error handling.


def _args(**overrides: object) -> argparse.Namespace:
    defaults: dict = {
        "model": "test-model",
        "temperature": 0.0,
        "max_tokens": 128,
        "thinking_kwarg": "enable_thinking",
        "repetition_penalty": None,
        "repeats": 3,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url="http://test/v1", transport=httpx.MockTransport(handler), timeout=5)


_VALID_TOOL_CALL: dict = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "run_splunk_query",
                            "arguments": '{"query": "search index=main"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"completion_tokens": 10},
}


def test_run_cell_fires_one_untimed_warmup() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_VALID_TOOL_CALL)

    async def go() -> dict:
        async with _mock_client(handler) as client:
            return await runner.run_cell(
                client, _args(repeats=3), 1, "off", "small", False, REQUIRED, history=[]
            )

    cell = asyncio.run(go())
    # 3 timed repeats are summarized; the warm-up is the 4th request nobody counts.
    assert cell["n_requests"] == 3
    assert calls == 4


def test_one_request_malformed_stream_body_is_recorded_not_raised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # Truncated SSE data line: json.loads used to raise straight through
        # one_request and abort the whole matrix.
        return httpx.Response(200, content=b'data: {"choices": [\n\n')

    async def go() -> dict:
        async with _mock_client(handler) as client:
            return await runner.one_request(
                client, _args(), [{"role": "user", "content": "hi"}], "off", True, REQUIRED
            )

    record = asyncio.run(go())
    assert record["outcome"] == "bad_response_body"
    assert record["latency_ms"] is not None


def test_run_cell_survives_malformed_bodies() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{not json")

    async def go() -> dict:
        async with _mock_client(handler) as client:
            return await runner.run_cell(
                client, _args(repeats=3), 1, "off", "small", False, REQUIRED, history=[]
            )

    cell = asyncio.run(go())
    # Every request fails on the bad body, yet the cell still summarizes cleanly.
    assert cell["failures"]["bad_response_body"] == 3
    assert cell["valid_tool_call_rate"] == 0.0


def test_run_persists_each_cell_incrementally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_cell(client, args, concurrency, thinking, context, stream, required, history):
        return {"name": runner.cell_name(concurrency, thinking, context, stream), "n_requests": args.repeats}

    async def fake_run_multiturn(client, args, thinking, required):
        return {"thinking": thinking, "rounds": [], "first_degraded_round": None}

    monkeypatch.setattr(runner, "run_cell", fake_run_cell)
    monkeypatch.setattr(runner, "run_multiturn", fake_run_multiturn)

    partial = tmp_path / "out.json.partial.jsonl"
    args = runner.build_parser().parse_args(
        [
            "--base-url",
            "http://test/v1",
            "--model",
            "m",
            "--concurrency",
            "1",
            "--thinking",
            "off",
            "--context",
            "small",
            "--stream",
            "nostream",
            "--repeats",
            "2",
            "--multiturn-rounds",
            "1",
        ]
    )
    results = asyncio.run(runner._run(args, "2026-07-16T00:00:00Z", partial))

    # Each completed unit was flushed as it finished, so a late crash keeps them.
    kinds = [json.loads(line)["kind"] for line in partial.read_text().splitlines()]
    assert kinds == ["cell", "multiturn"]
    assert results["timestamp"] == "2026-07-16T00:00:00Z"
    assert len(results["cells"]) == 1
    assert len(results["multiturn"]) == 1
