from __future__ import annotations

from mlx_benchmarks.converters import get_converter
from mlx_benchmarks.converters.base import ConverterContext
from mlx_benchmarks.envelope import validate_envelope
from mlx_benchmarks.system import detect_system


def test_throughput_probe_round_trip() -> None:
    raw = {
        "model": "mlx-community/Qwen3.8-27B-4bit",
        "started_utc": "2026-08-24T11:41:57Z",
        "max_tokens": 256,
        "thinking": "off",
        "context_tokens_target": 21333,
        "campaign": {"id": "local-context-2026-08", "cell_id": "qwen38-64k-c1", "profile": "mtp"},
        "cell_status": "success",
        "context": {
            "configured_window_tokens": 131072,
            "requested_prompt_tokens": 64000,
            "actual_prompt_tokens": 64031,
            "output_reservation_tokens": 256,
        },
        "sequential_runs": [
            {"prompt_tokens": 64031, "completion_tokens": 47},
            {"prompt_tokens": 64031, "completion_tokens": 47},
        ],
        "sequential": {
            "n_ok": 4,
            "n_err": 0,
            "answered_rate": 1.0,
            "truncated_rate": 1.0,
            "finish_reasons": ["length"],
            "cumulative_tok_s": {"median": 54.11, "min": 51.84, "max": 56.13},
            "decode_tok_s": {"median": 42.97, "min": 41.35, "max": 44.53},
            "prefill_tok_s": {"median": 186.83, "min": 172.78, "max": 206.61},
            "ttft_s": {"median": 0.49, "min": 0.44, "max": 0.53},
            "total_s": {"median": 6.42, "min": 6.18, "max": 6.69},
        },
    }
    ctx = ConverterContext(
        suite="throughput",
        model=raw["model"],
        git_sha="deadbeef",
        concurrency=1,
        env_class="isolated",
        system=detect_system(),
    )

    envelope = get_converter("throughput-probe").build_envelope(raw, ctx)
    validate_envelope(envelope)

    by_metric = {result["metric"]: result for result in envelope["results"]}
    assert by_metric["throughput_total_toks_per_s"]["value"] == 54.11
    assert by_metric["throughput_output_toks_per_s"]["value"] == 42.97
    assert by_metric["ttft_p50_ms"]["value"] == 490.0
    assert by_metric["throughput_total_toks_per_s"]["tags"]["truncated_rate"] == "1.0"
    assert by_metric["throughput_total_toks_per_s"]["tags"]["context_tokens_target"] == "21333"
    assert by_metric["throughput_total_toks_per_s"]["tags"]["context_tokens_actual"] == "64031"
    assert by_metric["throughput_total_toks_per_s"]["raw"] == raw
    assert envelope["campaign"]["cell_id"] == "qwen38-64k-c1"
    assert envelope["context"]["actual_prompt_tokens"] == 64031


def _minimal_raw(**extra: object) -> dict:
    raw = {
        "model": "m",
        "thinking": "off",
        "sequential": {"cumulative_tok_s": {"median": 1.0}},
    }
    raw.update(extra)
    return raw


def _tags(raw: dict) -> dict[str, str]:
    ctx = ConverterContext(suite="throughput", model="m", git_sha="deadbeef", system=detect_system())
    envelope = get_converter("throughput-probe").build_envelope(raw, ctx)
    validate_envelope(envelope)
    return envelope["results"][0]["tags"]


def test_a_graded_think_value_reaches_the_tags() -> None:
    """The runner records the literal level it sent; the converter dropped it, so
    two arms differing only in effort published identically."""
    tags = _tags(_minimal_raw(thinking="on", think_kwarg="reasoning_effort", think_value="xhigh"))
    assert tags["think_value"] == "xhigh"
    assert tags["think_kwarg"] == "reasoning_effort"


def test_no_kwarg_sent_is_distinguishable_from_asking_for_off() -> None:
    """``thinking="off"`` is what a run records when it passes NO kwarg at all,
    while the model reasons at its own default. Without ``think_kwarg_sent`` the
    two are the same row, and the label is simply wrong for the second."""
    asked_off = _tags(
        _minimal_raw(thinking="off", think_kwarg="thinking", think_value=False, think_kwarg_sent=True)
    )
    asked_nothing = _tags(
        _minimal_raw(thinking="off", think_kwarg=None, think_value=None, think_kwarg_sent=False)
    )

    assert asked_off["thinking"] == asked_nothing["thinking"] == "off"
    assert asked_off["think_kwarg_sent"] == "True"
    assert asked_nothing["think_kwarg_sent"] == "False"
    # An absent condition is unstated, never a value: "None" would read as one.
    assert asked_nothing["think_value"] == "unstated"
