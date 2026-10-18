"""Regression coverage for viewer loading, evidence filtering, and series identity."""

from __future__ import annotations

import json
import time
from threading import Event, Lock

import pandas as pd
from test_charts import app


def _annotated(
    rows: list[dict[str, object]], index: dict[str, dict[str, object]] | None = None
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return app.add_evidence_metadata(frame, index or {}, None)


def test_invalid_and_item_rows_never_become_tasks() -> None:
    rows = _annotated(
        [
            {
                "timestamp": "2026-10-01T00:00:00Z",
                "suite": "agentic",
                "name": name,
                "metric": metric,
                "value": 0.5,
                "model": "m/a",
                "source_path": source,
            }
            for name, metric, source in (
                ("test_0", "score", "data/items.parquet"),
                ("test_0_assertion_1", "assertion_pass", "data/items.parquet"),
                ("task", "pass_at_1", "data/pass.parquet"),
                ("task", "pass_rate", "data/pass.parquet"),
                ("invalid-task", "score", "data/invalid.parquet"),
            )
        ],
        {
            "data/items.parquet": {"status": "scored"},
            "data/pass.parquet": {"status": "scored"},
            "data/invalid.parquet": {"status": "invalid"},
        },
    )

    clean = app.comparable_rows(rows)

    assert set(clean["name"]) == {"task"}
    assert set(clean["metric"]) == {"pass_rate"}
    assert app.evidence_view(rows, "invalid").empty
    assert app.task_choices(rows, "agentic") == ["task"]


def test_suites_without_multiple_series_are_not_offered() -> None:
    rows = _annotated(
        [
            {
                "timestamp": f"2026-10-0{index + 1}T00:00:00Z",
                "suite": suite,
                "name": "task",
                "metric": "score",
                "value": value,
                "model": model,
                "machine": machine,
                "source_path": f"data/{suite}-{index}.parquet",
            }
            for suite, items in (
                ("comparable", [("m/a", "Machine A", 0.4), ("m/b", "Machine B", 0.6)]),
                ("single", [("m/a", "Machine A", 0.4)]),
            )
            for index, (model, machine, value) in enumerate(items)
        ],
        {
            f"data/{suite}-{index}.parquet": {"status": "scored"}
            for suite, items in (
                ("comparable", [0, 1]),
                ("single", [0]),
            )
            for index in items
        },
    )

    assert app.suite_choices(rows) == ["comparable"]


def test_default_uses_newest_scored_non_saturated_two_machine_triple() -> None:
    records = []
    index = {}
    for suite, machines, values, day in (
        ("older", ["RTX PRO 6000 96GB", "Mac Studio"], [0.4, 0.6], 10),
        ("saturated", ["RTX 4080 16GB", "Mac Studio M4 Max 128GB"], [1.0, 1.0], 17),
        ("single", ["Mac Studio"], [0.8], 18),
        ("newest", ["RTX 4080 SUPER 16GB", "Mac Studio"], [0.5, 0.7], 16),
    ):
        for number, (machine, value) in enumerate(zip(machines, values, strict=True)):
            path = f"data/{suite}-{number}.parquet"
            records.append(
                {
                    "timestamp": f"2026-10-{day:02}T00:00:00Z",
                    "suite": suite,
                    "name": "aggregate",
                    "metric": "accuracy",
                    "value": value,
                    "model": f"m/{number}",
                    "machine": machine,
                    "source_path": path,
                }
            )
            index[path] = {"status": "scored"}

    rows = _annotated(records, index)

    assert app.best_default(rows) == ("newest", "aggregate", "accuracy")
    assert app.best_default(rows[rows["suite"].isin(["saturated", "single"])]) == (None, None, None)
    assert app.normalize_machine("RTX PRO 6000 96GB") == "RTX PRO 6000 Max-Q 96GB"
    assert app.normalize_machine("RTX 4080 16GB") == "RTX 4080 SUPER 16GB"
    assert app.normalize_machine("Mac Studio") == "Mac Studio M4 Max 128GB"
    assert app.normalize_machine("Apple M4 Max") == "Mac Studio M4 Max 128GB"


def test_latest_collapses_repeats_but_preserves_knob_series_and_trend() -> None:
    records = []
    for index, (day, value, mtp) in enumerate(((1, 10.0, False), (2, 12.0, False), (2, 9.0, True))):
        records.append(
            {
                "timestamp": f"2026-10-{day:02}T00:00:00Z",
                "suite": "throughput-serving",
                "name": "fixed-length",
                "metric": "output_tokens_per_second",
                "value": value,
                "unit": "tok/s",
                "model": "m/model",
                "machine": "RTX 4080 16GB",
                "engine": "vLLM",
                "campaign_run_speculative_mtp": mtp,
                "source_path": f"data/run-{index}.parquet",
                "evidence_status": "experimental",
            }
        )
    rows = _annotated(records)
    latest = app.bar_chart(rows, "throughput-serving", "fixed-length", "output_tokens_per_second", "tok/s")
    off_key = rows.loc[rows["mtp"] == "false", "series_key"].iloc[0]
    trend = app.trend_chart(
        rows, "throughput-serving", "fixed-length", "output_tokens_per_second", [off_key], "tok/s"
    )

    assert len(latest.data[0].y) == 2
    assert all("(run " not in label for label in latest.data[0].y)
    assert len(trend.data[0].x) == 2
    assert latest.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert latest.layout.plot_bgcolor == "rgba(0,0,0,0)"


def test_knob_aliases_and_summary_details_use_published_columns() -> None:
    rows = _annotated(
        [
            {
                "timestamp": "2026-10-01T00:00:00Z",
                "suite": "throughput-serving",
                "name": "fixed-length",
                "metric": "output_tokens_per_second",
                "value": 22.0,
                "model": "m/model",
                "campaign_hardware_machine": "Mac Studio",
                "campaign_software_engine": "vLLM",
                "campaign_software_engine_version": "0.31.0",
                "campaign_model_quantization": "NVFP4",
                "reasoning_effort": "xhigh",
                "campaign_run_thinking": True,
                "campaign_run_speculative_mtp": True,
                "campaign_run_kv_cache_dtype": "fp8",
                "power_limit_w": 300,
                "campaign_run_concurrent_agents": 8,
                "campaign_run_prompt_tokens": 4096,
                "campaign_run_output_tokens": 1024,
                "tags_json": json.dumps(
                    {"arrival": "staggered; request_rate=0.12", "request_rate": "0.12", "burstiness": 1}
                ),
                "campaign_environment_fan_duty_pct_json": json.dumps({"gpu": 100, "mac": None}),
                "campaign_environment_ambient_temp_f": 80,
                "campaign_run_patches_json": "[]",
                "tag_dedicated": "true",
                "source_path": "data/knobs.parquet",
            }
        ],
        {
            "data/knobs.parquet": {
                "status": "experimental",
                "variant": "MTP default",
                "caveat": "single run",
                "context_band": "64k",
            }
        },
    )
    row = rows.iloc[0]
    summary = app.summary_table(rows, "throughput-serving", "fixed-length", "output_tokens_per_second")

    assert row["machine"] == "Mac Studio M4 Max 128GB"
    assert row["engine_version"] == "0.31.0"
    assert row["reasoning_effort"] == "xhigh"
    assert row["thinking"] == "true"
    assert row["mtp"] == "true"
    assert row["arrival_pattern"] == "Staggered"
    assert row["request_rate"] == "0.12"
    assert row["burstiness"] == "1"
    assert row["context_band"] == "64k"
    assert row["fan_duty"] == "100"
    assert row["patches"] == "[]"
    assert "Power limit (W)" in summary and summary.loc[0, "Power limit (W)"] == "300 W"
    assert "Fan duty" in summary and summary.loc[0, "Fan duty"] == "100%"
    assert "Ambient (°F)" in summary and summary.loc[0, "Ambient (°F)"] == "80 °F"
    assert "Patches" in summary and summary.loc[0, "Patches"] == "[]"
    assert "Dedicated" in summary and summary.loc[0, "Dedicated"] == "true"
    assert "Variant" in summary and summary.loc[0, "Variant"] == "MTP default"
    assert "Caveat" in summary and summary.loc[0, "Caveat"] == "single run"


def test_local_data_dir_keeps_loading_parquets_and_neighboring_index(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_name = "run-canonical-run-2026-10-01-example.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": "2026-10-01T00:00:00Z",
                "suite": "throughput-serving",
                "name": "fixed-length",
                "metric": "output_tokens_per_second",
                "value": 10,
                "model": "m/model",
            }
        ]
    ).to_parquet(data_dir / file_name)
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "run-index-v1.json").write_text(
        json.dumps(
            {
                "runs": [{"path": "data/run-2026-10-01-example.parquet", "status": "scored"}],
                "historical_scored_before": "2026-08-25T00:00:00Z",
            }
        )
    )
    monkeypatch.setenv("MLX_BENCHMARKS_LOCAL_DATA_DIR", str(data_dir))
    monkeypatch.setattr(app, "_cache", None)
    monkeypatch.setattr(
        app, "snapshot_download", lambda **_: (_ for _ in ()).throw(AssertionError("network"))
    )

    loaded = app.load_data()

    assert loaded.loc[0, "evidence_status"] == "scored"
    assert loaded.loc[0, "source_path"] == "data/run-2026-10-01-example.parquet"


def test_expired_cache_returns_old_frame_while_refresh_runs(monkeypatch) -> None:
    old = pd.DataFrame({"value": ["cached"]})
    new = pd.DataFrame({"value": ["fresh"]})
    started = Event()
    release = Event()
    finished = Event()

    def slow_refresh() -> pd.DataFrame:
        started.set()
        release.wait(timeout=2)
        finished.set()
        return new

    monkeypatch.setattr(app, "_cache_lock", Lock())
    monkeypatch.setattr(app, "_refresh_lock", Lock())
    monkeypatch.setattr(app, "_cache", (time.time() - app.CACHE_TTL - 1, old))
    monkeypatch.setattr(app, "_load_dataset", slow_refresh)

    before = time.monotonic()
    served = app.load_data()
    elapsed = time.monotonic() - before
    assert served is old
    assert elapsed < 0.5
    assert started.wait(timeout=1)
    release.set()
    assert finished.wait(timeout=2)
    for _ in range(100):
        if app._cache is not None and app._cache[1] is new:
            break
        time.sleep(0.01)
    assert app._cache is not None and app._cache[1] is new


def test_filter_catalog_includes_all_requested_tuning_dimensions() -> None:
    assert {
        "reasoning_effort",
        "thinking",
        "mtp",
        "speculative_tokens",
        "kv_cache_dtype",
        "engine_version",
        "power_limit_w",
        "agents",
        "context_band",
        "prompt_tokens",
        "output_budget_tokens",
        "arrival_pattern",
        "request_rate",
        "burstiness",
        "quant_format",
    } <= set(app.FILTER_SPECS)


def test_empty_filter_selection_does_not_hide_available_rows() -> None:
    rows = pd.DataFrame({"machine": ["Mac Studio M4 Max 128GB", "RTX 4080 SUPER 16GB"]})

    filtered = app.filter_rows(rows, {"machine": ""})

    assert len(filtered) == 2


def test_empty_chart_has_transparent_dark_surface_without_axes() -> None:
    figure = app.empty_figure("No data")

    assert figure.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert figure.layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert figure.layout.xaxis.visible is False
    assert figure.layout.yaxis.visible is False
