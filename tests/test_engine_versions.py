"""Recorded upstream release data populates stable benchmark dimensions."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from urllib.error import URLError

import mlx_benchmarks.engine_versions as engine_versions
from mlx_benchmarks.dataset_schema import campaign_dimension_column
from mlx_benchmarks.envelope import validate_envelope
from mlx_benchmarks.publish import envelope_to_rows

FIXTURES = Path(__file__).parent / "fixtures" / "engine_releases"
LLAMA_ASSET = "https://github.com/ggml-org/llama.cpp/releases/download/v0.7.0/nightly-tag.txt"


def _urlopen(request: Any, *, timeout: int) -> io.BytesIO:
    url = request.full_url
    if url.endswith("/releases?per_page=100"):
        return io.BytesIO((FIXTURES / "llama_cpp.json").read_bytes())
    if url == LLAMA_ASSET:
        return io.BytesIO(b"b11669\n")
    if url == "https://pypi.org/pypi/vllm/json":
        return io.BytesIO((FIXTURES / "vllm.json").read_bytes())
    raise AssertionError(f"unexpected upstream URL {url} (timeout={timeout})")


def _llama_envelope(valid_envelope: dict[str, Any], version: str) -> dict[str, Any]:
    return {
        **valid_envelope,
        "timestamp": "2026-10-16T00:00:00Z",
        "campaign_dimensions": {"software": {"engine": "llama.cpp", "engine_version": version}},
    }


def test_llamacpp_daily_build_records_release_date_and_latest_status(
    valid_envelope: dict[str, Any], monkeypatch: Any
) -> None:
    monkeypatch.setattr(engine_versions, "urlopen", _urlopen)
    envelope = _llama_envelope(valid_envelope, "b11459")

    engine_versions.populate_engine_release_metadata(envelope)

    software = envelope["campaign_dimensions"]["software"]
    assert software["engine"] == "llama.cpp"
    assert software["engine_version"] == "b11459"
    assert software["engine_build"] == "b11459"
    assert software["engine_release_date"] == "2026-10-07"
    assert software["engine_latest_stable_at_run"] == "v0.7.0"
    assert software["engine_is_latest_stable"] is False
    validate_envelope(envelope)
    assert envelope_to_rows(envelope)[0][campaign_dimension_column("software", "engine_build")] == "b11459"


def test_llamacpp_stable_release_fills_tagged_build(valid_envelope: dict[str, Any], monkeypatch: Any) -> None:
    monkeypatch.setattr(engine_versions, "urlopen", _urlopen)
    envelope = _llama_envelope(valid_envelope, "v0.7.0")

    engine_versions.populate_engine_release_metadata(envelope)

    software = envelope["campaign_dimensions"]["software"]
    assert software["engine_version"] == "v0.7.0"
    assert software["engine_build"] == "b11669"
    assert software["engine_release_date"] == "2026-10-15"
    assert software["engine_latest_stable_at_run"] == "v0.7.0"
    assert software["engine_is_latest_stable"] is True


def test_vllm_pypi_release_date_and_missing_build_reason(
    valid_envelope: dict[str, Any], monkeypatch: Any
) -> None:
    monkeypatch.setattr(engine_versions, "urlopen", _urlopen)
    envelope = {
        **valid_envelope,
        "timestamp": "2026-10-16T00:00:00Z",
        "system": {**valid_envelope["system"], "engine": {"name": "vllm", "version": "0.31.0"}},
        "campaign_dimensions": {"software": {"backend": "CUDA"}},
    }

    engine_versions.populate_engine_release_metadata(envelope)

    software = envelope["campaign_dimensions"]["software"]
    assert software["engine"] == "vLLM"
    assert software["engine_version"] == "0.31.0"
    assert software["engine_build"] is None
    assert software["engine_release_date"] == "2026-10-05"
    assert software["engine_latest_stable_at_run"] == "0.31.0"
    assert software["engine_is_latest_stable"] is True
    assert envelope["dimension_null_reasons"]["software.engine_build"] == "not_applicable"
    validate_envelope(envelope)


def test_pypi_latest_release_uses_upload_timestamp(monkeypatch: Any) -> None:
    monkeypatch.setattr(engine_versions, "urlopen", _urlopen)
    metadata = engine_versions._pypi_metadata(
        "vllm",
        "0.31.0",
        engine_versions._datetime("2026-10-05T07:00:00Z"),
    )

    assert metadata["engine_latest_stable_at_run"] == "0.30.0"
    assert metadata["engine_is_latest_stable"] is False


def test_release_lookup_failure_keeps_publishable_null_reasons(
    valid_envelope: dict[str, Any], monkeypatch: Any
) -> None:
    def fail(request: Any, *, timeout: int) -> io.BytesIO:
        raise URLError("fixture failure")

    monkeypatch.setattr(engine_versions, "urlopen", fail)
    envelope = _llama_envelope(valid_envelope, "b11459")

    engine_versions.populate_engine_release_metadata(envelope)

    software = envelope["campaign_dimensions"]["software"]
    assert software["engine_build"] == "b11459"
    for field in ("engine_release_date", "engine_latest_stable_at_run", "engine_is_latest_stable"):
        assert software[field] is None
        assert envelope["dimension_null_reasons"][f"software.{field}"] == "not_applicable"
    validate_envelope(envelope)
