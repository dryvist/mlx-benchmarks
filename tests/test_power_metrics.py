"""Power metrics use the Prometheus-compatible HTTP API without extra dependencies."""

from __future__ import annotations

import io
import json
from urllib.parse import parse_qs, urlsplit

import mlx_benchmarks.power_metrics as power_metrics


def test_power_enrichment_queries_range_and_derives_energy(monkeypatch) -> None:
    start = 1792152000.0
    response = {
        "status": "success",
        "data": {
            "result": [
                {
                    "values": [[start, "100"], [start + 10, "200"]],
                }
            ]
        },
    }
    requests = []
    monkeypatch.setattr(
        power_metrics,
        "urlopen",
        lambda request, timeout: requests.append(request) or io.BytesIO(json.dumps(response).encode()),
    )
    envelope = {
        "timestamp": "2026-10-16T12:00:00Z",
        "system": {"os": "Ubuntu 26.04"},
        "results": [{"duration_seconds": 10}],
        "campaign_dimensions": {"run": {"phase_duration_seconds": 10, "measured_output_tokens": 50}},
    }

    power_metrics.enrich_power_metrics(envelope, "https://metrics.example/prometheus/api/v1/query_range")

    resource = envelope["campaign_dimensions"]["resource"]
    assert resource["power_avg_w"] == 150
    assert resource["power_max_w"] == 200
    assert resource["energy_j_per_output_token"] == 30
    assert resource["tokens_per_second_per_watt"] == 50 / 10 / 150
    request = requests[0]
    assert urlsplit(request.full_url).path == "/prometheus/api/v1/query_range"
    assert parse_qs(urlsplit(request.full_url).query)["query"] == ["sum(DCGM_FI_DEV_POWER_USAGE)"]
    assert parse_qs(urlsplit(request.full_url).query)["start"] == [f"{start:.3f}"]
    assert parse_qs(urlsplit(request.full_url).query)["end"] == [f"{start + 10:.3f}"]
    assert envelope["dimension_null_reasons"] == {}


def test_power_enrichment_queries_when_only_energy_is_missing(monkeypatch) -> None:
    start = 1792152000.0
    response = {"status": "success", "data": {"result": [{"values": [[start, "100"], [start + 10, "200"]]}]}}
    requests = []

    def open_response(request, timeout):
        requests.append(request)
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(power_metrics, "urlopen", open_response)
    envelope = {
        "timestamp": "2026-10-16T12:00:00Z",
        "system": {"os": "macOS"},
        "results": [{"duration_seconds": 10}],
        "campaign_dimensions": {
            "resource": {"power_avg_w": 150, "power_max_w": 200},
            "run": {"phase_duration_seconds": 10, "measured_output_tokens": 50},
        },
    }

    power_metrics.enrich_power_metrics(envelope, "https://metrics.example/api/v1")

    resource = envelope["campaign_dimensions"]["resource"]
    assert resource["energy_j_per_output_token"] == 30
    assert resource["tokens_per_second_per_watt"] == 50 / 10 / 150
    assert parse_qs(urlsplit(requests[0].full_url).query)["query"] == ["sum(macmon_sys_power_watts)"]
    assert envelope["dimension_null_reasons"] == {}


def test_power_enrichment_records_nullable_fields_when_url_is_unset(monkeypatch) -> None:
    monkeypatch.delenv(power_metrics.METRICS_URL_ENV, raising=False)
    envelope = {"timestamp": "2026-10-16T12:00:00Z", "system": {}, "results": []}

    power_metrics.enrich_power_metrics(envelope)

    resource = envelope["campaign_dimensions"]["resource"]
    assert {resource[field] for field in power_metrics.POWER_FIELDS} == {None}
    assert set(envelope["dimension_null_reasons"].values()) == {"METRICS_URL_UNSET"}
