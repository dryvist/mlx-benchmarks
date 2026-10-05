"""Verify schema.json is itself valid and that canonical examples pass."""

from __future__ import annotations

import pytest
from jsonschema import Draft7Validator

from mlx_benchmarks.dataset_schema import (
    CAMPAIGN_DIMENSION_TYPES,
    PARQUET_ROW_SCHEMA,
    campaign_dimension_column,
)
from mlx_benchmarks.envelope import (
    EnvelopeValidationError,
    load_schema,
    validate_envelope,
)


def test_schema_is_valid_draft7() -> None:
    Draft7Validator.check_schema(load_schema())


def test_schema_declares_id_and_examples() -> None:
    schema = load_schema()
    assert schema.get("$id"), "schema must declare $id so consumers can pin to a canonical URL"
    assert schema.get("examples"), "schema should ship at least one example"


def test_valid_envelope_passes(valid_envelope: dict) -> None:
    validate_envelope(valid_envelope)


def test_unknown_runtime_git_sha_can_be_null(valid_envelope: dict) -> None:
    validate_envelope({**valid_envelope, "git_sha": None})


def test_invalid_envelope_fails(invalid_envelope: dict) -> None:
    with pytest.raises(EnvelopeValidationError) as excinfo:
        validate_envelope(invalid_envelope)
    message = str(excinfo.value)
    # Known problems in the fixture: bad schema_version, bad timestamp, bad git_sha, bad trigger, bad suite, short system.
    assert "schema_version" in message or "suite" in message
    # Multiple errors collected, not just the first
    assert len(excinfo.value.errors) >= 2


def test_cluster_envelope_validates(cluster_envelope: dict) -> None:
    # A full two-node TB5 pipeline envelope: env_class + concurrency + serving +
    # system.topology all populated.
    validate_envelope(cluster_envelope)


def test_new_optional_top_level_fields_validate(valid_envelope: dict) -> None:
    env = {
        **valid_envelope,
        "env_class": "under-load",
        "concurrency": 8,
        "serving": {"stack": "vllm-mlx", "endpoint_port": 8000, "served_model": "m"},
    }
    validate_envelope(env)


def test_campaign_dimensions_are_optional_and_accept_explicit_nulls(valid_envelope: dict) -> None:
    # Legacy envelopes remain valid without campaign-specific dimensions.
    validate_envelope(valid_envelope)

    env = {
        **valid_envelope,
        "campaign_dimensions": {
            "hardware": {"machine": "Mac Studio", "pcie_generation": None},
            "software": {
                "backend": "Metal",
                "gpu_architectures": None,
                "build_flags": {"ENABLE_METAL": True},
            },
            "speed": {"ttft_p50_ms": 14.5, "itl_p99_ms": None},
        },
        "dimension_null_reasons": {
            "hardware.pcie_generation": "N/A_SOC",
            "software.gpu_architectures": "N/A_MLX",
            "speed.itl_p99_ms": "NOT_REPORTED",
        },
    }
    validate_envelope(env)


def test_campaign_dimensions_reject_unknown_fields(valid_envelope: dict) -> None:
    env = {
        **valid_envelope,
        "campaign_dimensions": {"hardware": {"invented_field": "value"}},
    }
    with pytest.raises(EnvelopeValidationError):
        validate_envelope(env)


def test_blocked_run_can_preserve_unknown_required_system_values_as_null(valid_envelope: dict) -> None:
    env = {
        **valid_envelope,
        "cell_status": "aborted",
        "system": {"os": None, "chip": None, "memory_gb": None},
    }
    validate_envelope(env)


def test_campaign_dimension_fields_match_the_fixed_parquet_schema() -> None:
    dimension_groups = load_schema()["properties"]["campaign_dimensions"]["properties"]
    assert set(dimension_groups) == set(CAMPAIGN_DIMENSION_TYPES)
    parquet_columns = set(PARQUET_ROW_SCHEMA.names)
    for group, fields in CAMPAIGN_DIMENSION_TYPES.items():
        properties = dimension_groups[group]["properties"]
        assert set(properties) == set(fields)
        for field in fields:
            assert "null" in properties[field]["type"]
            assert campaign_dimension_column(group, field) in parquet_columns


def test_env_class_enum_is_enforced(valid_envelope: dict) -> None:
    with pytest.raises(EnvelopeValidationError):
        validate_envelope({**valid_envelope, "env_class": "chaotic"})


def test_concurrency_minimum_is_enforced(valid_envelope: dict) -> None:
    with pytest.raises(EnvelopeValidationError):
        validate_envelope({**valid_envelope, "concurrency": 0})


def test_reasoning_effort_accepts_any_family_naming(valid_envelope: dict) -> None:
    """Not an enum on purpose: families name their own levels, and normalising
    one family's name into another's merges arms that are not the same arm. The
    schema's job is to carry the value, not to referee the vocabulary."""
    for level in ("low", "medium", "high", "xhigh", "max", "off", "unstated", "think-harder"):
        validate_envelope({**valid_envelope, "reasoning_effort": level})


def test_reasoning_effort_must_be_a_string(valid_envelope: dict) -> None:
    # A bare boolean is how effort got collapsed in the agentic harness; the
    # schema refuses it so that shape cannot reach a shard.
    with pytest.raises(EnvelopeValidationError):
        validate_envelope({**valid_envelope, "reasoning_effort": True})


def test_serving_rejects_unknown_key(valid_envelope: dict) -> None:
    with pytest.raises(EnvelopeValidationError):
        validate_envelope({**valid_envelope, "serving": {"stack": "x", "bogus": 1}})


def test_topology_parallelism_enum_is_enforced(valid_envelope: dict) -> None:
    env = {
        **valid_envelope,
        "system": {**valid_envelope["system"], "topology": {"parallelism": "quantum"}},
    }
    with pytest.raises(EnvelopeValidationError):
        validate_envelope(env)


def test_format_checker_rejects_non_iso_timestamp(valid_envelope: dict) -> None:
    """Targeted test: without format_checker=, jsonschema accepts any string
    for ``format: date-time``. The publisher contract — and the viewer's
    ``pd.to_datetime`` — requires real ISO-8601, so the validator must enforce it."""
    bad = dict(valid_envelope)
    bad["timestamp"] = "not-an-iso-date"
    with pytest.raises(EnvelopeValidationError) as excinfo:
        validate_envelope(bad)
    # Error must be scoped to the timestamp field specifically.
    paths = [list(e.absolute_path) for e in excinfo.value.errors]
    assert ["timestamp"] in paths
