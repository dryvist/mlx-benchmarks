"""Verify schema.json is itself valid and that canonical examples pass."""

from __future__ import annotations

import pytest
from jsonschema import Draft7Validator

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
