"""Public result projections must not contain host or network identifiers."""

from __future__ import annotations

from typing import Any

PRIVATE_IDENTIFIER_KEYS = frozenset(
    {"hostname", "host", "ip", "address", "node", "node_name", "host_ip", "node_ip"}
)


def remove_private_identifiers(value: Any) -> Any:
    """Remove host and network identity keys recursively from public metadata."""
    if isinstance(value, dict):
        return {
            key: remove_private_identifiers(item)
            for key, item in value.items()
            if key.casefold() not in PRIVATE_IDENTIFIER_KEYS
        }
    if isinstance(value, list):
        return [remove_private_identifiers(item) for item in value]
    return value


def contains_private_identifier_keys(value: Any) -> bool:
    """Return whether metadata contains a host or network identity key."""
    if isinstance(value, dict):
        return any(
            key.casefold() in PRIVATE_IDENTIFIER_KEYS or contains_private_identifier_keys(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(contains_private_identifier_keys(item) for item in value)
    return False
