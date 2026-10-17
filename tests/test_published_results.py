from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from huggingface_hub import HfApi

from mlx_benchmarks.published_results import build_published_results
from mlx_benchmarks.result_contract import MODEL_FIELDS


@pytest.mark.parametrize(
    ("license_link", "is_relative"),
    [
        ("LICENSE", True),
        ("./LICENSE", True),
        ("https://example.com/LICENSE", False),
    ],
)
def test_model_card_license_links_resolve_to_pinned_hub_blob(
    valid_envelope: dict[str, Any],
    published_metadata: dict[str, Any],
    mock_hf_registry: None,
    monkeypatch: pytest.MonkeyPatch,
    license_link: str,
    is_relative: bool,
) -> None:
    original_model_info = HfApi.model_info

    def model_info(self: Any, repo_id: str, revision: str | None = None, **kwargs: Any) -> Any:
        info = original_model_info(self, repo_id=repo_id, revision=revision, **kwargs)
        info.card_data = {**info.card_data, "license": "other", "license_link": license_link}
        return info

    monkeypatch.setattr(HfApi, "model_info", model_info)
    [row] = build_published_results(valid_envelope, metadata=published_metadata, api=HfApi())

    expected = (
        f"https://huggingface.co/{valid_envelope['model']}/blob/{valid_envelope['model_revision']}/LICENSE"
        if is_relative
        else license_link
    )
    assert row["license_link"] == expected


def test_model_card_nulls_get_not_applicable_reasons(
    valid_envelope: dict[str, Any],
    published_metadata: dict[str, Any],
    mock_hf_registry: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def model_info(self: Any, repo_id: str, revision: str | None = None, **kwargs: Any) -> Any:
        return SimpleNamespace(
            id=repo_id,
            sha=revision,
            pipeline_tag=None,
            card_data={},
            config={},
            safetensors=None,
            gguf=None,
            library_name=None,
            gated=None,
            tags=[],
        )

    monkeypatch.setattr(HfApi, "model_info", model_info)
    [row] = build_published_results(valid_envelope, metadata=published_metadata, api=HfApi())

    null_model_fields = {field for field in MODEL_FIELDS if row[field] is None}
    reasons = row["dimension_null_reasons"]
    assert null_model_fields
    assert all(reasons[field] == "not_applicable" for field in null_model_fields)
