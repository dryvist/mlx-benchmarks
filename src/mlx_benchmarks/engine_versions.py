"""Best-effort upstream release metadata for benchmark engine dimensions."""

from __future__ import annotations

import json
import logging
import re
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any, cast
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from mlx_benchmarks.envelope import CampaignDimensions, Envelope

log = logging.getLogger(__name__)

# The engine name remains the benchmark-facing label; the table selects its
# upstream release source without inferring a different engine from a version.
ENGINE_SOURCES = {
    "llama.cpp": ("github", "ggml-org/llama.cpp"),
    "vLLM": ("pypi", "vllm"),
    "SGLang": ("pypi", "sglang"),
    "MLX": ("pypi", "mlx-lm"),
}
_ENGINE_LABELS = {
    "llama.cpp": "llama.cpp",
    "vllm": "vLLM",
    "sglang": "SGLang",
    "mlx": "MLX",
    "strata": "Strata",
}

_RELEASE_FIELDS = (
    "engine_build",
    "engine_release_date",
    "engine_latest_stable_at_run",
    "engine_is_latest_stable",
)
_BUILD_RE = re.compile(r"^b\d+$")


def _request(url: str, *, accept: str = "application/json") -> bytes:
    request = Request(url, headers={"Accept": accept, "User-Agent": "mlx-benchmarks"})
    with urlopen(request, timeout=5) as response:
        return response.read()


def _json(url: str) -> Any:
    return json.loads(_request(url))


def _datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _version_key(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"[vV]?(\d+(?:\.\d+)*)(?:\.post(\d+))?", value)
    if not match:
        return None
    return (*tuple(int(part) for part in match.group(1).split(".")), int(match.group(2) or 0))


def _run_time(envelope: Envelope) -> datetime | None:
    return _datetime(envelope.get("timestamp"))


def _pypi_release_time(releases: dict[str, Any], version: str) -> datetime | None:
    files = releases.get(version)
    if not isinstance(files, list):
        return None
    uploaded = [
        parsed
        for item in files
        if isinstance(item, dict)
        and not item.get("yanked", False)
        and (parsed := _datetime(item.get("upload_time_iso_8601") or item.get("upload_time")))
    ]
    return min(uploaded) if uploaded else None


def _pypi_metadata(project: str, version: str | None, run_at: datetime | None) -> dict[str, Any]:
    data = _json(f"https://pypi.org/pypi/{quote(project, safe='')}/json")
    releases = data.get("releases") if isinstance(data, dict) else None
    if not isinstance(releases, dict):
        raise ValueError("PyPI returned no release map")

    stable: list[tuple[str, tuple[int, ...], datetime | None]] = []
    for key in releases:
        version_key = _version_key(key)
        if version_key is not None:
            stable.append((key, version_key, _pypi_release_time(releases, key)))
    eligible = [
        (key, version_key)
        for key, version_key, released in stable
        if released is not None and (run_at is None or released <= run_at)
    ]
    latest = max(eligible, key=lambda item: item[1])[0] if eligible else None
    recorded_version = version.removeprefix("v").removeprefix("V") if isinstance(version, str) else None
    release_time = _pypi_release_time(releases, recorded_version) if recorded_version else None
    return {
        "engine_version": recorded_version,
        "engine_release_date": release_time.date().isoformat() if release_time else None,
        "engine_latest_stable_at_run": latest,
        "engine_is_latest_stable": (
            _version_key(recorded_version) == _version_key(latest)
            if recorded_version is not None and latest is not None
            else None
        ),
    }


def _github_releases(repository: str) -> list[dict[str, Any]]:
    data = _json(f"https://api.github.com/repos/{repository}/releases?per_page=100")
    if not isinstance(data, list):
        raise ValueError("GitHub returned no release list")
    return [item for item in data if isinstance(item, dict)]


def _nightly_build(release: dict[str, Any]) -> str | None:
    assets = release.get("assets")
    if not isinstance(assets, list):
        return None
    asset = next(
        (item for item in assets if isinstance(item, dict) and item.get("name") == "nightly-tag.txt"),
        None,
    )
    url = asset.get("browser_download_url") if isinstance(asset, dict) else None
    if not isinstance(url, str):
        return None
    value = _request(url, accept="application/octet-stream").decode("utf-8").strip()
    return value or None


def _github_metadata(
    repository: str,
    version: str | None,
    build: str | None,
    run_at: datetime | None,
) -> dict[str, Any]:
    releases = _github_releases(repository)
    stable = [
        release
        for release in releases
        if not release.get("draft")
        and not release.get("prerelease")
        and _version_key(release.get("tag_name")) is not None
    ]
    eligible = [
        release
        for release in stable
        if (published := _datetime(release.get("published_at"))) and (run_at is None or published <= run_at)
    ]
    latest = (
        max(eligible, key=lambda release: _version_key(release.get("tag_name")) or ())["tag_name"]
        if eligible
        else None
    )

    selected = next(
        (release for release in releases if release.get("tag_name") in {version, build}),
        None,
    )
    latest_release = next((release for release in stable if release.get("tag_name") == latest), None)
    latest_build = None
    if latest_release:
        with suppress(OSError, URLError, UnicodeDecodeError, ValueError):
            latest_build = _nightly_build(latest_release)
    selected_build = None
    if selected in stable:
        with suppress(OSError, URLError, UnicodeDecodeError, ValueError):
            selected_build = latest_build if selected is latest_release else _nightly_build(selected)
    if selected is None and (_BUILD_RE.fullmatch(version or "") or _BUILD_RE.fullmatch(build or "")):
        for release in stable:
            try:
                asset_build = latest_build if release is latest_release else _nightly_build(release)
            except (OSError, URLError, UnicodeDecodeError, ValueError):
                continue
            if asset_build in {version, build}:
                selected = release
                selected_build = asset_build
                break

    selected_version = selected.get("tag_name") if selected else version
    selected_date = _datetime(selected.get("published_at")) if selected else None
    if isinstance(version, str) and _BUILD_RE.fullmatch(version) and selected_version != version:
        version = selected_version
    return {
        "engine_version": version or selected_version,
        "engine_build": build
        or (version if isinstance(version, str) and _BUILD_RE.fullmatch(version) else None)
        or selected_build,
        "engine_release_date": selected_date.date().isoformat() if selected_date else None,
        "engine_latest_stable_at_run": latest,
        "engine_is_latest_stable": selected_version == latest if selected_version and latest else None,
    }


def populate_engine_release_metadata(envelope: Envelope) -> None:
    """Fill missing engine release fields; upstream failures stay non-fatal."""
    raw_dimensions = envelope.get("campaign_dimensions")
    if raw_dimensions is not None and not isinstance(raw_dimensions, dict):
        return
    dimensions = cast(CampaignDimensions, raw_dimensions or {})
    raw_software = dimensions.get("software")
    if raw_software is not None and not isinstance(raw_software, dict):
        return
    software = raw_software if isinstance(raw_software, dict) else {}
    raw_reasons = envelope.get("dimension_null_reasons")
    if raw_reasons is not None and not isinstance(raw_reasons, dict):
        return
    reasons = raw_reasons if isinstance(raw_reasons, dict) else {}

    system = envelope.get("system") or {}
    system_engine = system.get("engine") if isinstance(system, dict) else None
    if not software.get("engine") and isinstance(system_engine, dict):
        software["engine"] = system_engine.get("name")
    if not software.get("engine_version") and isinstance(system_engine, dict):
        software["engine_version"] = system_engine.get("version")
    engine = software.get("engine")
    if isinstance(engine, str):
        engine = _ENGINE_LABELS.get(engine.casefold(), engine)
        software["engine"] = engine
    version = software.get("engine_version")
    if not isinstance(version, str) or not version:
        version = None
    commit = software.get("engine_commit")
    build = software.get("engine_build")
    if not isinstance(build, str) or not build:
        build = version if isinstance(version, str) and _BUILD_RE.fullmatch(version) else commit
    if not isinstance(build, str) or not build:
        build = None
    if version is None and build is not None and _version_key(build) is not None:
        version = build

    source = ENGINE_SOURCES.get(engine) if isinstance(engine, str) else None
    try:
        if source is None:
            metadata = {}
        elif source[0] == "github":
            metadata = _github_metadata(source[1], version, build, _run_time(envelope))
        else:
            metadata = _pypi_metadata(source[1], version, _run_time(envelope))
    except (OSError, URLError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        log.warning("engine release lookup failed for %s: %s", engine, type(exc).__name__)
        metadata = {}

    if "engine" not in software:
        software["engine"] = engine
    if metadata.get("engine_version") and (
        not version or (isinstance(version, str) and _BUILD_RE.fullmatch(version))
    ):
        version = metadata["engine_version"]
    software["engine_version"] = version
    software["engine_build"] = build or metadata.get("engine_build")
    for field in _RELEASE_FIELDS[1:]:
        software[field] = metadata.get(field)

    dimensions["software"] = software
    envelope["campaign_dimensions"] = dimensions
    for field in ("engine", "engine_version", *_RELEASE_FIELDS):
        key = f"software.{field}"
        if software.get(field) is None:
            reasons[key] = "not_applicable"
        else:
            reasons.pop(key, None)
    envelope["dimension_null_reasons"] = reasons
