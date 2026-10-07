"""Integration checks for the run-suite endpoint preflight."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = "mlx-community/test-model"


def _dry_run_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    proof_file = tmp_path / "curl-proof"
    (bin_dir / "curl").write_text(
        """#!/bin/sh
set -eu
case " $* " in
  *" $TEST_API_KEY "*) exit 90 ;;
esac
case " $* " in
  *" -H @- "*) ;;
  *) exit 91 ;;
esac
IFS= read -r header || exit 92
[ "$header" = "Authorization: Bearer $TEST_API_KEY" ] || exit 93
printf 'header-ok' > "$CURL_PROOF"
printf '{"data":[{"id":"mlx-community/test-model"}]}'
""",
        encoding="utf-8",
    )
    (bin_dir / "curl").chmod(0o755)
    (bin_dir / "pgrep").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    (bin_dir / "pgrep").chmod(0o755)

    cache_dir = tmp_path / "hf" / "hub" / "models--mlx-community--test-model"
    snapshot_dir = cache_dir / "snapshots" / "fixture"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "weights.safetensors").touch()

    env = os.environ.copy()
    env.update(
        {
            "BENCH_OUT_DIR": str(tmp_path / "out"),
            "CURL_PROOF": str(proof_file),
            "HF_HOME": str(tmp_path / "hf"),
            "MLX_EVAL_CONCURRENT": "1",
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
            "TEST_API_KEY": "fixture-key",
        }
    )
    return env, proof_file


def _run_suite(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "run-suite.sh"),
            "--dry-run",
            "--no-window",
            "--suites",
            "throughput",
            MODEL,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )


def test_endpoint_probe_uses_api_key_without_putting_it_in_curl_arguments(tmp_path: Path) -> None:
    env, proof_file = _dry_run_environment(tmp_path)
    env["OPENAI_API_KEY"] = env["TEST_API_KEY"]

    result = _run_suite(env)

    assert result.returncode == 0, result.stderr
    assert proof_file.read_text(encoding="utf-8") == "header-ok"
    assert env["TEST_API_KEY"] not in result.stdout + result.stderr


def test_endpoint_probe_requires_api_key_before_request(tmp_path: Path) -> None:
    env, proof_file = _dry_run_environment(tmp_path)
    env.pop("OPENAI_API_KEY", None)

    result = _run_suite(env)

    assert result.returncode != 0
    assert "OPENAI_API_KEY is required" in result.stderr
    assert not proof_file.exists()
