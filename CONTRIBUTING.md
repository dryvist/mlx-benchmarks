# Contributing

Thanks for your interest in mlx-benchmarks. This repo is small, opinionated,
and biased toward reusing upstream tools over writing new code.

## Developer setup

```sh
git clone https://github.com/JacobPEvans/mlx-benchmarks.git
cd mlx-benchmarks

# Install package + dev deps
uv sync --extra dev
# Or pip-based:
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Install the pre-commit hooks
.venv/bin/pre-commit install
```

For Nix users: `direnv allow` activates `flake.nix`.

## Local quality gates (what CI runs)

```sh
# Lint + format
.venv/bin/ruff check .
.venv/bin/ruff format --check .

# Type check (strict mode)
.venv/bin/pyright src/mlx_benchmarks

# Tests
.venv/bin/pytest tests space/tests

# Schema + TOML validator
.venv/bin/python scripts/validate_schema.py

# Dry-run publish against canonical fixture
.venv/bin/mlx-bench-publish tests/fixtures/lm_eval_results_sample.json \
  --kind lm-eval --suite reasoning --git-sha deadbeef --dry-run
```

All of these must pass before opening a PR. CI re-runs them on every push.

## Pull request workflow

1. Branch from `main` — use a `type/short-slug` naming pattern
   (`feat/add-lighteval`, `fix/lm-eval-timeout`, `docs/arch-diagram`).
2. Keep PRs focused. One conceptual change per PR.
3. Follow [Conventional Commits](https://www.conventionalcommits.org/) —
   `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, `ci:`, etc.
   `release-please` uses these to auto-generate `CHANGELOG.md`.
4. Include tests for any new converter logic, publisher behavior, or schema
   field. Fixtures live under `tests/fixtures/` and `examples/`.
5. If you add a new suite to the `schema.json` enum, also:
   - Add a valid envelope fixture demonstrating it.
   - Update `docs/schema.md`.
   - Update the viewer if it needs new chart handling.
   For breaking schema changes (version bump to v2+), follow [`docs/schema-migration.md`](docs/schema-migration.md).
6. Never manually edit `CHANGELOG.md` — `release-please` owns it.

## Writing a new converter

Implementations live in `src/mlx_benchmarks/converters/`. A converter is any
class that satisfies the `Converter` protocol from `base.py`:

```python
class MyConverter:
    kind = "my-tool"

    def build_envelope(self, raw: dict, ctx: ConverterContext) -> Envelope: ...
```

Register it in `converters/__init__.py::get_converter`. Add tests in
`tests/test_my_tool_converter.py` that round-trip a fixture through the
converter and call `validate_envelope(envelope)` to confirm schema compliance.

## What not to add

- Bespoke benchmark orchestration logic that duplicates an upstream tool.
- Suite enum values that don't correspond to a real measurement you plan to
  publish.
- Silent drops of malformed input — raise or record in `envelope.errors[]`.
- Hardcoded system metadata — use `detect_system()`.
- Runtime tooling that bypasses the pre-commit hooks or the CI quality gates.

## CI secrets (for forks)

Two workflows read repository secrets:

- `release-please.yml` — a GitHub App with write access to **contents**
  and **pull-requests** (App ID and private key).
- `deploy-space.yml` — a Hugging Face token with **write** scope on the
  target Space namespace, and `SPACE_REPO_ID` for the Space.

## Questions

Open an issue at
<https://github.com/JacobPEvans/mlx-benchmarks/issues>.
