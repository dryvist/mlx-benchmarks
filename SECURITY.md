# Security Policy

## Supported versions

`main` is the only supported branch. Published tags follow semver; security
fixes land on `main` and are released as patch versions.

## Reporting a vulnerability

Open a [private security advisory](https://github.com/JacobPEvans/mlx-benchmarks/security/advisories/new)
on GitHub rather than a public issue. If for some reason that is not
possible, email `jacob.p.evans@gmail.com` with details and a suggested
disclosure timeline.

Expect an acknowledgement within 72 hours.

## Credential handling

- `HF_TOKEN` is read only from the environment (never from a file in the
  repo). The publisher exits early if it is unset for a live run.
- Never commit `.env*` files with real tokens.
- CI workflows that need `HF_TOKEN` (only `deploy-space.yml`) reference it
  via `${{ secrets.HF_TOKEN }}` and pipe it through an `env:` block, never
  into a `run:` string.

## Unsafe code execution

The `coding` suite configs pass `HF_ALLOW_CODE_EVAL=1` and
`--confirm_run_unsafe_code` to lm-eval, which executes model-generated
Python to score HumanEval / MBPP. Assume that output is arbitrary code.

Run coding benchmarks in a throwaway VM or container if you are worried
about a model emitting malicious code against your shell. We do not run
coding benchmarks in GitHub Actions.

## Third-party action pinning

Per repository-level policy, trusted-org actions
(`actions/*`, `github/codeql-action/*`, `astral-sh/*`, `googleapis/*`) are
pinned to major-version tags. Untrusted external actions would be pinned by
commit SHA; none are currently used.

Renovate keeps these up to date — see `renovate.json`.

## Known behaviors that are not vulnerabilities

- The viewer accepts and renders any Parquet shard currently in the HF
  dataset. A malicious shard can distort charts but cannot escape the
  sandbox of the Gradio Space.
- `HF_TOKEN` is required to **publish** but not to read the public dataset.
