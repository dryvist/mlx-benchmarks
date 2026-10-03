# 2026-07-19 — prompt-eval framework selection + scaffold deliverables

Session outcome: the framework decision is recorded in
[`../prompt-eval-framework.md`](../prompt-eval-framework.md) (promptfoo in a
new `dryvist/llm-prompt-evals` repo; promptstack superseded; Langfuse and a
homelab runner as follow-on phases; Galileo as the API-app lane). This entry
preserves the two hand-off artifacts: the goal doc and the ready-to-run
Claude Code prompt. **Primary repo to run them in: `dryvist/llm-prompt-evals`**
(create it empty first — MIT license, main-branch protection per org norms).

## Artifact 1 — GOAL.md (drop into the new repo root; under 4k chars)

```markdown
# llm-prompt-evals — goal

Quantifiably compare slightly-different system prompts and make the result
readable at a glance. Engine: promptfoo (pinned). Prompt variants x providers
x tests produce a side-by-side matrix via `promptfoo view`, HTML/CSV export,
and PR comments from the official GitHub Action. Primary target: the homelab
LLM fabric (OpenAI-compatible endpoints). Also first-class: Anthropic API
(`anthropic:`) and OpenRouter (`openrouter:`). Volume stays under ~1k
evals/week; free tiers are fine.

## Non-negotiables

1. Single prompt source: canonical prompts come ONLY from the `catalog/`
   submodule (`dryvist/ai-llm-prompts`, pinned to a release commit,
   Renovate-bumped). Never copy a canonical prompt body into this repo.
   Experimental wording lives in `variants/<surface>/` until adopted
   upstream.
2. OKF loading: prompts load through `prompts/load_okf.py` (promptfoo Python
   prompt function) which strips YAML frontmatter and honors the prompt's
   `render` policy.
3. Assert ordering: deterministic assertions (contains/regex/is-json/
   latency/cost) come first and gate; `llm-rubric`/`g-eval` judgments come
   second. A rubric score never overrides a failed deterministic check.
4. Grader rule: the default `llm-rubric` grader is one cheap pinned
   OpenRouter model, overridable to a local fabric model for zero-cost runs.
   Never grade a model with itself.
5. Verdict language: results say a variant "leads/lags as of N runs" —
   never "best/worst". Re-run before adopting; single unreplicated runs
   do not decide adoption.
6. Security: no secrets, fabric hostnames, IPs, or ports committed —
   endpoints come from env vars. Fabric
   CI jobs run only on the homelab self-hosted runner and never for fork
   PRs (push/workflow_dispatch/same-repo labeled PRs only).
7. Pin everything: promptfoo version in package.json, catalog submodule to
   a release commit, grader model id. Renovate manages bumps.

## Layout

catalog/ (submodule) - prompts/load_okf.py - variants/<surface>/ -
providers/{local,cloud}.yaml - evals/<surface>/{promptfooconfig,tests}.yaml -
datasets/ (shared cases; migrate mlx-benchmarks promptstack probe banks:
reasoning, tool_call incl. negative bank, instruction, homelab_qa) -
scripts/report.sh - results/ (committed markdown only when recording an
adopt/reject decision) - .github/workflows/eval.yml

## Definition of done (initial scaffold)

- `promptfoo eval` green locally against the fabric endpoint AND in CI
  against Anthropic + OpenRouter.
- The hermes eval compares canonical `hermes.md` vs at least one variant
  with >=8 deterministic + >=2 rubric tests; PR comment renders the
  comparison.
- Wiki updated: an "LLM prompt evaluation" page in `dryvist/docs` and
  `dryvist/docs-starlight` mapping ai-llm-prompts -> llm-prompt-evals ->
  consumers, cross-linked from the Hermes/fabric pages; both link back to
  mlx-benchmarks `docs/prompt-eval-framework.md`.
```

## Artifact 2 — ready-to-run Claude Code prompt

Run in a fresh Claude Code session in `dryvist/llm-prompt-evals`:

```text
Read GOAL.md in the repo root and treat it as binding. Scaffold this repo as
the dryvist prompt-eval framework it describes:

1. Add dryvist/ai-llm-prompts as a git submodule at catalog/, pinned to its
   latest release commit; configure Renovate (git-submodules manager) to bump
   it. package.json pins promptfoo; add .env.example, .gitignore
   (.env, node_modules/, output/), README.md, AGENTS.md (distilled from
   GOAL.md's non-negotiables).
2. Implement prompts/load_okf.py: a promptfoo Python prompt function that
   loads an OKF markdown prompt by path or prompt:// resource id, strips
   YAML frontmatter, and returns the system prompt string. Unit-test the
   frontmatter stripping.
3. Build providers/local.yaml (OpenAI-compatible, base URL from
   LOCAL_FABRIC_BASE_URL env var, model from env) and providers/cloud.yaml
   (anthropic: and openrouter: entries; pinned cheap OpenRouter model as the
   default llm-rubric grader via defaultTest).
4. Build evals/hermes/: promptfooconfig.yaml comparing the canonical
   catalog/auto-ai-agent/hermes.md against variants/hermes/candidate-a.md
   (create one candidate with a small, clearly-commented wording change),
   with tests.yaml holding >=8 deterministic assertions + >=2 llm-rubric
   assertions. Port test cases from mlx-benchmarks
   configs/promptstack/probes/*.json (reasoning, tool_call including the
   negative bank, instruction, homelab_qa) into datasets/ as promptfoo test
   cases, preserving the negative tool-call semantics (a fabricated call
   must FAIL).
5. Wire .github/workflows/eval.yml using promptfoo/promptfoo-action with
   caching: cloud-provider matrix on same-repo pull requests (never fork
   secrets); a separate fabric job gated to push/workflow_dispatch/labeled
   same-repo PRs on the self-hosted homelab runner label, endpoints read
   from runner env only. Add scripts/report.sh producing
   output/latest.{json,html} plus a markdown summary.
6. Verify for real: run `promptfoo eval` against OpenRouter (and the fabric
   if reachable) and include the command + result in the PR body. Never
   commit secrets or fabric endpoints.
7. Open a PR here (conventional commits), then separate PRs to dryvist/docs
   and dryvist/docs-starlight adding the "LLM prompt evaluation" wiki page
   per GOAL.md's definition of done.
```

## Follow-on phases (run later, each in its owning repo)

- `ansible-proxmox-apps`: `langfuse_docker` role (web, worker, postgres,
  clickhouse, redis, minio; follow the `n8n_docker` compose pattern, with
  traefik and OpenBao secrets), then a homelab GitHub runner for the fabric
  CI job via the existing `github_runner` role.
- API apps: wire the Galileo free tier as the observability/eval lane.
- This repo: retire `harness/promptstack/` once `llm-prompt-evals` covers
  all four probe classes.
