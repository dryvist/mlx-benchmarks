#!/usr/bin/env bash
# Run the benchmark suites for one model, end to end, on this host.
#
# This is a THIN wrapper around docs/RUNBOOK.md Step 4. It does not reimplement
# any suite — it sequences the commands the runbook already specifies, marks
# each one so a long unattended run stays legible, and refuses to start when a
# precondition would waste hours.
#
# It deliberately does NOT orchestrate serving. The runbook's managed-window
# path (bootout the worker, serve solo, restore) is operator work with real
# blast radius; this script instead ASSERTS the endpoint already serves the
# target model and tells you what to change if it does not.
#
# Usage:
#   scripts/run-suite.sh <model-id> [options]
#
#   --suites a,b,c   subset of: throughput,coding,math,reasoning,agentic
#                    (default: all, in that order — throughput first)
#   --limit N        cap examples per lm-eval task (smoke runs)
#   --dry-run        validate everything and print commands; run nothing long
#   --no-window      skip the maintenance-window open/close
#   --base-url URL   default http://127.0.0.1:11434/v1
#
# Self-check (runs the whole flow in ~a minute, exercising every branch):
#   scripts/run-suite.sh --dry-run --limit 1 mlx-community/<some-cached-id>
set -euo pipefail

BASE_URL="http://127.0.0.1:11434/v1"
SUITES="throughput,coding,math,reasoning,agentic,factual,promptstack"
LIMIT=""
DRY_RUN=0
NO_WINDOW=0
MODEL=""

VIKUNJA_URL="${VIKUNJA_URL:-https://vikunja.jacobpevans.com/api/v1}"
VIKUNJA_MAINTENANCE_PROJECT=54

die() {
  echo "run-suite: $*" >&2
  exit 1
}

mark() {
  # The marker format a log-follower greps for. Keep START/DONE/FAIL literal.
  echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) $1 $2 ====="
}

while [ $# -gt 0 ]; do
  case "$1" in
    --suites) SUITES="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-window) NO_WINDOW=1; shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    -*) die "unknown option: $1" ;;
    *) [ -n "$MODEL" ] && die "unexpected argument: $1"; MODEL="$1"; shift ;;
  esac
done

[ -n "$MODEL" ] || die "model id required (e.g. mlx-community/Qwen3.8-27B-4bit)"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SLUG="$(echo "$MODEL" | tr '/.' '__')"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${BENCH_OUT_DIR:-$HOME/bench-runs}/${SLUG}-${STAMP}"

# ---------------------------------------------------------------- preflight --
# Each check below has cost hours at least once. None of them are optional.

# 1. Weights actually cached. Workers run HF_HUB_OFFLINE=1, so an uncached id
#    does not download — it 502s after minutes of retrying with nothing saying
#    why. Registration makes an id servable, never cached.
HF_HUB="${HF_HOME:-/Volumes/HuggingFace}/hub"
CACHE_DIR="$HF_HUB/models--${MODEL//\//--}"
[ -d "$CACHE_DIR" ] || die "not cached: $MODEL
  expected $CACHE_DIR
  fix: hf download $MODEL"

# A metadata-only stub passes the directory test and fails at load, so require
# real weight shards. Note -L: inside an HF cache the snapshot entries are
# SYMLINKS into blobs/, so a plain `-type f` matches nothing even for a fully
# downloaded model. Following links also makes this catch the genuinely broken
# case — a dangling symlink whose blob was pruned away.
if ! find -L "$CACHE_DIR" -name '*.safetensors' -type f 2>/dev/null | grep -q .; then
  die "cached but has NO usable weight files (metadata-only stub, or blobs pruned): $MODEL
  fix: hf download $MODEL"
fi

# 2. The endpoint serves this exact id. IPv4-literal on purpose: the reverse
#    proxy holds the same port on IPv6/TLS, so a plain localhost probe can hit
#    the wrong listener and report a confusing failure.
PROBE_URL="${BASE_URL%/v1}"
SERVED="$(curl -s4 --max-time 10 "$PROBE_URL/v1/models" 2>/dev/null || true)"
[ -n "$SERVED" ] || die "no response from $PROBE_URL/v1/models — is the model server up?"

if ! echo "$SERVED" | grep -q "\"$MODEL\""; then
  die "endpoint is not serving $MODEL
  served ids: $(echo "$SERVED" | grep -o '"id":"[^"]*"' | cut -d'"' -f4 | tr '\n' ' ')
  This script does not repoint serving. Change the host's resident/singleModel
  setting and converge, then re-run."
fi

# 3. Never benchmark a machine that is mid-cluster: ranks and a benchmark
#    contend for the same unified memory and both results are garbage.
if pgrep -f 'cluster-rank-launch|mlx-cluster.rank' >/dev/null 2>&1; then
  die "cluster ranks are running — tear the cluster down before benchmarking"
fi

# 4. Concurrency must match the endpoint's own limit. The mlx-eval wrapper
#    defaults to 1, which silently multiplies every duration below.
#
#    Default 2, not 4: the deployed --decode-concurrency is 2, and a probe
#    wider than the endpoint collects HTTP 429 on the excess and averages the
#    failures into the aggregate — it reads as a throughput regression when it
#    is a probe error (benchmark-traps.md Trap 11, hit live 2026-08-14).
#    Override only after confirming the endpoint's real limit.
CONCURRENCY="${MLX_EVAL_CONCURRENT:-2}"

# 5. Generation budget. MUST be large enough for a thinking model to finish
#    reasoning AND emit an answer. The throughput probe's own default is 256,
#    which a thinking-on model can spend entirely on reasoning — producing a
#    healthy-looking tok/s row with zero answer characters. 4096 matches what
#    the lm-eval qwen3 overlays already set for the same reason ("thinking text
#    consumes 1-3k tokens before code"). Lower it only for a non-thinking model,
#    and say so when you publish.
SUITE_MAX_TOKENS="${SUITE_MAX_TOKENS:-4096}"

# Optional flags as arrays so an empty LIMIT contributes zero words rather than
# an empty string argument (which lm-eval parses as a bad value, not as absent).
LIMIT_ARGS=()
AGENTIC_LIMIT_ARGS=()
if [ -n "$LIMIT" ]; then
  LIMIT_ARGS=(--limit "$LIMIT")
  AGENTIC_LIMIT_ARGS=(--repeats "$LIMIT")
fi

echo "run-suite: model=$MODEL"
echo "run-suite: suites=$SUITES concurrency=$CONCURRENCY limit=${LIMIT:-none}"
echo "run-suite: output=$OUT_DIR"
[ "$DRY_RUN" -eq 1 ] && echo "run-suite: DRY RUN — commands are printed, not executed"

mkdir -p "$OUT_DIR"

# ------------------------------------------------------- maintenance window --
WINDOW_TASK_ID=""

open_window() {
  [ "$NO_WINDOW" -eq 1 ] && return 0
  [ "$DRY_RUN" -eq 1 ] && return 0
  local pw jwt
  pw="$(bao kv get -field=svc_mcp_rw_password secret/apps/vikunja 2>/dev/null || true)"
  if [ -z "$pw" ]; then
    die "cannot open a maintenance window (no Vikunja credential).
  A benchmark saturates this host for hours; other agents need to see that.
  Re-run under the credential, or pass --no-window to proceed deliberately."
  fi
  jwt="$(curl -s --max-time 15 -X POST -H 'Content-Type: application/json' \
    -d "{\"username\":\"svc-mcp-rw\",\"password\":\"$pw\"}" \
    "$VIKUNJA_URL/login" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("token",""))' 2>/dev/null || true)"
  [ -n "$jwt" ] || die "Vikunja login failed — cannot open maintenance window"
  # Creation is PUT on this API version, not POST.
  WINDOW_TASK_ID="$(curl -s --max-time 15 -X PUT -H "Authorization: Bearer $jwt" \
    -H 'Content-Type: application/json' \
    -d "{\"title\":\"$(hostname -f)\",\"description\":\"benchmark run: $MODEL ($SLUG-$STAMP)\"}" \
    "$VIKUNJA_URL/projects/$VIKUNJA_MAINTENANCE_PROJECT/tasks" \
    | python3 -c 'import json,sys;print(json.load(sys.stdin).get("id",""))' 2>/dev/null || true)"
  VIKUNJA_JWT="$jwt"
  echo "run-suite: maintenance window opened (task ${WINDOW_TASK_ID:-unknown})"
}

close_window() {
  [ -n "$WINDOW_TASK_ID" ] || return 0
  curl -s --max-time 15 -X POST -H "Authorization: Bearer $VIKUNJA_JWT" \
    -H 'Content-Type: application/json' -d '{"done":true}' \
    "$VIKUNJA_URL/tasks/$WINDOW_TASK_ID" >/dev/null 2>&1 || true
  echo "run-suite: maintenance window closed (task $WINDOW_TASK_ID)"
}
trap close_window EXIT

# --------------------------------------------------------------- suite defs --
# One function per suite. Each echoes its command when dry-running so the
# self-check exercises the same code path the real run takes.

run_cmd() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  would run: $*"
    return 0
  fi
  "$@"
}

suite_throughput() {
  # First on purpose: it is the cheapest suite and the one that answers
  # "did switching models cost us tokens/sec". cumulative_tok_s is the
  # headline the published baselines compare on.
  run_cmd uv run harness/throughput/run.py \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --repeats "${THROUGHPUT_REPEATS:-4}" \
    --concurrency "$CONCURRENCY" \
    --max-tokens "$SUITE_MAX_TOKENS" \
    --output "$OUT_DIR/throughput.json"
}

suite_coding() {
  # The qwen3 overlay is mandatory: plain humaneval/mbpp score ~0 against a
  # chat-served model. Executes model-generated code — see SECURITY.md.
  export HF_ALLOW_CODE_EVAL=1 MLX_EVAL_CONCURRENT="$CONCURRENCY"
  run_cmd mlx-eval \
    --include_path configs/lm-eval/qwen3-tasks \
    --tasks humaneval_instruct_qwen3,mbpp_instruct_qwen3 \
    --confirm_run_unsafe_code --log_samples \
    "${LIMIT_ARGS[@]}" \
    --output_path "$OUT_DIR/coding"
}

suite_math() {
  # Read math_verify, not exact_match, when interpreting the result.
  export MLX_EVAL_CONCURRENT="$CONCURRENCY"
  run_cmd mlx-eval minerva_math500 \
    "${LIMIT_ARGS[@]}" \
    --output_path "$OUT_DIR/math"
}

suite_reasoning() {
  export MLX_EVAL_CONCURRENT="$CONCURRENCY"
  run_cmd mlx-eval --tasks arc_challenge_chat \
    "${LIMIT_ARGS[@]}" \
    --output_path "$OUT_DIR/reasoning"
}

suite_agentic() {
  run_cmd uv run harness/agentic/run.py \
    --base-url "$BASE_URL" \
    --api-key-env OPENAI_API_KEY \
    --model "$MODEL" \
    "${AGENTIC_LIMIT_ARGS[@]}" \
    --output "$OUT_DIR/agentic.json"
}

suite_factual() {
  # Hallucination / grounding: fact recall, fabricated numbers, tool-syntax
  # leakage. Scored by assertion, never by a model judge.
  run_cmd uv run harness/factual/run.py \
    --base-url "$BASE_URL" \
    --api-key-env OPENAI_API_KEY \
    --model "$MODEL" \
    --max-tokens "$SUITE_MAX_TOKENS" \
    --output "$OUT_DIR/factual.json"
}

suite_promptstack() {
  # Prompt regression: format-constraint adherence and the negative tool-call
  # probes (the model must NOT fabricate a call).
  run_cmd uv run harness/promptstack/run.py \
    --base-url "$BASE_URL" \
    --api-key-env OPENAI_API_KEY \
    --model "$MODEL" \
    --max-tokens "$SUITE_MAX_TOKENS" \
    --output "$OUT_DIR/promptstack.json"
}

# ------------------------------------------------------------------- driver --
open_window

FAILED=""
IFS=',' read -ra SUITE_LIST <<< "$SUITES"
for suite in "${SUITE_LIST[@]}"; do
  case "$suite" in
    throughput|coding|math|reasoning|agentic|factual|promptstack) ;;
    *) die "unknown suite: $suite" ;;
  esac
  mark START "$suite"
  # A failing suite must not abort the ones after it — hours of completed work
  # would be thrown away for one bad shard. Record and continue.
  if "suite_$suite"; then
    mark DONE "$suite"
  else
    mark FAIL "$suite"
    FAILED="$FAILED $suite"
  fi
done

# ------------------------------------------------------------------ publish --
# Dry-run only, always. A real publish needs the write token and is an explicit
# human step — never a side effect of finishing a benchmark.
if [ "$DRY_RUN" -eq 0 ]; then
  mark START publish-dryrun
  # mlx-bench-publish is a console script of THIS package, so it lives in the
  # project venv and is not on a bare PATH. Resolve it explicitly; if it is
  # genuinely absent, say so once instead of emitting "command not found" per
  # artifact and still reporting the suite clean.
  publish_bin=""
  for cand in "$REPO_ROOT/.venv/bin/mlx-bench-publish" "$(command -v mlx-bench-publish 2>/dev/null || true)"; do
    [ -n "$cand" ] && [ -x "$cand" ] && { publish_bin="$cand"; break; }
  done
  if [ -z "$publish_bin" ]; then
    echo "  FAILED: mlx-bench-publish not found (create the venv: uv sync)" >&2
    mark FAIL publish-dryrun
    FAILED="$FAILED publish-dryrun"
  else
    # --suite is required=True and --kind defaults to lm-eval, so invoking
    # without both argparse-errors on every artifact. That failure used to be
    # swallowed by `|| true` piped through tail, and the suite still reported
    # clean — a check that cannot fail reporting success. Derive both from the
    # filename. `set -o pipefail` (top of file) is what makes the `if !` below
    # see the publisher's status and not tail's; without it this would be the
    # same bug in a new place.
    publish_failed=0
    published_metadata="$OUT_DIR/published-metadata.json"
    if [ ! -f "$published_metadata" ]; then
      echo "  FAILED: required published metadata file is missing: $published_metadata" >&2
      publish_failed=1
    fi
    for f in "$OUT_DIR"/*.json; do
      [ -e "$f" ] || continue
      base="$(basename "$f" .json)"
      case "$base" in
        throughput) kind="throughput-probe" ;;
        agentic)    kind="agentic" ;;
        *)          kind="lm-eval" ;;
      esac
      echo "  $f (suite=$base kind=$kind)"
      if [ -f "$published_metadata" ] && ! "$publish_bin" "$f" --dry-run --suite "$base" --kind "$kind" \
        --published-metadata "$published_metadata" 2>&1 | tail -5; then
        publish_failed=1
      fi
    done
    if [ "$publish_failed" -ne 0 ]; then
      echo "  FAILED: at least one dry-run publish failed (see output above)" >&2
      mark FAIL publish-dryrun
      FAILED="$FAILED publish-dryrun"
    else
      mark DONE publish-dryrun
    fi
  fi
fi

echo
echo "run-suite: results in $OUT_DIR"
if [ -n "$FAILED" ]; then
  echo "run-suite: FAILED suites:$FAILED" >&2
  exit 1
fi
echo "run-suite: all requested suites completed"
