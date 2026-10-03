# 2026-10-03 — MacBook: publish-screen judge test and MiMo vs Qwen3.8-27B

Measurement only. Nothing on the serving stack was changed: no config edit, no service restart, no process killed. All numbers
are **under-load** class (production live, other clients present) and **PROVISIONAL** under the
[verdict policy](../verdict-policy.md): one session, not four runs five days apart. The Mac Studio half of this comparison
is not part of this entry.

## Models and training-data cutoffs

| Model | Served as | Weights | Card states a training-data cutoff | Labelled best guess |
| --- | --- | --- | --- | --- |
| MiMo-V2.6-Distill-Qwen-9B | `mlx-community/MiMo-V2.6-Distill-Qwen-9B-OptiQ-4bit` | 6.77 GB, mixed 4/8-bit | No | Late 2025 |
| Qwen3.8-27B | `mlx-community/Qwen3.8-27B-4bit` | 4-bit | No | Early to mid 2026 |

Both guesses are unverified. The MiMo card describes a supervised fine-tune of Qwen3.5-9B on a 77.4B-token mix and gives no
date, so the guess is inherited from the base model (Hub repo created 2026-02-27; the MiMo repo was created 2026-09-21).
The Qwen3.8-27B card gives no date; its Hub repo was created 2026-08-05.

## Instrument and method

| Field | Value |
| --- | --- |
| Host | MacBook, Apple M4 Max, 128 GiB unified memory |
| Serving | `mlx-lm-server` workers behind llama-swap, with a loopback wait-queue endpoint and a loopback direct endpoint in front |
| MiMo serving | thinking disabled in the server's chat-template arguments, concurrency limit 2 |
| 27B serving | reasoning effort medium, concurrency limit 1 |
| Judge test | installed `gh-guard` in `--scan` mode, judge endpoint and model overridden per run |
| Throughput | `harness/throughput/run.py`, `max_tokens 300`, thinking off, discarded probe then 3 measured runs |
| Tool calling | `harness/agentic/run.py`, thinking off |
| Probes | `harness/promptstack/run.py`, `hermes` surface, four probe banks (13 cases), thinking off, 3 repeats |
| Temperature | 0 everywhere |

## Judge screen test

The `gh-guard` judge answers `block` or `allow` to text bound for a public repository. The 8-case set used when its prompt was
last changed is not stored in the repo, so this entry uses a reconstruction: 4 disclosures (incident narrative, internal
topology, credential scope, hostnames and internal service names) and 4 clean texts (feature, docs edit, dependency bump with
prose, config field names), none of them one of the prompt's six few-shot examples. Four supplementary cases were added: a
proxy data flow, topology in a fenced diagram, a bare category label, and a policy rule that names a category.

Each case was scanned 3 times per model after one throwaway warm-up call. Verdicts were identical in every pass.

| Model | Guard payload | Disclosures flagged (of 4) | Clean allowed (of 4) | Supplementary correct (of 4) |
| --- | --- | --- | --- | --- |
| MiMo-V2.6-Distill-Qwen-9B | unmodified | 4 / 4 / 4 | 4 / 4 / 4 | 2 / 2 / 2 |
| Qwen3.8-27B | unmodified | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Qwen3.8-27B | with `enable_thinking=false` added | 4 / 4 / 4 | 4 / 4 / 4 | 4 / 4 / 4 |

- The two supplementary cases MiMo missed are the proxy data flow and the topology drawn as a fenced diagram.
- The unmodified 27B returned no verdict on any call: with `max_tokens 4` all four tokens were reasoning, `content` was null,
  and the guard failed closed (`judge-unavailable`). The third row used a scratch copy of the guard with one added payload
  field, `chat_template_kwargs.enable_thinking=false`, which the server applies per request.
- The guard's own suite (`run-gh-guard-tests.sh`, 22 cases): MiMo 22 passed, 0 failed. The 27B with the unmodified guard
  passed 15 and failed 7; with the added field it passed 16 and failed 6, all six `judge-unavailable` during model swaps
  (see conditions), so that figure does not score the model.

Judge latency, end-to-end `gh --scan` of a text of about 1.5k tokens, 20 calls, MiMo: p50 0.50 s, p95 0.68 s, max 1.11 s, 20 of
20 allowed. On the 8-case set MiMo took p50 0.25 s and p95 0.37 s per call. The 27B (added field) took 0.48 to 0.60 s on calls
that did not coincide with a model swap and 16 to 50 s on calls that did. Latency of the judge while the other model serves a
12k-token prompt was not measured.

## Throughput

MiMo, median [min-max] of 3 runs after the discarded probe. "Wait queue" and "direct" are the two loopback endpoints in front
of llama-swap. A cell whose runs diverge by more than 15 percent breaks the pair rule and is marked.

| Endpoint | Prompt tokens | Cumulative tok/s | Decode tok/s | TTFT s | Pair rule |
| --- | --- | --- | --- | --- | --- |
| wait queue | 88 | 53.97 [52.64-54.72] | 43.31 [42.80-43.44] | 0.31 [0.19-0.39] | holds |
| wait queue | 1997 | 495.39 [273.36-529.21] | 46.40 [46.29-46.46] | 3.29 [3.10-6.79] | breaks |
| wait queue | 11999 | 640.30 [560.87-644.92] | 44.30 [39.78-45.10] | 17.84 [17.81-20.70] | holds (13 percent) |
| wait queue | 40001 | 607.56 (1 run) | 30.54 | 64.78 | single run |
| direct | 88 | 50.71 [50.53-51.61] | 42.28 [42.08-42.45] | 0.60 [0.41-0.61] | holds |
| direct | 1997 | 549.55 [406.27-584.00] | 44.48 [36.78-48.16] | 2.92 [2.82-4.15] | breaks |
| direct | 11999 | 643.58 [558.93-666.07] | 43.96 [43.02-45.70] | 17.95 [17.21-20.65] | holds (14 percent) |

MiMo worker peak physical footprint (`vmmap --summary`): 20.1 GiB after the 40k run; it covers the worker's life including the
12k run.

Qwen3.8-27B, direct endpoint only, same settings. These runs shared the server with the other clients listed under
conditions, and each harness TTFT includes waits for queue slots and model swaps (the worker restarted 7 to 12 times per
cell), so the TTFT and cumulative columns do not describe the model. Decode rate is the usable column.

| Endpoint | Prompt tokens | Cumulative tok/s | Decode tok/s | TTFT s | Completion tokens per run |
| --- | --- | --- | --- | --- | --- |
| direct | 91 | 5.44 [5.27-6.67] | 23.94 [21.36-24.04] | 59.38 [46.16-60.25] | 300 |
| direct | 2000 | 25.90 [21.17-52.73] | 22.06 [21.87-25.42] | 76.34 [37.70-94.75] | 13 to 144 |
| direct | 12002 | 139.76 [137.91-153.03] | 20.77 [10.55-21.07] | 83.47 [77.27-84.59] | 29 to 61 |

The decode rate at 91 prompt tokens (300 completion tokens) is the reliable 27B figure, about 24 tok/s; the 2k and 12k runs
produced short answers, so their decode rates rest on 13 to 144 tokens. The 27B was not run through the wait-queue endpoint
for throughput: the laptop was on battery, then another client held the 27B's only slot for most of the session. No 27B
`vmmap` peak was captured.

The simultaneous pair at concurrency 2 is not reported: in every cell one of the two requests received the model server's
`concurrency_limit` 429, because another client held a slot.

## Tool calling (thinking off)

| Cell | MiMo | 27B |
| --- | --- | --- |
| conc4, large context, stream, 3 requests, wait queue | 1 of 3 valid; 2 were HTTP 429 | not run |
| conc4, large context, stream, 3 requests, direct | 1 of 3 valid; 2 were HTTP 429 | not run |
| conc1, large context (36k tokens), stream, 10 requests, wait queue | 10 of 10 valid; latency p50 86.5 s, p95 171.4 s | not run |
| multi-turn, 20 rounds, wait queue | 20 of 20 valid, no degraded round | 20 of 20 valid, no degraded round |
| multi-turn, 20 rounds, direct | 20 of 20 valid, no degraded round | not run |

- The conc4 cell sends all three requests at once. The two HTTP 429s came from the model server's per-model concurrency
  limit being used up by other clients, so that cell describes the sharing, not the model.
- The large-context prompt measured 36,552 tokens at the server, not 20,000.
- A third MiMo multi-turn run through the wait queue returned 19 HTTP errors in 20 rounds while the wait-queue endpoint was
  answering 503 for MiMo; that run was discarded and repeated.

## Probes (promptstack, `hermes` surface, 3 repeats)

Run through the wait-queue endpoint with the `base_plus_variant` prompt (the shared behavioural base plus the surface delta).
Each cell is 9 requests (3 tasks times 3 repeats) for reasoning, instruction and homelab_qa, and 12 for tool_call. Every
request returned HTTP 200. Success rate is the fraction of requests scored correct.

| Probe class | MiMo success | 27B success | Gap (points) | Other metric, MiMo / 27B |
| --- | --- | --- | --- | --- |
| reasoning | 0.00 | 1.00 | 100 | |
| tool_call | 0.50 | 1.00 | 50 | fabricated call on a negative task: 1.00 / 0.00 |
| instruction | 0.67 | 0.67 | 0 | constraint adherence 0.889 / 0.889 |
| homelab_qa | 1.00 | 1.00 | 0 | unsupported-claim rate 0.33 / 0.00 |

A direct single-request check of the MiMo reasoning failures returned wrong numeric answers (480, 16 and 12 against the
expected 640, 21 and 24), not an HTTP or scoring fault.

Median latency: MiMo 0.38 to 1.70 s, 27B 0.51 to 17.4 s (the 27B figures include queue and swap waits). With the `current`
prompt variant MiMo scored reasoning 0.67, tool_call 0.75 (fabricated call on a negative task 0.50), instruction 0.67 and
homelab_qa 1.00; the 27B was not run on that variant, so no gap is computed for it. The MiMo results are therefore prompt
variant sensitive.

## Not run

| Suite | Reason |
| --- | --- |
| Coding (`humaneval_instruct_qwen3`, `mbpp_instruct_qwen3`) | The suite executes model-generated code; SECURITY.md asks for a sandbox and this host has none. |
| Pipe replay | The saved background-job logs hold session transcripts, not the model request bodies a replay needs. |
| Judge latency under load | Not reached. |
| Concurrency 2 pairs | Invalid in every cell (see above); a quiet slot was never available. |
| 27B conc4 and conc1 large-context tool calling, 27B wait-queue throughput | Another client held the 27B's only slot for most of the session. |
| Probes at 10 repeats; 27B on the `current` prompt variant | Run at 3 repeats and one variant to fit the time the 27B slot was free. |
| Pair replication of every cell | One session; see the policy note at the top. |

## Conditions during the runs

- The serving stack keeps one model loaded at a time. A request for the other model unloads the loaded one; loading took about
  25 s for the 27B and about 5 s for MiMo. Other clients were active, so the loaded model changed often and the worker
  restarted 7 to 14 times per run.
- Other clients held concurrency slots, so some requests through the wait queue received HTTP 429.
- The 27B's measured prefill from worker-log timestamps was about 170 tok/s (2048 tokens per 12 s); MiMo's, from the harness,
  about 670 tok/s at 12k tokens. Harness TTFT for the 27B includes queue and swap waits and is not a compute figure.
- A change of power source made both loopback front endpoints return 503 for a while; runs that overlapped that window were
  discarded and repeated.
