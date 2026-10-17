# Run a benchmark block end to end

Use the reusable Semaphore templates for client creation, profile convergence,
suite execution, and cleanup. Pass survey values through each task; do not run
SSH, `pct`, `uv`, or suite commands by hand. Resolve new template IDs by their
canonical names after the catalog is applied. The workstation helper accepts
`sem.sh <project-id> <template-id> <message> '<JSON argument array>'`.

Run these jobs in order, waiting for each to finish before the next:

```text
sem.sh 2 "<proxmox-benchmark-client-id>" "client present" \
  '[
    "--extra-vars", "node=<NODE>",
    "--extra-vars", "vmid=<CLIENT_VMID>",
    "--extra-vars", "gpu_guest_vmid=<ENGINE_GUEST_VMID>",
    "--extra-vars", "state=present",
    "--extra-vars", "cores=<CORES>",
    "--extra-vars", "memory_mb=<MEMORY_MB>",
    "--extra-vars", "disk_gb=<DISK_GB>",
    "--extra-vars", "template=<LXC_TEMPLATE>",
    "--extra-vars", "engine_url=<ENGINE_URL>"
  ]'
sem.sh 4 "<137-or-97>" "profile converge <run-id>" \
  '["--extra-vars","llm_active_profile=<PROFILE>"]'
sem.sh 4 85 "baseline telemetry <run-id>" '[]'
sem.sh 4 "<ai-benchmark-run-id>" "benchmark cell <run-id> <cell-id>" \
  '[
    "--extra-vars", "node=<NODE>",
    "--extra-vars", "client_vmid=<CLIENT_VMID>",
    "--extra-vars", "engine_url=<ENGINE_URL>",
    "--extra-vars", "model=<MODEL>",
    "--extra-vars", "tokenizer=<TOKENIZER>",
    "--extra-vars", "suite=<SUITE>",
    "--extra-vars", "items_file=<ITEMS_FILE>",
    "--extra-vars", "reasoning_arm=<ARM>",
    "--extra-vars", "concurrency=<C>",
    "--extra-vars", "max_tokens=<OUTPUT_TOKENS>",
    "--extra-vars", "sandbox_pool=<POOL>",
    "--extra-vars", "input_tokens=<INPUT_TOKENS>",
    "--extra-vars", "request_rate=<RATE>",
    "--extra-vars", "num_prompts=<PROMPTS>",
    "--extra-vars", "seed=<SEED>",
    "--extra-vars", "run_id=<RUN_ID>",
    "--extra-vars", "publish=false"
  ]'
# Repeat the benchmark-run job once for every planned cell.
# Collect baseline and final telemetry with the same template.
sem.sh 4 85 "final telemetry <run-id>" '[]'
# Collect the reported raw artifact and publish it with the workstation
# command below.
sem.sh 2 "<proxmox-benchmark-client-id>" "client absent" \
  '[
    "--extra-vars", "node=<NODE>",
    "--extra-vars", "vmid=<CLIENT_VMID>",
    "--extra-vars", "gpu_guest_vmid=<ENGINE_GUEST_VMID>",
    "--extra-vars", "state=absent"
  ]'
```

Project 4 profile template `137` or `97` is selected by the lane owner. The
client and benchmark-run template IDs are read by canonical name after apply.
The client template uses its digest-pinned image and tool defaults unless the
`images` or `tools` survey values are overridden.
Set `items_file=` for BigCodeBench; for Promptfoo, use the config URL with its
immutable source commit in the path.

1. **Create the client.** Run `proxmox-benchmark-client` with `state=present`,
   the published node and client VMID, the serving guest VMID used for network
   settings, and the requested CPU, memory, disk, template, image, and tool
   values. Use the default digest-pinned BigCodeBench sandbox image and the
   pinned x86_64 vLLM client image from the tool defaults. The task starts the
   tagged client, installs its tools, pulls both images, and verifies the
   selected engine URL.
2. **Converge the serving profile.** Run the existing AI GPU profile template
   for the selected lane, with its normal profile survey. Respect lane
   ownership before scheduling the job. Then run Semaphore project 4, template
   85 to collect baseline power and fan telemetry. Repeat it after the last
   benchmark cell for final telemetry.
3. **Run suite cells.** Dispatch `ai-benchmark-run` once per cell, with a
   unique `run_id` per cell and a shared block prefix. Use
   `suite=bigcodebench-hard` with `item_limit` from 1 through 148 (default 30),
   `suite=promptfoo-<suite>` with an immutable raw GitHub `items_file`, or
   `suite=vllm-sweep` for one fixed-length serving cell. EvalScope records
   `reasoning_arm` explicitly: `off` maps to `enable_thinking=false`; `low` and
   `xhigh` map to their matching `reasoning_effort` values. Promptfoo and
   EvalScope run inside the client.

   BigCodeBench first checks three negative controls and three canonical
   ground-truth controls with the official pinned `bigcodebench.evaluate`
   image. The job stops unless all wrong controls fail and all canonical
   controls pass. EvalScope 1.13.0 streams generation with a configurable
   `evalscope_timeout_seconds` (default 3600); its score is not authoritative.
   The runner sanitizes the generated samples and records the final score with
   `bigcodebench.evaluate --execution local` from the same digest-pinned image.
   The official evaluator uses its published entrypoint directly.

   For throughput cells, use `concurrency` values 1, 4, 8, 12, and 16 wherever
   each request fits. Also run the capacity ceiling and ceiling + 1 from the
   serving engine's own capacity report. Run both batched (`request_rate=inf`)
   and staggered Poisson (`request_rate=R`) cells at each selected concurrency.
   Include the staggered agentic pair at 4,096 input / 1,024 output tokens and
   at least 32,768 input tokens. Give every cell a unique numeric `seed`; use
   `num_prompts` to set its fixed request count. Keep each cell within the
   model's context limit and below any saturated suite threshold.

   `suite=vllm-sweep` is a group of fixed-length `vllm bench serve` jobs using
   `vllm/vllm-openai-cpu:v0.31.0-x86_64`. Each job sets
   `input_tokens`, `max_tokens`, `request_rate`, `concurrency`, and `seed`, and
   saves detailed results with p50, p90, and p99 metrics. The upstream
   `vllm bench sweep serve` command starts a server from `--serve-cmd`; it is
   not used as a client against an already-running engine. See the [vLLM sweep
   command reference](https://docs.vllm.ai/en/stable/cli/bench/sweep/serve/)
   and [serving benchmark reference](https://docs.vllm.ai/en/stable/api/vllm/benchmarks/serve/).

   The task prints `host_uptime_s`, `guest_uptime_s`, and `engine_uptime_s` at
   block start and stores them in `run.json`. If direct guest uptime is not
   readable, keep it `null` and retain its reason. The task archives raw output
   on the selected node and prints the artifact path in the job log. The log
   reports the suite, return code, and output-line counts without echoing raw
   suite text.
4. **Collect and review.** Retrieve the reported archive from the node and
   review the result JSON and `run.json`. Keep each cell's uptime values and
   null reasons with its publication metadata.
5. **Publish from the workstation.** Semaphore publication stays disabled.
   Supply `HF_TOKEN_REPOS_ADMIN` through the approved local secret wrapper and
   map it to the publisher's `HF_TOKEN`. Dry-run the existing
   `mlx-bench-publish` command first. Add `host_uptime_s=<seconds>` and
   `guest_uptime_s=<seconds-or-null>` tags; include the null reason when needed.
   Never print the token. No dataset token is read by the Semaphore job.

   ```sh
   HF_TOKEN="$HF_TOKEN_REPOS_ADMIN" .venv/bin/mlx-bench-publish \
     "run-output/<cell>.json" --kind vllm --suite throughput \
     --published-metadata "run-output/<published-metadata>.json" \
     --tag "host_uptime_s=<seconds>" --tag "guest_uptime_s=<seconds-or-null>" --dry-run
   ```

   After reviewing the dry-run output and metadata, repeat the command without
   `--dry-run`.
6. **Remove the client.** Run `proxmox-benchmark-client` with
   `state=absent`. The playbook refuses to stop or destroy an LXC that lacks
   the `benchmark-client` tag.

The Studio-native variant runs its suite client natively on the Studio and
publishes from the workstation using the same result contract. The current
Semaphore catalog does not declare a Studio execution target; do not substitute
the Proxmox client template for a native run.
