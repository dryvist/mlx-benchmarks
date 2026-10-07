# Stage 0 System Load Test rows

Use `campaign.profile: stage0-system-load`. Record a separate envelope for
each measured phase. The publisher copies phase dimensions to every result row
in that envelope.

## Load phases

Each phase is 180 seconds. Run serial at concurrency 1, then `high_parallel`
phases at 2, 4, 8, 16, 32, 64, and higher powers of two until system throughput
plateaus. Use aggregate output tokens/s for generation, requests/s for decision
models, and embeddings/s for embedding models to identify the plateau. Set
`closed_loop` to `true` so a finished request is replaced immediately. Record
`streaming` to match the endpoint; stream generation responses when supported.
Keep generation prompt and output sizes fixed across the ladder.

For embeddings, use the same serial and parallel ladder. Keep
`embedding_items_per_request` and `embedding_input_tokens_per_request` fixed
across phases. Record `embeddings_per_second`, `embedding_tokens_per_second`,
and p50/p95/p99 embedding request latency. For decision models, record
`requests_per_second` and complete request latency p50/p95/p99. Generation
decisions also record aggregate output tokens/s and TTFT percentiles when the
server exposes them.

## Required Mac fields

The Stage 0 profile requires non-null `system.os`, `system.chip`, and
`system.memory_gb`, plus `campaign_dimensions.hardware.machine`,
`power_source`, and `power_mode`, and `campaign_dimensions.software.macos_version`.
Use the generic machine label `MacBook Pro M4 Max 128GB` or
`Mac Studio M4 Max 128GB`; do not use a host name. Keep power source and mode
as observed. Do not include outlet or circuit identifiers.

The publisher copies system and campaign values to each result row. This
profile requires 180-second phases, a closed loop, consistent serial or
high-parallel concurrency, and the appropriate decision or embedding
throughput and latency fields. These dimensions describe results; they do not
add a benchmark harness or claim that a phase has run.
