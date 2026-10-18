# Cache fields

The cache configuration, workload, and measurement fields use typed campaign
dimensions. Each field is nullable. Record a `dimension_null_reasons` entry for
explicit unknown or inapplicable values. Rate fields are fractions from 0 to 1,
except `kv_cache_usage_peak`, which is a percentage. Capacity fields use the
unit named in the field.

| Standard field | Campaign dimension | Type | Description |
| --- | --- | --- | --- |
| `host_ram_limit_gib` | `hardware.host_ram_limit_gib` | number | Host memory limit in GiB. |
| `prefix_caching` | `run.prefix_cache` | boolean | Whether prefix caching is enabled. |
| `kv_offload_backend` | `run.kv_offload_backend` | string enum | Backend: `none`, `native`, `lmcache`, or `llamacpp_cache_ram`. |
| `kv_offload_cpu_gib` | `run.kv_offload_cpu_gib` | number | KV offload capacity in host RAM, GiB. |
| `kv_offload_disk_gib` | `run.kv_offload_disk_gib` | number | KV offload capacity on disk, GiB. |
| `weight_cpu_offload_gib` | `run.weight_cpu_offload_gib` | number | Model-weight offload capacity in host RAM, GiB. |
| `llamacpp_cache_ram_mib` | `run.llamacpp_cache_ram_mib` | integer | llama.cpp cache-ram setting, MiB. |
| `gpu_memory_utilization` | `run.gpu_memory_utilization` | number | GPU memory reservation fraction. |
| `max_num_batched_tokens` | `run.max_num_batched_tokens` | integer | Maximum tokens per serving batch. |
| `gpu_kv_capacity_tokens` | `run.kv_pool_tokens` | integer | Total configured token capacity of the serving KV pool. |
| `cache_workload` | `run.cache_workload` | string enum | Pressure class: `fits_gpu`, `spills_to_ram`, `exceeds_all`, or `none`. |
| `num_users` | `run.num_users` | integer | Simulated users. |
| `num_rounds` | `run.num_rounds` | integer | Interaction rounds per user. |
| `shared_prompt_tokens` | `run.shared_prompt_tokens` | integer | Tokens in the prompt shared across users. |
| `user_history_tokens` | `run.user_history_tokens` | integer | Retained history tokens per user. |
| `answer_tokens` | `run.output_tokens` | integer | Requested output tokens per answer. |
| `request_rate` | `run.request_rate` | number | Request arrivals per second. |
| `prefix_cache_hit_rate` | `speed.prefix_cache_hit_rate` | number | Prefix-cache hit fraction. |
| `external_prefix_cache_hit_rate` | `speed.external_prefix_cache_hit_rate` | number | External prefix-cache hit fraction. |
| `kv_cache_usage_peak` | `speed.kv_cache_usage_peak_pct` | number | Peak KV-cache occupancy, percent. |
| `kv_block_idle_before_evict_p50_s` | `speed.kv_block_idle_before_evict_p50_s` | number | Median block idle time before eviction, seconds. |
| `host_ram_peak_gib` | `resource.host_ram_peak_gib` | number | Peak host RAM use, GiB. |
| `preemptions` | `resource.preemptions` | integer | Requests preempted by the serving engine. |
| `ttft_first_round_p50_ms` | `speed.ttft_first_round_p50_ms` | number | First-round median time to first token, ms. |
| `ttft_later_rounds_p50_ms` | `speed.ttft_later_rounds_p50_ms` | number | Later-round median time to first token, ms. |
| `ttft_later_rounds_p99_ms` | `speed.ttft_later_rounds_p99_ms` | number | Later-round 99th-percentile time to first token, ms. |

The standard names reuse existing schema equivalents for `prefix_caching`
(`run.prefix_cache`), `gpu_kv_capacity_tokens` (`run.kv_pool_tokens`),
`answer_tokens` (`run.output_tokens`), and `kv_cache_usage_peak`
(`speed.kv_cache_usage_peak_pct`). Other standard names use matching dimension
names.
