# 2026-10-17 — Late benchmark findings

## Flash-Next weight fit

The [NVFP4 artifact](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
is about 124 GiB, so it does not fit a 96 GB GPU. Retire its vLLM profiles for
this memory class. Flash-Next on a 96 GB GPU uses llama.cpp with a low-bit GGUF;
UD-Q2 remains the pick. See a [published UD-Q2 128k result](https://huggingface.co/datasets/JacobPEvans/mlx-benchmarks/viewer/default/train?row=8).

## Nemotron 3.5 Bolt GGUF

llama.cpp Metal v0.7.0 fails with missing kernel
`kernel_mul_mm_id_map0_ne20_9`; none of three requests succeeded. Use Bolt with
CUDA llama.cpp for now. No successful Metal result was published.

## SciCode XHIGH context

Later SciCode sub-steps include earlier outputs, so prompts grow to about
33–40k tokens. A 64k slot leaves about 4.5k tokens for generation and truncates
75% of XHIGH calls. Treat the Qwen3.8-27B NVFP4 result (0/20 main, 13.4% sub)
as invalid for XHIGH. SciCode XHIGH needs a slot of at least 128k.

## EvalScope BigCodeBench sandbox

In EvalScope 1.13.0, configure the image at
`sandbox.default_config.image`; `sandbox.image` is the wrong key. Pin
`bigcodebench/bigcodebench-evaluate@sha256:a3cd34ec3840a49d6b7afb240f4bdd47c350bc5991043fd0a91773830f7cd405`,
pull it before evaluation, and set `pool_size` at least as high as evaluation
concurrency. “Pool initialized with 0 sandboxes” means there is no runnable
sandbox. The smoke evaluated one item, `BigCodeBench/13`; that item executed
and passed.

## Inspect `.eval` archives

Inspect `.eval` archives use Zstandard ZIP compression, which Python's
standard-library `zipfile` cannot read. No workaround is in the repository;
this remains an open gap.

## Published row fields

Published rows do not contain serving-port fields. [PR #377](https://github.com/dryvist/mlx-benchmarks/pull/377)
removed the field and scrubbed existing rows.
