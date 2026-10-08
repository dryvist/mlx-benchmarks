# Model currency and architecture facts

Updated 2026-10-07. `OLD` means a newer runnable architecture generation exists in
that model family. Artificial Analysis scores and version numbers alone do not
set lifecycle. Per-run results stay in the ranking catalog and journals.

Full source URLs for candidate models are in
[`candidates.toml`](../configs/shootout/candidates.toml), in `facts_source` and
`supersession_evidence`. Catalogued sizes are identified below; they do not
claim a fresh local load unless stated.

## Current Qwen successor

**Qwen3.8-Flash-Next** uses the experimental Qwen4 architecture. Qwen calls it
an early preview of Qwen4. Released 2026-08-26; open weights; 176B by QwenLM
components (125B transformer plus 51B n-gram), 180B in HF metadata, 6B active.
The MLX-OptiQ 2-bit artifact is 80.8 GB on disk and has a documented coding
run; it requires SSD streaming and `mlx-optiq>=0.5.14`. Qwen source records:
[release](https://github.com/QwenLM/Qwen3.8/blob/main/README.md),
[MLX card](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-OptiQ-2bit).

## OLD Qwen records

The following entries remain for historical results. Each is OLD under the
runnable Qwen4-generation model above. Source and artifact links are recorded
in each model's facts record or the linked model inventory.

- **OLD Qwen3-Next-80B-A3B Instruct / Thinking** — Qwen3 hybrid MoE; released
  2025-09-11; open, Apache-2.0; 80B/3B active. MLX 4-bit: Instruct 44.8 GB,
  Thinking about 45 GB.
- **OLD Qwen3-30B-A3B-Instruct-2507** — Qwen3 MoE; 2025-07-30; open,
  Apache-2.0; 30.5B/3.3B active. MLX 4-bit 17.2 GB; 8-bit 32.4 GB.
- **OLD Qwen3-Coder-30B-A3B** — Qwen3 MoE; 2025-07-22; open, Apache-2.0;
  30.5B/3.3B active. MLX 4-bit 17.2 GB; 8-bit 32.4 GB. Resident record
  protected; current residency unverified.
- **OLD Qwen3-Coder-Next-80B-A3B** — Qwen3-Next hybrid MoE; 2026-02-02;
  open, Apache-2.0; 80B/3B active. Candidate MLX 4-bit 44.8 GB.
- **OLD Qwen3.5-27B** — Qwen3.5 dense hybrid; 2026-02-24; open,
  Apache-2.0; 27B. MLX 4-bit 16.1 GB.
- **OLD Qwen3.6-27B** — Qwen3.6 dense hybrid; 2026-04-22; open,
  Apache-2.0; 27B. MLX 4-bit 16.1 GB.
- **OLD Qwen3.6-35B-A3B** — Qwen3.6 hybrid MoE; 2026-04-16; open,
  Apache-2.0; 35B/3B active. MLX 4-bit 20.4 GB, OptiQ 4-bit 24.7 GB,
  8-bit 37.7 GB. Served family protected.
- **OLD Qwen3.5-122B-A10B** — Qwen3.5 hybrid MoE; 2026-02-24; open,
  Apache-2.0; 122B/10B active. MLX 4-bit 69.6 GB.
- **OLD Qwen3.5-35B-A3B** — Qwen3.5 hybrid MoE; 2026-02-24; open,
  Apache-2.0; 35B/3B active. MLX 4-bit 20.4 GB.
- **OLD Qwen3.5-9B** — Qwen3.5 dense hybrid; 2026-03-02; open,
  Apache-2.0; 9B. MLX 4-bit artifact 5.95 GB; catalog sizing 5.2 GB.
  The served record remains protected.
- **OLD Qwen3.8-27B** — Qwen3.8 dense; 2026-08-14; open, Apache-2.0;
  27B official card / 28B HF metadata. MLX 4-bit 16.08 GB and 8-bit
  29.53 GB; loaded 4-bit record protected. MTP is an auxiliary artifact.
- **OLD Qwen3-235B-A22B-Instruct-2507** — Qwen3 MoE; 2025-07-21; open,
  Apache-2.0; 235B/22B active. Catalogued MLX 4-bit 132.26 GB; fresh load
  unverified and above single-host memory.
- **OLD Qwen3-4B-Instruct-2507** — Qwen3 dense; 2025-08-06; open,
  Apache-2.0; 4B. MLX 4-bit 2.28 GB.

## Other family decisions

- **OLD Ling-2.6-Flash** — Bailing hybrid MoE; 2026-04-23; open, MIT;
  104B/7.4B active. Candidate DWQ 4-bit 58.6 GB. Successor: Ling-3.0-Flash.
- **Ling-3.0-Flash** — Bailing hybrid KDA + MLA; 2026-08-02; open, MIT;
  124B/5.1B active. MLX 4-bit 70.03 GB has Apple-silicon load and generation
  validation. [Artifact](https://huggingface.co/TensorFold/Ling-3.0-flash-MLX-4bit).
- **MiniMax-M2** — M2 sparse MoE; 2025-10-27; weights available, license
  `other`; 230B/10B active. REAP-pruned MLX MXFP4: 139B/10B active, 73.9 GB.
- **MiniMax-M2.7** — M2 architecture; 2026-03-18; weights available,
  license `other`; 229B, active count unverified. MLX quant size unverified.
- **MiniMax-M3** — MSA; 2026-06-01; weights available, license `other`;
  about 428B/23B active (HF: 427B). MLX 4-bit 241 GB and 3-bit 186 GB;
  neither is target-fit.
- **GLM-5.3-Flash** — GLM 5.3 / `glm5_next`; 2026-08-26; open, MIT;
  320B/18B active (HF: 321B). NVIDIA NVFP4 is 190.4 GiB; fitting MLX quant
  unverified, so GLM-4.7-Flash is NOT PROVEN OLD for this target.
- **GLM-4.7-Flash** — GLM 4.7 / `glm4_moe_lite`; 2026-01-19; open, MIT;
  30B/3.6B active. Candidate MLX 8-bit 31.8 GB.
