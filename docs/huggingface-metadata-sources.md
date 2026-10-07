# Hugging Face metadata sources

Read on 2026-10-07. The published-result contract follows the Hub's model-card
metadata, ModelInfo API, evaluation-result sidecar, and Benchmark `eval.yaml`
formats.

- [Model Cards](https://huggingface.co/docs/hub/model-cards):
  `pipeline_tag`, `library_name`, license fields, `base_model`,
  `base_model_relation`, and `tags`.
- [Evaluation Results](https://huggingface.co/docs/hub/eval-results):
  `.eval_results/*.yaml` records, Benchmark tags, and dataset task IDs.
- [Evaluation result YAML schema](https://github.com/huggingface/hub-docs/blob/main/eval_results.yaml):
  exact sidecar keys and nesting.
- [Model card metadata schema](https://github.com/huggingface/hub-docs/blob/main/modelcard.md):
  exact model-card frontmatter keys.
- [HfApi reference](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api):
  ModelInfo `sha`, `pipeline_tag`, `library_name`, `card_data`, `safetensors`,
  `gguf`, `config`, `gated`, and `tags`.
- [Benchmark evaluation documentation](https://huggingface.co/docs/hub/en/eval-results):
  registered `evaluation_framework`; task `id` and optional `config`/`split`.
- [Evaluation framework registry](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/eval.ts):
  supported `evaluation_framework` values.

The sidecar format has no dedicated metric key. The publisher preserves the
metric in the result row and records it in sidecar `notes`, while retaining the
Hub's sidecar keys unchanged.
