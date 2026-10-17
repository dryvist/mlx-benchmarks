# Evaluation result fields

| Field | Type | Description |
| --- | --- | --- |
| `output_tokens_mean` | number \| null | Mean output tokens per sample. |
| `output_tokens_median` | number \| null | Median output tokens per sample. |
| `output_tokens_p90` | number \| null | Nearest-rank p90 output tokens per sample. |
| `reasoning_tokens_mean` | number \| null | Mean reasoning tokens per sample when logged. |
| `tokens_per_correct` | number \| null | Total output tokens divided by estimated correct items. |
| `truncation_rate` | number \| null | Fraction of samples that reached the output-token limit. |
| `reasoning_arm` | string | Declared arm, or `unstated` if absent. |

Evaluation rows also publish `tokens_per_task` as a `results[]` metric. Use
observed output-token usage per scored task; scored rows include
`tokens_per_correct`. Compare token cost with the quality score and matching
tok/s. These fields use the existing v1 contract.

EvalScope and lm-eval read sibling sample logs. Inspect AI reads per-sample
`model_usage` from `.eval` logs. Converters prefer API usage counts and fall back
to tokenizing saved response text. The p90 uses nearest rank;
`tokens_per_correct` is total output tokens divided by the rounded estimate
from score ratio times sample count. Missing values are `null` with a
`<field>_unavailable_reason` tag.
