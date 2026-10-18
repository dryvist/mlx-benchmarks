---
title: MLX Benchmarks Viewer
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.26.0"
app_file: app.py
python_version: "3.11"
pinned: true
license: apache-2.0
short_description: Interactive viewer for the MLX Benchmarks dataset
tags:
  - benchmarks
  - mlx
  - apple-silicon
  - lm-eval
  - visualization
models:
  - mlx-community/Qwen3.5-9B-MLX-4bit
datasets:
  - JacobPEvans/mlx-benchmarks
---

Interactive Gradio viewer for the
[`JacobPEvans/mlx-benchmarks`](https://huggingface.co/datasets/JacobPEvans/mlx-benchmarks)
dataset — a collection of benchmark runs of MLX-quantized and locally-hosted
LLMs on Apple Silicon, all serialized under the envelope v1 schema.

## Installation

Requires Python 3.11+.

```sh
cd space
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Launch the viewer locally:

```sh
python app.py
```

Gradio prints a local URL (default `http://127.0.0.1:7860`). The app loads
canonical Parquet shards and the run index from the HF dataset. It prepares
both viewer modes, then caches the prepared data for 600 seconds. After that
TTL expires, the viewer keeps serving the cached rows and starts one background
refresh. The **Refresh data** button uses the same path. See
[`space/app.py`](https://github.com/JacobPEvans/mlx-benchmarks/blob/main/space/app.py)
for the cache implementation. The viewer offers three charts:

- **Bar chart — latest run** — one bar per model for a given (suite, task, metric)
- **Trend — over time** — score trajectory per model across runs
- **Summary table** — pivot of models x tasks for the selected (suite, metric)

Select **Refresh data** to load the latest dataset snapshot.

## Performance and remeasurement

These measurements were recorded on 2026-10-18 UTC against the live public
Space, dataset, and datasets-server endpoints. Network timings vary by run.

Measurements while the Space was already running:

- The current filter sample showed a cold page-to-data load in about 3.1 s,
  mode switches in about 0.3 s, and suite switches in 0.3–0.4 s.
- datasets-server `default` first rows returned HTTP 200 in 0.41 s; `history`
  first rows returned HTTP 200 in 0.25 s.
- `data/recent/latest.parquet` downloaded 268,957 bytes in 0.71 s. The dataset
  page loaded in 0.76 s.
- The fresh viewer snapshot had 3,060 rows: 637 scored, 2,410 experimental,
  and 13 recovered. The 313-entry remote run index passed validation. No scored
  row had an empty machine, engine, or engine version.

The first measured server cold start after the [PR #416 rewrite](https://github.com/JacobPEvans/mlx-benchmarks/pull/416)
took about 18 s to show data. Before the [PR #440](https://github.com/JacobPEvans/mlx-benchmarks/pull/440)
and [PR #445](https://github.com/JacobPEvans/mlx-benchmarks/pull/445)
precompute fixes, a mode switch still showed `processing` after 6 s.

The current targets are:

| Check | Target |
| --- | --- |
| Cold page load | Show data within 30 s. |
| Mode switch | Complete within 5 s. |
| Interaction after the cache TTL | Return the cached frame while the background refresh runs; do not wait for a full reload. |

The current code starts the refresh in a background thread when cached data
reaches its 600-second TTL. It serves the existing frame until the refresh
completes. A Space wake or restart with the current code has not been measured.
Neither has loading the full history through the `datasets` library.

### Repeat the measurements

From the repository root, run the timing script against three fresh browser
pages while the Space is running:

```sh
uvx --with playwright python -I scripts/measure_viewer_performance.py
```

The script prints page-to-data, mode-switch, and evidence-filter timings for
each page. To capture the same visual states for comparison, pass a screenshot
directory:

```sh
mkdir -p /tmp/mlx-viewer-shots
uvx --with playwright python -I scripts/capture_viewer_screenshots.py /tmp/mlx-viewer-shots
```

Time the public dataset endpoints and page with `curl`:

```sh
curl -sS -L -o /dev/null -w 'HTTP %{http_code}; %{time_total}s\n' \
  'https://datasets-server.huggingface.co/first-rows?dataset=JacobPEvans%2Fmlx-benchmarks&config=default&split=train'
curl -sS -L -o /dev/null -w 'HTTP %{http_code}; %{time_total}s\n' \
  'https://datasets-server.huggingface.co/first-rows?dataset=JacobPEvans%2Fmlx-benchmarks&config=history&split=train'
curl -sS -L -o /dev/null -w 'HTTP %{http_code}; %{size_download} bytes; %{time_total}s\n' \
  'https://huggingface.co/datasets/JacobPEvans/mlx-benchmarks/resolve/main/data/recent/latest.parquet?download=true'
curl -sS -L -o /dev/null -w 'HTTP %{http_code}; %{time_total}s\n' \
  'https://huggingface.co/datasets/JacobPEvans/mlx-benchmarks'
```

Validate the live run-index entries from the repository root:

```sh
.venv/bin/python scripts/validate_run_index.py --remote
```

The current-code Space cold start after sleep or restart and a full-history
load through the `datasets` library remain unmeasured.

## Deployment

This Space is synced automatically from the
[JacobPEvans/mlx-benchmarks](https://github.com/JacobPEvans/mlx-benchmarks)
GitHub repository via the `deploy-space.yml` workflow on every `main` push
that touches `space/`.

## Contributing

Issues and PRs live in the upstream
[GitHub repo](https://github.com/JacobPEvans/mlx-benchmarks). See the
repository `CONTRIBUTING.md` for the full developer workflow.

## API

The viewer is a Gradio UI — it has no stable programmatic API. To query the
underlying data yourself, read the HF dataset directly:

```python
import pandas as pd
from huggingface_hub import HfFileSystem

fs = HfFileSystem()
paths = fs.glob("datasets/JacobPEvans/mlx-benchmarks/data/*.parquet")
df = pd.concat([pd.read_parquet(f"hf://{p}") for p in paths], ignore_index=True)
```

## License

Apache-2.0. See [LICENSE](https://github.com/JacobPEvans/mlx-benchmarks/blob/main/LICENSE).

## Source

Full source, tests, and developer docs:
<https://github.com/JacobPEvans/mlx-benchmarks>
