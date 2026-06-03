# VLM Offline Eval Benchmark

A two-phase offline benchmarking framework for Vision-Language Models (VLMs) on video action-recognition tasks. Videos are hosted on HuggingFace Hub and streamed on demand — no local dataset copy required.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Dataset](#dataset)
  - [Upload (first time only)](#upload-first-time-only)
  - [Available Configs](#available-configs)
- [Running Benchmarks](#running-benchmarks)
  - [Single Benchmark Run](#single-benchmark-run)
  - [Single Judge Run](#single-judge-run)
  - [Batch Run (multiple models × datasets)](#batch-run)
- [Output Format](#output-format)
- [Supported Models](#supported-models)
- [Environment Variables](#environment-variables)

---

## Project Structure

```
vlm_offline_eval_benchmark/
│
├── scripts/
│   ├── upload_to_huggingface.py   # One-time upload of local videos to HF Hub
│   ├── hf_loading_script.py       # HF dataset loading script (uploaded to Hub)
│   ├── run_benchmark.py           # Entry point: single inference run
│   ├── run_judge.py               # Entry point: single judge run
│   ├── run_batch_eval.py          # Entry point: multi-model × multi-dataset batch
│   └── gpu_test.py                # GPU availability / performance check
│
├── src/vlm_eval/
│   ├── hf_dataset.py              # HF streaming dataset loader
│   ├── config.py                  # BenchmarkConfig, JudgeConfig dataclasses
│   ├── cli.py                     # argparse entry points (benchmark/judge/batch)
│   ├── video.py                   # Frame sampling (from path or raw bytes)
│   ├── metrics.py                 # VideoResult dataclass + summarize_results()
│   ├── paths.py                   # Run directory naming, find_latest_run()
│   ├── hardware.py                # GPU VRAM / power utilities
│   ├── logging_utils.py           # Logging configuration
│   │
│   ├── inference/
│   │   ├── gemma.py               # HuggingFaceVLM wrapper (Gemma via transformers)
│   │   └── runner.py              # Inference loop: stream → sample frames → generate
│   │
│   ├── judge/
│   │   ├── runner.py              # Judge loop: text metrics + LLM-as-a-judge
│   │   ├── prompts.py             # LLM judge prompt template
│   │   ├── text_metrics.py        # BLEU, ROUGE-L, CIDEr computation
│   │   └── parser.py              # Load predictions.jsonl / legacy log files
│   │
│   └── llm/
│       ├── factory.py             # Route judge model to openai / medusa backend
│       ├── openai_backend.py      # OpenAI API backend
│       └── medusa_backend.py      # Internal Medusa API backend
│
├── runs/                          # Auto-created; all benchmark + judge outputs
├── requirements.txt
└── pyproject.toml
```

---

## Setup

### Option A — uv (recommended)

```shell
uv sync
uv run python scripts/gpu_test.py   # confirm GPU is available
```

### Option B — pip

```shell
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                    # install package in editable mode
python3 scripts/gpu_test.py
```

### Environment

Create a `.env` file (or export variables in your shell):

```dotenv
HF_TOKEN=hf_...                         # HuggingFace token (required to access private repos)
HF_DATASET_REPO=gnitoahc/vlm-eval-videos  # dataset repo (default already set)
OPENAI_API_KEY=sk-...                   # required when using gpt-4o as judge
```

---

## Dataset

Videos live in a single HuggingFace dataset repository as per-label `.tar.gz` archives. They are streamed one at a time during inference — the full dataset is never downloaded locally.

### Upload (first time only)

If you have local videos under `dataset/` that have not yet been uploaded:

```shell
python scripts/upload_to_huggingface.py \
  --repo_id gnitoahc/vlm-eval-videos \
  --dataset_root ./dataset
```

This script will:
1. Pack each `dataset/{label}/` directory into an in-memory `.tar.gz`
2. Upload archives to `data/{label}.tar.gz` in the HF repo
3. Upload the loading script (`hf_loading_script.py`) and a `README.md` with YAML dataset-card metadata

### Available Configs

| Config name | Action depicted |
|---|---|
| `climbing_ladder` | Person climbing a ladder |
| `face_planting` | Person face-planting |
| `falling_off_bike` | Person falling off a bike |
| `falling_off_chair` | Person falling off a chair |

Each example returned by the dataset has three fields:

```python
{
    "video_id":    str,   # YouTube ID, e.g. "03jyaZxUUgk"
    "label":       str,   # action label, e.g. "climbing_ladder"
    "video_bytes": bytes, # raw MP4 bytes
}
```

---

## Running Benchmarks

### Single Benchmark Run

Streams videos for one dataset config, runs the VLM, writes `predictions.jsonl`.

```shell
uv run python scripts/run_benchmark.py \
  --dataset falling_off_chair \
  --model_id google/gemma-4-E4B-it \
  --num_frames 8 \
  --sample_size 50

# Optional flags:
#   --dataset_repo  gnitoahc/vlm-eval-videos  (override repo)
#   --seed          42                          (reproducible sampling)
#   --prompt        "Describe the action."      (override default prompt)
#   --output_root   runs/                       (output directory root)
```

### Single Judge Run

Evaluates predictions from a prior benchmark run using text metrics (BLEU, ROUGE-L, CIDEr) and optionally an LLM judge.

```shell
# Auto-find the latest run for this model + dataset:
uv run python scripts/run_judge.py \
  --dataset falling_off_chair \
  --model_id google/gemma-4-E4B-it \
  --judge_model gpt-4o

# Point at a specific run directory:
uv run python scripts/run_judge.py \
  --run_dir runs/gemma-4-E4B_8frames_falling_off_chair_20250603-120000 \
  --judge_model gpt-4o

# Text metrics only (no LLM call):
uv run python scripts/run_judge.py \
  --dataset falling_off_chair \
  --model_id google/gemma-4-E4B-it \
  --skip_llm_judge
```

### Batch Run

Runs the full benchmark → judge pipeline for every combination of datasets × models. Edit `scripts/run_batch_eval.py` to change the lists, then:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python scripts/run_batch_eval.py
```

Default configuration in the script:

```python
--datasets  climbing_ladder face_planting falling_off_bike falling_off_chair
--model_ids google/gemma-4-E2B-it  bear7011/gemma4-e2b-webvid4K_FT
            google/gemma-4-E4B-it  bear7011/gemma4-e4b-webvid4K_FT
--num_frames 8
--judge_model gpt-4o
```

---

## Output Format

Each run creates a directory under `runs/`:

```
runs/gemma-4-E4B_8frames_falling_off_chair_20250603-120000/
├── config.json          # all input parameters
├── benchmark.log        # per-video inference log
├── predictions.jsonl    # one VideoResult JSON per line
├── summary.json         # aggregate throughput / latency metrics
├── judge.log            # per-video judge log
├── judge_results.jsonl  # per-video scores + text metrics
└── judge_summary.json   # aggregate judge score + text metric averages
```

**`summary.json` output fields:**

| Field | Description |
|---|---|
| `average_query_latency_ms` | Mean inference time per video |
| `frames_per_second` | Sampled frames processed per second |
| `equivalent_real_time_latency` | Inference time / video duration |
| `peak_vram_usage_gb` | Peak GPU memory |
| `throughput_tokens_per_sec` | Generated tokens per second |
| `power_consumption_watts` | Average GPU power draw |
| `ttft_ms` | Time to first token |

**`judge_summary.json` output fields:**

| Field | Description |
|---|---|
| `average_score` | Mean LLM judge score (0–10) |
| `text_metrics.bleu` | Average sentence BLEU |
| `text_metrics.corpus_bleu` | Corpus-level BLEU |
| `text_metrics.rouge_l` | Average ROUGE-L F1 |
| `text_metrics.cider` | Average CIDEr |

---

## Supported Models

### VLM (inference)

| Model ID | Description |
|---|---|
| `google/gemma-4-E2B-it` | Gemma 4 2B instruction-tuned |
| `google/gemma-4-E4B-it` | Gemma 4 4B instruction-tuned |
| `bear7011/gemma4-e2b-webvid4K_FT` | Gemma 4 2B fine-tuned on WebVid-4K |
| `bear7011/gemma4-e4b-webvid4K_FT` | Gemma 4 4B fine-tuned on WebVid-4K |

Any HuggingFace model compatible with `AutoModelForCausalLM` + `AutoProcessor` can be swapped in via `--model_id`.

### Judge LLM

| Model | Backend |
|---|---|
| `gpt-4o`, `gpt-4o-mini`, `gpt-5` | `openai` |
| `gpt-oss-20b`, `gpt-oss-120b` | `medusa` |
| `Google-Gemma-3-27B`, `Llama-3.1-70B` | `medusa` |

The backend is auto-detected from the model name. Pass `--backend openai` or `--backend medusa` to override.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | HuggingFace auth token |
| `HF_DATASET_REPO` | `gnitoahc/vlm-eval-videos` | Dataset repository ID |
| `OPENAI_API_KEY` | — | Required for OpenAI judge models |
| `CUDA_VISIBLE_DEVICES` | all | Restrict GPU visibility |
