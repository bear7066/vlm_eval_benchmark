# VLM Offline Eval Benchmark

A two-phase offline benchmarking framework for evaluating Vision-Language Models (VLMs) on video action-recognition tasks, with automated LLM-as-a-judge scoring.

## Setup

```shell
# Install dependencies
uv sync

# Verify GPU availability
uv run python scripts/gpu_test.py
```

### Environment variables

Create a `.env` file in the project root:

```dotenv
HF_TOKEN=<your_huggingface_token>

# Required for OpenAI judge models (gpt-4o, gpt-4o-mini, …)
OPENAI_API_KEY=<your_openai_key>

# Required for internal Medusa judge models
OUTER_MEDUSA_ENDPOINT=<endpoint_url>
OUTER_MEDUSA_API_KEY=<api_key>
```

---

## Batch Evaluation

Run inference + judging across all dataset/model combinations in one shot:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python scripts/run_batch_eval.py
```

Edit `scripts/run_batch_eval.py` to configure `--datasets` and `--model_ids` before running. Results are written under `runs/`.

---

## Single-Run Commands

### 1. Inference (Benchmark)

```shell
uv run python scripts/run_benchmark.py \
  --dataset climbing_ladder \
  --model_id google/gemma-4-E4B-it \
  --num_frames 8
```

Outputs `runs/<run_id>/predictions.jsonl` and `runs/<run_id>/summary.json`.

**Available datasets:** `climbing_ladder`, `face_planting`, `falling_off_bike`, `falling_off_chair`

**Available VLM models:**
- `google/gemma-4-E2B-it`
- `google/gemma-4-E4B-it`
- `bear7011/gemma4-e2b-webvid4K_FT`
- `bear7011/gemma4-e4b-webvid4K_FT`

### 2. Judge

```shell
uv run python scripts/run_judge.py \
  --dataset climbing_ladder \
  --model_id google/gemma-4-E4B-it \
  --judge_model gpt-4o
```

Outputs `runs/<run_id>/judge_results.jsonl` and `runs/<run_id>/judge_summary.json`. Each result includes `bleu`, `rouge_l`, and `cider` text metrics alongside the LLM score.

To compute text metrics only (skip the LLM judge call):

```shell
uv run python scripts/run_judge.py \
  --dataset climbing_ladder \
  --model_id google/gemma-4-E4B-it \
  --skip_llm_judge
```

**Available judge models:** `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`, `gpt-5`, `gpt-oss-20b`, `gpt-oss-120b`, `Google-Gemma-3-27B`, `Llama-3.1-70B`, `Llama-3.1-405B-Instruct-FP8`

---

## CLI Entry Points

After `uv sync`, the following commands are available directly:

| Command                | Equivalent script           |
| ---------------------- | --------------------------- |
| `uv run vlm-benchmark` | `scripts/run_benchmark.py`  |
| `uv run vlm-judge`     | `scripts/run_judge.py`      |
| `uv run vlm-batch`     | `scripts/run_batch_eval.py` |
| `uv run vlm-gpu-test`  | `scripts/gpu_test.py`       |

---

## Project Structure

```
src/vlm_eval/
  config.py           # BenchmarkConfig and JudgeConfig dataclasses
  inference/
    runner.py         # Inference loop
    gemma.py          # HuggingFace VLM wrapper
  judge/
    runner.py         # Judge loop
    prompts.py        # LLM judge prompt template
    text_metrics.py   # BLEU / ROUGE-L / CIDEr computation
  llm/
    factory.py        # Judge LLM backend routing (OpenAI / Medusa)
  video.py            # Frame sampling via decord
  metrics.py          # VideoResult dataclass + summarize_results()
  paths.py            # Run directory naming helpers
scripts/
  run_batch_eval.py   # Main entry point for multi-model, multi-dataset runs
  run_benchmark.py    # Single inference run
  run_judge.py        # Single judge run
  gpu_test.py         # GPU availability check
runs/                 # All benchmark outputs (predictions, judge results, summaries)
```
