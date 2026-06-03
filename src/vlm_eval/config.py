from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vlm_eval.hf_dataset import DEFAULT_DATASET_REPO


DEFAULT_PROMPT = (
    "Describe the main action briefly in 2~6 words."
)


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset: str               # HF config name, e.g. "climbing_ladder"
    model_id: str
    dataset_repo: str = DEFAULT_DATASET_REPO
    num_frames: int = 8
    sample_size: int = 1000
    output_root: Path = Path("runs")
    run_id: str | None = None
    seed: int | None = None
    prompt: str = DEFAULT_PROMPT
    max_new_tokens: int = 150


@dataclass(frozen=True)
class JudgeConfig:
    judge_model: str
    backend: str | None = None
    run_dir: Path | None = None
    predictions_path: Path | None = None
    legacy_log_path: Path | None = None
    output_root: Path = Path("runs")
    dataset: str | None = None       # HF config name, used to find the latest run
    model_id: str | None = None
    num_frames: int | None = None
    skip_llm_judge: bool = False
