from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROMPT = (
    "These are uniformly sampled frames from a video. "
    "Analyze what action is happening with English."
)


@dataclass(frozen=True)
class BenchmarkConfig:
    video_dir: Path
    model_id: str
    sample_fps: float = 2.0
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
    video_dir: Path | None = None
    model_id: str | None = None
    sample_fps: float | None = None
    num_frames: int | None = None
    skip_llm_judge: bool = False
