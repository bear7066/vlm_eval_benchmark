"""TensorRT-LLM sweep orchestration.

Kept separate from :mod:`realtime_eval.pipeline.sweep` (the HuggingFace path) by
design: this only swaps the model loader for :class:`TensorRTVLM` and tags the
run as ``backend="tensorrt"``. Everything else — the timed loop, the JSONL/JSON
writers, aggregation — is reused from the HuggingFace pipeline.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from vlm_eval.hardware import get_hardware_name
from vlm_eval.paths import model_name_from_id, slugify

from realtime_eval.core.config import SweepConfig
from realtime_eval.core.dataset import discover_videos
from realtime_eval.core.metrics import RealtimeResult, aggregate
from realtime_eval.core.trt_backend import TensorRTVLM
from realtime_eval.pipeline.runner import run_config
from realtime_eval.pipeline.sweep import _append_jsonl, _write_json

logger = logging.getLogger(__name__)


def load_trt_model(model_id: str, hf_token: str | None = None) -> TensorRTVLM:
    """Load a TensorRT-LLM-served VLM once for reuse across a config's repeats.

    Args:
        model_id: HuggingFace model ID / TensorRT-LLM checkpoint.
        hf_token: Optional HuggingFace access token.

    Returns:
        A ready :class:`realtime_eval.core.trt_backend.TensorRTVLM`.
    """
    return TensorRTVLM(model_id, hf_token=hf_token)


def run_sweep_rt(
    videos_root: Path,
    config: SweepConfig,
    video_limit: int | None = None,
) -> Path:
    """Run the full real-time sweep using the TensorRT-LLM backend.

    Mirrors :func:`realtime_eval.pipeline.sweep.run_sweep` but loads each model
    through :func:`load_trt_model` and writes to a ``sweep_rt_*`` run dir tagged
    ``backend="tensorrt"``.

    Args:
        videos_root: Directory of labeled videos (or a single video file).
        config: Sweep grid and timing parameters.
        video_limit: Optional cap on number of videos used.

    Returns:
        Path to the created run directory.
    """
    load_dotenv()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    hf_token = os.environ.get("HF_TOKEN")

    videos = discover_videos(Path(videos_root), limit=video_limit)
    hardware_name = get_hardware_name()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config.output_root) / slugify(f"sweep_rt_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        run_dir / "config.json",
        {
            "backend": "tensorrt",
            "videos_root": str(videos_root),
            "num_videos": len(videos),
            "hardware_name": hardware_name,
            "model_ids": list(config.model_ids),
            "num_frames_grid": list(config.num_frames_grid),
            "max_new_tokens_grid": list(config.max_new_tokens_grid),
            "repeats": config.repeats,
            "warmup": config.warmup,
            "realtime_threshold": config.realtime_threshold,
            "prompt": config.prompt,
        },
    )

    results_path = run_dir / "results.jsonl"
    logger.info("TensorRT sweep run dir: %s", run_dir)
    logger.info("Hardware: %s | videos: %d", hardware_name, len(videos))

    all_results: list[RealtimeResult] = []
    for model_id in config.model_ids:
        logger.info("Loading model: %s", model_id)
        try:
            model = load_trt_model(model_id, hf_token=hf_token)
        except Exception as exc:
            logger.error("Failed to load %s, skipping: %s", model_id, exc)
            continue

        for num_frames in config.num_frames_grid:
            for max_new_tokens in config.max_new_tokens_grid:
                logger.info(
                    "Config: %s | frames=%d | max_new_tokens=%d",
                    model_name_from_id(model_id),
                    num_frames,
                    max_new_tokens,
                )
                results = run_config(
                    model=model,
                    model_id=model_id,
                    videos=videos,
                    num_frames=num_frames,
                    max_new_tokens=max_new_tokens,
                    prompt=config.prompt,
                    repeats=config.repeats,
                    warmup=config.warmup,
                    power_interval_sec=config.power_sample_interval_sec,
                )
                for result in results:
                    _append_jsonl(results_path, result.to_dict())
                all_results.extend(results)

        del model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    summaries = aggregate(all_results, threshold=config.realtime_threshold)
    _write_json(
        run_dir / "summary.json",
        {
            "backend": "tensorrt",
            "hardware_name": hardware_name,
            "realtime_threshold": config.realtime_threshold,
            "configs": [s.to_dict() for s in summaries],
        },
    )
    logger.info("Wrote %d results to %s", len(all_results), results_path)
    return run_dir
