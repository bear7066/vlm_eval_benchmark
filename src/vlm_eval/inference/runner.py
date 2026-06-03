from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

from vlm_eval.config import BenchmarkConfig
from vlm_eval.hardware import get_hardware_name, get_peak_vram_gb, reset_peak_memory_stats
from vlm_eval.hf_dataset import load_video_dataset
from vlm_eval.inference.gemma import HuggingFaceVLM
from vlm_eval.logging_utils import configure_logging, quiet_third_party_loggers
from vlm_eval.metrics import VideoResult, summarize_results
from vlm_eval.paths import build_run_id, ensure_run_dir, slugify
from vlm_eval.video import sample_frames_from_bytes


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
        f.write("\n")


def _append_jsonl(path: Path, data: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=_json_default)
        f.write("\n")


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def run_benchmark(config: BenchmarkConfig) -> Path | None:
    _load_dotenv_if_available()
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    quiet_third_party_loggers()

    label = config.dataset
    run_id = slugify(config.run_id) if config.run_id else build_run_id(config.model_id, label, config.num_frames)
    try:
        run_dir = ensure_run_dir(Path(config.output_root), run_id)
    except FileExistsError:
        run_dir = Path(config.output_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(run_dir / "benchmark.log", mode="a")

    config_data = asdict(config)
    config_data["output_root"] = str(config.output_root)
    config_data["run_id"] = run_id
    _write_json(run_dir / "config.json", config_data)

    predictions_path = run_dir / "predictions.jsonl"
    if predictions_path.exists():
        predictions_path.unlink()
    predictions_path.touch()

    logging.info("Run directory: %s", run_dir)
    logging.info("Loading dataset: %s / %s", config.dataset_repo, label)

    try:
        ds = load_video_dataset(label, config.dataset_repo)
    except Exception as exc:
        logging.error("Failed to load dataset: %s", exc)
        return None

    # Shuffle for reproducibility then cap at sample_size
    if config.seed is not None:
        ds = ds.shuffle(seed=config.seed, buffer_size=500)
    ds = ds.take(config.sample_size)

    logging.info("Loading model and processor: %s", config.model_id)
    hf_token = os.environ.get("HF_TOKEN")
    hardware_name = get_hardware_name()

    try:
        model = HuggingFaceVLM(config.model_id, hf_token=hf_token)
    except Exception as exc:
        logging.error("Failed to load model: %s", exc)
        return None

    reset_peak_memory_stats()

    logging.info("Starting benchmark.")
    logging.info("Hardware Name : %s", hardware_name)
    logging.info("Fixed Frames  : %s", config.num_frames)

    results: list[VideoResult] = []
    index = 0

    for example in ds:
        index += 1
        video_id = example.get("video_id") or "unknown"
        logging.info("")
        logging.info("=" * 60)
        logging.info("[%s] Processing video: %s", index, video_id)

        frames, video_duration_sec, total_video_frames, original_fps = sample_frames_from_bytes(
            example["video_bytes"],
            num_frames=config.num_frames,
        )
        if frames is None:
            result = VideoResult(
                video=video_id,
                label=label,
                status="error",
                error="Could not decode video bytes",
            )
            results.append(result)
            _append_jsonl(predictions_path, result.to_dict())
            continue

        try:
            generated = model.generate_from_frames(
                frames=frames,
                prompt_text=config.prompt,
                max_new_tokens=config.max_new_tokens,
            )

            elapsed_sec = generated["elapsed_sec"]
            sampled_fps = len(frames) / elapsed_sec if elapsed_sec > 0 else 0.0

            result = VideoResult(
                video=video_id,
                label=label,
                status="success",
                response=generated["response"],
                query_latency_sec=elapsed_sec,
                query_latency_ms=generated["elapsed_ms"],
                ttft_ms=generated["ttft_ms"],
                video_duration_sec=video_duration_sec,
                total_video_frames=total_video_frames,
                original_fps=original_fps,
                sampled_frames=len(frames),
                tokens=generated["tokens"],
                throughput_tps=generated["throughput_tps"],
                frames_per_second=sampled_fps,
                average_power_watts=generated["average_power_watts"],
            )

            if video_duration_sec is not None:
                logging.info("Video duration: %.2f sec", video_duration_sec)
            else:
                logging.info("Video duration: N/A")
            logging.info("Query Latency: %.2f ms", result.query_latency_ms)
            if result.ttft_ms is not None:
                logging.info("TTFT: %.2f ms", result.ttft_ms)
            else:
                logging.info("TTFT: N/A")
            logging.info("Frames Per Second: %.2f", result.frames_per_second)
            logging.info("Throughput: %.2f tokens/sec", result.throughput_tps)
            logging.info("Model answer: %s", result.response)
            logging.info("=" * 60)

        except Exception as exc:
            logging.error("Inference failed: %s", exc)
            result = VideoResult(
                video=video_id,
                label=label,
                status="error",
                error=str(exc),
                video_duration_sec=video_duration_sec,
                total_video_frames=total_video_frames,
                original_fps=original_fps,
                sampled_frames=len(frames),
            )

        results.append(result)
        _append_jsonl(predictions_path, result.to_dict())

    sample_size = index
    summary = summarize_results(
        results=results,
        sample_size=sample_size,
        hardware_name=hardware_name,
        model_id=config.model_id,
        dataset=label,
        num_frames=config.num_frames,
        peak_vram_gb=get_peak_vram_gb(),
    )
    _write_json(run_dir / "summary.json", summary)

    logging.info("")
    logging.info("==================== Benchmark Summary ====================")
    logging.info("")
    logging.info("Input")
    logging.info("\tModel          : %s", config.model_id)
    logging.info("\tHardware       : %s", hardware_name)
    logging.info("\tDataset        : %s / %s", config.dataset_repo, label)
    logging.info("\tFixed Frames   : %s", config.num_frames)

    output = summary["output"]
    logging.info("")
    logging.info("Output")
    logging.info("\tAverage Query Latency: %s ms", _format_optional(output["average_query_latency_ms"]))
    logging.info("\tFrames Per Second (FPS): %s", _format_optional(output["frames_per_second"]))
    logging.info(
        "\tEquivalent Real-time Latency (RT Latency): %s",
        _format_optional(output["equivalent_real_time_latency"]),
    )
    logging.info("\tPeak VRAM Usage: %s GB", _format_optional(output["peak_vram_usage_gb"]))
    logging.info("\tThroughput: %s tokens/sec", _format_optional(output["throughput_tokens_per_sec"]))
    logging.info("\tPower Consumption: %s W", _format_optional(output["power_consumption_watts"]))
    logging.info("\tTTFT: %s ms", _format_optional(output["ttft_ms"]))
    logging.info("")
    logging.info("Successful videos: %s / %s", output["successful_videos"], sample_size)
    if output["average_video_duration_sec"] is not None:
        logging.info("Average video duration: %.4f sec", output["average_video_duration_sec"])
    logging.info("Predictions written to: %s", predictions_path)
    logging.info("Summary written to: %s", run_dir / "summary.json")

    return run_dir


def _format_optional(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"
