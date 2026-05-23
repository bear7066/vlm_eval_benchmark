from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class VideoResult:
    video: str
    label: str
    status: str
    response: str = ""
    error: str | None = None
    query_latency_sec: float | None = None
    query_latency_ms: float | None = None
    ttft_ms: float | None = None
    video_duration_sec: float | None = None
    total_video_frames: int | None = None
    original_fps: float | None = None
    sampled_frames: int | None = None
    tokens: int | None = None
    throughput_tps: float | None = None
    frames_per_second: float | None = None
    average_power_watts: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_results(
    results: list[VideoResult],
    sample_size: int,
    hardware_name: str,
    model_id: str,
    video_dir: str,
    num_frames: int,
    peak_vram_gb: float | None,
) -> dict[str, Any]:
    successful = [result for result in results if result.status == "success"]
    total_time = sum(result.query_latency_sec or 0.0 for result in successful)
    total_tokens = sum(result.tokens or 0 for result in successful)
    total_sampled_frames = sum(result.sampled_frames or 0 for result in successful)

    duration_values = [
        result.video_duration_sec
        for result in successful
        if result.video_duration_sec is not None
    ]
    power_values = [
        result.average_power_watts
        for result in successful
        if result.average_power_watts is not None
    ]
    ttft_values = [
        result.ttft_ms
        for result in successful
        if result.ttft_ms is not None
    ]

    avg_query_latency_sec = total_time / len(successful) if successful else None
    avg_video_duration = sum(duration_values) / len(duration_values) if duration_values else None

    equivalent_real_time_latency = None
    if avg_query_latency_sec is not None and avg_video_duration and avg_video_duration > 0:
        equivalent_real_time_latency = avg_query_latency_sec / avg_video_duration

    return {
        "input": {
            "model": model_id,
            "hardware_name": hardware_name,
            "video_dir": video_dir,
            "num_frames": num_frames,
            "sample_size": sample_size,
        },
        "output": {
            "average_query_latency_ms": (
                avg_query_latency_sec * 1000.0 if avg_query_latency_sec is not None else None
            ),
            "frames_per_second": (
                total_sampled_frames / total_time if total_time > 0 else None
            ),
            "equivalent_real_time_latency": equivalent_real_time_latency,
            "peak_vram_usage_gb": peak_vram_gb,
            "throughput_tokens_per_sec": (
                total_tokens / total_time if total_time > 0 else None
            ),
            "power_consumption_watts": (
                sum(power_values) / len(power_values) if power_values else None
            ),
            "ttft_ms": sum(ttft_values) / len(ttft_values) if ttft_values else None,
            "successful_videos": len(successful),
            "attempted_videos": len(results),
            "requested_sample_size": sample_size,
            "average_video_duration_sec": avg_video_duration,
        },
    }
