from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = (".mp4", ".mkv")


def find_videos(video_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for extension in VIDEO_EXTENSIONS:
        paths.extend(video_dir.rglob(f"*{extension}"))
    return sorted(paths)


def get_video_duration(video_reader: Any) -> float | None:
    try:
        fps = video_reader.get_avg_fps()
        total_frames = len(video_reader)
        if fps and fps > 0:
            return total_frames / fps
    except Exception:
        pass
    return None


def sample_frames(video_path: Path, sample_fps: float = 2.0):
    import decord
    import numpy as np
    from PIL import Image

    try:
        video_reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    except Exception as exc:
        logging.error("Could not read video %s: %s", video_path, exc)
        return None, None, None, None

    total_frames = len(video_reader)
    if total_frames == 0:
        return None, None, None, None

    try:
        original_fps = video_reader.get_avg_fps()
    except Exception:
        original_fps = None

    video_duration_sec = get_video_duration(video_reader)

    if not original_fps or original_fps <= 0:
        logging.error("Could not determine FPS, skipping %s", video_path)
        return None, None, None, None

    if sample_fps <= 0:
        logging.error("sample_fps must be > 0, got %s", sample_fps)
        return None, None, None, None

    duration_sec = video_duration_sec if video_duration_sec is not None else total_frames / original_fps
    timestamps = np.arange(0, duration_sec, 1.0 / sample_fps)
    indices = np.floor(timestamps * original_fps).astype(int)
    indices = np.clip(indices, 0, total_frames - 1)
    indices = np.unique(indices)

    if len(indices) == 0:
        indices = np.array([0], dtype=int)

    frames = video_reader.get_batch(indices).asnumpy()
    pil_frames = [Image.fromarray(frame) for frame in frames]
    return pil_frames, video_duration_sec, total_frames, original_fps
