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


def sample_frames(video_path: Path, num_frames: int = 8):
    import decord  # 用 decord 讀取影片 frame。
    import numpy as np  # 用 numpy 計算平均分布的 frame index。
    from PIL import Image  # 將 numpy frame 轉成 PIL Image，供模型 processor 使用。

    try:  # 嘗試建立影片 reader。
        video_reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0))  # 用 CPU backend 開啟影片。
    except Exception as exc:  # 如果影片無法讀取，記錄錯誤並回傳失敗。
        logging.error("Could not read video %s: %s", video_path, exc)  # 寫入影片讀取失敗的原因。
        return None, None, None, None  # 回傳空結果，讓呼叫端跳過這支影片。

    total_frames = len(video_reader)  # 取得影片總 frame 數。
    if total_frames == 0:  # 如果影片沒有任何 frame，無法抽樣。
        return None, None, None, None  # 回傳空結果，讓呼叫端跳過這支影片。

    try:  # 嘗試讀取原始影片 FPS。
        original_fps = video_reader.get_avg_fps()  # 取得影片平均 FPS。
    except Exception:  # 如果讀取 FPS 失敗，先設為 None。
        original_fps = None  # 固定張數抽樣不依賴 FPS，所以仍可繼續抽 frame。

    video_duration_sec = get_video_duration(video_reader)  # 計算影片長度，讀不到 FPS 時會是 None。

    if num_frames <= 0:  # 固定抽樣張數必須大於 0。
        logging.error("num_frames must be > 0, got %s", num_frames)  # 記錄錯誤的抽樣設定。
        return None, None, None, None  # 回傳空結果，避免產生無效 frame index。

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)  # 從頭到尾平均切出固定數量的 frame index。
    indices = np.clip(indices, 0, total_frames - 1)  # 確保所有 index 都落在有效 frame 範圍內。
    indices = np.unique(indices)  # 移除重複 index，避免同一張 frame 被抽多次。

    if len(indices) == 0:  # 如果沒有產生任何 index，至少保留第一張 frame。
        indices = np.array([0], dtype=int)  # fallback 成只取第 0 張 frame。

    frames = video_reader.get_batch(indices).asnumpy()  # 一次讀出所有選中的 frames，並轉成 numpy array。
    pil_frames = [Image.fromarray(frame) for frame in frames]  # 將每張 numpy frame 轉成 PIL Image。
    return pil_frames, video_duration_sec, total_frames, original_fps  # 回傳抽出的 frames 與影片 metadata。
