from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any
from vlm_eval.paths import display_label_from_video_path


def normalize_label(label: str) -> str:
    return label.replace("_", " ").strip() or "Unknown Action"


def load_predictions_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("status") == "success" and item.get("response"):
                item.setdefault("label", display_label_from_video_path(item.get("video", "")))
                item["label"] = normalize_label(str(item["label"]))
                items.append(item)
    return items


def parse_legacy_log_file(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []

    parsed_items: list[dict[str, Any]] = []
    current_video: str | None = None
    current_answers: list[str] = []

    def flush_current() -> None:
        nonlocal current_video, current_answers
        if current_video and current_answers:
            parsed_items.append(
                {
                    "video": current_video,
                    "response": " ".join(current_answers).strip(),
                    "label": display_label_from_video_path(current_video),
                    "status": "success",
                }
            )
        current_video = None
        current_answers = []

    video_pattern = re.compile(r"\[\d+/\d+\]\s*(?:處理影片|Processing video):\s*(.+)$")
    answer_pattern = re.compile(r"(?:模型回答|Model answer)\s*:\s*(.+)$")

    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("=" * 60):
            flush_current()
            continue

        video_match = video_pattern.search(line)
        if video_match:
            flush_current()
            current_video = video_match.group(1).strip()
            continue

        answer_match = answer_pattern.search(line)
        if answer_match and current_video:
            current_answers.append(answer_match.group(1).strip())

    flush_current()

    for item in parsed_items:
        item["label"] = normalize_label(str(item["label"]))

    return parsed_items


def extract_score(judge_text: str) -> int | None:
    match = re.search(r"Score:\s*(\d+)", judge_text)
    if match:
        return int(match.group(1))
    return None
