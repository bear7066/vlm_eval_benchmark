from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def model_name_from_id(model_id: str) -> str:
    return model_id.split("/")[-1].replace("-it", "")


def label_from_video_dir(video_dir: Path) -> str:
    """Legacy helper — still used by judge/parser.py for old log files."""
    name = Path(video_dir).resolve().name
    return name if name and name != "." else "default_ground_truth"


def display_label_from_video_path(video_path: str | Path) -> str:
    label = Path(video_path).parent.name
    if not label or label in {".", ".."}:
        return "Unknown Action"
    return label.replace("_", " ")


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "run"


def build_run_id(
    model_id: str,
    label: str,
    num_frames: int,
    now: datetime | None = None,
) -> str:
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    model_name = slugify(model_name_from_id(model_id))
    return f"{model_name}_{num_frames}frames_{slugify(label)}_{timestamp}"


def ensure_run_dir(output_root: Path, run_id: str) -> Path:
    run_dir = output_root / slugify(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_run(
    output_root: Path,
    model_id: str | None = None,
    dataset: str | None = None,
    num_frames: int | None = None,
) -> Path | None:
    if not output_root.exists():
        return None

    candidates: list[Path] = []

    for config_path in output_root.glob("*/config.json"):
        try:
            config = read_json(config_path)
        except Exception:
            continue

        if model_id is not None and config.get("model_id") != model_id:
            continue
        if dataset is not None and config.get("dataset") != dataset:
            continue
        if num_frames is not None and int(config.get("num_frames", -1)) != num_frames:
            continue

        candidates.append(config_path.parent)

    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)
