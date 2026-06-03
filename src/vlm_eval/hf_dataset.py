from __future__ import annotations

import os

# Override via HF_DATASET_REPO env var once the dataset is uploaded.
DEFAULT_DATASET_REPO: str = os.getenv("HF_DATASET_REPO", "gnitoahc/vlm-eval-videos")

VALID_CONFIGS = frozenset(
    ["climbing_ladder", "face_planting", "falling_off_bike", "falling_off_chair"]
)


def load_video_dataset(config_name: str, dataset_repo: str = DEFAULT_DATASET_REPO):
    """Return a streaming HF IterableDataset for one action config.

    Each example: {"video_id": str, "label": str, "video_bytes": bytes}

    Set HF_TOKEN env var (or ~/.huggingface/token) for private repos.
    """
    if config_name not in VALID_CONFIGS:
        raise ValueError(
            f"Unknown dataset config '{config_name}'. "
            f"Valid options: {sorted(VALID_CONFIGS)}"
        )

    from datasets import load_dataset
    from datasets.features.video import Video

    ds = load_dataset(
        dataset_repo,
        config_name,
        streaming=True,
        split="train",
    )

    # The repo stores videos in WebDataset (tar.gz) format.  Auto-detection
    # assigns: mp4 → Video(decode=True), __key__ → str, __url__ → str.
    # decode=True requires torchcodec; cast to decode=False to keep raw bytes.
    ds = ds.cast_column("mp4", Video(decode=False))

    # Reshape to the pipeline's expected schema: video_id, label, video_bytes.
    # __key__ is "{label}_{video_id}", e.g. "falling_off_chair_-5hw88bD4mE".
    return ds.map(
        lambda ex: _reshape_example(ex, config_name),
        remove_columns=["mp4", "__key__", "__url__"],
    )


def _reshape_example(example: dict, label: str) -> dict:
    key = example.get("__key__") or ""
    prefix = label + "_"
    video_id = key[len(prefix):] if key.startswith(prefix) else key

    mp4 = example.get("mp4") or {}
    if isinstance(mp4, dict):
        video_bytes = mp4.get("bytes") or b""
    elif isinstance(mp4, (bytes, bytearray)):
        video_bytes = bytes(mp4)
    else:
        video_bytes = b""

    return {"video_id": video_id, "label": label, "video_bytes": video_bytes}
