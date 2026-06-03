"""Upload local dataset videos to a HuggingFace Hub dataset repository.

Converts each per-label directory of MP4 files into sharded Parquet files
(standard HuggingFace format — no custom loading script required) and uploads
them under data/{label}/, then uploads a README with YAML dataset-card metadata.

Usage:
    python scripts/upload_to_huggingface.py --repo_id owner/vlm-eval-videos
    python scripts/upload_to_huggingface.py --repo_id owner/vlm-eval-videos \\
        --dataset_root ./dataset --token $HF_TOKEN

The HF_TOKEN env var (or ~/.huggingface/token) is used for authentication if
--token is not supplied.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_LABELS = [
    "climbing_ladder",
    "face_planting",
    "falling_off_bike",
    "falling_off_chair",
]

_PARQUET_SCHEMA = pa.schema([
    ("video_id", pa.string()),
    ("label", pa.string()),
    ("video_bytes", pa.binary()),
])

# Number of videos per Parquet shard. Smaller shards stream more efficiently.
_SHARD_SIZE = 50

_README_TEMPLATE = """\
---
license: other
tags:
  - video
  - action-recognition
dataset_info:
{dataset_info_block}
configs:
{configs_block}
---

# VLM Eval Videos

Short MP4 clips of four physical-action categories scraped from YouTube,
used for offline VLM benchmark evaluation.

## Configs / subsets

| Config name | Description | # videos |
|---|---|---|
{table_rows}

## Loading

```python
from datasets import load_dataset

ds = load_dataset(
    "{repo_id}",
    "climbing_ladder",        # or any config name above
    streaming=True,
    split="train",
)

for example in ds:
    video_id    = example["video_id"]     # str   — YouTube ID
    label       = example["label"]        # str   — action label
    video_bytes = example["video_bytes"]  # bytes — raw MP4
```
"""


def _build_readme(repo_id: str, label_counts: dict[str, int]) -> str:
    # Explicit feature types prevent datasets from auto-detecting video_bytes as
    # a Video feature (which would require torchcodec for decoding).
    dataset_info_lines: list[str] = []
    for label in _LABELS:
        count = label_counts.get(label, 0)
        dataset_info_lines += [
            f"  - config_name: {label}",
            f"    features:",
            f"      - name: video_id",
            f"        dtype: string",
            f"      - name: label",
            f"        dtype: string",
            f"      - name: video_bytes",
            f"        dtype: binary",
            f"    splits:",
            f"      - name: train",
            f"        num_examples: {count}",
        ]
    dataset_info_block = "\n".join(dataset_info_lines)

    configs_block_lines: list[str] = []
    for label in _LABELS:
        configs_block_lines += [
            f"  - config_name: {label}",
            f"    data_files:",
            f"      - split: train",
            f"        path: data/{label}/*.parquet",
        ]
    configs_block = "\n".join(configs_block_lines)

    table_rows = "\n".join(
        f"| `{label}` | Videos of '{label.replace('_', ' ')}' | {label_counts.get(label, '?')} |"
        for label in _LABELS
    )

    return _README_TEMPLATE.format(
        dataset_info_block=dataset_info_block,
        configs_block=configs_block,
        table_rows=table_rows,
        repo_id=repo_id,
    )


def _iter_parquet_shards(label_dir: Path, label: str):
    """Yield (shard_index, parquet_bytes, row_count) for each shard of _SHARD_SIZE videos."""
    mp4_files = sorted(label_dir.glob("*.mp4"))
    for shard_idx, offset in enumerate(range(0, len(mp4_files), _SHARD_SIZE)):
        chunk = mp4_files[offset : offset + _SHARD_SIZE]

        video_ids: list[str] = []
        labels: list[str] = []
        video_bytes_list: list[bytes] = []

        for mp4 in chunk:
            raw = mp4.read_bytes()
            if not raw:
                log.warning("Skipping empty file: %s", mp4.name)
                continue
            stem = mp4.stem  # e.g. "climbing_ladder_03jyaZxUUgk"
            video_id = stem[len(label) + 1 :] if stem.startswith(label + "_") else stem
            video_ids.append(video_id)
            labels.append(label)
            video_bytes_list.append(raw)

        table = pa.table(
            {
                "video_id": pa.array(video_ids, type=pa.string()),
                "label": pa.array(labels, type=pa.string()),
                "video_bytes": pa.array(video_bytes_list, type=pa.binary()),
            },
            schema=_PARQUET_SCHEMA,
        )

        buf = io.BytesIO()
        pq.write_table(table, buf)
        yield shard_idx, buf.getvalue(), len(chunk)


def upload(repo_id: str, dataset_root: Path, token: str | None) -> None:
    api = HfApi(token=token)

    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    log.info("Repository: https://huggingface.co/datasets/%s", repo_id)

    label_counts: dict[str, int] = {}

    for label in _LABELS:
        label_dir = dataset_root / label
        if not label_dir.exists():
            log.warning("Skipping '%s' — directory not found: %s", label, label_dir)
            continue

        log.info("[%s] Converting to Parquet shards...", label)
        total = 0
        for shard_idx, parquet_bytes, row_count in _iter_parquet_shards(label_dir, label):
            total += row_count
            shard_name = f"train-{shard_idx:05d}-of-{{}}.parquet"  # name updated after loop
            path_in_repo = f"data/{label}/train-{shard_idx:05d}.parquet"
            api.upload_file(
                path_or_fileobj=io.BytesIO(parquet_bytes),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Upload {label} shard {shard_idx} ({row_count} videos)",
            )
            log.info(
                "[%s] Uploaded shard %d — %d videos, %.1f MB",
                label, shard_idx, row_count, len(parquet_bytes) / 1_048_576,
            )

        label_counts[label] = total
        log.info("[%s] Done — %d videos total.", label, total)

    readme_content = _build_readme(repo_id, label_counts)
    api.upload_file(
        path_or_fileobj=io.BytesIO(readme_content.encode()),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add dataset card with YAML metadata",
    )
    log.info("Uploaded README.md")
    log.info("Done. View at: https://huggingface.co/datasets/%s", repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload VLM eval videos to HuggingFace Hub.")
    parser.add_argument("--repo_id", required=True, help="HF repo, e.g. owner/vlm-eval-videos")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("./dataset"),
        help="Local root containing per-label subdirectories (default: ./dataset)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (default: $HF_TOKEN env var)",
    )
    args = parser.parse_args()
    upload(args.repo_id, args.dataset_root, args.token)


if __name__ == "__main__":
    main()
