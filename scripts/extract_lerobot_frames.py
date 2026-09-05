#!/usr/bin/env python3
"""Extract canonical JPEG frames from LeRobot v2.1 videos."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
from PIL import Image


DEFAULT_VIDEO_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)


def _video_path(root: Path, info: dict, episode_index: int, video_key: str) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    return root / info["video_path"].format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
        video_key=video_key,
    )


def _frame_dir(root: Path, info: dict, episode_index: int, video_key: str) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    return (
        root
        / "frames"
        / f"chunk-{episode_index // chunks_size:03d}"
        / video_key
        / f"episode_{episode_index:06d}"
    )


def _is_complete(path: Path, expected_frames: int) -> bool:
    if not path.is_dir():
        return False
    frames = sorted(path.glob("frame_*.jpg"))
    return (
        len(frames) == expected_frames
        and frames[0].name == "frame_000001.jpg"
        and frames[-1].name == f"frame_{expected_frames:06d}.jpg"
    )


def _extract_one(job: tuple[str, str, int, str, int, bool]) -> tuple[str, int]:
    root_str, info_json, episode_index, video_key, expected_frames, overwrite = job
    root = Path(root_str)
    info = json.loads(info_json)
    source = _video_path(root, info, episode_index, video_key)
    destination = _frame_dir(root, info, episode_index, video_key)

    if _is_complete(destination, expected_frames) and not overwrite:
        return f"skip {destination}", expected_frames
    if destination.exists() and not overwrite:
        raise RuntimeError(
            f"Partial frame directory exists: {destination}. "
            "Use --overwrite to replace it."
        )
    if not source.is_file():
        raise FileNotFoundError(source)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        count = 0
        with av.open(str(source)) as container:
            for frame in container.decode(video=0):
                count += 1
                image = Image.fromarray(frame.to_ndarray(format="rgb24"), mode="RGB")
                image.save(
                    temporary / f"frame_{count:06d}.jpg",
                    format="JPEG",
                    quality=95,
                )
        if count != expected_frames:
            raise RuntimeError(
                f"Frame count mismatch for {source}: decoded={count}, "
                f"metadata={expected_frames}"
            )
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
        return f"write {destination}", count
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_jobs(
    dataset_root: Path,
    video_keys: tuple[str, ...],
    max_episodes: int | None,
    overwrite: bool,
) -> list[tuple[str, str, int, str, int, bool]]:
    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    with info_path.open("r", encoding="utf-8") as file:
        info = json.load(file)
    version = str(info.get("codebase_version", "")).lower()
    if not version.startswith("v2"):
        raise ValueError(
            f"{dataset_root} reports codebase_version={version!r}; "
            "canonical extraction supports LeRobot v2.x only."
        )
    with episodes_path.open("r", encoding="utf-8") as file:
        episodes = [json.loads(line) for line in file if line.strip()]
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    info_json = json.dumps(info, sort_keys=True)
    return [
        (
            str(dataset_root),
            info_json,
            int(episode["episode_index"]),
            video_key,
            int(episode["length"]),
            overwrite,
        )
        for episode in episodes
        for video_key in video_keys
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decode LeRobot v2.1 videos and save the JPEG frames consumed by "
            "the canonical Stage 1 and Stage 2 recipes."
        )
    )
    parser.add_argument(
        "dataset_roots",
        nargs="+",
        type=Path,
        help="Concrete LeRobot dataset roots containing meta/info.json.",
    )
    parser.add_argument(
        "--video-key",
        action="append",
        dest="video_keys",
        help="Video key to extract. Repeat for multiple keys.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    video_keys = tuple(args.video_keys or DEFAULT_VIDEO_KEYS)
    jobs = []
    for root in args.dataset_roots:
        jobs.extend(
            _load_jobs(
                root.resolve(),
                video_keys=video_keys,
                max_episodes=args.max_episodes,
                overwrite=args.overwrite,
            )
        )
    if not jobs:
        raise RuntimeError("No episodes found.")

    completed = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(_extract_one, job) for job in jobs]
        for future in as_completed(futures):
            message, count = future.result()
            completed += 1
            print(f"[{completed}/{len(jobs)}] {message} frames={count}", flush=True)


if __name__ == "__main__":
    main()
