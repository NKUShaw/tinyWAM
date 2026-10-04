"""Evaluate existing Stage 1 checkpoints; no training or checkpoint mutation."""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from slim.compat.legacy import to_public_config
from slim.data.dataset import LeRobotBaseDataset
from slim.model import SLIMModel
from slim.model.checkpoint import load_weights


def load_source_stats(run: Path, sources: list[dict]) -> dict:
    """Training writes one flat q01/q99 payload per source, not one aggregate file."""
    stats = {}
    for source in sources:
        key = LeRobotBaseDataset._source_key(source)
        path = run / f"action_stats_{key}.json"
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        stats[key] = {"q01": payload["q01"], "q99": payload["q99"]}
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Resolved run config; defaults to checkpoint run/config.yaml")
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batches < 1 or args.batch_size < 2:
        parser.error("--batches must be positive and --batch-size must be at least 2")
    logging.basicConfig(level=logging.INFO)
    run = args.checkpoint.resolve().parents[1]
    cfg = to_public_config(OmegaConf.load(args.config or run / "config.yaml"))
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = cfg.data
    sources = OmegaConf.to_container(data.sources, resolve=True)
    stats = load_source_stats(run, sources)
    dataset = LeRobotBaseDataset(
        data_sources=sources, split="val",
        val_ratio=float(data.val_ratio),
        eval_monitor_episode_ratio=float(data.get("eval_monitor_episode_ratio", 0.05)),
        split_seed=int(data.split_seed), action_horizon=int(cfg.model.action_horizon),
        video_backend=data.get("video_backend", "torchvision_av"),
        action_normalization=data.get("action_normalization", "q01_q99"),
        action_dim=int(cfg.model.action_dim), state_dim=int(cfg.model.state_dim),
        action_normalized_dims=data.get("action_normalized_dims"),
        action_indices=data.get("action_indices"), state_indices=data.get("state_indices"),
        action_key=data.get("action_key", "action"), state_key=data.get("state_key", "observation.state"),
        action_stats=stats, skip_lerobot_episodes=data.get("skip_lerobot_episodes"),
        include_future_image=True, episode_shuffle=bool(data.get("episode_shuffle", True)),
    )
    # A fixed shuffled subset avoids repeatedly auditing the first adjacent trajectory.
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
                        num_workers=0, collate_fn=list)
    model = SLIMModel(cfg)
    load_weights(model, args.checkpoint, strict=True)
    model.to(device=args.device, dtype=torch.bfloat16 if args.bf16 else torch.float32).eval()
    sums, counts = {}, {}
    samples = 0
    batches = 0
    for batch in loader:
        if batches >= args.batches:
            break
        if len(batch) < 2:
            continue
        metrics = model.evaluate_dynamics(batch)
        samples += len(batch)
        batches += 1
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value * len(batch)
            counts[key] = counts.get(key, 0) + len(batch)
    if not samples:
        raise RuntimeError("No evaluation batch with at least two samples was available")
    payload = {
        "checkpoint": str(args.checkpoint.resolve()), "seed": seed,
        "samples": samples, "batches": batches,
        "future_loss_type": cfg.model.transformer.get("future_loss_type", "cosine"),
        "validation_is_holdout": float(data.val_ratio) > 0,
        "metrics": {key: sums[key] / counts[key] for key in sums},
    }
    result = json.dumps(payload, indent=2, ensure_ascii=False)
    print(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
