"""Checkpoint loading and action-stat discovery."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from slim.compat.legacy import to_runtime_config


def find_action_stats(run_dir: Path) -> Path:
    candidates = [
        run_dir / "action_stats.json",
        run_dir / "dataset_statistics.json",
        *sorted(run_dir.glob("action_stats_*.json")),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No action statistics found under {run_dir}")


def load_run_metadata(checkpoint: str | Path):
    checkpoint = Path(checkpoint)
    run_dir = checkpoint.parents[1]
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = to_runtime_config(OmegaConf.load(config_path))
    with find_action_stats(run_dir).open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    return config, stats


def load_weights(model, checkpoint: str | Path, strict: bool = True):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state.pop("vision_encoder.model.mask_token", None)
    unexpected_language = [key for key in state if key.startswith("lang_encoder.")]
    for key in unexpected_language:
        state.pop(key)
    result = model.load_state_dict(state, strict=False)
    missing = [key for key in result.missing_keys if not key.endswith(".mask_token")]
    unexpected = list(result.unexpected_keys)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return result
