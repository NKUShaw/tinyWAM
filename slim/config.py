"""Configuration loading with lightweight YAML inheritance."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def load_config(path: str | Path):
    path = Path(path)
    config = OmegaConf.load(path)
    parents = config.pop("defaults", [])
    merged = OmegaConf.create({})
    for parent in parents:
        merged = OmegaConf.merge(merged, load_config(path.parent / str(parent)))
    return OmegaConf.merge(merged, config)
