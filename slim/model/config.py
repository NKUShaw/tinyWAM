"""Runtime model configuration helpers."""

from __future__ import annotations

import dataclasses

from omegaconf import OmegaConf

from slim.compat.legacy import to_runtime_config


def merge_runtime_defaults(defaults_type, config):
    config = to_runtime_config(config)
    defaults = OmegaConf.create(dataclasses.asdict(defaults_type()))
    config.framework = OmegaConf.merge(defaults, config.get("framework", {}))
    return config
