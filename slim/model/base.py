"""Common policy model interface."""

from __future__ import annotations

from transformers import PretrainedConfig, PreTrainedModel


class PolicyModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self):
        super().__init__(PretrainedConfig())

    @classmethod
    def from_checkpoint(cls, checkpoint, **kwargs):
        from slim.model.checkpoint import load_run_metadata, load_weights
        from slim.model.slim_model import SLIMModel

        config, stats = load_run_metadata(checkpoint)
        model = SLIMModel(config, **kwargs)
        load_weights(model, checkpoint, strict=True)
        model.norm_stats = stats
        return model
