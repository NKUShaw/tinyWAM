from types import SimpleNamespace

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from slim.training import stage1, stage2
from slim.training.common import steps_from_epochs
from slim.training.optimizer import build_parameter_groups


class _OptimizerFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_model = nn.ModuleDict(
            {"proj": nn.Linear(4, 4), "norm": nn.LayerNorm(4)}
        )
        self.vision_encoder = nn.ModuleDict(
            {"proj": nn.Linear(4, 4), "norm": nn.LayerNorm(4)}
        )
        self.base = nn.Linear(4, 4)


class _CheckpointFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.ema_enabled = False


def test_training_entries_build_accelerator_from_config(monkeypatch):
    captured = []

    def fake_accelerator(**kwargs):
        captured.append(kwargs)
        return kwargs

    monkeypatch.setattr(stage1, "Accelerator", fake_accelerator)
    monkeypatch.setattr(stage2, "Accelerator", fake_accelerator)
    cfg = OmegaConf.create({"training": {"gradient_accumulation_steps": 3}})

    stage1._build_accelerator(cfg)
    stage2._build_accelerator(cfg)

    assert [
        item["gradient_accumulation_plugin"].num_steps for item in captured
    ] == [3, 3]


def test_steps_from_epochs_uses_actual_accelerator_accumulation():
    cfg = OmegaConf.create(
        {
            "data": {"per_device_batch_size": 8},
            "training": {
                "gradient_accumulation_steps": 99,
                "max_epochs": 3,
                "max_train_steps": 1000,
            },
        }
    )
    accelerator = SimpleNamespace(num_processes=4, gradient_accumulation_steps=2)
    dataset = list(range(1000))

    steps_per_epoch, max_train_steps = steps_from_epochs(
        cfg, dataset, accelerator
    )

    assert steps_per_epoch == 16
    assert max_train_steps == 48


def test_optimizer_groups_exclude_bias_and_norm_from_weight_decay():
    model = _OptimizerFixture()
    cfg = OmegaConf.create(
        {
            "training": {
                "learning_rate": {
                    "base": 2.5e-5,
                    "action_model": 1e-4,
                    "vision_encoder": 1e-5,
                },
                "optimizer": {"weight_decay": 0.01},
            }
        }
    )

    groups = build_parameter_groups(model, cfg)
    assert [group["name"] for group in groups] == [
        "action_model",
        "action_model_no_decay",
        "vision_encoder",
        "vision_encoder_no_decay",
        "base",
        "base_no_decay",
    ]
    assert [group["weight_decay"] for group in groups] == [
        0.01,
        0.0,
        0.01,
        0.0,
        0.01,
        0.0,
    ]
    assert [group["lr"] for group in groups] == [
        1e-4,
        1e-4,
        1e-5,
        1e-5,
        2.5e-5,
        2.5e-5,
    ]

    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    grouped_parameters = [
        parameter for group in groups for parameter in group["params"]
    ]
    assert len(grouped_parameters) == len({id(parameter) for parameter in grouped_parameters})
    assert {id(parameter) for parameter in grouped_parameters} == set(parameter_names)

    no_decay_names = {
        parameter_names[id(parameter)]
        for group in groups
        if group["weight_decay"] == 0.0
        for parameter in group["params"]
    }
    assert no_decay_names == {
        "action_model.proj.bias",
        "action_model.norm.weight",
        "action_model.norm.bias",
        "vision_encoder.proj.bias",
        "vision_encoder.norm.weight",
        "vision_encoder.norm.bias",
        "base.bias",
    }


def test_stage2_checkpoint_loader_only_skips_declared_prefixes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stage2,
        "logger",
        SimpleNamespace(info=lambda *args, **kwargs: None),
    )
    model = _CheckpointFixture()
    checkpoint = dict(model.state_dict())
    checkpoint["ema_vision_encoder.weight"] = torch.ones(2, 2)
    checkpoint["_ema_fp32_0"] = torch.ones(2, 2)
    checkpoint_path = tmp_path / "stage1.pt"
    torch.save(checkpoint, checkpoint_path)

    stage2._load_initial_weights(
        model,
        str(checkpoint_path),
        skip_prefixes=("ema_vision_encoder.", "_ema_fp32_"),
    )

    checkpoint["unexpected.weight"] = torch.ones(2, 2)
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(RuntimeError, match="unexpected.weight"):
        stage2._load_initial_weights(
            model,
            str(checkpoint_path),
            skip_prefixes=("ema_vision_encoder.", "_ema_fp32_"),
        )
