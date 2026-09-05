"""CPU integration checks with synthetic features; no model/data downloads."""
from __future__ import annotations

import logging
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from slim.config import load_config
from slim.compat.legacy import to_public_config, to_runtime_config
from slim.model.control_latent import FactorizedControlEncoder, batch_variance_loss
from slim.model.initialization import load_control_warm_start
from slim.model import slim_model
from slim.model.checkpoint import load_weights


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class FakeVision(nn.Module):
    hidden_size = 12
    num_patches = 4

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(12, 12)

    def forward(self, images):
        x = torch.stack([torch.stack(views) for views in images]).to(self.proj.weight)
        patches = self.proj(x)
        return SimpleNamespace(patch_tokens=patches, cls_tokens=patches.mean(2))


def tiny_config(*, enabled=True, slots=0, stage=1, delta=False, dynamics=False):
    n = 2 * (slots or 4)
    return OmegaConf.create({
        "model": {
            "name": "SLIM",
            "vision_encoder": {"backbone_name": "fake", "freeze_backbone": True,
                               "num_image_views": 2, "vision_condition_mode": "dense_patch"},
            "ema": {"enabled": stage == 1 or dynamics, "momentum": 0.5},
            "control_latent": {"enabled": enabled, "control_dim": 6, "residual_dim": 6,
                               "tokens_per_view": slots, "num_heads": 2,
                               "decorrelation_weight": 0.01},
            "policy_dynamics": {"enabled": dynamics, "loss_weight": 0.1, "every_n_steps": 4},
            "action_dim": 3, "state_dim": 7, "action_horizon": 2,
            "use_language_condition": True, "language_embedding_dim": 5,
            "max_language_tokens": 3, "repeated_diffusion_steps": 2,
            "decoder_hidden_dim": 24, "inference_steps": 2,
            "transformer": {"hidden_dim": 12, "num_layers": 2, "num_heads": 2,
                            "ffn_ratio": 2, "num_future_tokens": n,
                            "max_state_tokens": 20, "max_action_tokens": 6,
                            "num_action_register_tokens": 1, "use_state_condition": True,
                            "future_loss_type": "norm_l1", "idm_loss_weight": 0.125,
                            "fdm_loss_weight": 1.0, "future_delta_loss_weight": 0.1 if delta else 0},
        },
        "data": {},
        "training": {"stage": stage, "objective": "idm_fdm" if stage == 1 else "policy"},
    })


def make_model(monkeypatch, **kwargs):
    monkeypatch.setattr(slim_model, "build_vision_encoder", lambda **unused: FakeVision())
    return slim_model.SLIMModel(tiny_config(**kwargs))


def examples(batch=3):
    rng = np.random.default_rng(11)
    return [{
        "image": [torch.randn(4, 12), torch.randn(4, 12)],
        "future_image": [torch.randn(4, 12), torch.randn(4, 12)],
        "action": rng.normal(size=(2, 3)).astype(np.float32),
        "state": rng.normal(size=(1, 7)).astype(np.float32),
        "lang_embs": rng.normal(size=(3, 5)).astype(np.float32), "lang_length": 2,
    } for _ in range(batch)]


@pytest.mark.parametrize("slots", [0, 2])
def test_policy_cannot_read_residual_and_reconstruction_receives_gradients(slots):
    encoder = FactorizedControlEncoder(12, 2, 4, 5, {
        "control_dim": 6, "residual_dim": 6, "tokens_per_view": slots, "num_heads": 2,
        "decorrelation_weight": 0.01,
    })
    x = torch.randn(3, 8, 12)
    before, _ = encoder(x)
    with torch.no_grad():
        encoder.residual_proj.weight.add_(100)
    after, losses = encoder(x, compute_aux=True)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert after.shape == (3, 2 * (slots or 4), 12)
    (after[..., 0].square().mean() + losses["representation_loss"]).backward()
    for param in (encoder.control_proj.weight, encoder.residual_proj.weight,
                  encoder.reconstruction[0].weight):
        assert param.grad is not None and param.grad.abs().sum() > 0
    if slots:
        assert encoder.queries.grad.abs().sum() > 0
        assert encoder.reconstruct_attention.in_proj_weight.grad.abs().sum() > 0


def test_fixed_query_diversity_does_not_hide_batch_collapse():
    repeated_slots = torch.randn(1, 8, 12).expand(4, -1, -1)
    assert batch_variance_loss(repeated_slots, 0.5) > 0.45


@pytest.mark.parametrize("slots", [0, 2])
def test_stage1_loss_backward_and_ema_includes_control_encoder(monkeypatch, slots):
    model = make_model(monkeypatch, slots=slots, delta=True).train()
    output = model(examples())
    for key in ("idm_loss", "fdm_loss", "future_loss", "delta_loss", "representation_loss"):
        assert torch.isfinite(output[key])
    torch.testing.assert_close(output["action_loss"],
        0.125 * output["idm_loss"] + output["fdm_loss"] + output["representation_loss"])
    output["action_loss"].backward()
    assert model.control_encoder.control_proj.weight.grad.abs().sum() > 0
    assert model.action_model.delta_decoder.layer2.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.vision_encoder.parameters())
    assert all(p.grad is None for p in model.ema_control_encoder.parameters())
    old = model.ema_control_encoder.control_proj.weight.detach().clone()
    with torch.no_grad():
        model.control_encoder.control_proj.weight.add_(0.2)
    model.update_ema()
    torch.testing.assert_close(model.ema_control_encoder.control_proj.weight, old + 0.1)
    assert not model.vision_encoder.training and not model.ema_control_encoder.training


def test_bf16_ema_keeps_fp32_shadow_for_new_module(monkeypatch):
    model = make_model(monkeypatch).to(torch.bfloat16)
    with torch.no_grad():
        model.control_encoder.control_proj.weight.add_(0.125)
    model.update_ema()
    assert all(getattr(model, name).dtype == torch.float32 for name in model._ema_fp32_buffer_names)
    pairs = model._ema_parameter_pairs()
    assert len(pairs) == len(model._ema_fp32_buffer_names)


def test_stage2_dynamics_is_optional_and_adds_real_loss(monkeypatch):
    model = make_model(monkeypatch, slots=2, stage=2, delta=True, dynamics=True).train()
    batch = examples()
    torch.manual_seed(10)
    policy = model(batch, run_policy_dynamics=False)
    torch.manual_seed(10)
    combined = model(batch, run_policy_dynamics=True)
    assert "fdm_loss" not in policy
    torch.testing.assert_close(combined["action_loss"],
                               policy["action_loss"] + 0.1 * combined["fdm_loss"])
    combined["action_loss"].backward()
    assert model.action_model.state_decoder.layer2.weight.grad.abs().sum() > 0
    assert model.control_encoder.queries.grad.abs().sum() > 0
    without_future = [{k: v for k, v in x.items() if k != "future_image"} for x in batch]
    model(without_future, run_policy_dynamics=False)
    with pytest.raises(ValueError, match="future_image"):
        model(without_future, run_policy_dynamics=True)


def test_fdm_eval_uses_training_prediction_and_is_action_sensitive(monkeypatch):
    model = make_model(monkeypatch, enabled=False).eval()
    trunk = model.action_model
    current, future, actions = torch.randn(3, 8, 12), torch.randn(3, 8, 12), torch.randn(3, 2, 3)
    train_loss = trunk.forward_fdm(current, actions, future)
    eval_loss = trunk.eval_future_latent(current, future, actions=actions)
    torch.testing.assert_close(train_loss, eval_loss, rtol=0, atol=0)
    assert not torch.equal(trunk.predict_future(current, actions),
                           trunk.predict_future(current, actions.roll(1, 0)))
    with pytest.raises(ValueError, match="action"):
        trunk.eval_future_latent(current, future)
    assert np.isfinite(model.eval_future_latent(examples()))


def test_checkpoint_roundtrip_and_dense_warm_start_are_explicit(monkeypatch, tmp_path):
    baseline = make_model(monkeypatch, enabled=False)
    dense_path = tmp_path / "dense.pt"
    torch.save(baseline.state_dict(), dense_path)
    target = make_model(monkeypatch, slots=2, delta=True)
    query_before = target.action_model.future_mask_tokens.weight.detach().clone()
    load_control_warm_start(target, dense_path, logging.getLogger(__name__))
    torch.testing.assert_close(target.action_model.trunk.blocks[0].to_qkv_state.weight,
                               baseline.action_model.trunk.blocks[0].to_qkv_state.weight)
    torch.testing.assert_close(target.action_model.future_mask_tokens.weight, query_before)
    torch.testing.assert_close(target.control_encoder.control_proj.weight,
                               target.ema_control_encoder.control_proj.weight)
    checkpoint = tmp_path / "control.pt"
    torch.save(target.state_dict(), checkpoint)
    restored = make_model(monkeypatch, slots=2, delta=True)
    load_weights(restored, checkpoint)
    torch.manual_seed(20)
    batch = examples()
    torch.manual_seed(30)
    before = target.eval().predict_action(batch)["normalized_actions"]
    torch.manual_seed(30)
    after = restored.eval().predict_action(batch)["normalized_actions"]
    np.testing.assert_array_equal(before, after)
    with pytest.raises(ValueError, match="strict"):
        load_control_warm_start(restored, checkpoint, logging.getLogger(__name__))


def test_disabled_control_keeps_original_checkpoint_keys(monkeypatch):
    model = make_model(monkeypatch, enabled=False)
    assert not any("control_encoder" in key or "delta_decoder" in key for key in model.state_dict())
    assert model.action_model.num_future_tokens == 8


def test_experiment_configs_roundtrip_and_stage_pairs(monkeypatch):
    for key in ("LIBERO_DATA_ROOT", "DINOV2_MODEL_DIR", "DINOV3_VITS16_MODEL_DIR", "T5_MODEL_DIR"):
        monkeypatch.setenv(key, "/tmp/not-used")
    root = Path(__file__).parents[1] / "configs/libero/control_latent"
    for path in root.glob("*.yaml"):
        cfg = load_config(path)
        public = to_public_config(to_runtime_config(cfg))
        if "frozen_baseline" in path.name:
            continue
        assert public.model.control_latent.enabled
        assert public.model.vision_encoder.freeze_backbone
        slots = cfg.model.control_latent.tokens_per_view
        assert cfg.model.transformer.num_future_tokens == 2 * (slots or 196)
        assert public.model.control_latent == cfg.model.control_latent
        if path.name.startswith("stage2"):
            assert cfg.training.stage == 2 and cfg.training.objective == "policy"
            assert cfg.data.sources[0].data_mix == "libero_all"
            if "dynamics" in path.name:
                assert public.model.policy_dynamics.enabled and cfg.model.ema.enabled


def test_auxiliary_schedule_uses_optimizer_steps():
    from slim.training.stage2 import policy_dynamics_step, _require_stage2_slim_config
    cfg = tiny_config(stage=2, dynamics=True)
    _require_stage2_slim_config(cfg)
    assert [policy_dynamics_step(cfg, n) for n in range(9)] == [
        True, False, False, False, True, False, False, False, True,
    ]
    cfg.model.policy_dynamics.every_n_steps = 0
    with pytest.raises(ValueError, match="positive"):
        _require_stage2_slim_config(cfg)


@pytest.mark.parametrize("dynamics", [False, True])
def test_strict_stage1_to_stage2_loading_includes_representation(monkeypatch, tmp_path, dynamics):
    from slim.training import stage1, stage2
    monkeypatch.setattr(stage1, "logger", logging.getLogger(__name__))
    monkeypatch.setattr(stage2, "logger", logging.getLogger(__name__))
    first = make_model(monkeypatch, slots=2, delta=True)
    with torch.no_grad():
        first.control_encoder.control_proj.weight.add_(0.125)
    first.update_ema()
    checkpoint = tmp_path / "stage1.pt"
    torch.save(first.state_dict(), checkpoint)
    second = make_model(monkeypatch, slots=2, delta=True, stage=2, dynamics=dynamics)
    skipped = () if dynamics else ("ema_vision_encoder.", "ema_control_encoder.", "_ema_fp32_")
    stage2._load_initial_weights(second, str(checkpoint), skip_prefixes=skipped)
    torch.testing.assert_close(first.control_encoder.control_proj.weight,
                               second.control_encoder.control_proj.weight)
    if dynamics:
        torch.testing.assert_close(first.ema_control_encoder.control_proj.weight,
                                   second.ema_control_encoder.control_proj.weight)
    with pytest.raises(RuntimeError, match="Strict"):
        incompatible = make_model(monkeypatch, slots=0, delta=True)
        stage1._load_initial_weights(incompatible, str(checkpoint))


def test_dynamics_audit_reads_saved_per_source_statistics(tmp_path):
    from scripts.evaluate_latent_dynamics import load_source_stats
    sources = [{"root_dir": "/unused", "data_mix": "libero_all_90"},
               {"root_dir": "/unused", "name": "second"}]
    for key in ("libero_all_90", "second"):
        (tmp_path / f"action_stats_{key}.json").write_text(json.dumps({
            "q01": [-1, -1, -1], "q99": [1, 1, 1], "action_dim": 3,
        }))
    stats = load_source_stats(tmp_path, sources)
    assert set(stats) == {"libero_all_90", "second"}
    assert stats["second"]["q01"] == [-1, -1, -1]


def test_dynamics_diagnostics_do_not_update_weights(monkeypatch):
    model = make_model(monkeypatch, slots=2, delta=True).eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    metrics = model.evaluate_dynamics(examples())
    assert set(metrics) == {"fdm_future_loss", "identity_loss", "shuffled_action_loss",
                            "action_sensitivity_gap", "delta_loss", "zero_delta_loss"}
    assert all(np.isfinite(value) for value in metrics.values())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(before[key], value, rtol=0, atol=0)


def test_mixed_precision_optimizer_step_updates_new_parameters(monkeypatch):
    model = make_model(monkeypatch, slots=2, stage=2, delta=True, dynamics=True).train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    before = model.control_encoder.queries.detach().clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(examples(), run_policy_dynamics=True)
    assert torch.isfinite(output["action_loss"])
    output["action_loss"].backward()
    optimizer.step()
    model.update_ema()
    assert not torch.equal(before, model.control_encoder.queries)
    assert all(torch.isfinite(p).all() for p in model.parameters())
