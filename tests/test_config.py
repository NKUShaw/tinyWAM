import os
from pathlib import Path

from slim.compat.legacy import to_public_config, to_runtime_config
from slim.config import load_config
from slim.training.common import pin_run_to_resume_state


ROOT = Path(__file__).parents[1]
os.environ.setdefault("LIBERO_DATA_ROOT", "/tmp/libero")
os.environ.setdefault("DINOV2_MODEL_DIR", "/tmp/dinov2")
os.environ.setdefault("DINOV3_VITS16_MODEL_DIR", "/tmp/dinov3-vits16")
os.environ.setdefault("T5_MODEL_DIR", "/tmp/t5")


def test_public_config_round_trip():
    public = load_config(ROOT / "configs/libero/stage1_idm0125_fdm1_h8.yaml")
    runtime = to_runtime_config(public)
    assert runtime.datasets.vla_data.data_sources[0].data_mix == "libero_all_90"
    restored = to_public_config(runtime)
    assert restored.model.name == "SLIM"
    assert restored.training.objective == "idm_fdm"
    assert restored.model.action_horizon == 8
    assert restored.data.sources[0].data_mix == "libero_all_90"


def test_ablation_inheritance():
    config = load_config(
        ROOT / "configs/libero/ablations/stage1_fdm_only_h8.yaml"
    )
    assert config.training.objective == "fdm"
    assert config.model.transformer.idm_loss_weight == 0.0
    assert config.model.transformer.fdm_loss_weight == 1.0


def test_dinov3_tiny_configs_keep_architecture_hyperparameters_explicit():
    stage1 = load_config(ROOT / "configs/libero/stage1_dinov3_vits16_d384_h6.yaml")
    stage2 = load_config(ROOT / "configs/libero/stage2_dinov3_vits16_d384_h6_40ep.yaml")

    for config in (stage1, stage2):
        assert config.model.vision_encoder.backbone_name == "dinov3_vits16_lvd1689m"
        assert config.model.transformer.hidden_dim == 384
        assert config.model.transformer.num_heads == 6
        assert config.model.transformer.num_layers == 16
        assert config.model.transformer.ffn_ratio == 4.0
        assert config.model.transformer.num_future_tokens == 392
        assert config.model.transformer.max_state_tokens >= 1 + 2 * 392


def test_canonical_stage_specific_data_and_diffusion_settings():
    stage1 = load_config(ROOT / "configs/libero/stage1_idm0125_fdm1_h8.yaml")
    stage1_frames = load_config(
        ROOT / "configs/libero/stage1_idm0125_fdm1_h8_frames.yaml"
    )
    stage2 = load_config(ROOT / "configs/libero/stage2_policy_h8_40ep.yaml")
    stage2_frames = load_config(
        ROOT / "configs/libero/stage2_policy_h8_40ep_frames.yaml"
    )
    stage2_video = load_config(
        ROOT / "configs/libero/stage2_policy_h8_40ep_video.yaml"
    )
    stage2_with_ema = load_config(
        ROOT / "configs/libero/ablations/stage2_with_ema_h8_40ep.yaml"
    )

    assert stage1.model.vision_encoder.vision_condition_mode == "dense_patch"
    assert stage2.model.vision_encoder.vision_condition_mode == "dense_patch"
    assert stage1.model.repeated_diffusion_steps == 1
    assert stage2.model.repeated_diffusion_steps == 4
    assert stage1.model.transformer.idm_loss_weight == 0.125
    assert stage1.data.video_backend == "torchvision_av"
    assert stage1_frames.data.video_backend == "frames"
    assert stage2.data.video_backend == "torchvision_av"
    assert stage2_frames.data.video_backend == "frames"
    assert stage2_video.data.video_backend == "torchvision_av"
    assert stage1.model.ema.enabled is True
    assert stage2.model.ema.enabled is False
    assert list(stage2.resume_skip_prefixes) == [
        "ema_vision_encoder.",
        "_ema_fp32_",
    ]
    assert stage2_with_ema.model.ema.enabled is True


def test_explicit_resume_reuses_original_run(tmp_path):
    config = load_config(ROOT / "configs/libero/stage2_policy_h8_40ep.yaml")
    state = tmp_path / "run_001" / "states" / "step_00001000"
    state.mkdir(parents=True)
    pin_run_to_resume_state(config, state)
    assert config.run.root == str(tmp_path)
    assert config.run.name == "run_001"
    assert config.run.timestamp is False
