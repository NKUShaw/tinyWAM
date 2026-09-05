from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SLIMDefaultConfig:
    name: str = "SLIM"
    # Shuffle episodes (not frames) inside the dataset to keep video-container
    # cache warm. DataLoader-level shuffle is always disabled.
    episode_shuffle: bool = True
    dino: dict = field(
        default_factory=lambda: {
            "backbone_name": "dinov2_vitb14",
            "model_path": None,
            "freeze_backbone": True,
            "num_image_views": 2,
            "vision_condition_mode": "dense_patch",
            "wrist_detail": {
                "enabled": False,
                "high_resolution": 256,
                "bottleneck_dim": 96,
                "freeze_except_adapter": False,
            },
            "image_augment": {
                "enabled": False,
                "brightness": 0.3,
                "contrast": 0.3,
                "saturation": 0.2,
                "hue": 0.05,
                "crop_scale_min": 0.9,
                "crop_scale_max": 1.0,
            },
        }
    )
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "use_future_image_condition": False,
            "use_language_condition": False,
            "max_lang_tokens": 32,
            "language_embedding_dim": 512,
            "offline_lang_emb_root": None,
            "language_encoder_path": None,
            "hidden_size": 1024,
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 8,
            "repeated_diffusion_steps": 1,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "diffusion_model_cfg": {
                "cross_attention_dim": 768,
                "output_dim": 1024,
            },
        }
    )
