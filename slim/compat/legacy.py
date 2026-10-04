"""Translation for checkpoints produced by the original research repository."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _container(config: Any) -> dict:
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)
    return deepcopy(config)


def is_legacy_config(config: Any) -> bool:
    data = _container(config)
    return "framework" in data or "datasets" in data or "trainer" in data


def to_runtime_config(config: Any) -> DictConfig:
    """Return the stable internal schema used by existing checkpoints.

    Public configurations use ``model``, ``data`` and ``training``. The
    runtime schema intentionally preserves historical module layout so model
    parameter names remain byte-for-byte compatible.
    """

    data = _container(config)
    if is_legacy_config(data):
        return OmegaConf.create(data)

    model = deepcopy(data["model"])
    dataset = deepcopy(data["data"])
    training = deepcopy(data["training"])
    if "sources" in dataset:
        dataset["data_sources"] = dataset.pop("sources")

    transformer = model.pop("transformer")
    vision = model.pop("vision_encoder")
    ema = model.pop("ema", {"enabled": True, "momentum": 0.999})
    control_latent = model.pop("control_latent", {"enabled": False})
    policy_dynamics = model.pop("policy_dynamics", {"enabled": False})

    action_model = {
        "use_future_image_condition": bool(training["stage"] == 1),
        "use_language_condition": model.pop("use_language_condition", True),
        "language_encoder_path": model.pop("language_encoder_path", None),
        "offline_lang_emb_root": model.pop("language_cache", None),
        "max_lang_tokens": model.pop("max_language_tokens", 32),
        "language_embedding_dim": model.pop("language_embedding_dim", 512),
        "action_dim": model.pop("action_dim"),
        "state_dim": model.pop("state_dim"),
        "action_horizon": model.pop("action_horizon"),
        "action_model_type": "DiT-B",
        "repeated_diffusion_steps": model.pop("repeated_diffusion_steps", 1),
        "num_inference_timesteps": model.pop("inference_steps", 4),
        "mot": {
            "task_mode": training.pop("objective"),
            **transformer,
        },
        "diffusion_model_cfg": {
            "cross_attention_dim": transformer["hidden_dim"],
            "output_dim": model.pop("decoder_hidden_dim", 1024),
        },
    }
    if action_model["language_encoder_path"] is None:
        action_model.pop("language_encoder_path")
    if action_model["offline_lang_emb_root"] is None:
        action_model.pop("offline_lang_emb_root")

    runtime = {
        "run_id": data.get("run", {}).get("name", "slim"),
        "run_id_auto_timestamp": data.get("run", {}).get("timestamp", True),
        "run_root_dir": data.get("run", {}).get("root", "checkpoints"),
        "seed": data.get("seed", 42),
        "trackers": data.get("logging", {}).get("trackers", ["jsonl"]),
        "wandb_entity": data.get("logging", {}).get("entity"),
        "wandb_project": data.get("logging", {}).get("project", "slim"),
        "is_debug": data.get("debug", False),
        "framework": {
            "name": "SLIM",
            "dino": vision,
            "ema": ema,
            "control_latent": control_latent,
            "policy_dynamics": policy_dynamics,
            "action_model": action_model,
        },
        "datasets": {"vla_data": dataset},
        "trainer": training,
    }
    runtime["trainer"].pop("stage", None)
    return OmegaConf.create(runtime)


def to_public_config(config: Any) -> DictConfig:
    """Convert an old run config into the public schema."""

    data = _container(config)
    if not is_legacy_config(data):
        return OmegaConf.create(data)

    fw = data["framework"]
    action = fw["action_model"]
    transformer = deepcopy(action["mot"])
    objective = transformer.pop("task_mode")
    public_data = deepcopy(data["datasets"]["vla_data"])
    if "data_sources" in public_data:
        public_data["sources"] = public_data.pop("data_sources")

    public = {
        "seed": data.get("seed", 42),
        "run": {
            "name": data.get("run_id", "slim"),
            "root": data.get("run_root_dir", "checkpoints"),
            "timestamp": data.get("run_id_auto_timestamp", True),
        },
        "model": {
            "name": "SLIM",
            "vision_encoder": deepcopy(fw["dino"]),
            "ema": deepcopy(fw.get("ema", {})),
            "control_latent": deepcopy(fw.get("control_latent", {"enabled": False})),
            "policy_dynamics": deepcopy(fw.get("policy_dynamics", {"enabled": False})),
            "action_dim": action["action_dim"],
            "state_dim": action["state_dim"],
            "action_horizon": action["action_horizon"],
            "use_language_condition": action.get("use_language_condition", True),
            "max_language_tokens": action.get("max_lang_tokens", 32),
            "language_embedding_dim": action.get("language_embedding_dim", 512),
            "language_encoder_path": action.get("language_encoder_path"),
            "language_cache": action.get("offline_lang_emb_root"),
            "inference_steps": action.get("num_inference_timesteps", 4),
            "repeated_diffusion_steps": action.get("repeated_diffusion_steps", 1),
            "decoder_hidden_dim": action.get("diffusion_model_cfg", {}).get("output_dim", 1024),
            "transformer": transformer,
        },
        "data": public_data,
        "training": {
            **deepcopy(data["trainer"]),
            "stage": 1 if objective != "policy" else 2,
            "objective": objective,
        },
        "logging": {
            "trackers": data.get("trackers", ["jsonl"]),
            "project": data.get("wandb_project", "slim"),
            "entity": data.get("wandb_entity"),
        },
    }
    return OmegaConf.create(public)
