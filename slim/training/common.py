from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler

from slim.data import LeRobotBaseDataset, collate_fn


TRAINER_STATE_FILE = "trainer_state.json"


def pin_run_to_resume_state(cfg, resume_state: str | Path | None):
    """Keep explicit full-state resumes in the original run directory."""
    if not resume_state:
        return cfg
    state_dir = Path(resume_state).expanduser().resolve()
    if state_dir.parent.name != "states":
        raise ValueError(
            "resume_state must use the layout <run_dir>/states/step_XXXXXXXX"
        )
    run_dir = state_dir.parent.parent
    cfg.run.root = str(run_dir.parent)
    cfg.run.name = run_dir.name
    cfg.run.timestamp = False
    return cfg


def maybe_append_timestamp_to_run_id(cfg):
    if bool(cfg.run.get("timestamp", False)):
        cfg.run.name = f"{cfg.run.name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    return cfg


def sync_run_id_across_ranks(cfg):
    if not bool(cfg.run.get("timestamp", False)):
        return cfg
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return maybe_append_timestamp_to_run_id(cfg)
    rank = int(os.environ.get("RANK", "0"))
    holder = [None]
    if rank == 0:
        holder[0] = f"{cfg.run.name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    dist.broadcast_object_list(holder, src=0)
    cfg.run.name = str(holder[0])
    return cfg


def diagnose_and_filter_param_groups(model, param_groups, logger):
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[diag] total_trainable_params={total_trainable}")
    normalized = []
    for idx, group in enumerate(param_groups):
        name = group.get("name", f"group_{idx}")
        params = list(group.get("params", []))
        trainable = [p for p in params if p.requires_grad]
        logger.info(
            f"[diag] param_group[{idx}] name={name} "
            f"params_all={sum(p.numel() for p in params)} params_trainable={sum(p.numel() for p in trainable)}"
        )
        if trainable:
            new_group = dict(group)
            new_group["params"] = trainable
            normalized.append(new_group)
    if total_trainable == 0 or not normalized:
        raise RuntimeError("[diag] no trainable parameters after filtering")
    return normalized


def _state_step(state_dir: Path) -> int:
    match = re.fullmatch(r"step_(\d+)", state_dir.name)
    if not match:
        return -1
    return int(match.group(1))


def resolve_resume_state(
    run_dir: str | Path,
    resume_state: str | Path | None = None,
    auto_resume: bool = False,
    logger=None,
) -> Path | None:
    """Resolve an explicit or latest accelerator state directory."""
    if resume_state:
        state_dir = Path(resume_state)
        if not state_dir.exists():
            raise FileNotFoundError(f"resume_state not found: {state_dir}")
        if not state_dir.is_dir():
            raise ValueError(f"resume_state must be a directory: {state_dir}")
        return state_dir
    if not auto_resume:
        return None

    states_root = Path(run_dir) / "states"
    if not states_root.exists():
        if logger is not None:
            logger.info(f"[resume_state] no states directory found: {states_root}")
        return None
    candidates = [p for p in states_root.iterdir() if p.is_dir() and _state_step(p) >= 0]
    if not candidates:
        if logger is not None:
            logger.info(f"[resume_state] no step_* state directories found under {states_root}")
        return None
    return max(candidates, key=_state_step)


def read_trainer_state(state_dir: str | Path) -> dict:
    metadata_path = Path(state_dir) / TRAINER_STATE_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing trainer metadata: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_training_state(
    accelerator,
    run_dir: str | Path,
    completed_steps: int,
    steps_per_epoch: int,
    max_train_steps: int,
    logger=None,
) -> Path:
    """Save full Accelerator state plus trainer metadata.

    All ranks must enter this function because DeepSpeed/Accelerate may write
    per-rank optimizer/model shards.
    """
    state_dir = Path(run_dir) / "states" / f"step_{int(completed_steps):08d}"
    accelerator.save_state(str(state_dir))
    if accelerator.is_main_process:
        payload = {
            "completed_steps": int(completed_steps),
            "steps_per_epoch": int(steps_per_epoch),
            "max_train_steps": int(max_train_steps),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        metadata_path = state_dir / TRAINER_STATE_FILE
        tmp_path = metadata_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp_path.replace(metadata_path)
        latest_path = Path(run_dir) / "states" / "latest_state.txt"
        latest_tmp = latest_path.with_suffix(".tmp")
        latest_tmp.write_text(str(state_dir), encoding="utf-8")
        latest_tmp.replace(latest_path)
        if logger is not None:
            logger.info(f"[resume_state] saved full training state: {state_dir}")
    accelerator.wait_for_everyone()
    return state_dir


def build_resume_train_iterator(
    accelerator,
    train_loader,
    completed_steps: int,
    steps_per_epoch: int,
    logger=None,
):
    """Create a train iterator positioned at the saved step within the epoch."""
    if completed_steps <= 0 or steps_per_epoch <= 0:
        return train_loader, iter(train_loader)
    grad_accum = max(int(getattr(accelerator, "gradient_accumulation_steps", 1)), 1)
    batches_to_skip = (int(completed_steps) % int(steps_per_epoch)) * grad_accum
    if batches_to_skip <= 0:
        return train_loader, iter(train_loader)
    if logger is not None:
        logger.info(
            f"[resume_state] skipping {batches_to_skip} train batches "
            f"within current epoch after completed_steps={completed_steps}"
        )
    skipped_loader = accelerator.skip_first_batches(train_loader, batches_to_skip)
    return skipped_loader, iter(skipped_loader)


def build_dataloaders(cfg, include_future_image: bool, logger):
    stats_dir = Path(cfg.data.get("action_stats_dir", Path(cfg.run.root) / cfg.run.name))
    base_kwargs = dict(
        action_horizon=cfg.model.action_horizon,
        val_ratio=cfg.data.val_ratio,
        eval_monitor_episode_ratio=float(cfg.data.get("eval_monitor_episode_ratio", 0.05)),
        split_seed=cfg.data.split_seed,
        video_backend=cfg.data.get("video_backend", "torchvision_av"),
        action_normalization=cfg.data.get("action_normalization", "q01_q99"),
        action_dim=cfg.model.get("action_dim", 7),
        state_dim=cfg.model.get("state_dim", cfg.model.get("action_dim", 7)),
        action_normalized_dims=cfg.data.get("action_normalized_dims", None),
        action_indices=cfg.data.get("action_indices", None),
        state_indices=cfg.data.get("state_indices", None),
        action_key=cfg.data.get("action_key", "action"),
        state_key=cfg.data.get("state_key", "observation.state"),
        action_stats_dir=stats_dir,
        skip_lerobot_episodes=cfg.data.get("skip_lerobot_episodes", None),
        include_future_image=include_future_image,
        use_fake_data=cfg.data.get("use_fake_data", False),
        episode_shuffle=bool(cfg.data.get("episode_shuffle", True)),
    )
    data_sources = OmegaConf.to_container(cfg.data.sources, resolve=True)
    is_dist = dist.is_initialized() and dist.get_world_size() > 1
    rank = dist.get_rank() if is_dist else 0
    if rank == 0:
        train_ds = LeRobotBaseDataset(split="train", data_sources=data_sources, **base_kwargs)
    else:
        train_ds = None
    if is_dist:
        dist.barrier()
    if rank != 0:
        train_ds = LeRobotBaseDataset(split="train", data_sources=data_sources, **base_kwargs)
    val_ds = LeRobotBaseDataset(split="val", data_sources=data_sources, action_stats=train_ds.get_action_stats(), **base_kwargs)

    train_num_workers = cfg.data.get("num_workers", 4)
    eval_num_workers = cfg.data.get("eval_num_workers", train_num_workers)
    prefetch_factor = cfg.data.get("prefetch_factor", 1)
    persistent_workers = cfg.data.get("persistent_workers", False)
    timeout = cfg.data.get("dataloader_timeout_s", 300)

    train_kwargs = dict(batch_size=cfg.data.per_device_batch_size, num_workers=train_num_workers, collate_fn=collate_fn, timeout=timeout)
    val_kwargs = dict(batch_size=cfg.data.get("eval_batch_size", cfg.data.per_device_batch_size), num_workers=eval_num_workers, collate_fn=collate_fn, timeout=timeout)
    if train_num_workers > 0:
        train_kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
    if eval_num_workers > 0:
        val_kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

    sampler = None
    if train_ds.has_nonuniform_sample_weights:
        weights = train_ds.sample_weights
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_ds), replacement=True)
        logger.info("[dataloader] Using WeightedRandomSampler.")
    return DataLoader(train_ds, sampler=sampler, **train_kwargs), DataLoader(val_ds, **val_kwargs), train_ds


def _mse_slice(pred: np.ndarray, gt: np.ndarray, start: int, end: int) -> float:
    return float(((pred[:, :, start:end] - gt[:, :, start:end]) ** 2).mean())


def default_eval_action_slices(action_dim: int) -> dict[str, tuple[int, int]]:
    """Per-dimension MSE groups keyed as mse_<name> (without prefix)."""
    if action_dim == 7:
        return {
            "xyz": (0, 3),
            "rot": (3, 6),
            "gripper": (6, 7),
        }
    if action_dim == 16:
        # Dual-arm RobotWin EEF action: xyz(3) + rotation(4) + gripper(1) per arm.
        return {
            "arm0_xyz": (0, 3),
            "arm0_rot": (3, 7),
            "arm0_grip": (7, 8),
            "arm1_xyz": (8, 11),
            "arm1_rot": (11, 15),
            "arm1_grip": (15, 16),
        }
    if action_dim > 7 and action_dim % 7 == 0:
        slices: dict[str, tuple[int, int]] = {}
        for arm_idx in range(action_dim // 7):
            base = arm_idx * 7
            prefix = f"arm{arm_idx}"
            slices[f"{prefix}_xyz"] = (base, base + 3)
            slices[f"{prefix}_rot"] = (base + 3, base + 6)
            slices[f"{prefix}_gripper"] = (base + 6, base + 7)
        return slices
    return {}


def resolve_eval_action_slices(cfg, action_dim: int) -> dict[str, tuple[int, int]]:
    if cfg is not None:
        custom = cfg.model.get("eval_action_slices")
        if custom is not None:
            raw = OmegaConf.to_container(custom, resolve=True)
            return {str(name): (int(bounds[0]), int(bounds[1])) for name, bounds in raw.items()}
    return default_eval_action_slices(action_dim)


def format_eval_metrics_log(metrics: dict) -> str:
    parts = [f"total={metrics['mse_total']:.6f}"]
    for key in sorted(metrics):
        if not key.startswith("mse_") or key == "mse_total":
            continue
        parts.append(f"{key[4:]}={metrics[key]:.6f}")
    return " ".join(parts)


def compute_eval_metrics(model, batch, cfg=None):
    gt = np.array([x["action"] for x in batch], dtype=np.float32)
    pred = model.predict_action(batch)["normalized_actions"].astype(np.float32)
    action_dim = int(gt.shape[-1])
    if cfg is not None:
        cfg_dim = cfg.model.get("action_dim")
        if cfg_dim is not None:
            action_dim = int(cfg_dim)
    metrics = {"mse_total": float(((pred - gt) ** 2).mean())}
    for name, (start, end) in resolve_eval_action_slices(cfg, action_dim).items():
        if end > action_dim:
            continue
        metrics[f"mse_{name}"] = _mse_slice(pred, gt, start, end)
    metrics["pred_action_first"] = pred[0, 0].tolist()
    metrics["gt_action_first"] = gt[0, 0].tolist()
    return metrics


def scalar_or_none(x):
    return None if x is None else float(x.detach().cpu().item())


def steps_from_epochs(cfg, train_ds, accelerator):
    world_size = max(int(getattr(accelerator, "num_processes", 1)), 1)
    grad_accum = max(
        int(getattr(accelerator, "gradient_accumulation_steps", 1)),
        1,
    )
    global_batch = max(int(cfg.data.per_device_batch_size) * world_size * grad_accum, 1)
    steps_per_epoch = max(int(math.ceil(len(train_ds) / global_batch)), 1)
    max_epochs = cfg.training.get("max_epochs", None)
    max_train_steps = max(int(max_epochs * steps_per_epoch), 1) if max_epochs is not None and max_epochs > 0 else int(cfg.training.max_train_steps)
    return steps_per_epoch, max_train_steps
