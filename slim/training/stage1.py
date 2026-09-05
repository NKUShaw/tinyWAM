"""Stage 1: Action-Grounded Masked Trajectory Prediction."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import GradientAccumulationPlugin, set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import get_scheduler

from slim.training.metrics import build_monitor_collapse_payload
from slim.training.common import (
    build_resume_train_iterator,
    build_dataloaders,
    compute_eval_metrics,
    diagnose_and_filter_param_groups,
    format_eval_metrics_log,
    pin_run_to_resume_state,
    read_trainer_state,
    resolve_resume_state,
    save_training_state,
    scalar_or_none,
    steps_from_epochs,
    sync_run_id_across_ranks,
)


from slim.model import SLIMModel
from slim.config import load_config
from slim.training.optimizer import build_parameter_groups, normalize_overrides

logger = get_logger(__name__)


def _build_accelerator(cfg) -> Accelerator:
    grad_accum_steps = int(cfg.training.get("gradient_accumulation_steps", 1))
    if grad_accum_steps <= 0:
        raise ValueError("training.gradient_accumulation_steps must be positive")
    return Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=grad_accum_steps,
            sync_each_batch=True,
        ),
        deepspeed_plugin=DeepSpeedPlugin(),
    )


def _sync_ema_vision_from_online(model) -> None:
    """Initialize EMA vision from online weights when the checkpoint lacks EMA."""
    if not getattr(model, "ema_enabled", False):
        return
    if getattr(model, "ema_vision_encoder", None) is None:
        return
    if hasattr(model, "sync_ema_from_online"):
        model.sync_ema_from_online()
        return
    model.ema_vision_encoder.load_state_dict(model.vision_encoder.state_dict())
    if hasattr(model, "refresh_ema_fp32_shadow"):
        model.refresh_ema_fp32_shadow()
    logger.info("[init_checkpoint] initialized EMA vision from online vision weights")


def _load_initial_weights(
    model,
    ckpt_path: str,
    init_mode: str = "slim_policy",
    skip_prefixes: tuple[str, ...] = (),
):
    if init_mode == "control_warm_start":
        from slim.model.initialization import load_control_warm_start
        return load_control_warm_start(model, ckpt_path, logger, skip_prefixes)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model_sd = model.state_dict()
    if init_mode == "vision_only":
        prefixes = ("vision_encoder.",)
    elif init_mode == "slim_policy":
        prefixes = None
    else:
        raise ValueError(f"Unsupported SLIM init_mode={init_mode}")

    filtered = {}
    dropped_keys = []
    loaded_ema_param_keys = 0
    for key, value in state_dict.items():
        if skip_prefixes and key.startswith(skip_prefixes):
            dropped_keys.append(key)
            continue
        if prefixes is not None and not key.startswith(prefixes):
            dropped_keys.append(key)
            continue
        if key not in model_sd:
            dropped_keys.append(key)
            continue
        if hasattr(value, "shape") and value.shape != model_sd[key].shape:
            logger.warning(
                f"[init_checkpoint] skip shape mismatch {key}: ckpt={tuple(value.shape)} "
                f"model={tuple(model_sd[key].shape)}"
            )
            dropped_keys.append(key)
            continue
        filtered[key] = value
        if key.startswith("ema_vision_encoder."):
            loaded_ema_param_keys += 1
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if unexpected:
        raise RuntimeError(f"[init_checkpoint] unexpected keys: {unexpected}")
    required_missing = [
        key
        for key in missing
        if not key.endswith(".mask_token")
        and not (skip_prefixes and key.startswith(skip_prefixes))
    ]
    unexpected_dropped = [
        key
        for key in dropped_keys
        if not (skip_prefixes and key.startswith(skip_prefixes))
    ]
    if init_mode == "slim_policy":
        if unexpected_dropped or required_missing:
            raise RuntimeError(
                "Strict Stage initialization failed: "
                f"dropped={unexpected_dropped}, missing={required_missing}"
            )
    logger.info(
        f"[init_checkpoint] loaded={len(filtered)} dropped={len(dropped_keys)} "
        f"missing={len(missing)} mode={init_mode}"
    )
    if skip_prefixes:
        logger.info(f"[init_checkpoint] skipped prefixes: {list(skip_prefixes)}")
    if getattr(model, "ema_enabled", False):
        if loaded_ema_param_keys > 0:
            if hasattr(model, "refresh_ema_fp32_shadow"):
                model.refresh_ema_fp32_shadow()
            logger.info(
                f"[init_checkpoint] kept EMA vision weights from checkpoint keys={loaded_ema_param_keys} "
                "and refreshed fp32 shadow"
            )
        else:
            _sync_ema_vision_from_online(model)


def main(
    cfg,
    init_checkpoint: str | None = None,
    resume_state: str | None = None,
    auto_resume: bool = False,
):
    cfg = pin_run_to_resume_state(cfg, resume_state)
    accelerator = _build_accelerator(cfg)
    cfg = sync_run_id_across_ranks(cfg)
    base_seed = int(cfg.get("seed", 42))
    set_seed(base_seed, device_specific=True)
    logger.info(
        f"[seed] base_seed={base_seed} process_index={accelerator.process_index} "
        f"effective_seed={base_seed + accelerator.process_index} "
        f"split_seed={int(cfg.data.get('split_seed', 42))}"
    )

    run_dir = Path(cfg.run.root) / cfg.run.name
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    resume_state_dir = resolve_resume_state(
        run_dir, resume_state=resume_state, auto_resume=auto_resume, logger=logger
    )
    if init_checkpoint and resume_state_dir is not None:
        raise ValueError(
            "`--init-checkpoint` warm-starts model weights only. Do not combine it with "
            "`--resume_state`/`--auto_resume`, which restores model, optimizer, scheduler, "
            "and dataloader position."
        )

    train_loader, val_loader, train_ds = build_dataloaders(cfg, include_future_image=True, logger=logger)
    if accelerator.is_main_process:
        OmegaConf.save(cfg, run_dir / "config.yaml")
        train_ds.save_action_stats()
        train_ds.save_action_stats(run_dir)

    model = SLIMModel(cfg)
    print("mixed_precision =", accelerator.state.mixed_precision)
    print("gradient_accumulation_steps =", accelerator.gradient_accumulation_steps)
    print("ds bf16 =", accelerator.state.deepspeed_plugin.deepspeed_config.get("bf16", {}))
    print("param dtype =", next(model.parameters()).dtype)

    if init_checkpoint:
        skip_prefixes = tuple(str(x) for x in cfg.get("resume_skip_prefixes", []) or [])
        _load_initial_weights(
            model,
            init_checkpoint,
            init_mode=str(cfg.get("init_mode", "slim_policy")),
            skip_prefixes=skip_prefixes,
        )

    ema_enabled = bool(cfg.model.get("ema", {}).get("enabled", False))
    logger.info(f"[stage1] EMA vision module: {ema_enabled}")

    param_groups = diagnose_and_filter_param_groups(model, build_parameter_groups(model, cfg), logger)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.training.learning_rate.base,
        betas=tuple(cfg.training.optimizer.betas),
        weight_decay=cfg.training.optimizer.weight_decay,
        eps=cfg.training.optimizer.eps,
    )
    steps_per_epoch, max_train_steps = steps_from_epochs(cfg, train_ds, accelerator)
    logger.info(f"[train] steps_per_epoch={steps_per_epoch} max_train_steps={max_train_steps}")
    scheduler = get_scheduler(
        name=cfg.training.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.num_warmup_steps,
        num_training_steps=max_train_steps,
        scheduler_specific_kwargs=cfg.training.scheduler_specific_kwargs,
    )
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    completed_steps = 0
    if resume_state_dir is not None:
        accelerator.load_state(str(resume_state_dir))
        trainer_state = read_trainer_state(resume_state_dir)
        completed_steps = int(trainer_state["completed_steps"])
        if completed_steps >= max_train_steps:
            logger.warning(
                f"[resume_state] completed_steps={completed_steps} >= "
                f"max_train_steps={max_train_steps}; training loop will exit."
            )
        logger.info(f"[resume_state] loaded full training state from {resume_state_dir}")
        if completed_steps > 0:
            scheduler.step(completed_steps)
            logger.info(
                f"[resume_state] restored scheduler to step={completed_steps} "
                f"lrs={scheduler.get_last_lr()}"
            )

    if accelerator.is_main_process and "wandb" in cfg.logging.trackers:
        wandb.init(
            name=cfg.run.name,
            dir=str(run_dir / "wandb"),
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            group="slim-stage1",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    model.train()
    train_loader, train_iter = build_resume_train_iterator(
        accelerator,
        train_loader,
        completed_steps=completed_steps,
        steps_per_epoch=steps_per_epoch,
        logger=logger,
    )
    pbar = tqdm(
        total=max_train_steps,
        initial=min(completed_steps, max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    while completed_steps < max_train_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        with accelerator.accumulate(model):
            loss_dict = model.forward(batch)
            total_loss = loss_dict["action_loss"]
            accelerator.backward(total_loss)
            if accelerator.sync_gradients and cfg.training.gradient_clipping is not None:
                accelerator.clip_grad_norm_(model.parameters(), cfg.training.gradient_clipping)
            optimizer.step()
            if accelerator.sync_gradients:
                scheduler.step()
            optimizer.zero_grad()

            if ema_enabled and accelerator.sync_gradients:
                accelerator.unwrap_model(model).update_ema()

        if accelerator.sync_gradients:
            completed_steps += 1
            pbar.update(1)
        else:
            continue

        if completed_steps % cfg.training.logging_frequency == 0 and accelerator.is_main_process:
            payload = {
                "train/total_loss": scalar_or_none(total_loss),
                "train/idm_loss": scalar_or_none(loss_dict.get("idm_loss")),
                "train/fdm_loss": scalar_or_none(loss_dict.get("fdm_loss")),
                "train/lr": float(scheduler.get_last_lr()[0]),
            }
            for key in ("future_loss", "delta_loss", "representation_loss", "reconstruction_loss",
                        "decorrelation_loss", "variance_loss"):
                if key in loss_dict:
                    payload[f"train/{key}"] = scalar_or_none(loss_dict[key])
            if "wandb" in cfg.logging.trackers:
                wandb.log(payload, step=completed_steps)
            logger.info(f"[train] step={completed_steps} {payload}")

        if completed_steps % cfg.training.eval_interval == 0 and completed_steps > 0:
            model.eval()
            try:
                val_batch = next(iter(val_loader))
            except StopIteration:
                model.train()
                continue
            model_unwrapped = accelerator.unwrap_model(model)
            metrics = compute_eval_metrics(model_unwrapped, val_batch, cfg=cfg)
            fdm_future_loss = None
            if all("future_image" in x for x in val_batch):
                try:
                    fdm_future_loss = model_unwrapped.eval_future_latent(val_batch)
                except Exception:
                    pass
            if accelerator.is_main_process:
                log_payload = {f"eval/{k}": v for k, v in metrics.items() if not k.endswith("_first")}
                log_payload["eval/epoch"] = completed_steps / float(steps_per_epoch)
                if fdm_future_loss is not None:
                    log_payload["eval/fdm_future_loss"] = fdm_future_loss
                with torch.no_grad():
                    log_payload.update(build_monitor_collapse_payload(loss_dict, include_svd=True))
                if "wandb" in cfg.logging.trackers:
                    wandb.log(log_payload, step=completed_steps)
                future_mse_str = f" fdm_future_loss={fdm_future_loss:.6f}" if fdm_future_loss is not None else ""
                logger.info(
                    f"[eval] step={completed_steps} epoch={log_payload['eval/epoch']:.2f} "
                    f"{format_eval_metrics_log(metrics)}"
                    f"{future_mse_str}\n"
                    f"pred_action_first={metrics['pred_action_first']}\n"
                    f"gt_action_first={metrics['gt_action_first']}"
                )
            model.train()

        if completed_steps % 500 == 0 and completed_steps > 0:
            gc.collect()
        saved_full_state_this_step = False
        if completed_steps % cfg.training.save_interval == 0 and completed_steps > 0:
            save_training_state(
                accelerator,
                run_dir,
                completed_steps=completed_steps,
                steps_per_epoch=steps_per_epoch,
                max_train_steps=max_train_steps,
                logger=logger,
            )
            saved_full_state_this_step = True
            if accelerator.is_main_process:
                torch.save(
                    accelerator.get_state_dict(model),
                    run_dir / "checkpoints" / f"steps_{completed_steps}_pytorch_model.pt",
                )
        if completed_steps % steps_per_epoch == 0 and completed_steps > 0:
            if not saved_full_state_this_step:
                save_training_state(
                    accelerator,
                    run_dir,
                    completed_steps=completed_steps,
                    steps_per_epoch=steps_per_epoch,
                    max_train_steps=max_train_steps,
                    logger=logger,
                )
            if accelerator.is_main_process:
                torch.save(
                    accelerator.get_state_dict(model),
                    run_dir / "checkpoints" / f"epoch_{completed_steps // steps_per_epoch}_pytorch_model.pt",
                )

    if accelerator.is_main_process and "wandb" in cfg.logging.trackers:
        wandb.finish()
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default="",
        help="Warm-start model weights from a pytorch_model.pt checkpoint. "
        "This does not restore optimizer/scheduler state.",
    )
    parser.add_argument(
        "--resume-state",
        dest="resume_state",
        type=str,
        default="",
        help="Path to a full Accelerator state directory, e.g. run_dir/states/step_000010000.",
    )
    parser.add_argument(
        "--auto-resume",
        dest="auto_resume",
        action="store_true",
        help="Resume from the latest run_dir/states/step_* directory if present.",
    )
    args, clipargs = parser.parse_known_args()
    cfg = OmegaConf.merge(load_config(args.config), OmegaConf.from_dotlist(normalize_overrides(clipargs)))
    main(
        cfg,
        init_checkpoint=args.init_checkpoint if args.init_checkpoint else None,
        resume_state=args.resume_state if args.resume_state else None,
        auto_resume=bool(args.auto_resume),
    )
