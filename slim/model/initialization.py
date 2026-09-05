"""Explicit initialization of the new representation from an existing dense policy."""
from __future__ import annotations

import torch


def load_control_warm_start(model, checkpoint, logger, skip_prefixes=()):
    """Reuse compatible dense-model weights, while reporting intentional reinitialization.

    Use strict slim_policy initialization for Stage 1 -> Stage 2 within the new
    architecture. This opt-in migration accepts only the original dense model.
    """
    if not getattr(model, "control_latent_enabled", False):
        raise ValueError("control_warm_start requires model.control_latent.enabled=true")
    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if any(key.startswith("control_encoder.") for key in source):
        raise ValueError("A control-latent checkpoint must use strict slim_policy initialization")
    target = model.state_dict()
    fresh = ["control_encoder.", "ema_control_encoder.", "ema_vision_encoder.",
             "_ema_fp32_", "action_model.delta_decoder."]
    mask_key = "action_model.future_mask_tokens.weight"
    if mask_key in source and source[mask_key].shape != target[mask_key].shape:
        # Future positions move when the current token count changes. Reinitialize
        # both the future queries and the positional table rather than slicing them.
        fresh += ["action_model.future_mask_tokens.", "action_model.state_pos_embed."]
    fresh = tuple(fresh) + tuple(skip_prefixes)
    ignored_source = fresh + ("lang_encoder.",)
    filtered, incompatible = {}, []
    for key, value in source.items():
        if key.startswith(ignored_source) or key.endswith(".mask_token"):
            continue
        if key not in target or target[key].shape != value.shape:
            incompatible.append(key)
        else:
            filtered[key] = value
    missing = [key for key in target if key not in filtered
               and not key.startswith(fresh) and not key.endswith(".mask_token")]
    if incompatible or missing:
        raise RuntimeError(f"Incompatible dense warm start: dropped={incompatible}, missing={missing}")
    model.load_state_dict(filtered, strict=False)
    model.sync_ema_from_online()
    logger.info(f"[control_warm_start] loaded={len(filtered)} fresh_prefixes={fresh}")
    return {"loaded_keys": len(filtered), "fresh_prefixes": fresh}
