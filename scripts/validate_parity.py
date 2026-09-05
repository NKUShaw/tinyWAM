#!/usr/bin/env python
"""Compare the independent package with a source research checkout."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from slim.compat.legacy import to_public_config, to_runtime_config
from slim.model import SLIMModel
from slim.model.checkpoint import load_weights
from slim.model.slim_transformer import SLIMTransformer


def _tensor_values(output):
    return {
        key: value.detach().float().cpu()
        for key, value in output.items()
        if isinstance(value, torch.Tensor) and value.ndim == 0
    }


def _load_state(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state.pop("vision_encoder.model.mask_token", None)
    for key in list(state):
        if key.startswith("lang_encoder."):
            state.pop(key)
    return state


def _compare_state_keys(old_model, new_model):
    old_keys = set(old_model.state_dict())
    new_keys = set(new_model.state_dict())
    if old_keys != new_keys:
        raise AssertionError(
            f"State keys differ: only_source={sorted(old_keys-new_keys)}, "
            f"only_slim={sorted(new_keys-old_keys)}"
        )


def _compare_transformer_keys(runtime, state):
    print("Constructing SLIM Transformer for checkpoint key validation...", flush=True)
    transformer = SLIMTransformer(runtime)
    expected = {f"action_model.{key}" for key in transformer.state_dict()}
    checkpoint_keys = {key for key in state if key.startswith("action_model.")}
    if expected != checkpoint_keys:
        raise AssertionError(
            "SLIM Transformer checkpoint keys differ: "
            f"only_checkpoint={sorted(checkpoint_keys-expected)}, "
            f"only_slim={sorted(expected-checkpoint_keys)}"
        )
    return len(expected)


def _compare_samples(source_dataset, new_dataset, index):
    source = source_dataset[index]
    new = new_dataset[index]
    for key in ("action", "state"):
        np.testing.assert_array_equal(np.asarray(source[key]), np.asarray(new[key]))
    for key in ("image", "future_image"):
        if key not in source:
            continue
        if len(source[key]) != len(new[key]):
            raise AssertionError(f"{key} view count differs")
        for source_image, new_image in zip(source[key], new[key]):
            np.testing.assert_array_equal(np.asarray(source_image), np.asarray(new_image))
    return new


def _dataset_kwargs(public):
    data = public.data
    return {
        "split": "train",
        "data_sources": OmegaConf.to_container(data.sources, resolve=True),
        "action_horizon": int(public.model.action_horizon),
        "val_ratio": float(data.val_ratio),
        "eval_monitor_episode_ratio": float(data.eval_monitor_episode_ratio),
        "split_seed": int(data.split_seed),
        "include_future_image": True,
        "video_backend": str(data.get("video_backend", "torchvision_av")),
        "action_normalization": str(data.action_normalization),
        "action_dim": int(public.model.action_dim),
        "state_dim": int(public.model.state_dim),
        "action_stats_dir": data.get("action_stats_dir"),
        "episode_shuffle": bool(data.episode_shuffle),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--keys-only", action="store_true")
    parser.add_argument("--slim-load-only", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    sys.path.insert(0, str(source_root))

    legacy = OmegaConf.load(checkpoint.parents[1] / "config.yaml")
    public = to_public_config(legacy)
    runtime = to_runtime_config(public)

    if args.data_only:
        from LIM.datasets import LeRobotBaseDataset as SourceDataset
        from slim.data import LeRobotBaseDataset

        source_kwargs = _dataset_kwargs(public)
        print("Constructing source dataset...", flush=True)
        source_dataset = SourceDataset(**source_kwargs)
        print("Constructing SLIM dataset...", flush=True)
        new_dataset = LeRobotBaseDataset(
            **source_kwargs, action_stats=source_dataset.get_action_stats()
        )
        sample = _compare_samples(source_dataset, new_dataset, args.sample_index)
        print(
            json.dumps(
                {
                    "action_shape": list(np.asarray(sample["action"]).shape),
                    "future_views": len(sample.get("future_image", [])),
                    "image_views": len(sample["image"]),
                    "source_samples": len(source_dataset),
                    "state_shape": list(np.asarray(sample["state"]).shape),
                    "slim_samples": len(new_dataset),
                    "status": "ok",
                },
                indent=2,
            )
        )
        return

    if args.keys_only:
        state = _load_state(checkpoint)
        transformer_keys = _compare_transformer_keys(runtime, state)
        print(
            json.dumps(
                {
                    "checkpoint_keys": len(state),
                    "transformer_keys": transformer_keys,
                    "status": "ok",
                },
                indent=2,
            )
        )
        return

    if args.slim_load_only:
        print("Constructing SLIM model...", flush=True)
        new_model = SLIMModel(public)
        print("Strictly loading checkpoint into SLIM...", flush=True)
        result = load_weights(new_model, checkpoint, strict=True)
        print(
            json.dumps(
                {
                    "missing_keys": list(result.missing_keys),
                    "unexpected_keys": list(result.unexpected_keys),
                    "state_keys": len(new_model.state_dict()),
                    "status": "ok",
                },
                indent=2,
            )
        )
        return

    from LIM.datasets import LeRobotBaseDataset as SourceDataset
    from LIM.models.dino_xmot_framework import DINO_XMoT as SourceModel
    from slim.data import LeRobotBaseDataset

    print("Constructing source model...", flush=True)
    source_model = SourceModel(legacy)
    print("Constructing SLIM model...", flush=True)
    new_model = SLIMModel(public)
    print("Comparing model state keys...", flush=True)
    _compare_state_keys(source_model, new_model)
    print("Loading checkpoint into both models...", flush=True)
    state = _load_state(checkpoint)
    source_model.load_state_dict(state, strict=False)
    new_model.load_state_dict(state, strict=False)

    source_kwargs = _dataset_kwargs(public)
    print("Constructing source and SLIM datasets...", flush=True)
    source_dataset = SourceDataset(**source_kwargs)
    new_dataset = LeRobotBaseDataset(
        **source_kwargs, action_stats=source_dataset.get_action_stats()
    )
    sample = _compare_samples(source_dataset, new_dataset, args.sample_index)
    batch = [sample]

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    tolerance = 1e-6 if dtype == torch.float32 else 1e-3
    device = torch.device(args.device)

    source_model = source_model.to(device=device, dtype=dtype).train()
    torch.manual_seed(123)
    source_output = _tensor_values(source_model(batch))
    source_model.eval()
    torch.manual_seed(456)
    source_actions = source_model.predict_action(batch)["normalized_actions"]
    del source_model
    gc.collect()
    torch.cuda.empty_cache()

    new_model = new_model.to(device=device, dtype=dtype).train()
    torch.manual_seed(123)
    new_output = _tensor_values(new_model(batch))
    new_model.eval()
    torch.manual_seed(456)
    new_actions = new_model.predict_action(batch)["normalized_actions"]

    common = sorted(set(source_output) & set(new_output))
    for key in common:
        torch.testing.assert_close(
            source_output[key], new_output[key], atol=tolerance, rtol=tolerance
        )
    np.testing.assert_allclose(
        source_actions, new_actions, atol=tolerance, rtol=tolerance
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "dtype": args.dtype,
                "tolerance": tolerance,
                "compared_losses": common,
                "state_keys": len(state),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
