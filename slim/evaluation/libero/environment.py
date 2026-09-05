from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import cv2 as cv
import numpy as np

from slim.model.checkpoint import find_action_stats
from slim.serving.client import WebsocketClientPolicy


class ModelClient:
    """Client wrapper that adapts server outputs for LIBERO eval."""

    def __init__(
        self,
        policy_ckpt_path: str,
        action_stats_path: Optional[str] = None,
        action_chunk_size: int = 8,
        image_size: list[int] = [256, 256],
        host: str = "127.0.0.1",
        port: int = 10093,
    ) -> None:
        self.client = WebsocketClientPolicy(host, port)
        self.image_size = image_size
        self.action_chunk_size = int(action_chunk_size)
        self.action_norm_stats = self._load_action_stats(policy_ckpt_path, action_stats_path)
        self.raw_actions: Optional[np.ndarray] = None
        self.task_description: Optional[str] = None

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.raw_actions = None

    def step(self, example: dict, step: int = 0) -> dict:
        task_description = example.get("lang", None)
        if task_description != self.task_description:
            self.reset(task_description)

        resized = [self._resize_image(img) for img in example["image"]]
        request = {
            "examples": [
                {
                    "image": resized,
                    "lang": example["lang"],
                    "dataset_name": example.get("dataset_name", ""),
                }
            ],
        }
        if "state" in example:
            request["examples"][0]["state"] = np.asarray(example["state"], dtype=np.float32)

        if (step % self.action_chunk_size == 0) or (self.raw_actions is None):
            response = self.client.predict_action(request)
            status = response.get("status", None)
            if status != "ok":
                error_msg = response.get("error", {}).get("message", "Unknown server error")
                raise RuntimeError(
                    f"Policy server inference failed (status={status}): {error_msg}. Full response: {response}"
                )
            normalized_actions = np.asarray(response["data"]["normalized_actions"], dtype=np.float32)[0]
            self.raw_actions = self.unnormalize_actions(normalized_actions, self.action_norm_stats)

        action = self.raw_actions[step % self.action_chunk_size]
        raw_action = {
            "world_vector": action[:3],
            "rotation_delta": action[3:6],
            "open_gripper": action[6:7],
        }
        return {"raw_action": raw_action}

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        actions = normalized_actions.astype(np.float32).copy()
        q01 = np.asarray(action_norm_stats["q01"], dtype=np.float32)
        q99 = np.asarray(action_norm_stats["q99"], dtype=np.float32)
        scale = np.maximum(q99[:6] - q01[:6], 1e-6)
        actions[:, :6] = 0.5 * (actions[:, :6] + 1.0) * scale + q01[:6]
        # Gripper is intentionally left as model output; downstream binarization handles it.
        return actions

    @staticmethod
    def _resolve_action_stats_path(policy_ckpt_path: str, action_stats_path: Optional[str]) -> Path:
        if action_stats_path:
            return Path(action_stats_path)
        ckpt_path = Path(policy_ckpt_path)
        # Expected layout:
        # <run_dir>/checkpoints/steps_xxx_pytorch_model.pt
        # <run_dir>/action_stats_libero_all.json
        run_dir = ckpt_path.parent.parent
        return find_action_stats(run_dir)

    def _load_action_stats(self, policy_ckpt_path: str, action_stats_path: Optional[str]) -> Dict[str, np.ndarray]:
        stats_path = self._resolve_action_stats_path(policy_ckpt_path, action_stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Action stats file not found. Tried: "
                f"{Path(policy_ckpt_path).parent.parent / 'action_stats.json'} and "
                f"{Path(policy_ckpt_path).parent.parent / 'dataset_statistics.json'}. "
                "Please pass --action_stats_path explicitly."
            )
        with open(stats_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return {
            "q01": np.asarray(payload["q01"], dtype=np.float32),
            "q99": np.asarray(payload["q99"], dtype=np.float32),
        }

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
