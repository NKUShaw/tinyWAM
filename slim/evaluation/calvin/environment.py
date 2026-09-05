from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import websockets.sync.client
from calvin_agent.models.calvin_base_model import CalvinBaseModel

from slim.serving import msgpack_numpy


class CalvinWebsocketClient:
    """WebSocket client compatible with the Python 3.8 / websockets 13 stack."""

    def __init__(self, host: str = "127.0.0.1", port: int = 10093) -> None:
        self._packer = msgpack_numpy.Packer()
        self._connection = websockets.sync.client.connect(
            f"ws://{host}:{port}",
            compression=None,
            max_size=None,
            open_timeout=150,
        )
        self._metadata = msgpack_numpy.unpackb(self._connection.recv())

    def predict_action(self, request: Dict) -> Dict:
        self._connection.send(self._packer.pack(request))
        response = self._connection.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Policy server returned a text error: {response}")
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        self._connection.close()


class CalvinModelClient(CalvinBaseModel):
    """Adapt SLIM action chunks to CALVIN's relative-action model interface."""

    def __init__(
        self,
        action_stats_path: str,
        dataset_name: str = "calvin_ABC_D_lerobot",
        action_horizon: int = 12,
        exec_stride: int = 12,
        host: str = "127.0.0.1",
        port: int = 10093,
    ) -> None:
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if exec_stride <= 0 or exec_stride > action_horizon:
            raise ValueError(
                f"exec_stride must be in [1, {action_horizon}], got {exec_stride}"
            )
        self.client = CalvinWebsocketClient(host=host, port=port)
        self.dataset_name = str(dataset_name)
        self.action_horizon = int(action_horizon)
        self.exec_stride = int(exec_stride)
        self.action_stats = self._load_action_stats(action_stats_path)
        self._chunk: Optional[np.ndarray] = None
        self._chunk_step = 0

    def reset(self) -> None:
        self._chunk = None
        self._chunk_step = 0

    def step(self, obs: dict, goal: str) -> np.ndarray:
        if self._chunk is None or self._chunk_step >= self.exec_stride:
            self._chunk = self._request_chunk(obs, goal)
            self._chunk_step = 0

        index = min(self._chunk_step, self._chunk.shape[0] - 1)
        action = self._chunk[index].copy()
        action[6] = 1.0 if action[6] > 0 else -1.0
        self._chunk_step += 1
        return action.astype(np.float32)

    def close(self) -> None:
        self.client.close()

    def _request_chunk(self, obs: dict, goal: str) -> np.ndarray:
        rgb = obs["rgb_obs"]
        state = np.asarray(obs["robot_obs"], dtype=np.float32).reshape(1, -1)
        request = {
            "examples": [
                {
                    "image": [
                        np.ascontiguousarray(rgb["rgb_static"]),
                        np.ascontiguousarray(rgb["rgb_gripper"]),
                    ],
                    "lang": str(goal),
                    "dataset_name": self.dataset_name,
                    "state": state,
                }
            ]
        }
        response = self.client.predict_action(request)
        if response.get("status") != "ok":
            message = response.get("error", {}).get("message", "unknown server error")
            raise RuntimeError(f"Policy server inference failed: {message}")

        normalized = np.asarray(
            response["data"]["normalized_actions"], dtype=np.float32
        )[0]
        if normalized.ndim != 2 or normalized.shape[1] != 7:
            raise ValueError(
                f"Expected normalized actions with shape [H, 7], got {normalized.shape}"
            )
        return self._unnormalize(normalized)

    def _unnormalize(self, normalized_actions: np.ndarray) -> np.ndarray:
        actions = normalized_actions.astype(np.float32).copy()
        q01 = self.action_stats["q01"]
        q99 = self.action_stats["q99"]
        dims = self.action_stats["action_normalized_dims"]
        scale = np.maximum(q99[dims] - q01[dims], 1e-6)
        actions[:, dims] = 0.5 * (actions[:, dims] + 1.0) * scale + q01[dims]
        return actions

    @staticmethod
    def _load_action_stats(path: str) -> Dict[str, np.ndarray]:
        stats_path = Path(path)
        if not stats_path.is_file():
            raise FileNotFoundError(f"Action stats not found: {stats_path}")
        with stats_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        q01 = np.asarray(payload["q01"], dtype=np.float32)
        q99 = np.asarray(payload["q99"], dtype=np.float32)
        if q01.shape != (7,) or q99.shape != (7,):
            raise ValueError(
                f"CALVIN requires 7D action stats, got q01={q01.shape}, q99={q99.shape}"
            )
        normalized_dims = np.asarray(
            payload.get("action_normalized_dims", [0, 1, 2, 3, 4, 5]),
            dtype=np.int64,
        )
        return {
            "q01": q01,
            "q99": q99,
            "action_normalized_dims": normalized_dims,
        }
