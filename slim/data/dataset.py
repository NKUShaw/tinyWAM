from __future__ import annotations

import collections
import gc
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import av
import numpy as np
from PIL import Image
from pyarrow import parquet as pq
from torch.utils.data import Dataset

from slim.data.mixtures import (
    NAMED_MIXTURES,
)


class _VideoContainerCache:
    """Per-process LRU cache for open av.Container objects.

    Each DataLoader worker is a separate process, so this module-level
    instance is naturally isolated - no locking needed.  Caching
    containers avoids repeated open/seek/close on the same video file
    across consecutive samples from the same episode.
    """

    def __init__(self, maxsize: int = 4) -> None:
        self._cache: collections.OrderedDict[str, av.container.InputContainer] = (
            collections.OrderedDict()
        )
        self._maxsize = maxsize

    def get(self, path: str) -> av.container.InputContainer:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        container = av.open(path)
        if len(self._cache) >= self._maxsize:
            _, evicted = self._cache.popitem(last=False)
            try:
                evicted.close()
            except Exception:
                pass
            gc.collect()
        self._cache[path] = container
        return container

    def invalidate(self, path: str) -> None:
        container = self._cache.pop(path, None)
        if container is not None:
            try:
                container.close()
            except Exception:
                pass

    def close_all(self) -> None:
        for container in self._cache.values():
            try:
                container.close()
            except Exception:
                pass
        self._cache.clear()
        gc.collect()


# Episodes that failed video decode in this DataLoader worker (process-local).
_BAD_EPISODE_INDICES: set[int] = set()

# Module-level singleton - one instance per worker process.
# With episode-level shuffle each worker only switches videos at episode
# boundaries, so a small cache (16) is plenty. Bump it for safety.
_CONTAINER_CACHE = _VideoContainerCache(maxsize=16)


def _is_lerobot_v3(info: Dict[str, Any]) -> bool:
    """Return whether an info.json describes a file-sharded LeRobot v3 dataset."""
    version = str(info.get("codebase_version", "")).lower()
    data_path = str(info.get("data_path", ""))
    return version.startswith("v3") or "{file_index" in data_path


def _episode_data_path(
    root: Path,
    info: Dict[str, Any],
    episode_meta: Dict[str, Any],
) -> Path:
    episode_index = int(episode_meta["episode_index"])
    if _is_lerobot_v3(info):
        return root / info["data_path"].format(
            chunk_index=int(episode_meta["data_chunk_index"]),
            file_index=int(episode_meta["data_file_index"]),
        )
    chunks_size = int(info.get("chunks_size", 1000))
    return root / info["data_path"].format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )


def _read_episode_columns(
    episode_meta: Dict[str, Any],
    columns: Sequence[str],
):
    """Read one episode from either a v2 per-episode or v3 shared parquet file."""
    root = Path(episode_meta["root"])
    info = episode_meta["lerobot_info"]
    parquet_path = _episode_data_path(root, info, episode_meta)
    schema_names = set(pq.read_schema(parquet_path).names)
    missing_columns = set(columns).difference(schema_names)
    if missing_columns:
        raise KeyError(
            f"Missing configured columns {sorted(missing_columns)} in {parquet_path}. "
            f"Available columns: {sorted(schema_names)}"
        )

    read_columns = list(dict.fromkeys(columns))
    filters = None
    if _is_lerobot_v3(info):
        if "episode_index" not in schema_names:
            raise KeyError(f"Missing episode_index in LeRobot v3 shard: {parquet_path}")
        filters = [("episode_index", "=", int(episode_meta["episode_index"]))]
    table = pq.read_table(parquet_path, columns=read_columns, filters=filters)
    expected_length = int(episode_meta["length"])
    if table.num_rows != expected_length:
        raise RuntimeError(
            f"Episode length mismatch for episode={episode_meta['episode_index']} in "
            f"{parquet_path}: metadata={expected_length}, parquet={table.num_rows}"
        )
    return parquet_path, table


class _EpisodeSampleIndex:
    """Memory-efficient flat (episode, frame) index backed by prefix sums."""

    def __init__(
        self,
        episodes: Sequence[Dict[str, Any]],
        action_horizon: int,
        episode_order: Sequence[int],
    ) -> None:
        self._episode_order = np.asarray(episode_order, dtype=np.int32)
        counts = np.asarray(
            [
                max(0, int(episodes[int(index)]["length"]) - int(action_horizon))
                for index in self._episode_order
            ],
            dtype=np.int64,
        )
        keep = counts > 0
        self._episode_order = self._episode_order[keep]
        self._cumulative_counts = np.cumsum(counts[keep], dtype=np.int64)

    def __len__(self) -> int:
        if len(self._cumulative_counts) == 0:
            return 0
        return int(self._cumulative_counts[-1])

    def __bool__(self) -> bool:
        return len(self) > 0

    def __getitem__(self, index: int) -> Tuple[int, int]:
        total = len(self)
        index = int(index)
        if index < 0:
            index += total
        if index < 0 or index >= total:
            raise IndexError(index)
        position = int(np.searchsorted(self._cumulative_counts, index, side="right"))
        previous = 0 if position == 0 else int(self._cumulative_counts[position - 1])
        return int(self._episode_order[position]), int(index - previous)

    def __iter__(self):
        previous = 0
        for episode_index, cumulative in zip(
            self._episode_order, self._cumulative_counts
        ):
            count = int(cumulative) - previous
            for frame_index in range(count):
                yield int(episode_index), frame_index
            previous = int(cumulative)


def collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch


def _project_last_dim(array: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad or truncate an array's last dimension to target_dim."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.shape[-1] == target_dim:
        return arr.astype(np.float32, copy=False)
    out = np.zeros((*arr.shape[:-1], target_dim), dtype=np.float32)
    copy_dim = min(int(arr.shape[-1]), int(target_dim))
    if copy_dim > 0:
        out[..., :copy_dim] = arr[..., :copy_dim]
    return out


def _select_last_dim(
    array: np.ndarray,
    indices: Optional[Sequence[int]],
    *,
    field_name: str,
) -> np.ndarray:
    """Select/reorder the last dimension, allowing duplicate indices."""
    arr = np.asarray(array, dtype=np.float32)
    if indices is None:
        return arr
    idx = np.asarray([int(x) for x in indices], dtype=np.int64)
    if idx.size == 0:
        raise ValueError(f"{field_name}_indices must not be empty")
    if np.any(idx < 0) or np.any(idx >= arr.shape[-1]):
        raise IndexError(
            f"{field_name}_indices {idx.tolist()} out of range for shape {arr.shape}"
        )
    return arr[..., idx]


def _load_episode_actions_for_stats(
    args: Tuple[Dict[str, Any], int, Optional[Tuple[int, ...]], str]
) -> np.ndarray:
    """Load one configured action field for stats computation.

    Must be a module-level function for :class:`ProcessPoolExecutor` workers.
    """
    episode_meta, action_dim, action_indices, action_key = args
    _, table = _read_episode_columns(episode_meta, [action_key])
    data = table.to_pandas()
    # Match :meth:`_load_episode_arrays` stacking behavior.
    action = np.stack(data[action_key].to_numpy()).astype(np.float32)
    if action.ndim == 1:
        action = action.reshape(1, -1)
    action = _select_last_dim(action, action_indices, field_name="action")
    return np.ascontiguousarray(_project_last_dim(action, int(action_dim)))


def _load_action_group_for_stats(
    args: Tuple[List[Dict[str, Any]], int, Optional[Tuple[int, ...]], str]
) -> np.ndarray:
    """Load actions once per parquet shard instead of once per v3 episode."""
    episode_group, action_dim, action_indices, action_key = args
    if not episode_group:
        return np.zeros((0, int(action_dim)), dtype=np.float32)
    first = episode_group[0]
    path = _episode_data_path(
        Path(first["root"]), first["lerobot_info"], first
    )
    if _is_lerobot_v3(first["lerobot_info"]):
        table = pq.read_table(path, columns=[action_key, "episode_index"])
        wanted = np.asarray(
            [int(episode["episode_index"]) for episode in episode_group],
            dtype=np.int64,
        )
        episode_indices = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        keep = np.isin(episode_indices, wanted, assume_unique=False)
        action = np.asarray(table[action_key].to_pylist(), dtype=np.float32)[keep]
    else:
        _, table = _read_episode_columns(first, [action_key])
        action = np.asarray(table[action_key].to_pylist(), dtype=np.float32)
    if action.ndim == 1:
        action = action.reshape(1, -1)
    action = _select_last_dim(action, action_indices, field_name="action")
    return np.ascontiguousarray(_project_last_dim(action, int(action_dim)))


class LeRobotBaseDataset(Dataset):
    """Multi-source LeRobot dataset loader with per-source video key configuration.

    Accepts a list of data_sources, each specifying its own root directory,
    video keys, optional named mixture, and sampling weight. All sources share
    a unified action normalization (q01/q99 computed jointly across sources).

    Each source dict schema::

        {
            "root_dir": str | Path,       # dataset root (or parent when data_mix is set)
            "video_keys": List[str],       # camera keys for this source (len == num_image_views)
            "data_mix": str | None,        # optional named mixture or comma-separated subdir names
            "weight": float,               # relative sampling weight (default 1.0)
            "name": str | None,            # display name for logging (optional)
        }

    Duplicate entries in video_keys are allowed and cause the same frame to be
    decoded twice - this is the intended way to pad single-camera datasets to
    match a fixed num_image_views.
    """

    # Named mixture mappings: {mix_name: [subdir_name, ...]} relative to root_dir.
    NAMED_MIXTURES: Dict[str, List[str]] = {**NAMED_MIXTURES}

    # Episodes to skip by dataset name (per LeRobot subdataset folder name).
    DEFAULT_SKIP_LEROBOT_EPISODES: Dict[str, Tuple[int, ...]] = {
        "libero_goal_no_noops_1.0.0_lerobot": (82,),
    }

    def __init__(
        self,
        data_sources: List[Dict[str, Any]],
        split: str = "train",
        val_ratio: float = 0.05,
        eval_monitor_episode_ratio: float = 0.05,
        split_seed: int = 42,
        action_horizon: int = 8,
        video_backend: str = "torchvision_av",
        action_normalization: str = "q01_q99",
        action_dim: int = 7,
        state_dim: int = 7,
        action_normalized_dims: Optional[Sequence[int]] = None,
        action_indices: Optional[Sequence[int]] = None,
        state_indices: Optional[Sequence[int]] = None,
        action_key: str = "action",
        state_key: str = "observation.state",
        action_stats: Optional[Dict[str, Dict[str, List[float]]]] = None,
        action_stats_dir: str | Path | None = None,
        skip_lerobot_episodes: Optional[Dict[str, Sequence[int]]] = None,
        include_future_image: bool = False,
        use_fake_data: bool = False,
        episode_shuffle: bool = True,
    ) -> None:
        if not data_sources:
            raise ValueError("data_sources must contain at least one source dict.")

        self.data_sources = data_sources
        self.split = split
        self.val_ratio = val_ratio
        self.eval_monitor_episode_ratio = float(eval_monitor_episode_ratio)
        self.split_seed = split_seed
        self.action_horizon = action_horizon
        self.video_backend = str(video_backend).strip().lower()
        if self.video_backend not in ("torchvision_av", "frames"):
            raise ValueError(
                f"Unsupported video_backend={video_backend!r}. Choose 'torchvision_av' or 'frames'."
            )
        self.action_normalization = action_normalization
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {self.action_dim}")
        if self.state_dim <= 0:
            raise ValueError(f"state_dim must be > 0, got {self.state_dim}")
        self.action_indices = (
            tuple(int(x) for x in action_indices)
            if action_indices is not None
            else None
        )
        self.state_indices = (
            tuple(int(x) for x in state_indices)
            if state_indices is not None
            else None
        )
        self.action_key = str(action_key).strip()
        self.state_key = str(state_key).strip()
        if not self.action_key or not self.state_key:
            raise ValueError("action_key and state_key must be non-empty parquet column names")
        if action_normalized_dims is None:
            # Preserve the original 7D behavior: normalize xyz/rpy and keep the
            # Keep the gripper target raw for standard seven-dimensional actions.
            normalized_dims = range(6) if self.action_dim == 7 else range(self.action_dim)
        else:
            normalized_dims = [int(x) for x in action_normalized_dims]
        self.action_normalized_dims = tuple(sorted(set(normalized_dims)))
        bad_dims = [d for d in self.action_normalized_dims if d < 0 or d >= self.action_dim]
        if bad_dims:
            raise ValueError(
                f"action_normalized_dims contains dims outside action_dim={self.action_dim}: {bad_dims}"
            )
        self.use_fake_data = use_fake_data
        self.action_stats_dir = Path(action_stats_dir) if action_stats_dir is not None else None
        self.skip_lerobot_episodes = self._build_skip_episode_map(skip_lerobot_episodes)
        self.include_future_image = bool(include_future_image)
        self.episode_shuffle = bool(episode_shuffle)

        self.curr_traj_id: Optional[Tuple[str, int]] = None
        self.curr_traj_data: Optional[Dict[str, np.ndarray]] = None
        # {source_key -> {"q01": ndarray, "q99": ndarray}}
        self._per_source_stats: Dict[str, Dict[str, np.ndarray]] = {}
        self._all_episodes: List[Dict[str, Any]] = []

        if self.use_fake_data:
            self._fake_len = 64
            self._fake_num_views = len(data_sources[0]["video_keys"])
            return

        self._all_episodes = self._load_all_episodes()
        if not self._all_episodes:
            raise RuntimeError(
                f"No episodes found across {len(data_sources)} data source(s). "
                "Check root_dir and data_mix settings."
            )
        if action_stats is not None:
            # Caller passes pre-computed per-source stats (e.g. val split reusing train stats).
            self._per_source_stats = {
                key: {
                    "q01": np.asarray(v["q01"], dtype=np.float32),
                    "q99": np.asarray(v["q99"], dtype=np.float32),
                }
                for key, v in action_stats.items()
            }
            mismatched = [key for key, stats in self._per_source_stats.items() if not self._stats_match_action_dim(stats)]
            if mismatched:
                raise ValueError(
                    f"Provided action_stats do not match action_dim={self.action_dim} "
                    f"for source(s): {mismatched}"
                )
        elif self.action_normalization == "q01_q99":
            self._per_source_stats = self._load_or_compute_per_source_stats()
        self.episodes = self._select_split_episodes()
        # Episode-level shuffle keeps the video-container cache warm: consecutive
        # samples within the same episode reuse the same open av.Container.
        ep_seed = self.split_seed if (self.split == "train" and self.episode_shuffle) else None
        self.sample_index = self._build_sample_index(episode_shuffle_seed=ep_seed)
        if self.video_backend == "frames" and not self.use_fake_data:
            self._validate_frames_layout()

    def _validate_frames_layout(self) -> None:
        """Smoke-check that pre-extracted frames exist for the first episode."""
        if not self.episodes or not self.sample_index:
            return
        ep_meta = self.episodes[self.sample_index[0][0]]
        video_key = ep_meta["video_keys"][0]
        path = self._build_frame_path(ep_meta, video_key=video_key, frame_index=0)
        if not path.is_file():
            raise FileNotFoundError(
                f"video_backend=frames but frame not found: {path}. "
                "Expected layout: {{root}}/frames/chunk-XXX/{{video_key}}/episode_XXXXXX/frame_XXXXXX.jpg"
            )
        print(f"[SLIM][DATASET] video_backend=frames validated: {path}", flush=True)

    # ------------------------------------------------------------------
    # Source resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _source_key(source: Dict[str, Any]) -> str:
        """Derive a stable, human-readable key for a source dict.

        Priority:
        1. Explicit ``name`` field in the source dict.
        2. ``data_mix`` value when set (e.g. ``libero_all``, ``bridge``).
        3. Basename of ``root_dir`` (for sources with ``data_mix: null``).
        """
        if source.get("name"):
            return str(source["name"])
        data_mix = source.get("data_mix")
        if data_mix:
            return str(data_mix)
        return Path(source["root_dir"]).name

    def _resolve_source_roots(self, source: Dict[str, Any]) -> List[Path]:
        """Expand one source dict into a list of concrete dataset root paths."""
        root = Path(source["root_dir"])
        data_mix = source.get("data_mix")
        if not data_mix:
            return [root]
        if data_mix in self.NAMED_MIXTURES:
            return [root / name for name in self.NAMED_MIXTURES[data_mix]]
        # Comma-separated subdirectory names.
        return [root / name.strip() for name in data_mix.split(",") if name.strip()]

    # ------------------------------------------------------------------
    # Skip-episode map
    # ------------------------------------------------------------------

    def _build_skip_episode_map(
        self, user_skip: Optional[Dict[str, Sequence[int]]]
    ) -> Dict[str, set]:
        merged: Dict[str, set] = {
            ds_name: set(int(ep) for ep in eps)
            for ds_name, eps in self.DEFAULT_SKIP_LEROBOT_EPISODES.items()
        }
        if user_skip is None:
            return merged
        for ds_name, eps in user_skip.items():
            if ds_name not in merged:
                merged[ds_name] = set()
            merged[ds_name].update(int(ep) for ep in eps)
        return merged

    # ------------------------------------------------------------------
    # Episode loading
    # ------------------------------------------------------------------

    def _load_all_episodes(self) -> List[Dict[str, Any]]:
        all_episodes: List[Dict[str, Any]] = []
        for source in self.data_sources:
            video_keys = list(source["video_keys"])
            weight = float(source.get("weight", 1.0))
            source_key = self._source_key(source)
            for root in self._resolve_source_roots(source):
                dataset_name = root.name
                info_path = root / "meta" / "info.json"
                episodes_path = root / "meta" / "episodes.jsonl"
                episodes_dir = root / "meta" / "episodes"
                if not info_path.exists():
                    print(
                        f"[SLIM][DATASET] skip source root={root}: missing meta/info.json",
                        flush=True,
                    )
                    continue

                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)

                if _is_lerobot_v3(info):
                    episode_files = sorted(episodes_dir.glob("**/*.parquet"))
                    if not episode_files:
                        print(
                            f"[SLIM][DATASET] skip source root={root}: no v3 episode parquet files",
                            flush=True,
                        )
                        continue
                    metadata_columns = [
                        "episode_index",
                        "tasks",
                        "length",
                        "data/chunk_index",
                        "data/file_index",
                    ]
                    for video_key in video_keys:
                        metadata_columns.extend(
                            [
                                f"videos/{video_key}/chunk_index",
                                f"videos/{video_key}/file_index",
                                f"videos/{video_key}/from_timestamp",
                            ]
                        )
                    for episode_file in episode_files:
                        schema_names = set(pq.read_schema(episode_file).names)
                        missing = set(metadata_columns).difference(schema_names)
                        if missing:
                            raise KeyError(
                                f"Missing LeRobot v3 episode metadata columns {sorted(missing)} "
                                f"in {episode_file}"
                            )
                        rows = pq.read_table(
                            episode_file, columns=metadata_columns
                        ).to_pylist()
                        for row in rows:
                            video_segments = {
                                video_key: {
                                    "chunk_index": int(
                                        row[f"videos/{video_key}/chunk_index"]
                                    ),
                                    "file_index": int(
                                        row[f"videos/{video_key}/file_index"]
                                    ),
                                    "from_timestamp": float(
                                        row[f"videos/{video_key}/from_timestamp"]
                                    ),
                                }
                                for video_key in video_keys
                            }
                            episode_index = int(row["episode_index"])
                            if episode_index in self.skip_lerobot_episodes.get(
                                dataset_name, set()
                            ):
                                continue
                            all_episodes.append(
                                {
                                    "root": str(root),
                                    "dataset_name": dataset_name,
                                    "source_key": source_key,
                                    "lerobot_info": info,
                                    "episode_index": episode_index,
                                    "length": int(row["length"]),
                                    "tasks": row.get("tasks", []),
                                    "video_keys": video_keys,
                                    "weight": weight,
                                    "data_chunk_index": int(row["data/chunk_index"]),
                                    "data_file_index": int(row["data/file_index"]),
                                    "video_segments": video_segments,
                                }
                            )
                    continue

                if not episodes_path.exists():
                    print(
                        f"[SLIM][DATASET] skip source root={root}: missing meta/episodes.jsonl",
                        flush=True,
                    )
                    continue
                with open(episodes_path, "r", encoding="utf-8") as f:
                    rows = (json.loads(line) for line in f if line.strip())
                    for row in rows:
                        episode_index = int(row["episode_index"])
                        if episode_index in self.skip_lerobot_episodes.get(dataset_name, set()):
                            print(
                                "[SLIM][SKIP_EPISODE] "
                                f"split={self.split} dataset={dataset_name} "
                                f"episode_index={episode_index} skipped by skip_lerobot_episodes",
                                flush=True,
                            )
                            continue
                        all_episodes.append(
                            {
                                "root": str(root),
                                "dataset_name": dataset_name,
                                "source_key": source_key,
                                "lerobot_info": info,
                                "episode_index": episode_index,
                                "length": int(row["length"]),
                                "tasks": row.get("tasks", []),
                                "video_keys": video_keys,
                                "weight": weight,
                            }
                        )

        return all_episodes

    # ------------------------------------------------------------------
    # Train / val split
    # ------------------------------------------------------------------

    def _select_split_episodes(self) -> List[Dict[str, Any]]:
        all_episodes = self._all_episodes
        if not all_episodes:
            return []
        rng = np.random.default_rng(self.split_seed)
        perm = rng.permutation(len(all_episodes)).tolist()
        n = len(perm)

        # val_ratio <= 0: use all episodes for train; val split is an overlapping subset
        # of the same perm (train-monitor only, not a held-out validation set).
        if float(self.val_ratio) <= 0.0:
            if self.split == "train":
                selected = perm
            elif self.split == "val":
                k = max(1, int(round(n * self.eval_monitor_episode_ratio)))
                k = min(k, n)
                selected = perm[:k]
            else:
                raise ValueError(f"Unsupported split={self.split}. expected train|val")
            return [all_episodes[i] for i in selected]

        cut = int(round(n * (1.0 - self.val_ratio)))
        cut = min(max(cut, 1), n - 1) if n > 1 else 1

        if self.split == "train":
            selected = perm[:cut]
        elif self.split == "val":
            selected = perm[cut:]
        else:
            raise ValueError(f"Unsupported split={self.split}. expected train|val")
        return [all_episodes[i] for i in selected]

    # ------------------------------------------------------------------
    # Sample index
    # ------------------------------------------------------------------

    def _build_sample_index(self, episode_shuffle_seed: Optional[int] = None) -> _EpisodeSampleIndex:
        """Build a flat list of (episode_idx, frame_t) pairs.

        Episodes are optionally shuffled first so that DataLoader frame-level
        iteration stays within the same episode for many consecutive steps.
        This keeps the per-worker video-container cache effective and avoids
        the random-seek I/O spike caused by frame-level shuffle.

        Frame-level shuffle within an episode is intentionally *not* applied
        here; the DataLoader ``shuffle=True`` or ``WeightedRandomSampler``
        still provide cross-episode randomness.
        """
        episode_order = list(range(len(self.episodes)))
        if episode_shuffle_seed is not None:
            rng = np.random.default_rng(episode_shuffle_seed)
            rng.shuffle(episode_order)
        return _EpisodeSampleIndex(
            self.episodes,
            action_horizon=self.action_horizon,
            episode_order=episode_order,
        )

    @property
    def has_nonuniform_sample_weights(self) -> bool:
        return len({round(float(ep["weight"]), 6) for ep in self.episodes}) > 1

    @property
    def sample_weights(self) -> List[float]:
        """Per-sample weights reflecting each source's configured weight.

        Pass this to torch.utils.data.WeightedRandomSampler when sources
        differ in size and you want balanced cross-source sampling.
        """
        return [self.episodes[ei]["weight"] for ei, _ in self.sample_index]

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_episode_arrays(self, episode_meta: Dict[str, Any]) -> Dict[str, np.ndarray]:
        root = Path(episode_meta["root"])
        episode_index = int(episode_meta["episode_index"])
        key = (str(root), episode_index)
        if self.curr_traj_id == key and self.curr_traj_data is not None:
            return self.curr_traj_data

        info = episode_meta["lerobot_info"]
        parquet_path = _episode_data_path(root, info, episode_meta)
        schema_names = set(pq.read_schema(parquet_path).names)
        required_columns = {self.state_key, self.action_key, "timestamp"}
        missing_columns = required_columns.difference(schema_names)
        if missing_columns:
            raise KeyError(
                f"Missing configured state/action columns {sorted(missing_columns)} in {parquet_path}. "
                f"Available columns: {sorted(schema_names)}"
            )
        _, table = _read_episode_columns(
            episode_meta, [self.state_key, self.action_key, "timestamp"]
        )
        data = table.to_pandas()
        state = np.stack(data[self.state_key].to_numpy()).astype(np.float32)
        action = np.stack(data[self.action_key].to_numpy()).astype(np.float32)
        if "timestamp" not in data.columns:
            raise RuntimeError(
                f"Missing required 'timestamp' column in parquet shard: {parquet_path}. "
                "Timestamp-driven video decoding requires explicit timestamps."
            )
        timestamp = data["timestamp"].to_numpy().astype(np.float32)

        out = {"state": state, "action": action, "timestamp": timestamp}
        self.curr_traj_id = key
        self.curr_traj_data = out
        return out

    def _build_video_path(self, episode_meta: Dict[str, Any], video_key: str) -> Path:
        root = Path(episode_meta["root"])
        info = episode_meta["lerobot_info"]
        if _is_lerobot_v3(info):
            segment = episode_meta["video_segments"][video_key]
            return root / info["video_path"].format(
                video_key=video_key,
                chunk_index=int(segment["chunk_index"]),
                file_index=int(segment["file_index"]),
            )
        episode_index = int(episode_meta["episode_index"])
        chunks_size = int(info.get("chunks_size", 1000))
        episode_chunk = episode_index // chunks_size
        return root / info["video_path"].format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
            video_key=video_key,
        )

    def _build_frame_path(
        self,
        episode_meta: Dict[str, Any],
        video_key: str,
        frame_index: int,
    ) -> Path:
        """Path to a pre-extracted JPEG frame (1-based frame numbering on disk)."""
        root = Path(episode_meta["root"])
        info = episode_meta["lerobot_info"]
        episode_index = int(episode_meta["episode_index"])
        chunks_size = int(info.get("chunks_size", 1000))
        episode_chunk = episode_index // chunks_size
        frame_num = int(frame_index) + 1
        return (
            root
            / "frames"
            / f"chunk-{episode_chunk:03d}"
            / video_key
            / f"episode_{episode_index:06d}"
            / f"frame_{frame_num:06d}.jpg"
        )

    def _load_rgb_frame_from_disk(
        self,
        episode_meta: Dict[str, Any],
        frame_index: int,
        video_key: str,
    ) -> Image.Image:
        path = self._build_frame_path(
            episode_meta=episode_meta,
            video_key=video_key,
            frame_index=frame_index,
        )
        with Image.open(path) as img:
            return img.convert("RGB")

    def _load_rgb_frame(
        self,
        episode_meta: Dict[str, Any],
        video_key: str,
        *,
        frame_index: int,
        frame_ts: float | None = None,
    ) -> Image.Image:
        if self.video_backend == "frames":
            return self._load_rgb_frame_from_disk(
                episode_meta=episode_meta,
                frame_index=frame_index,
                video_key=video_key,
            )
        if frame_ts is None:
            raise ValueError("frame_ts is required when video_backend=torchvision_av")
        video_path = str(self._build_video_path(episode_meta=episode_meta, video_key=video_key))
        target_ts = float(frame_ts)
        if _is_lerobot_v3(episode_meta["lerobot_info"]):
            target_ts += float(
                episode_meta["video_segments"][video_key]["from_timestamp"]
            )
        frame_array = self._decode_frame_cached(video_path, target_ts)
        return Image.fromarray(frame_array, mode="RGB")

    @staticmethod
    def _decode_frame_cached(video_path: str, target_ts: float) -> np.ndarray:
        """Decode a single frame using a process-local cached av.Container."""
        container = _CONTAINER_CACHE.get(video_path)
        stream = container.streams.video[0]
        time_base = float(stream.time_base)

        target_pts = int(target_ts / time_base)
        try:
            container.seek(target_pts, stream=stream, backward=True, any_frame=False)
        except av.error.FFmpegError:
            _CONTAINER_CACHE.close_all()
            container = _CONTAINER_CACHE.get(video_path)
            stream = container.streams.video[0]
            time_base = float(stream.time_base)
            target_pts = int(target_ts / time_base)
            container.seek(target_pts, stream=stream, backward=True, any_frame=False)

        closest_frame: Optional[av.VideoFrame] = None
        closest_diff = float("inf")
        try:
            for frame in container.decode(video=0):
                current_ts = float(frame.pts * time_base)
                diff = abs(current_ts - target_ts)
                if diff < closest_diff:
                    if closest_frame is not None:
                        del closest_frame
                    closest_diff = diff
                    closest_frame = frame
                if current_ts > target_ts + 1.0:
                    break
        except av.error.FFmpegError:
            _CONTAINER_CACHE.close_all()
            if closest_frame is not None:
                del closest_frame
            raise

        if closest_frame is None:
            raise av.error.FFmpegError(
                -1, f"No frame found near timestamp {target_ts} in {video_path}"
            )

        frame_array = closest_frame.to_ndarray(format="rgb24")
        del closest_frame
        return frame_array

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------

    def _project_action(self, action: np.ndarray) -> np.ndarray:
        selected = _select_last_dim(
            action,
            self.action_indices,
            field_name="action",
        )
        return _project_last_dim(selected, self.action_dim)

    def _project_state(self, state: np.ndarray) -> np.ndarray:
        selected = _select_last_dim(
            state,
            self.state_indices,
            field_name="state",
        )
        return _project_last_dim(selected, self.state_dim)

    def _stats_payload(self, stats: Dict[str, np.ndarray]) -> Dict[str, Any]:
        return {
            "q01": np.asarray(stats["q01"], dtype=np.float32).tolist(),
            "q99": np.asarray(stats["q99"], dtype=np.float32).tolist(),
            "normalization": self.action_normalization,
            "action_dim": self.action_dim,
            "action_normalized_dims": list(self.action_normalized_dims),
            "action_indices": list(self.action_indices) if self.action_indices is not None else None,
            "action_key": self.action_key,
            "state_key": self.state_key,
            "note": "q01/q99 are length action_dim; only action_normalized_dims are normalized",
        }

    def _stats_match_action_dim(self, stats: Dict[str, np.ndarray]) -> bool:
        return (
            int(np.asarray(stats["q01"]).shape[0]) == self.action_dim
            and int(np.asarray(stats["q99"]).shape[0]) == self.action_dim
        )

    def _stats_cache_is_compatible(
        self,
        payload: Dict[str, Any],
        stats: Dict[str, np.ndarray],
    ) -> bool:
        cached_indices = payload.get("action_indices")
        expected_indices = (
            list(self.action_indices) if self.action_indices is not None else None
        )
        return (
            self._stats_match_action_dim(stats)
            and str(payload.get("action_key", "action")) == self.action_key
            and str(payload.get("normalization", "q01_q99"))
            == self.action_normalization
            and list(payload.get("action_normalized_dims", []))
            == list(self.action_normalized_dims)
            and cached_indices == expected_indices
        )

    def _compute_stats_for_episodes(
        self, episodes: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """Compute q01/q99 action stats over a list of episode metas (parallel)."""
        n_eps = len(episodes)
        env_w = os.environ.get("SLIM_ACTION_STATS_NUM_WORKERS", "").strip()
        num_workers = max(1, min(32, os.cpu_count() or 8)) if env_w == "" else max(1, int(env_w))
        use_parallel = num_workers > 1 and n_eps >= 32

        raw_actions: List[np.ndarray] = []
        if use_parallel:
            episode_groups: Dict[str, List[Dict[str, Any]]] = {}
            for episode in episodes:
                path = _episode_data_path(
                    Path(episode["root"]), episode["lerobot_info"], episode
                )
                episode_groups.setdefault(str(path), []).append(episode)
            print(
                f"[SLIM][ACTION_STATS]   parallel: {num_workers} workers, "
                f"{len(episode_groups)} parquet shard(s)",
                flush=True,
            )
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                worker_args = (
                    (group, self.action_dim, self.action_indices, self.action_key)
                    for group in episode_groups.values()
                )
                for actions in ex.map(_load_action_group_for_stats, worker_args, chunksize=1):
                    raw_actions.append(actions)
        else:
            for ep_meta in episodes:
                a = self._load_episode_arrays(ep_meta)["action"]
                raw_actions.append(self._project_action(a))

        all_actions = (
            np.concatenate(raw_actions, axis=0)
            if raw_actions
            else np.zeros((1, self.action_dim), dtype=np.float32)
        )
        q01 = np.zeros(self.action_dim, dtype=np.float32)
        q99 = np.ones(self.action_dim, dtype=np.float32)
        if self.action_normalized_dims:
            dims = np.asarray(self.action_normalized_dims, dtype=np.int64)
            q01[dims] = np.percentile(all_actions[:, dims], 1, axis=0).astype(np.float32)
            q99[dims] = np.percentile(all_actions[:, dims], 99, axis=0).astype(np.float32)
        return {"q01": q01, "q99": q99}

    def _load_or_compute_per_source_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Load per-source stats from action_stats_dir if available, else compute."""
        # Group episodes by source_key.
        by_source: Dict[str, List[Dict[str, Any]]] = {}
        for ep in self._all_episodes:
            key = ep["source_key"]
            by_source.setdefault(key, []).append(ep)

        result: Dict[str, Dict[str, np.ndarray]] = {}
        for key, episodes in by_source.items():
            stats_path = (
                self.action_stats_dir / f"action_stats_{key}.json"
                if self.action_stats_dir is not None
                else None
            )
            if stats_path is not None and stats_path.exists():
                with open(stats_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                loaded_stats = {
                    "q01": np.asarray(payload["q01"], dtype=np.float32),
                    "q99": np.asarray(payload["q99"], dtype=np.float32),
                }
                if self._stats_cache_is_compatible(payload, loaded_stats):
                    result[key] = loaded_stats
                    print(f"[SLIM][ACTION_STATS] Loaded cached stats for source '{key}': {stats_path}", flush=True)
                    continue
                print(
                    f"[SLIM][ACTION_STATS] Ignoring cached stats for source '{key}' "
                    f"because its action field, projection, normalization, or dimensions "
                    f"do not match: {stats_path}",
                    flush=True,
                )
            if key not in result:
                n = len(episodes)
                print(
                    f"[SLIM][ACTION_STATS] Computing stats for source '{key}' ({n} episodes) ...",
                    flush=True,
                )
                stats = self._compute_stats_for_episodes(episodes)
                result[key] = stats
                if stats_path is not None:
                    stats_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(stats_path, "w", encoding="utf-8") as f:
                        json.dump(self._stats_payload(stats), f, indent=2)
                    print(f"[SLIM][ACTION_STATS] Saved stats for source '{key}': {stats_path}", flush=True)
        return result

    def normalize_actions(self, actions: np.ndarray, source_key: str) -> np.ndarray:
        if self.action_normalization == "none" or not self._per_source_stats:
            return actions.astype(np.float32)
        if self.action_normalization != "q01_q99":
            raise ValueError(f"Unsupported action_normalization={self.action_normalization}")
        stats = self._per_source_stats.get(source_key)
        if stats is None:
            raise KeyError(
                f"No action stats for source_key={source_key!r}. "
                f"Available keys: {list(self._per_source_stats)}"
            )
        out = actions.astype(np.float32).copy()
        if not self._stats_match_action_dim(stats):
            raise ValueError(
                f"Action stats shape does not match action_dim={self.action_dim}: "
                f"q01={np.asarray(stats['q01']).shape}, q99={np.asarray(stats['q99']).shape}"
            )
        if self.action_normalized_dims:
            dims = np.asarray(self.action_normalized_dims, dtype=np.int64)
            q01 = stats["q01"][dims]
            q99 = stats["q99"][dims]
            scale = np.maximum(q99 - q01, 1e-6)
            out[..., dims] = 2.0 * ((out[..., dims] - q01) / scale) - 1.0
            out[..., dims] = np.clip(out[..., dims], -1.0, 1.0)
        return out

    def get_action_stats(self) -> Optional[Dict[str, Dict[str, List[float]]]]:
        """Return {source_key: {"q01": [...], "q99": [...]}} for val-split reuse."""
        if not self._per_source_stats:
            return None
        return {
            key: {
                "q01": v["q01"].tolist(),
                "q99": v["q99"].tolist(),
            }
            for key, v in self._per_source_stats.items()
        }

    def save_action_stats(self, stats_dir: str | Path | None = None) -> None:
        """Write per-source stats files to stats_dir (defaults to self.action_stats_dir)."""
        target = Path(stats_dir) if stats_dir is not None else self.action_stats_dir
        if target is None or not self._per_source_stats:
            return
        target.mkdir(parents=True, exist_ok=True)
        for key, stats in self._per_source_stats.items():
            p = target / f"action_stats_{key}.json"
            with open(p, "w", encoding="utf-8") as f:
                json.dump(
                    self._stats_payload(stats),
                    f,
                    indent=2,
                )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self.use_fake_data:
            return self._fake_len
        return len(self.sample_index)

    def _build_sample(self, sample_index: int) -> Dict[str, Any]:
        ep_idx, t = self.sample_index[sample_index]
        ep_meta = self.episodes[ep_idx]
        arrays = self._load_episode_arrays(ep_meta)
        actions_raw = arrays["action"][t : t + self.action_horizon]
        states_raw = arrays["state"][t]

        actions = self._project_action(actions_raw)
        actions = self.normalize_actions(actions, ep_meta["source_key"])
        state = self._project_state(states_raw)[None, :]
        frame_ts = float(arrays["timestamp"][t])

        # Use per-episode video_keys so each source can have different cameras.
        video_keys = ep_meta["video_keys"]
        frames = [
            self._load_rgb_frame(ep_meta, video_key=vk, frame_index=t, frame_ts=frame_ts)
            for vk in video_keys
        ]
        task_text = ep_meta["tasks"][0] if ep_meta["tasks"] else ""

        sample = {
            "image": frames,
            "action": actions,
            "state": state,
            "lang": task_text,
            "dataset_name": ep_meta["dataset_name"],
        }
        if self.include_future_image:
            offset = max(int(self.action_horizon), 1)
            future_t = min(t + offset, int(ep_meta["length"]) - 1)
            future_ts = float(arrays["timestamp"][future_t])
            sample["future_image"] = [
                self._load_rgb_frame(
                    ep_meta,
                    video_key=vk,
                    frame_index=future_t,
                    frame_ts=future_ts,
                )
                for vk in video_keys
            ]
        return sample

    def _log_frame_load_error(self, sample_index: int, error: BaseException) -> None:
        ep_idx, t = self.sample_index[sample_index]
        ep_meta = self.episodes[ep_idx]
        video_keys = ep_meta["video_keys"]
        if self.video_backend == "frames":
            bad_paths = [self._build_frame_path(ep_meta, vk, t) for vk in video_keys]
            media_label = "frame_paths"
        else:
            bad_paths = [self._build_video_path(ep_meta, vk) for vk in video_keys]
            media_label = "video_paths"
        print(
            "\n" + "=" * 100 + "\n"
            + "[SLIM][FRAME_LOAD_ERROR] skipping corrupted sample/episode\n"
            + f"split={self.split} sample_index={sample_index} "
            + f"episode_index={ep_meta['episode_index']} t={t}\n"
            + f"video_backend={self.video_backend}\n"
            + f"{media_label}:\n  - "
            + "\n  - ".join(str(p) for p in bad_paths)
            + f"\nerror={type(error).__name__}: {error}\n"
            + "=" * 100,
            flush=True,
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.use_fake_data:
            fake = np.random.uniform(-1, 1, size=(self.action_horizon, self.action_dim)).astype(np.float32)
            state = np.zeros((1, self.state_dim), dtype=np.float32)
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), mode="RGB")
            sample = {
                "image": [image] * self._fake_num_views,
                "action": fake,
                "state": state,
                "lang": "fake task",
                "dataset_name": "fake_dataset",
            }
            if self.include_future_image:
                sample["future_image"] = [image] * self._fake_num_views
            return sample

        total = len(self.sample_index)
        if total <= 0:
            raise RuntimeError("Dataset is empty.")

        start_index = index % total
        max_decode_failures = min(
            total,
            max(1, int(os.environ.get("SLIM_VIDEO_DECODE_MAX_RETRIES", "16"))),
        )
        last_error: Optional[BaseException] = None
        decode_failures = 0
        offset = 0

        while offset < total and decode_failures < max_decode_failures:
            curr_index = (start_index + offset) % total
            offset += 1
            ep_idx, _t = self.sample_index[curr_index]
            if ep_idx in _BAD_EPISODE_INDICES:
                continue

            try:
                return self._build_sample(curr_index)
            except (MemoryError, av.error.MemoryError) as e:
                gc.collect()
                raise RuntimeError(
                    "Video decode ran out of host memory. "
                    "Please reduce DataLoader pressure (num_workers/prefetch_factor/batch size)."
                ) from e
            except av.error.FFmpegError as e:
                if self.video_backend == "frames":
                    raise
                last_error = e
                decode_failures += 1
                _BAD_EPISODE_INDICES.add(ep_idx)
                _CONTAINER_CACHE.close_all()
                ep_meta = self.episodes[ep_idx]
                for vk in ep_meta["video_keys"]:
                    _CONTAINER_CACHE.invalidate(
                        str(self._build_video_path(ep_meta, vk))
                    )
                self._log_frame_load_error(curr_index, e)
            except (FileNotFoundError, OSError) as e:
                if self.video_backend != "frames":
                    raise
                last_error = e
                decode_failures += 1
                _BAD_EPISODE_INDICES.add(ep_idx)
                self._log_frame_load_error(curr_index, e)

        raise RuntimeError(
            f"Failed to load a training sample after scanning {offset} indices "
            f"and {decode_failures} video decode failures "
            f"(start_index={start_index}, bad_episodes_in_worker={len(_BAD_EPISODE_INDICES)}). "
            "Set SLIM_VIDEO_DECODE_MAX_RETRIES higher or add bad episodes to skip_lerobot_episodes."
        ) from last_error
