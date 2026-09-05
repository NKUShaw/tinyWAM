import numpy as np

from slim.data.dataset import LeRobotBaseDataset


def _dataset_for_stats_check():
    dataset = LeRobotBaseDataset.__new__(LeRobotBaseDataset)
    dataset.action_dim = 7
    dataset.action_key = "action"
    dataset.action_indices = None
    dataset.action_normalization = "q01_q99"
    dataset.action_normalized_dims = tuple(range(6))
    return dataset


def test_action_stats_cache_requires_matching_projection_metadata():
    dataset = _dataset_for_stats_check()
    stats = {
        "q01": np.zeros(7, dtype=np.float32),
        "q99": np.ones(7, dtype=np.float32),
    }
    payload = {
        "action_key": "action",
        "action_indices": None,
        "normalization": "q01_q99",
        "action_normalized_dims": list(range(6)),
    }

    assert dataset._stats_cache_is_compatible(payload, stats)

    payload["action_indices"] = list(range(7))
    assert not dataset._stats_cache_is_compatible(payload, stats)
