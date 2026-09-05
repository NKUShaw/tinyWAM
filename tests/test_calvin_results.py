import json

import pytest

from slim.evaluation.calvin.results import aggregate_shards, summarize, write_shard


def test_calvin_summary():
    summary = summarize([5, 3, 1, 0])

    assert summary["avg_length"] == 2.25
    assert summary["success_rates"] == {
        "1": 0.75,
        "2": 0.5,
        "3": 0.5,
        "4": 0.25,
        "5": 0.25,
    }


def test_calvin_shards_are_strictly_aggregated(tmp_path):
    write_shard(
        tmp_path,
        num_sequences=4,
        num_shards=2,
        shard_index=0,
        results={0: 5, 2: 1},
    )
    write_shard(
        tmp_path,
        num_sequences=4,
        num_shards=2,
        shard_index=1,
        results={1: 3, 3: 0},
    )

    summary = aggregate_shards(tmp_path, num_sequences=4, num_shards=2)

    assert summary["results"] == [5, 3, 1, 0]
    assert json.loads((tmp_path / "summary.json").read_text()) == summary


def test_calvin_aggregation_rejects_missing_sequences(tmp_path):
    write_shard(
        tmp_path,
        num_sequences=2,
        num_shards=1,
        shard_index=0,
        results={0: 1},
    )

    with pytest.raises(ValueError, match="incomplete"):
        aggregate_shards(tmp_path, num_sequences=2, num_shards=1)
