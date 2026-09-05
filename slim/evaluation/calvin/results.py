"""Result persistence and strict shard aggregation for CALVIN evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence


def summarize(results: Sequence[int]) -> dict:
    if not results:
        raise ValueError("Cannot summarize an empty CALVIN evaluation")
    normalized = [int(result) for result in results]
    if any(result < 0 or result > 5 for result in normalized):
        raise ValueError("CALVIN sequence results must be integers in [0, 5]")
    success_rates = {
        str(length): sum(result >= length for result in normalized) / len(normalized)
        for length in range(1, 6)
    }
    return {
        "num_sequences": len(normalized),
        "success_rates": success_rates,
        "avg_length": sum(normalized) / len(normalized),
        "results": normalized,
    }


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_shard(
    output_dir: str | Path,
    *,
    num_sequences: int,
    num_shards: int,
    shard_index: int,
    results: Mapping[int, int],
) -> Path:
    payload = {
        "num_sequences": int(num_sequences),
        "num_shards": int(num_shards),
        "shard_index": int(shard_index),
        "results": {str(index): int(value) for index, value in sorted(results.items())},
    }
    path = Path(output_dir) / f"shard_{shard_index:03d}.json"
    write_json_atomic(path, payload)
    return path


def aggregate_shards(
    output_dir: str | Path,
    *,
    num_sequences: int,
    num_shards: int,
) -> dict:
    if num_sequences <= 0:
        raise ValueError("num_sequences must be positive")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")

    output_dir = Path(output_dir)
    merged: dict[int, int] = {}
    for shard_index in range(num_shards):
        path = output_dir / f"shard_{shard_index:03d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing CALVIN result shard: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_metadata = {
            "num_sequences": num_sequences,
            "num_shards": num_shards,
            "shard_index": shard_index,
        }
        for key, expected in expected_metadata.items():
            if int(payload.get(key, -1)) != expected:
                raise ValueError(
                    f"Invalid {key} in {path}: expected {expected}, "
                    f"got {payload.get(key)!r}"
                )
        for raw_index, raw_result in payload.get("results", {}).items():
            index = int(raw_index)
            result = int(raw_result)
            if index < 0 or index >= num_sequences:
                raise ValueError(f"Out-of-range sequence index {index} in {path}")
            if index % num_shards != shard_index:
                raise ValueError(f"Sequence index {index} belongs to a different shard")
            if result < 0 or result > 5:
                raise ValueError(f"Out-of-range sequence result {result} in {path}")
            if index in merged:
                raise ValueError(f"Duplicate sequence index {index} across shards")
            merged[index] = result

    missing = sorted(set(range(num_sequences)) - set(merged))
    if missing:
        raise ValueError(
            f"CALVIN shard aggregation is incomplete: {len(missing)} sequences "
            f"missing, beginning with {missing[:10]}"
        )

    summary = summarize([merged[index] for index in range(num_sequences)])
    write_json_atomic(output_dir / "summary.json", summary)
    return summary
