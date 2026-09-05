"""Evaluate a SLIM policy server on CALVIN's long-horizon benchmark."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

# Official CALVIN still uses aliases removed in NumPy 1.24.
for _name, _value in {
    "float": float,
    "int": int,
    "bool": bool,
    "object": object,
    "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)

import hydra
from omegaconf import OmegaConf
from tqdm import tqdm

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
from calvin_env.envs.play_table_env import get_env

from slim.evaluation.calvin.environment import CalvinModelClient
from slim.evaluation.calvin.results import (
    aggregate_shards,
    summarize,
    write_json_atomic,
    write_shard,
)


EPISODE_LENGTH = 360
TASK_OBSERVATION_SPACE = {
    "rgb_obs": ["rgb_static", "rgb_gripper"],
    "depth_obs": [],
}


def get_calvin_config_dir() -> Path:
    import calvin_agent

    return Path(calvin_agent.__file__).resolve().parents[1] / "conf"


def make_environment(dataset_path: str):
    validation_dir = Path(dataset_path) / "validation"
    if not validation_dir.is_dir():
        raise FileNotFoundError(
            f"CALVIN validation directory not found: {validation_dir}"
        )
    return get_env(
        validation_dir,
        obs_space=TASK_OBSERVATION_SPACE,
        show_gui=False,
    )


def rollout(env, model, task_oracle, subtask: str, annotation: str) -> bool:
    observation = env.get_obs()
    start_info = env.get_info()
    model.reset()

    for _ in range(EPISODE_LENGTH):
        action = model.step(observation, annotation)
        observation, _, _, current_info = env.step(action)
        completed = task_oracle.get_task_info_for_set(
            start_info, current_info, {subtask}
        )
        if completed:
            return True
    return False


def evaluate_sequence(
    env,
    model,
    task_oracle,
    validation_annotations,
    initial_state,
    sequence,
) -> int:
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    completed = 0
    for subtask in sequence:
        annotation = str(validation_annotations[subtask][0])
        if not rollout(env, model, task_oracle, subtask, annotation):
            break
        completed += 1
    return completed


def evaluate(args) -> dict:
    if args.num_sequences <= 0:
        raise ValueError("--num-sequences must be positive")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")

    random.seed(args.seed)
    np.random.seed(args.seed)

    config_dir = get_calvin_config_dir()
    task_config = OmegaConf.load(
        config_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_config)
    annotations = OmegaConf.load(
        config_dir / "annotations/new_playtable_validation.yaml"
    )
    sequences = get_sequences(args.num_sequences)

    env = make_environment(args.dataset_path)
    model = CalvinModelClient(
        action_stats_path=args.action_stats,
        dataset_name=args.dataset_name,
        action_horizon=args.action_horizon,
        exec_stride=args.exec_stride,
        host=args.host,
        port=args.port,
    )
    result_by_index = {}
    sequence_indices = range(
        args.shard_index,
        args.num_sequences,
        args.num_shards,
    )
    try:
        for sequence_index in tqdm(
            sequence_indices,
            desc=f"CALVIN shard {args.shard_index + 1}/{args.num_shards}",
        ):
            initial_state, sequence = sequences[sequence_index]
            result_by_index[sequence_index] = evaluate_sequence(
                env,
                model,
                task_oracle,
                annotations,
                initial_state,
                sequence,
            )
    finally:
        model.close()
        if hasattr(env, "close"):
            env.close()
            # CALVIN's destructor calls close() a second time without clearing
            # the client id, which otherwise raises a harmless PyBullet error.
            if hasattr(env, "cid"):
                env.cid = -1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.num_shards > 1:
        output_path = write_shard(
            output_dir,
            num_sequences=args.num_sequences,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            results=result_by_index,
        )
        print(
            f"Wrote {len(result_by_index)} sequence results to {output_path}"
        )
        return {
            "num_sequences": len(result_by_index),
            "shard_index": args.shard_index,
            "results": result_by_index,
        }

    summary = summarize([result_by_index[index] for index in range(args.num_sequences)])
    output_path = output_dir / "summary.json"
    write_json_atomic(output_path, summary)

    rates = summary["success_rates"]
    print(
        "CALVIN "
        + " ".join(f"SR@{length}={rates[str(length)]:.3f}" for length in range(1, 6))
        + f" Avg. Length={summary['avg_length']:.3f}"
    )
    print(f"Wrote {output_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a running SLIM policy server on CALVIN."
    )
    parser.add_argument("--dataset-path")
    parser.add_argument("--action-stats")
    parser.add_argument("--dataset-name", default="calvin_ABC_D_lerobot")
    parser.add_argument("--num-sequences", type=int, default=1000)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Strictly aggregate completed shard files without running simulation.",
    )
    parser.add_argument("--action-horizon", type=int, default=12)
    parser.add_argument("--exec-stride", type=int, default=12)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--output-dir", default="evaluation_outputs/calvin")
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.aggregate:
        summary = aggregate_shards(
            args.output_dir,
            num_sequences=args.num_sequences,
            num_shards=args.num_shards,
        )
        rates = summary["success_rates"]
        print(
            "CALVIN "
            + " ".join(
                f"SR@{length}={rates[str(length)]:.3f}" for length in range(1, 6)
            )
            + f" Avg. Length={summary['avg_length']:.3f}"
        )
        print(f"Wrote {Path(args.output_dir) / 'summary.json'}")
    else:
        if not args.dataset_path or not args.action_stats:
            raise SystemExit(
                "--dataset-path and --action-stats are required unless --aggregate is used"
            )
        evaluate(args)
