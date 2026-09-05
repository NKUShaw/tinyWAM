from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pathlib
import re
from dataclasses import asdict, dataclass, field

import imageio
import numpy as np
import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from slim.evaluation.libero.environment import ModelClient

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LIBERO_PLUS_TASKS = ("libero_10", "libero_spatial", "libero_object", "libero_goal")
LIBERO_PLUS_TASK_COUNTS = {
    "libero_10": 2519,
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
}


@dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    checkpoint: str = ""
    action_stats_path: str = ""
    action_chunk_size: int = 8
    resize_h: int = 256
    resize_w: int = 256
    image_views: str = "dual"
    dataset_name: str = ""
    send_state: bool = True

    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    episode_start_idx: int = 0
    episode_end_idx: int = -1
    task_start_idx: int = 0
    task_end_idx: int = -1
    task_ids: list[int] = field(default_factory=list)
    task_ids_file: str = ""
    resume: int = 1

    video_out_path: str = "evaluation_outputs/libero_plus"
    save_video: int = 1
    seed: int = 7


def eval_libero_plus(args: Args) -> None:
    logging.info("Arguments: %s", json.dumps(asdict(args), indent=2))
    np.random.seed(args.seed)

    if args.task_suite_name == "all":
        root = pathlib.Path(args.video_out_path)
        for suite_name in LIBERO_PLUS_TASKS:
            suite_args = Args(**asdict(args))
            suite_args.task_suite_name = suite_name
            suite_args.video_out_path = str(root / suite_name)
            _eval_one_suite(suite_args)
        return

    _eval_one_suite(args)


def _eval_one_suite(args: Args) -> None:
    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name not in benchmark_dict:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    expected_count = LIBERO_PLUS_TASK_COUNTS.get(args.task_suite_name)
    if expected_count is not None and expected_count != num_tasks_in_suite:
        logging.warning(
            "LIBERO-plus suite %s reports %d tasks, expected %d from local constants.",
            args.task_suite_name,
            num_tasks_in_suite,
            expected_count,
        )

    video_out_dir = pathlib.Path(args.video_out_path)
    video_out_dir.mkdir(parents=True, exist_ok=True)
    task_ids = _resolve_task_ids(args, num_tasks_in_suite)
    max_steps = _max_steps_for_suite(args.task_suite_name)

    logging.info(
        "Starting LIBERO-plus suite=%s tasks=%d output=%s",
        args.task_suite_name,
        len(task_ids),
        video_out_dir,
    )

    client_model = ModelClient(
        policy_ckpt_path=args.checkpoint,
        action_stats_path=(args.action_stats_path if args.action_stats_path else None),
        action_chunk_size=args.action_chunk_size,
        host=args.host,
        port=args.port,
        image_size=[args.resize_h, args.resize_w],
    )

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_ids, desc=f"{args.task_suite_name}_tasks", disable=True):
        task = task_suite.get_task(task_id)
        task_description = task.language
        initial_states = task_suite.get_task_init_states(task_id)
        pending_episodes = _resolve_episode_ids(args, len(initial_states))
        if args.resume:
            pending_episodes = [
                episode_idx
                for episode_idx in pending_episodes
                if not any(
                    video_out_dir.glob(
                        f"rollout_{args.task_suite_name}_task{task_id}_episode{episode_idx}_*.txt"
                    )
                )
            ]
        if not pending_episodes:
            logging.info("Skip completed task_id=%d (%s)", task_id, task_description)
            continue

        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        task_episodes, task_successes = 0, 0
        try:
            for episode_idx in pending_episodes:
                done = _rollout_episode(
                    args=args,
                    client_model=client_model,
                    env=env,
                    initial_state=initial_states[episode_idx],
                    task_description=task_description,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    max_steps=max_steps,
                    video_out_dir=video_out_dir,
                )
                task_episodes += 1
                total_episodes += 1
                if done:
                    task_successes += 1
                    total_successes += 1
        finally:
            try:
                env.close()
            except Exception:
                pass

        if task_episodes > 0:
            logging.info(
                "[suite=%s task_id=%d] success_rate=%.4f (%d/%d)",
                args.task_suite_name,
                task_id,
                task_successes / max(task_episodes, 1),
                task_successes,
                task_episodes,
            )

    logging.info(
        "[suite=%s] total_success_rate=%.4f (%d/%d)",
        args.task_suite_name,
        total_successes / max(total_episodes, 1),
        total_successes,
        total_episodes,
    )


def _rollout_episode(
    *,
    args: Args,
    client_model: ModelClient,
    env,
    initial_state,
    task_description: str,
    task_id: int,
    episode_idx: int,
    max_steps: int,
    video_out_dir: pathlib.Path,
) -> bool:
    logging.info("Task %d episode %d: %s", task_id, episode_idx, task_description)
    client_model.reset(task_description=task_description)
    env.reset()
    obs = env.set_init_state(initial_state)

    done = False
    step = 0
    replay_images: list[np.ndarray] = []

    for t in range(max_steps + args.num_steps_wait):
        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            if done:
                break
            continue

        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        if args.save_video:
            replay_images.append(img)
        if args.image_views == "dual":
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            images = [img, wrist_img]
        elif args.image_views == "single":
            images = [img]
        else:
            raise ValueError(f"Unsupported image_views={args.image_views}, expected single|dual")

        state = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"][:1],
            )
        ).astype(np.float32)

        example = {
            "image": images,
            "lang": str(task_description),
            "dataset_name": args.dataset_name,
        }
        if args.send_state:
            example["state"] = np.expand_dims(state, axis=0)
        response = client_model.step(example=example, step=step)
        raw_action = response["raw_action"]
        world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
        rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
        open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
        gripper = _binarize_gripper_open(open_gripper)

        if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
            raise ValueError(
                "Invalid action sizes: "
                f"world_vector={world_vector_delta.shape}, "
                f"rotation_delta={rotation_delta.shape}, gripper={open_gripper.shape}"
            )

        delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)
        obs, _, done, _ = env.step(delta_action.tolist())
        if done:
            break
        step += 1

    suffix = "success" if done else "failure"
    video_path = video_out_dir / f"rollout_{args.task_suite_name}_task{task_id}_episode{episode_idx}_{suffix}.mp4"
    txt_path = video_out_dir / f"rollout_{args.task_suite_name}_task{task_id}_episode{episode_idx}_{suffix}.txt"
    opposite_suffix = "failure" if done else "success"
    opposite_txt_path = (
        video_out_dir
        / f"rollout_{args.task_suite_name}_task{task_id}_episode{episode_idx}_{opposite_suffix}.txt"
    )
    opposite_txt_path.unlink(missing_ok=True)
    if args.save_video:
        if replay_images:
            imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)
        else:
            logging.warning("No replay frames captured for task_id=%d episode=%d", task_id, episode_idx)
    txt_path.write_text(
        "\n".join(
            [
                f"Task suite: {args.task_suite_name}",
                f"Task ID: {task_id}",
                f"Task: {task_description}",
                f"Episode: {episode_idx}",
                f"Success: {done}",
                f"Video: {video_path if args.save_video else 'disabled'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    logging.info("Saved rollout marker: %s", txt_path)
    return bool(done)


def _resolve_task_ids(args: Args, num_tasks_in_suite: int) -> list[int]:
    requested: list[int] = []
    requested.extend(int(x) for x in args.task_ids)
    if args.task_ids_file:
        raw = pathlib.Path(args.task_ids_file).read_text(encoding="utf-8")
        requested.extend(int(x) for x in re.split(r"[\s,]+", raw.strip()) if x)

    if requested:
        valid = [task_id for task_id in dict.fromkeys(requested) if 0 <= task_id < num_tasks_in_suite]
        invalid = [task_id for task_id in requested if task_id < 0 or task_id >= num_tasks_in_suite]
        if invalid:
            logging.warning("Ignoring invalid task ids: %s", invalid[:20])
        if not valid:
            raise ValueError(f"No valid task ids for suite with {num_tasks_in_suite} tasks")
        return valid

    start = max(0, int(args.task_start_idx))
    end = int(args.task_end_idx)
    if end < 0 or end >= num_tasks_in_suite:
        end = num_tasks_in_suite - 1
    if start > end:
        raise ValueError(f"Invalid task range {start}..{end} for suite with {num_tasks_in_suite} tasks")
    return list(range(start, end + 1))


def _resolve_episode_ids(args: Args, num_initial_states: int) -> list[int]:
    start = max(0, int(args.episode_start_idx))
    if args.episode_end_idx is None or int(args.episode_end_idx) < 0:
        end = int(args.num_trials_per_task) - 1
    else:
        end = min(int(args.episode_end_idx), int(args.num_trials_per_task) - 1)
    end = min(end, num_initial_states - 1)
    if start > end:
        return []
    return list(range(start, end + 1))


def _max_steps_for_suite(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220 + 120
    if task_suite_name == "libero_object":
        return 280 + 120
    if task_suite_name == "libero_goal":
        return 300 + 120
    if task_suite_name == "libero_10":
        return 520 + 120
    if task_suite_name == "libero_90":
        return 400 + 120
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


def _get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float32).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Evaluate SLIM on LIBERO-plus.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--action_stats_path", type=str, default="")
    parser.add_argument("--action_chunk_size", type=int, default=8)
    parser.add_argument("--resize_h", type=int, default=256)
    parser.add_argument("--resize_w", type=int, default=256)
    parser.add_argument("--image_views", type=str, default="dual", choices=["single", "dual"])
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument(
        "--send-state",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--task_suite_name", type=str, default="libero_goal")
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--num_trials_per_task", type=int, default=1)
    parser.add_argument("--episode_start_idx", type=int, default=0)
    parser.add_argument("--episode_end_idx", type=int, default=-1)
    parser.add_argument("--task_start_idx", type=int, default=0)
    parser.add_argument("--task_end_idx", type=int, default=-1)
    parser.add_argument("--task_ids", type=int, nargs="*", default=[])
    parser.add_argument("--task_ids_file", type=str, default="")
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--video_out_path", type=str, default="evaluation_outputs/libero_plus")
    parser.add_argument("--save_video", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    ns = parser.parse_args()
    return Args(**vars(ns))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    eval_libero_plus(parse_args())
