from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pathlib
from dataclasses import asdict, dataclass

import imageio
import numpy as np
import tqdm
import yaml
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from slim.evaluation.libero.environment import ModelClient
from slim.compat.legacy import to_public_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LIBERO_SUITE_TO_DATASET_NAME = {
    "libero_object": "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal": "libero_goal_no_noops_1.0.0_lerobot",
    "libero_spatial": "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_10": "libero_10_no_noops_1.0.0_lerobot",
}


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    checkpoint: str = ""
    action_stats_path: str = ""
    action_chunk_size: int = 8
    resize_h: int = 256
    resize_w: int = 256
    image_views: str = "dual"  # "single" or "dual"
    state_representation: str = "auto"  # auto, axis_angle, or quaternion
    send_state: bool = True

    task_suite_name: str = "libero_goal"
    task_start_idx: int = 0
    task_end_idx: int = -1
    num_steps_wait: int = 20
    num_trials_per_task: int = 50
    episode_start_idx: int = 0
    episode_end_idx: int = -1
    video_out_path: str = "evaluation_outputs/libero"
    save_video: int = 1
    seed: int = 7


def eval_libero(args: Args) -> None:
    logging.info("Arguments: %s", json.dumps(asdict(args), indent=2))
    np.random.seed(args.seed)

    state_representation, state_key, config_path = _resolve_state_representation(
        requested=args.state_representation,
        checkpoint=args.checkpoint,
    )
    expected_state_dim = 8 if state_representation == "quaternion" else 7
    logging.info(
        "State representation: %s (dim=%d, state_key=%s, config=%s)",
        state_representation,
        expected_state_dim,
        state_key or "<unknown>",
        str(config_path) if config_path is not None else "<not found>",
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    max_steps = _max_steps_for_suite(args.task_suite_name)
    client_model = ModelClient(
        policy_ckpt_path=args.checkpoint,
        action_stats_path=(args.action_stats_path if args.action_stats_path else None),
        action_chunk_size=args.action_chunk_size,
        host=args.host,
        port=args.port,
        image_size=[args.resize_h, args.resize_w],
    )

    total_episodes, total_successes = 0, 0
    eval_dataset_name = _dataset_name_for_suite(args.task_suite_name)
    task_start_idx = max(0, int(args.task_start_idx))
    task_end_idx = (
        num_tasks_in_suite - 1
        if int(args.task_end_idx) < 0
        else min(int(args.task_end_idx), num_tasks_in_suite - 1)
    )
    if task_start_idx > task_end_idx:
        raise ValueError(
            f"Invalid task range start={task_start_idx} end={task_end_idx} "
            f"for suite with {num_tasks_in_suite} tasks."
        )
    for task_id in tqdm.tqdm(
        range(task_start_idx, task_end_idx + 1), desc="tasks", disable=True
    ):
        task = task_suite.get_task(task_id)
        logging.info("Starting task %d/%d: %s", task_id + 1, num_tasks_in_suite, task.language)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        start_idx = max(0, int(args.episode_start_idx))
        if args.episode_end_idx is None or int(args.episode_end_idx) < 0:
            end_idx = int(args.num_trials_per_task) - 1
        else:
            end_idx = min(int(args.episode_end_idx), int(args.num_trials_per_task) - 1)

        if start_idx > end_idx:
            logging.warning(
                "Skip task %d: invalid episode range start=%d end=%d num_trials_per_task=%d",
                task_id,
                start_idx,
                end_idx,
                args.num_trials_per_task,
            )
            try:
                env.close()
            except Exception:
                pass
            continue

        for episode_idx in tqdm.tqdm(range(start_idx, end_idx + 1), desc=f"task_{task_id}_episodes", disable=True):
            client_model.reset(task_description=task_description)
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            previous_quat = None

            step = 0
            done = False
            replay_images = []
            step_pbar = tqdm.tqdm(
                range(max_steps + args.num_steps_wait),
                desc=f"task_{task_id}_ep_{episode_idx}_steps",
                leave=False,
                disable=True,
            )
            for t in step_pbar:
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    continue

                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                replay_images.append(img)
                if args.image_views == "dual":
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    images = [img, wrist_img]
                elif args.image_views == "single":
                    images = [img]
                else:
                    raise ValueError(f"Unsupported image_views={args.image_views}, expected single|dual")

                state, previous_quat = _build_robot_state(
                    obs=obs,
                    representation=state_representation,
                    previous_quat=previous_quat,
                )
                if state.shape != (expected_state_dim,):
                    raise ValueError(
                        f"Invalid {state_representation} state shape: "
                        f"expected {(expected_state_dim,)}, got {state.shape}"
                    )
                example_dict = {
                    "image": images,
                    "lang": str(task_description),
                    "dataset_name": eval_dataset_name,
                }
                if args.send_state:
                    example_dict["state"] = np.expand_dims(state, axis=0)
                response = client_model.step(example=example_dict, step=step)
                raw_action = response["raw_action"]
                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    raise ValueError(
                        f"Invalid action sizes: "
                        f"world_vector={world_vector_delta.shape}, rotation_delta={rotation_delta.shape}, "
                        f"gripper={open_gripper.shape}"
                    )

                delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)
                obs, _, done, _ = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                step += 1

            task_episodes += 1
            total_episodes += 1
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4"
            marker_path = (
                pathlib.Path(args.video_out_path)
                / f"rollout_{args.task_suite_name}_task{task_id}_episode{episode_idx}_{suffix}.txt"
            )
            opposite_suffix = "failure" if done else "success"
            opposite_marker = (
                pathlib.Path(args.video_out_path)
                / f"rollout_{args.task_suite_name}_task{task_id}_episode{episode_idx}_{opposite_suffix}.txt"
            )
            opposite_marker.unlink(missing_ok=True)
            marker_path.write_text(
                json.dumps(
                    {
                        "suite": args.task_suite_name,
                        "task_id": task_id,
                        "episode": episode_idx,
                        "success": bool(done),
                        "task": task_description,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if args.save_video:
                imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)
                logging.info("Saved replay video to: %s", video_path)

        logging.info(
            "[task=%s] success_rate=%.4f (%d/%d)",
            task_description,
            float(task_successes) / float(max(task_episodes, 1)),
            task_successes,
            task_episodes,
        )
        try:
            env.close()
        except Exception:
            pass

    logging.info(
        "Total success rate: %.4f (%d/%d)",
        float(total_successes) / float(max(total_episodes, 1)),
        total_successes,
        total_episodes,
    )


def _max_steps_for_suite(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _dataset_name_for_suite(task_suite_name: str) -> str:
    dataset_name = LIBERO_SUITE_TO_DATASET_NAME.get(task_suite_name, "")
    if not dataset_name:
        logging.warning(
            "No offline dataset_name mapping for task_suite=%s. Language will likely fallback to online encoding.",
            task_suite_name,
        )
    return dataset_name


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
    """Copied from robosuite's quaternion-to-axis-angle helper."""
    quat = np.asarray(quat, dtype=np.float32).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)

    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def _find_checkpoint_config(checkpoint: str) -> pathlib.Path | None:
    checkpoint_path = pathlib.Path(checkpoint).expanduser()
    candidates = []
    if checkpoint_path.is_dir():
        candidates.extend([checkpoint_path / "config.yaml", checkpoint_path.parent / "config.yaml"])
    else:
        candidates.extend(
            [
                checkpoint_path.parent.parent / "config.yaml",
                checkpoint_path.parent / "config.yaml",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _read_state_key(config_path: pathlib.Path) -> str | None:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config = to_public_config(config)
    state_key = config.get("data", {}).get("state_key")
    return str(state_key).strip() if state_key else None


def _resolve_state_representation(
    requested: str,
    checkpoint: str,
) -> tuple[str, str | None, pathlib.Path | None]:
    requested = str(requested).strip().lower()
    if requested not in {"auto", "axis_angle", "quaternion"}:
        raise ValueError(
            f"Unsupported state_representation={requested}; expected auto|axis_angle|quaternion"
        )

    config_path = _find_checkpoint_config(checkpoint)
    state_key = None
    if config_path is not None:
        try:
            state_key = _read_state_key(config_path)
        except Exception as exc:
            logging.warning("Failed to read checkpoint config %s: %s", config_path, exc)

    if requested != "auto":
        return requested, state_key, config_path

    if state_key and "quat" in state_key.lower():
        return "quaternion", state_key, config_path
    if state_key == "observation.state":
        return "axis_angle", state_key, config_path

    logging.warning(
        "Could not determine state representation from checkpoint config/state_key; "
        "falling back to legacy axis_angle. Use --state_representation to override."
    )
    return "axis_angle", state_key, config_path


def _normalize_continuous_quaternion(
    quat: np.ndarray,
    previous_quat: np.ndarray | None,
) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(-1)
    if quat.shape != (4,):
        raise ValueError(f"Expected xyzw quaternion with shape (4,), got {quat.shape}")
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1.0e-8:
        raise ValueError(f"Invalid quaternion norm: {norm}")
    quat = quat / norm
    if previous_quat is None:
        if quat[3] < 0.0:
            quat = -quat
    elif float(np.dot(previous_quat, quat)) < 0.0:
        quat = -quat
    return quat.astype(np.float32)


def _build_robot_state(
    obs: dict,
    representation: str,
    previous_quat: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    position = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)[:1]
    if position.shape != (3,) or gripper.shape != (1,):
        raise ValueError(f"Invalid position/gripper shapes: {position.shape}, {gripper.shape}")

    if representation == "quaternion":
        quat = _normalize_continuous_quaternion(obs["robot0_eef_quat"], previous_quat)
        return np.concatenate([position, quat, gripper]).astype(np.float32), quat
    if representation == "axis_angle":
        axis_angle = _quat2axisangle(obs["robot0_eef_quat"])
        return np.concatenate([position, axis_angle, gripper]).astype(np.float32), None
    raise ValueError(f"Unsupported state representation: {representation}")


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--action_stats_path", type=str, default="")
    parser.add_argument("--action_chunk_size", type=int, default=8)
    parser.add_argument("--resize_h", type=int, default=256)
    parser.add_argument("--resize_w", type=int, default=256)
    parser.add_argument("--image_views", type=str, default="dual", choices=["single", "dual"])
    parser.add_argument(
        "--state_representation",
        type=str,
        default="auto",
        choices=["auto", "axis_angle", "quaternion"],
    )
    parser.add_argument(
        "--send-state",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--task_suite_name", type=str, default="libero_goal")
    parser.add_argument("--task_start_idx", type=int, default=0)
    parser.add_argument("--task_end_idx", type=int, default=-1)
    parser.add_argument("--num_steps_wait", type=int, default=20)
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--episode_start_idx", type=int, default=0)
    parser.add_argument("--episode_end_idx", type=int, default=-1)
    parser.add_argument("--video_out_path", type=str, default="evaluation_outputs/libero")
    parser.add_argument("--save_video", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    ns = parser.parse_args()
    return Args(**vars(ns))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    eval_libero(parse_args())
