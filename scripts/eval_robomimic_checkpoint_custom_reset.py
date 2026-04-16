#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import h5py
import imageio
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "repos" / "robomimic") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "repos" / "robomimic"))

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import RolloutPolicy
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper


TASK_OBJECT_STATE_INDICES = {
    "Stack": {
        "object": np.array([10, 11, 12], dtype=np.int64),
        "target": np.array([17, 18, 19], dtype=np.int64),
    },
}


def load_reset_state(source_demo_path: Path, source_episode_idx: int) -> dict:
    with h5py.File(source_demo_path, "r") as f:
        demos = list(f["data"].keys())
        if source_episode_idx < 0 or source_episode_idx >= len(demos):
            raise IndexError(
                f"source episode index {source_episode_idx} out of range for demos={demos}"
            )
        ep = demos[source_episode_idx]
        group = f[f"data/{ep}"]
        state = {
            "states": group["states"][0],
            "model": group.attrs["model_file"],
        }
        if "ep_meta" in group.attrs:
            state["ep_meta"] = group.attrs["ep_meta"]
        return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a robomimic checkpoint from a custom source-demo reset state, "
            "optionally recording videos and rollout datasets."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-demo", required=True)
    parser.add_argument("--source-episode", type=int, default=0)
    parser.add_argument("--object-translation", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--target-translation", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--n-rollouts", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=800)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--video-skip", type=int, default=5)
    parser.add_argument("--camera-names", nargs="+", default=["agentview"])
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--dataset-obs", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def apply_stack_translation(
    reset_state: dict,
    object_translation: np.ndarray,
    target_translation: np.ndarray,
) -> dict:
    translated = dict(reset_state)
    translated["states"] = np.asarray(reset_state["states"], dtype=np.float64).copy()
    translated["states"][TASK_OBJECT_STATE_INDICES["Stack"]["object"]] += np.asarray(
        object_translation[:3], dtype=np.float64
    )
    translated["states"][TASK_OBJECT_STATE_INDICES["Stack"]["target"]] += np.asarray(
        target_translation[:3], dtype=np.float64
    )
    return translated


def extract_eef_pos(obs: dict) -> np.ndarray | None:
    value = obs.get("robot0_eef_pos", None)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        return arr.copy()
    return arr.reshape(-1, arr.shape[-1])[-1].copy()


def rollout_from_custom_reset(
    *,
    policy: RolloutPolicy,
    env: EnvBase | EnvWrapper,
    reset_state: dict,
    horizon: int,
    render: bool = False,
    video_writer=None,
    video_skip: int = 5,
    return_obs: bool = False,
    camera_names: list[str] | None = None,
) -> tuple[dict, dict, dict]:
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    camera_names = camera_names or ["agentview"]
    policy.start_episode()
    env.reset()
    obs = env.reset_to(reset_state)
    state_dict = env.get_state()

    success = False
    video_count = 0
    total_reward = 0.0
    steps_taken = 0
    traj = dict(
        actions=[],
        rewards=[],
        dones=[],
        states=[],
        initial_state_dict=deepcopy(reset_state),
    )
    if return_obs:
        traj.update(dict(obs=[], next_obs=[]))

    actions_hist: list[np.ndarray] = []
    eef_hist: list[np.ndarray] = []
    eef_pos = extract_eef_pos(obs)
    if eef_pos is not None:
        eef_hist.append(eef_pos)

    try:
        for step_i in range(int(horizon)):
            action = policy(ob=obs)
            next_obs, reward, done, _ = env.step(action)

            total_reward += float(reward)
            success = bool(env.is_success()["task"])
            steps_taken = step_i + 1

            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % max(1, int(video_skip)) == 0:
                    frames = []
                    for camera_name in camera_names:
                        frames.append(
                            env.render(
                                mode="rgb_array",
                                height=512,
                                width=512,
                                camera_name=camera_name,
                            )
                        )
                    video_writer.append_data(np.concatenate(frames, axis=1))
                video_count += 1

            traj["actions"].append(action)
            traj["rewards"].append(reward)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                traj["obs"].append(obs)
                traj["next_obs"].append(next_obs)

            action_np = np.asarray(action, dtype=np.float32)
            actions_hist.append(action_np.copy())
            eef_pos = extract_eef_pos(next_obs)
            if eef_pos is not None:
                eef_hist.append(eef_pos)

            if done or success:
                break

            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as exc:
        print(f"WARNING: got rollout exception {exc}")

    stats = dict(Return=total_reward, Horizon=steps_taken, Success_Rate=float(success))

    if return_obs:
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    for key in traj:
        if key == "initial_state_dict":
            continue
        if isinstance(traj[key], dict):
            for subkey in traj[key]:
                traj[key][subkey] = np.asarray(traj[key][subkey])
        else:
            traj[key] = np.asarray(traj[key])

    actions_np = np.asarray(actions_hist, dtype=np.float32)
    eef_np = np.asarray(eef_hist, dtype=np.float32) if eef_hist else None
    diagnostics = {
        "steps": int(actions_np.shape[0]),
        "success": bool(success),
        "return": float(total_reward),
        "action_xyz_meanabs": np.mean(np.abs(actions_np[:, :3]), axis=0).round(6).tolist()
        if actions_np.size
        else None,
        "action_rot_meanabs": np.mean(np.abs(actions_np[:, 3:6]), axis=0).round(6).tolist()
        if actions_np.size
        else None,
        "action_grip_meanabs": np.mean(np.abs(actions_np[:, 6:]), axis=0).round(6).tolist()
        if actions_np.size
        else None,
        "eef_span": (eef_np.max(axis=0) - eef_np.min(axis=0)).round(6).tolist() if eef_np is not None else None,
        "eef_start": eef_np[0].round(6).tolist() if eef_np is not None else None,
        "eef_end": eef_np[-1].round(6).tolist() if eef_np is not None else None,
        "first10_xyz": actions_np[:10, :3].round(6).tolist() if actions_np.size else [],
    }
    return stats, traj, diagnostics


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    source_demo = Path(args.source_demo).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not source_demo.exists():
        raise FileNotFoundError(f"source demo not found: {source_demo}")

    write_video = args.video_path is not None
    write_dataset = args.dataset_path is not None
    if args.render and write_video:
        raise ValueError("render and video recording cannot both be enabled")
    if args.render and len(args.camera_names) != 1:
        raise ValueError("on-screen rendering only supports exactly one camera")

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=str(checkpoint),
        device=device,
        verbose=False,
    )
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=None,
        render=bool(args.render),
        render_offscreen=write_video,
        verbose=False,
    )

    reset_state = load_reset_state(source_demo, int(args.source_episode))
    env_name = getattr(env, "name", None) or ckpt_dict.get("env_metadata", {}).get("env_name")
    object_translation = np.asarray(args.object_translation, dtype=np.float32)
    target_translation = np.asarray(args.target_translation, dtype=np.float32)
    if np.any(np.abs(object_translation) > 0) or np.any(np.abs(target_translation) > 0):
        if env_name not in TASK_OBJECT_STATE_INDICES:
            raise ValueError(f"custom translation is not implemented for env={env_name}")
        reset_state = apply_stack_translation(
            reset_state=reset_state,
            object_translation=object_translation,
            target_translation=target_translation,
        )

    video_writer = None
    if write_video:
        video_path = Path(args.video_path).expanduser().resolve()
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(str(video_path), fps=20)

    data_writer = None
    data_grp = None
    total_samples = 0
    if write_dataset:
        dataset_path = Path(args.dataset_path).expanduser().resolve()
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        data_writer = h5py.File(dataset_path, "w")
        data_grp = data_writer.create_group("data")

    rollout_stats = []
    diagnostics_per_rollout = []
    try:
        for rollout_idx in range(int(args.n_rollouts)):
            rollout_seed = int(args.seed) + rollout_idx
            np.random.seed(rollout_seed)
            torch.manual_seed(rollout_seed)

            stats, traj, diagnostics = rollout_from_custom_reset(
                policy=policy,
                env=env,
                reset_state=reset_state,
                horizon=int(args.horizon),
                render=bool(args.render),
                video_writer=video_writer,
                video_skip=int(args.video_skip),
                return_obs=bool(write_dataset and args.dataset_obs),
                camera_names=list(args.camera_names),
            )
            rollout_stats.append(stats)
            diagnostics["rollout_index"] = rollout_idx
            diagnostics["seed"] = rollout_seed
            diagnostics_per_rollout.append(diagnostics)

            if data_grp is not None:
                ep_data_grp = data_grp.create_group(f"demo_{rollout_idx}")
                ep_data_grp.create_dataset("actions", data=np.asarray(traj["actions"]))
                ep_data_grp.create_dataset("states", data=np.asarray(traj["states"]))
                ep_data_grp.create_dataset("rewards", data=np.asarray(traj["rewards"]))
                ep_data_grp.create_dataset("dones", data=np.asarray(traj["dones"]))
                if args.dataset_obs:
                    for key in traj["obs"]:
                        ep_data_grp.create_dataset(f"obs/{key}", data=np.asarray(traj["obs"][key]))
                        ep_data_grp.create_dataset(f"next_obs/{key}", data=np.asarray(traj["next_obs"][key]))
                if "model" in traj["initial_state_dict"]:
                    ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]
                if "ep_meta" in traj["initial_state_dict"]:
                    ep_data_grp.attrs["ep_meta"] = traj["initial_state_dict"]["ep_meta"]
                ep_data_grp.attrs["num_samples"] = int(traj["actions"].shape[0])
                total_samples += int(traj["actions"].shape[0])
    finally:
        if video_writer is not None:
            video_writer.close()
        if data_writer is not None and data_grp is not None:
            data_grp.attrs["total"] = total_samples
            data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)
            data_writer.close()

    rollout_stats_dict = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = {
        key: float(np.mean(rollout_stats_dict[key]))
        for key in rollout_stats_dict
    }
    avg_rollout_stats["Num_Success"] = float(np.sum(rollout_stats_dict["Success_Rate"]))

    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    payload = {
        "checkpoint": str(checkpoint),
        "source_demo": str(source_demo),
        "source_episode": int(args.source_episode),
        "n_rollouts": int(args.n_rollouts),
        "horizon": int(args.horizon),
        "seed": int(args.seed),
        "object_translation": object_translation.round(6).tolist(),
        "target_translation": target_translation.round(6).tolist(),
        "avg_rollout_stats": avg_rollout_stats,
        "per_rollout": diagnostics_per_rollout,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
