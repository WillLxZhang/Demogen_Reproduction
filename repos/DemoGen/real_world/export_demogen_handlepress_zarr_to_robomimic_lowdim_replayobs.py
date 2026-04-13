#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import h5py
import numpy as np
import zarr


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMOGEN_ROOT = REPO_ROOT / "demo_generation"
DIFFUSION_POLICIES_ROOT = REPO_ROOT / "diffusion_policies"
if str(DEMOGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMOGEN_ROOT))
if str(DIFFUSION_POLICIES_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICIES_ROOT))

from demo_generation.handlepress_robosuite_wrapper import HandlePressRobosuite3DEnv
import diffusion_policies.env.robosuite.robosuite_wrapper as robosuite_wrapper
from scipy.spatial.transform import Rotation as R


ACTION_FRAME_ROT_OFFSET = R.from_rotvec([0.0, 0.0, -1.5707]).as_matrix().astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a solved HandlePress DemoGen zarr dataset into a robomimic-compatible low-dim HDF5."
        )
    )
    parser.add_argument("--generated-zarr", required=True)
    parser.add_argument("--source-low-dim-hdf5", required=True)
    parser.add_argument("--output-hdf5", required=True)
    parser.add_argument("--include-source-demos", action="store_true")
    parser.add_argument("--copy-source-rewards", action="store_true")
    parser.add_argument("--write-next-obs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--control-steps", type=int, default=1)
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sorted_demo_keys(group: h5py.Group) -> list[str]:
    def key_fn(name: str) -> Tuple[int, str]:
        match = re.search(r"(\d+)$", name)
        return (int(match.group(1)) if match else -1, name)

    return sorted(list(group.keys()), key=key_fn)


def iter_episode_bounds(episode_ends: Iterable[int]) -> Iterator[Tuple[int, int]]:
    start = 0
    for end in episode_ends:
        end = int(end)
        yield start, end
        start = end


def action_to_action_dict(actions: np.ndarray) -> dict[str, np.ndarray]:
    rel_pos = actions[:, :3].astype(np.float32)
    rel_rot_axis_angle = actions[:, 3:6].astype(np.float32)
    rel_rot_6d = R.from_rotvec(rel_rot_axis_angle.astype(np.float64)).as_matrix()[:, :, :2].reshape(-1, 6)
    gripper = actions[:, 6:7].astype(np.float32)
    return {
        "rel_pos": rel_pos,
        "rel_rot_axis_angle": rel_rot_axis_angle,
        "rel_rot_6d": rel_rot_6d.astype(np.float32),
        "gripper": gripper,
    }


def slice_source_array(group: h5py.Group, key: str, length: int, default: np.ndarray | None = None) -> np.ndarray:
    if key in group:
        arr = group[key][()]
        if arr.shape[0] < length:
            raise ValueError(f"Source key '{key}' shorter than requested length: {arr.shape[0]} < {length}")
        return np.asarray(arr[:length])
    if default is None:
        raise KeyError(f"Missing key '{key}' in source group")
    return default


def build_reset_state(source_episode_group: h5py.Group) -> dict:
    state = {
        "states": np.asarray(source_episode_group["states"][0], dtype=np.float64),
        "model": source_episode_group.attrs["model_file"],
    }
    if "ep_meta" in source_episode_group.attrs:
        state["ep_meta"] = source_episode_group.attrs["ep_meta"]
    return state


def build_generated_obs_replay(
    env: HandlePressRobosuite3DEnv,
    source_episode_group: h5py.Group,
    generated_actions: np.ndarray,
    object_translation: np.ndarray,
) -> dict[str, np.ndarray]:
    reset_state = build_reset_state(source_episode_group)
    reset_state["object_translation"] = np.asarray(object_translation, dtype=np.float32).copy()
    obs = env.reset_to(reset_state)
    if obs is None:
        raise RuntimeError("env.reset_to(...) did not return an observation")

    eef_pos_seq = []
    eef_quat_seq = []
    gripper_qpos_seq = []
    object_obs_seq = []

    for action in np.asarray(generated_actions, dtype=np.float32):
        raw_obs = env.get_observation()
        eef_pos_seq.append(np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32))
        eef_quat_seq.append(np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float32))
        gripper_qpos_seq.append(np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32))
        object_obs_seq.append(np.asarray(raw_obs["object"], dtype=np.float32))
        env.step(action)

    return {
        "robot0_eef_pos": np.asarray(eef_pos_seq, dtype=np.float32),
        "robot0_eef_quat": np.asarray(eef_quat_seq, dtype=np.float32),
        "robot0_gripper_qpos": np.asarray(gripper_qpos_seq, dtype=np.float32),
        "object": np.asarray(object_obs_seq, dtype=np.float32),
    }


def write_episode(
    out_data_grp: h5py.Group,
    episode_name: str,
    obs: dict[str, np.ndarray],
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    model_file: str,
    source_episode_idx: int | None,
    object_translation: np.ndarray | None,
    write_next_obs: bool,
) -> None:
    ep_grp = out_data_grp.create_group(episode_name)
    ep_grp.create_dataset("actions", data=np.asarray(actions, dtype=np.float32))
    ep_grp.create_dataset("rewards", data=np.asarray(rewards, dtype=np.float32))
    ep_grp.create_dataset("dones", data=np.asarray(dones, dtype=np.int32))

    for key, value in obs.items():
        ep_grp.create_dataset(f"obs/{key}", data=np.asarray(value, dtype=np.float32))

    if write_next_obs:
        for key, value in obs.items():
            next_value = np.concatenate([value[1:], value[-1:]], axis=0)
            ep_grp.create_dataset(f"next_obs/{key}", data=np.asarray(next_value, dtype=np.float32))

    action_dict = action_to_action_dict(np.asarray(actions, dtype=np.float32))
    for key, value in action_dict.items():
        ep_grp.create_dataset(f"action_dict/{key}", data=value)

    ep_grp.attrs["num_samples"] = int(actions.shape[0])
    ep_grp.attrs["model_file"] = model_file
    if source_episode_idx is not None:
        ep_grp.attrs["source_episode_idx"] = int(source_episode_idx)
    if object_translation is not None:
        ep_grp.attrs["object_translation"] = json.dumps(np.asarray(object_translation, dtype=np.float32).tolist())


def main() -> None:
    args = parse_args()

    generated_zarr = Path(args.generated_zarr).expanduser().resolve()
    source_low_dim = Path(args.source_low_dim_hdf5).expanduser().resolve()
    output_hdf5 = Path(args.output_hdf5).expanduser().resolve()

    if not generated_zarr.exists():
        raise FileNotFoundError(f"Generated zarr not found: {generated_zarr}")
    if not source_low_dim.exists():
        raise FileNotFoundError(f"Source low_dim.hdf5 not found: {source_low_dim}")
    if args.control_steps <= 0:
        raise ValueError("--control-steps must be positive")

    ensure_parent(output_hdf5)
    if output_hdf5.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_hdf5}")
        output_hdf5.unlink()

    zarr_root = zarr.open(str(generated_zarr), mode="r")
    zarr_data = zarr_root["data"]
    zarr_meta = zarr_root["meta"]

    episode_ends = np.asarray(zarr_meta["episode_ends"][:], dtype=np.int64)
    generated_actions = np.asarray(zarr_data["action"][:], dtype=np.float32)
    generated_source_episode_idx = np.asarray(zarr_meta["source_episode_idx"][:], dtype=np.int64)
    generated_object_translation = np.asarray(zarr_meta["object_translation"][:], dtype=np.float32)

    replay_env = None

    try:
        with h5py.File(source_low_dim, "r") as src_f, h5py.File(output_hdf5, "w") as out_f:
            src_data = src_f["data"]
            src_keys = sorted_demo_keys(src_data)
            src_env_args = src_data.attrs["env_args"]

            out_data = out_f.create_group("data")
            out_data.attrs["env_args"] = src_env_args
            out_data.attrs["demogen_generated_zarr"] = str(generated_zarr)
            out_data.attrs["source_low_dim_hdf5"] = str(source_low_dim)
            out_data.attrs["obs_keys"] = json.dumps(
                ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
            )

            total_samples = 0
            next_demo_idx = 0

            robosuite_wrapper.N_CONTROL_STEPS = int(args.control_steps)
            replay_env = HandlePressRobosuite3DEnv(str(source_low_dim), render=False)

            if args.include_source_demos:
                for src_idx, src_key in enumerate(src_keys):
                    src_ep = src_data[src_key]
                    src_obs = src_ep["obs"]
                    src_actions = np.asarray(src_ep["actions"][()], dtype=np.float32)
                    source_obs = {
                        "robot0_eef_pos": np.asarray(src_obs["robot0_eef_pos"][()], dtype=np.float32),
                        "robot0_eef_quat": np.asarray(src_obs["robot0_eef_quat"][()], dtype=np.float32),
                        "robot0_gripper_qpos": np.asarray(src_obs["robot0_gripper_qpos"][()], dtype=np.float32),
                        "object": np.asarray(src_obs["object"][()], dtype=np.float32),
                    }
                    rewards = (
                        np.asarray(src_ep["rewards"][()], dtype=np.float32)
                        if args.copy_source_rewards and "rewards" in src_ep
                        else np.zeros(src_actions.shape[0], dtype=np.float32)
                    )
                    dones = np.zeros(src_actions.shape[0], dtype=np.int32)
                    dones[-1] = 1
                    write_episode(
                        out_data_grp=out_data,
                        episode_name=f"demo_{next_demo_idx}",
                        obs=source_obs,
                        actions=src_actions,
                        rewards=rewards,
                        dones=dones,
                        model_file=src_ep.attrs["model_file"],
                        source_episode_idx=src_idx,
                        object_translation=np.zeros(3, dtype=np.float32),
                        write_next_obs=args.write_next_obs,
                    )
                    total_samples += int(src_actions.shape[0])
                    next_demo_idx += 1

            for gen_idx, (start, end) in enumerate(iter_episode_bounds(episode_ends)):
                gen_actions_ep = generated_actions[start:end]
                src_idx = int(generated_source_episode_idx[gen_idx])
                if src_idx < 0 or src_idx >= len(src_keys):
                    raise IndexError(
                        f"source_episode_idx[{gen_idx}]={src_idx} is out of range for {len(src_keys)} source demos"
                    )

                src_key = src_keys[src_idx]
                src_ep = src_data[src_key]
                translation = np.asarray(generated_object_translation[gen_idx], dtype=np.float32)

                obs = build_generated_obs_replay(
                    env=replay_env,
                    source_episode_group=src_ep,
                    generated_actions=gen_actions_ep,
                    object_translation=translation,
                )
                rewards = (
                    np.asarray(slice_source_array(src_ep, "rewards", gen_actions_ep.shape[0]), dtype=np.float32)
                    if args.copy_source_rewards and "rewards" in src_ep
                    else np.zeros(gen_actions_ep.shape[0], dtype=np.float32)
                )
                dones = np.zeros(gen_actions_ep.shape[0], dtype=np.int32)
                dones[-1] = 1

                write_episode(
                    out_data_grp=out_data,
                    episode_name=f"demo_{next_demo_idx}",
                    obs=obs,
                    actions=gen_actions_ep,
                    rewards=rewards,
                    dones=dones,
                    model_file=src_ep.attrs["model_file"],
                    source_episode_idx=src_idx,
                    object_translation=translation,
                    write_next_obs=args.write_next_obs,
                )
                total_samples += int(gen_actions_ep.shape[0])
                next_demo_idx += 1

            out_data.attrs["total"] = total_samples
    finally:
        if replay_env is not None:
            replay_env.close()

    print(f"Saved robomimic low-dim dataset to: {output_hdf5}")
    print(f"Included source demos: {args.include_source_demos}")
    print(f"Total exported episodes: {next_demo_idx}")
    print(f"Total exported samples: {total_samples}")
    print(f"Replay control steps: {args.control_steps}")


if __name__ == "__main__":
    main()
