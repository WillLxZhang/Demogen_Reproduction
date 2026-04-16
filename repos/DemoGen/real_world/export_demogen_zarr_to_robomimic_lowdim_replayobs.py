#!/usr/bin/env python3
"""
Export a DemoGen generated zarr dataset into a robomimic-compatible low-dim HDF5.

This fork preserves the legacy exporter and adds replay-based generated
observation reconstruction so that generated `object` observations come from
the true replayed trajectory instead of the source-plus-translation
approximation.

Typical usage:

    conda run -n demogen python repos/DemoGen/real_world/export_demogen_zarr_to_robomimic_lowdim_replayobs.py \
      --generated-zarr repos/DemoGen/data/datasets/generated/lift_v28_originalschedule_phasecopy_statedelta_halfcorr_v9_s220_test_9.zarr \
      --source-low-dim-hdf5 data/raw/lift_keyboard_1/1774355871_95818/low_dim.hdf5 \
      --output-hdf5 data/processed/robomimic/lift_v28_demogen_lowdim_replayobs.hdf5 \
      --include-source-demos \
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import zarr
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]
DIFFUSION_POLICIES_ROOT = REPO_ROOT / "diffusion_policies"
if str(DIFFUSION_POLICIES_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICIES_ROOT))

import diffusion_policies.env.robosuite.robosuite_wrapper as robosuite_wrapper
from diffusion_policies.env.robosuite.robosuite_wrapper import Robosuite3DEnv
from diffusion_policies.env.robosuite.dataset_meta import load_env_name_from_dataset


# Keep this aligned with the source-zarr conversion scripts.
ACTION_FRAME_ROT_OFFSET = R.from_rotvec([0.0, 0.0, -1.5707]).as_matrix().astype(np.float32)
TASK_OBJECT_STATE_INDICES = {
    "Lift": np.array([10, 11, 12], dtype=np.int64),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export DemoGen generated zarr episodes back into a robomimic-compatible "
            "low-dimensional HDF5 dataset with replay-based generated observations."
        )
    )
    parser.add_argument("--generated-zarr", required=True, help="Path to DemoGen generated zarr dataset")
    parser.add_argument("--source-low-dim-hdf5", required=True, help="Path to source robomimic low_dim.hdf5")
    parser.add_argument("--output-hdf5", required=True, help="Path to output robomimic-compatible HDF5")
    parser.add_argument(
        "--include-source-demos",
        action="store_true",
        help="Prepend the original source demos into the exported HDF5 to create a source+generated dataset.",
    )
    parser.add_argument(
        "--copy-source-rewards",
        action="store_true",
        help="Copy rewards from the source low_dim.hdf5 instead of writing zeros.",
    )
    parser.add_argument(
        "--write-next-obs",
        action="store_true",
        help=(
            "Write next_obs by shifting each generated episode. BC / Diffusion Policy do not need this, "
            "so it is off by default to keep the export simple."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    parser.add_argument(
        "--generated-obs-mode",
        choices=["replay", "approx"],
        default="replay",
        help=(
            "How to reconstruct generated low-dim observations. "
            "'replay' replays the generated action sequence and writes the true observed object state. "
            "'approx' keeps the legacy source-object-plus-translation behavior as a fallback."
        ),
    )
    parser.add_argument(
        "--control-steps",
        type=int,
        default=1,
        help="Internal robosuite control repeats to use when replaying generated episodes.",
    )
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


def agent_pos_to_eef_quat(agent_pos: np.ndarray) -> np.ndarray:
    action_frame_rot = R.from_rotvec(agent_pos[:, 3:6].astype(np.float64))
    state_rot_mats = action_frame_rot.as_matrix() @ ACTION_FRAME_ROT_OFFSET.T
    return R.from_matrix(state_rot_mats).as_quat().astype(np.float32)


def agent_pos_to_gripper_qpos(agent_pos: np.ndarray) -> np.ndarray:
    gap = agent_pos[:, 6:7].astype(np.float32)
    return np.concatenate([gap / 2.0, -gap / 2.0], axis=1).astype(np.float32)


def make_object_obs(object_pos: np.ndarray, object_quat: np.ndarray, eef_pos: np.ndarray) -> np.ndarray:
    object_pos = np.asarray(object_pos, dtype=np.float32)
    object_quat = np.asarray(object_quat, dtype=np.float32)
    gripper_to_object = object_pos - eef_pos.astype(np.float32)
    return np.concatenate([object_pos, object_quat, gripper_to_object], axis=1).astype(np.float32)


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


def build_generated_obs_approx(
    source_obs_group: h5py.Group,
    generated_agent_pos: np.ndarray,
    translation: np.ndarray,
) -> dict[str, np.ndarray]:
    eef_pos = generated_agent_pos[:, :3].astype(np.float32)
    eef_quat = agent_pos_to_eef_quat(generated_agent_pos)
    gripper_qpos = agent_pos_to_gripper_qpos(generated_agent_pos)
    source_object_obs = source_obs_group["object"][()]
    if source_object_obs.shape[0] < generated_agent_pos.shape[0]:
        raise ValueError(
            f"Source object obs is shorter than generated episode: "
            f"{source_object_obs.shape[0]} < {generated_agent_pos.shape[0]}"
        )
    source_object_obs = source_object_obs[: generated_agent_pos.shape[0]]
    object_pos = source_object_obs[:, :3].astype(np.float32) + translation[None, :]
    object_quat = source_object_obs[:, 3:7].astype(np.float32)
    object_obs = make_object_obs(object_pos=object_pos, object_quat=object_quat, eef_pos=eef_pos)
    return {
        "robot0_eef_pos": eef_pos,
        "robot0_eef_quat": eef_quat,
        "robot0_gripper_qpos": gripper_qpos,
        "object": object_obs,
    }


def load_env_name(dataset_path: Path) -> str:
    return load_env_name_from_dataset(dataset_path)


def build_reset_state(source_episode_group: h5py.Group) -> dict:
    state = {
        "states": np.asarray(source_episode_group["states"][0], dtype=np.float64),
        "model": source_episode_group.attrs["model_file"],
    }
    if "ep_meta" in source_episode_group.attrs:
        state["ep_meta"] = source_episode_group.attrs["ep_meta"]
    return state


def apply_object_translation_to_reset_state(
    reset_state: dict,
    env_name: str,
    object_translation: np.ndarray,
) -> dict:
    object_state_indices = TASK_OBJECT_STATE_INDICES.get(env_name)
    if object_state_indices is None:
        raise NotImplementedError(
            f"Replay-based generated obs export is not implemented for env '{env_name}'. "
            "Add task-specific object state indices or use --generated-obs-mode approx."
        )
    translated = dict(reset_state)
    translated["states"] = np.asarray(reset_state["states"], dtype=np.float64).copy()
    translated["states"][object_state_indices] += np.asarray(
        object_translation[: len(object_state_indices)], dtype=np.float64
    )
    return translated


def capture_lift_object_pose(env: Robosuite3DEnv) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(env.env, "cube_body_id"):
        raise AttributeError(
            "Expected Lift env with env.env.cube_body_id when replaying generated observations."
        )
    body_id = env.env.cube_body_id
    object_pos = np.asarray(env.env.sim.data.body_xpos[body_id][:3], dtype=np.float32).copy()
    object_quat = T.convert_quat(
        np.asarray(env.env.sim.data.body_xquat[body_id], dtype=np.float32),
        to="xyzw",
    ).astype(np.float32)
    return object_pos, object_quat


def build_generated_obs_replay(
    env: Robosuite3DEnv,
    env_name: str,
    source_episode_group: h5py.Group,
    generated_actions: np.ndarray,
    object_translation: np.ndarray,
) -> dict[str, np.ndarray]:
    reset_state = build_reset_state(source_episode_group)
    reset_state = apply_object_translation_to_reset_state(
        reset_state=reset_state,
        env_name=env_name,
        object_translation=object_translation,
    )
    obs = env.reset_to(reset_state)
    if obs is None:
        raise RuntimeError("env.reset_to(...) did not return an observation")

    eef_pos_seq = []
    eef_quat_seq = []
    gripper_qpos_seq = []
    object_obs_seq = []

    for action in np.asarray(generated_actions, dtype=np.float32):
        raw_obs = env.get_observation()
        eef_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32)
        eef_quat = np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float32)
        gripper_qpos = np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32)

        if env_name == "Lift":
            object_pos, object_quat = capture_lift_object_pose(env)
            object_obs = make_object_obs(
                object_pos=object_pos[None, :],
                object_quat=object_quat[None, :],
                eef_pos=eef_pos[None, :],
            )[0]
        else:
            raise NotImplementedError(
                f"Replay-based generated obs export is not implemented for env '{env_name}'. "
                "Add task-specific object pose capture or use --generated-obs-mode approx."
            )

        eef_pos_seq.append(eef_pos)
        eef_quat_seq.append(eef_quat)
        gripper_qpos_seq.append(gripper_qpos)
        object_obs_seq.append(object_obs)

        env.step(action)

    return {
        "robot0_eef_pos": np.asarray(eef_pos_seq, dtype=np.float32),
        "robot0_eef_quat": np.asarray(eef_quat_seq, dtype=np.float32),
        "robot0_gripper_qpos": np.asarray(gripper_qpos_seq, dtype=np.float32),
        "object": np.asarray(object_obs_seq, dtype=np.float32),
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
    if "data" not in zarr_root or "meta" not in zarr_root:
        raise KeyError(f"Invalid DemoGen zarr structure: {generated_zarr}")

    zarr_data = zarr_root["data"]
    zarr_meta = zarr_root["meta"]

    required_data_keys = {"agent_pos", "action"}
    missing_keys = sorted(required_data_keys - set(zarr_data.keys()))
    if missing_keys:
        raise KeyError(f"Generated zarr missing required keys: {missing_keys}")
    if "episode_ends" not in zarr_meta:
        raise KeyError("Generated zarr missing meta/episode_ends")

    episode_ends = np.asarray(zarr_meta["episode_ends"][:], dtype=np.int64)
    generated_agent_pos = np.asarray(zarr_data["agent_pos"][:], dtype=np.float32)
    generated_actions = np.asarray(zarr_data["action"][:], dtype=np.float32)
    generated_source_episode_idx = (
        np.asarray(zarr_meta["source_episode_idx"][:], dtype=np.int64)
        if "source_episode_idx" in zarr_meta
        else None
    )
    generated_object_translation = (
        np.asarray(zarr_meta["object_translation"][:], dtype=np.float32)
        if "object_translation" in zarr_meta
        else None
    )
    env_name = load_env_name(source_low_dim)
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

            if args.generated_obs_mode == "replay":
                robosuite_wrapper.N_CONTROL_STEPS = int(args.control_steps)
                replay_env = Robosuite3DEnv(str(source_low_dim), render=False)

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
                gen_agent_pos_ep = generated_agent_pos[start:end]
                gen_actions_ep = generated_actions[start:end]
                if generated_source_episode_idx is None:
                    raise KeyError("Generated zarr is missing meta/source_episode_idx, which is required for write-back")
                src_idx = int(generated_source_episode_idx[gen_idx])
                if src_idx < 0 or src_idx >= len(src_keys):
                    raise IndexError(f"source_episode_idx[{gen_idx}]={src_idx} is out of range for {len(src_keys)} source demos")

                src_key = src_keys[src_idx]
                src_ep = src_data[src_key]
                translation = (
                    np.asarray(generated_object_translation[gen_idx], dtype=np.float32)
                    if generated_object_translation is not None
                    else np.zeros(3, dtype=np.float32)
                )
                if args.generated_obs_mode == "replay":
                    obs = build_generated_obs_replay(
                        env=replay_env,
                        env_name=env_name,
                        source_episode_group=src_ep,
                        generated_actions=gen_actions_ep,
                        object_translation=translation,
                    )
                else:
                    obs = build_generated_obs_approx(
                        source_obs_group=src_ep["obs"],
                        generated_agent_pos=gen_agent_pos_ep,
                        translation=translation,
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
    print(f"Generated obs mode: {args.generated_obs_mode}")
    print(f"Replay control steps: {args.control_steps}")


if __name__ == "__main__":
    main()
