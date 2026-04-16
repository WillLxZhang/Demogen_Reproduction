#!/usr/bin/env python3
"""
Export a two-phase DemoGen generated zarr dataset into a robomimic-compatible
low-dim HDF5 dataset with replay-based generated observations.

This fork mirrors the Lift replayobs exporter but supports Stack-style
two-object translations by replaying each generated action sequence and writing
the true observed `obs/object` back into the exported HDF5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import zarr


REPO_ROOT = Path(__file__).resolve().parents[1]
DIFFUSION_POLICIES_ROOT = REPO_ROOT / "diffusion_policies"
SCRIPTS_ROOT = Path(__file__).resolve().parents[3] / "scripts"
if str(DIFFUSION_POLICIES_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICIES_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import diffusion_policies.env.robosuite.robosuite_wrapper as robosuite_wrapper
from diffusion_policies.env.robosuite.robosuite_wrapper import Robosuite3DEnv

from export_demogen_zarr_to_robomimic_lowdim_replayobs import (
    build_reset_state,
    ensure_parent,
    iter_episode_bounds,
    load_env_name,
    slice_source_array,
    sorted_demo_keys,
    write_episode,
)
from relalign_task_spec import (
    apply_translation_to_reset_state as apply_relalign_translation,
    split_translation,
    zero_translation_for_env,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export two-phase DemoGen generated zarr episodes back into a "
            "robomimic-compatible low-dimensional HDF5 dataset with replay-based "
            "generated observations."
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
        "--control-steps",
        type=int,
        default=1,
        help="Internal robosuite control repeats to use when replaying generated episodes.",
    )
    return parser.parse_args()
def apply_translation_to_reset_state(
    *,
    reset_state: dict,
    env_name: str,
    object_translation: np.ndarray | None,
    target_translation: np.ndarray | None,
) -> dict:
    return apply_relalign_translation(
        reset_state=reset_state,
        env_name=env_name,
        object_translation=object_translation,
        target_translation=target_translation,
    )


def build_generated_obs_replay(
    *,
    env: Robosuite3DEnv,
    env_name: str,
    source_episode_group: h5py.Group,
    generated_actions: np.ndarray,
    translation: np.ndarray | None,
) -> dict[str, np.ndarray]:
    reset_state = build_reset_state(source_episode_group)
    object_translation, target_translation = split_translation(translation)
    reset_state = apply_translation_to_reset_state(
        reset_state=reset_state,
        env_name=env_name,
        object_translation=object_translation,
        target_translation=target_translation,
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
        if "object" not in raw_obs:
            raise KeyError(
                f"Expected raw observation to contain 'object' for env '{env_name}', "
                f"got keys={sorted(raw_obs.keys())}"
            )
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
            out_data.attrs["generated_obs_mode"] = "replay"
            out_data.attrs["obs_keys"] = json.dumps(
                ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
            )

            total_samples = 0
            next_demo_idx = 0

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
                        object_translation=zero_translation_for_env(env_name),
                        write_next_obs=args.write_next_obs,
                    )
                    total_samples += int(src_actions.shape[0])
                    next_demo_idx += 1

            for gen_idx, (start, end) in enumerate(iter_episode_bounds(episode_ends)):
                gen_actions_ep = generated_actions[start:end]
                if generated_source_episode_idx is None:
                    raise KeyError("Generated zarr is missing meta/source_episode_idx")
                src_idx = int(generated_source_episode_idx[gen_idx])
                if src_idx < 0 or src_idx >= len(src_keys):
                    raise IndexError(f"source_episode_idx[{gen_idx}]={src_idx} is out of range")

                src_key = src_keys[src_idx]
                src_ep = src_data[src_key]
                translation = (
                    np.asarray(generated_object_translation[gen_idx], dtype=np.float32)
                    if generated_object_translation is not None
                    else zero_translation_for_env(env_name)
                )
                obs = build_generated_obs_replay(
                    env=replay_env,
                    env_name=env_name,
                    source_episode_group=src_ep,
                    generated_actions=gen_actions_ep,
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

    print(f"Saved robomimic replayobs low-dim dataset to: {output_hdf5}")
    print(f"Included source demos: {args.include_source_demos}")
    print(f"Total exported episodes: {next_demo_idx}")
    print(f"Total exported samples: {total_samples}")


if __name__ == "__main__":
    main()
