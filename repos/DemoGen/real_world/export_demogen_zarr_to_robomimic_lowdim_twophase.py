#!/usr/bin/env python3
"""
Export a two-phase DemoGen generated zarr dataset into a robomimic-compatible
low-dim HDF5 dataset.

This fork keeps the original exporter untouched and only extends the object
observation reconstruction to support two translated entities, such as Stack's
cubeA / cubeB layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import zarr

from export_demogen_zarr_to_robomimic_lowdim import (
    action_to_action_dict,
    agent_pos_to_eef_quat,
    agent_pos_to_gripper_qpos,
    ensure_parent,
    iter_episode_bounds,
    slice_source_array,
    sorted_demo_keys,
    write_episode,
)


def parse_slice(raw: str) -> slice:
    parts = [p.strip() for p in raw.split(":")]
    if len(parts) != 2:
        raise ValueError(f"Expected slice like '0:3', got {raw}")
    start, end = int(parts[0]), int(parts[1])
    if end <= start:
        raise ValueError(f"Invalid slice {raw}")
    return slice(start, end)


def infer_object_obs_slices_from_env(env_name: str, object_obs_dim: int) -> dict[str, str]:
    env_name = str(env_name)
    if env_name == "Stack":
        if object_obs_dim < 23:
            raise ValueError(
                f"Stack object obs dim is too small for auto slice inference: {object_obs_dim}"
            )
        return {
            "object_pos_slice": "0:3",
            "target_pos_slice": "7:10",
            "object_to_target_slice": "14:17",
            "gripper_to_object_slice": "17:20",
            "gripper_to_target_slice": "20:23",
        }
    raise ValueError(
        f"Auto object slice inference is not configured for env={env_name}. "
        "Pass all slice flags explicitly."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export two-phase DemoGen generated zarr episodes back into a "
            "robomimic-compatible low-dimensional HDF5 dataset."
        )
    )
    parser.add_argument("--generated-zarr", required=True, help="Path to DemoGen generated zarr dataset")
    parser.add_argument("--source-low-dim-hdf5", required=True, help="Path to source robomimic low_dim.hdf5")
    parser.add_argument("--output-hdf5", required=True, help="Path to output robomimic-compatible HDF5")
    parser.add_argument("--object-pos-slice", default="auto")
    parser.add_argument("--target-pos-slice", default="auto")
    parser.add_argument("--object-to-target-slice", default="auto")
    parser.add_argument("--gripper-to-object-slice", default="auto")
    parser.add_argument("--gripper-to-target-slice", default="auto")
    parser.add_argument(
        "--include-source-demos",
        action="store_true",
        help="Prepend the original source demos into the exported HDF5.",
    )
    parser.add_argument(
        "--copy-source-rewards",
        action="store_true",
        help="Copy rewards from the source low_dim.hdf5 instead of writing zeros.",
    )
    parser.add_argument(
        "--write-next-obs",
        action="store_true",
        help="Write next_obs by shifting each generated episode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    return parser.parse_args()


def make_two_phase_object_obs(
    source_object_obs: np.ndarray,
    eef_pos: np.ndarray,
    translation: np.ndarray,
    object_pos_slice: slice,
    target_pos_slice: slice,
    object_to_target_slice: slice,
    gripper_to_object_slice: slice,
    gripper_to_target_slice: slice,
) -> np.ndarray:
    source_object_obs = np.asarray(source_object_obs, dtype=np.float32).copy()
    translation = np.asarray(translation, dtype=np.float32).reshape(-1)

    if translation.shape == (3,):
        source_object_obs[:, object_pos_slice] += translation[None, :]
        if source_object_obs.shape[1] >= gripper_to_object_slice.stop:
            source_object_obs[:, gripper_to_object_slice] = (
                source_object_obs[:, object_pos_slice] - eef_pos.astype(np.float32)
            )
        return source_object_obs.astype(np.float32)

    if translation.shape != (6,):
        raise ValueError(f"Expected translation shape (3,) or (6,), got {translation.shape}")

    object_translation = translation[:3]
    target_translation = translation[3:6]

    source_object_obs[:, object_pos_slice] += object_translation[None, :]
    source_object_obs[:, target_pos_slice] += target_translation[None, :]

    if source_object_obs.shape[1] >= object_to_target_slice.stop:
        source_object_obs[:, object_to_target_slice] = (
            source_object_obs[:, target_pos_slice] - source_object_obs[:, object_pos_slice]
        )
    if source_object_obs.shape[1] >= gripper_to_object_slice.stop:
        source_object_obs[:, gripper_to_object_slice] = (
            source_object_obs[:, object_pos_slice] - eef_pos.astype(np.float32)
        )
    if source_object_obs.shape[1] >= gripper_to_target_slice.stop:
        source_object_obs[:, gripper_to_target_slice] = (
            source_object_obs[:, target_pos_slice] - eef_pos.astype(np.float32)
        )

    return source_object_obs.astype(np.float32)


def build_generated_obs(
    source_obs_group: h5py.Group,
    generated_agent_pos: np.ndarray,
    translation: np.ndarray,
    object_pos_slice: slice,
    target_pos_slice: slice,
    object_to_target_slice: slice,
    gripper_to_object_slice: slice,
    gripper_to_target_slice: slice,
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
    object_obs = make_two_phase_object_obs(
        source_object_obs=source_object_obs,
        eef_pos=eef_pos,
        translation=translation,
        object_pos_slice=object_pos_slice,
        target_pos_slice=target_pos_slice,
        object_to_target_slice=object_to_target_slice,
        gripper_to_object_slice=gripper_to_object_slice,
        gripper_to_target_slice=gripper_to_target_slice,
    )
    return {
        "robot0_eef_pos": eef_pos,
        "robot0_eef_quat": eef_quat,
        "robot0_gripper_qpos": gripper_qpos,
        "object": object_obs,
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

    with h5py.File(source_low_dim, "r") as src_f, h5py.File(output_hdf5, "w") as out_f:
        src_data = src_f["data"]
        src_keys = sorted_demo_keys(src_data)
        src_env_args = src_data.attrs["env_args"]
        env_args = json.loads(src_env_args)
        env_name = env_args["env_name"].split("_")[0]
        first_src_key = src_keys[0]
        first_object_obs_dim = int(src_data[first_src_key]["obs"]["object"].shape[-1])

        raw_slices = {
            "object_pos_slice": args.object_pos_slice,
            "target_pos_slice": args.target_pos_slice,
            "object_to_target_slice": args.object_to_target_slice,
            "gripper_to_object_slice": args.gripper_to_object_slice,
            "gripper_to_target_slice": args.gripper_to_target_slice,
        }
        if any(v == "auto" for v in raw_slices.values()):
            inferred = infer_object_obs_slices_from_env(
                env_name=env_name,
                object_obs_dim=first_object_obs_dim,
            )
            for key, value in inferred.items():
                if raw_slices[key] == "auto":
                    raw_slices[key] = value

        object_pos_slice = parse_slice(raw_slices["object_pos_slice"])
        target_pos_slice = parse_slice(raw_slices["target_pos_slice"])
        object_to_target_slice = parse_slice(raw_slices["object_to_target_slice"])
        gripper_to_object_slice = parse_slice(raw_slices["gripper_to_object_slice"])
        gripper_to_target_slice = parse_slice(raw_slices["gripper_to_target_slice"])

        out_data = out_f.create_group("data")
        out_data.attrs["env_args"] = src_env_args
        out_data.attrs["demogen_generated_zarr"] = str(generated_zarr)
        out_data.attrs["source_low_dim_hdf5"] = str(source_low_dim)
        out_data.attrs["obs_keys"] = json.dumps(
            ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
        )
        out_data.attrs["object_pos_slice"] = raw_slices["object_pos_slice"]
        out_data.attrs["target_pos_slice"] = raw_slices["target_pos_slice"]
        out_data.attrs["object_to_target_slice"] = raw_slices["object_to_target_slice"]
        out_data.attrs["gripper_to_object_slice"] = raw_slices["gripper_to_object_slice"]
        out_data.attrs["gripper_to_target_slice"] = raw_slices["gripper_to_target_slice"]

        total_samples = 0
        next_demo_idx = 0

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
                    object_translation=np.zeros(6, dtype=np.float32),
                    write_next_obs=args.write_next_obs,
                )
                total_samples += int(src_actions.shape[0])
                next_demo_idx += 1

        for gen_idx, (start, end) in enumerate(iter_episode_bounds(episode_ends)):
            gen_agent_pos_ep = generated_agent_pos[start:end]
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
                else np.zeros(6, dtype=np.float32)
            )
            obs = build_generated_obs(
                source_obs_group=src_ep["obs"],
                generated_agent_pos=gen_agent_pos_ep,
                translation=translation,
                object_pos_slice=object_pos_slice,
                target_pos_slice=target_pos_slice,
                object_to_target_slice=object_to_target_slice,
                gripper_to_object_slice=gripper_to_object_slice,
                gripper_to_target_slice=gripper_to_target_slice,
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

    print(f"Saved robomimic low-dim dataset to: {output_hdf5}")
    print(f"Included source demos: {args.include_source_demos}")
    print(f"Total exported episodes: {next_demo_idx}")
    print(f"Total exported samples: {total_samples}")


if __name__ == "__main__":
    main()
