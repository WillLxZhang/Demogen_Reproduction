"""
Convert robomimic / robosuite HDF5 exports into a DemoGen source zarr dataset
for two-phase tasks such as Stack.

This fork keeps the replay_h1 semantics from the validated one-stage path, and
adds a minimal amount of Stack-specific metadata:
- dual mask export for object / target
- configurable low-dim position slices for object / target centers
- source zarr attrs that document the two-object semantics without changing
  downstream baseline keys
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import zarr
from termcolor import cprint
import json

from convert_robomimic_hdf5_to_zarr_exec_motion import (
    build_exec_action,
    build_forward_delta,
    build_state,
    build_workspace_bounds,
    compute_exec_scale_stats,
    crop_workspace,
    ensure_parent,
    load_camera_info,
    reconstruct_point_cloud,
    sample_point_cloud,
    save_mask_assets,
    sorted_demo_keys,
    summarize_xyz,
)


DIFFUSION_POLICIES_ROOT = Path(__file__).resolve().parents[1] / "diffusion_policies"
if str(DIFFUSION_POLICIES_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICIES_ROOT))

import diffusion_policies.env.robosuite.robosuite_wrapper as robosuite_wrapper
from diffusion_policies.env.robosuite.robosuite_wrapper import Robosuite3DEnv


def parse_slice(raw: str) -> slice:
    parts = [p.strip() for p in raw.split(":")]
    if len(parts) != 2:
        raise ValueError(f"Expected slice like '0:3', got {raw}")
    start, end = int(parts[0]), int(parts[1])
    if end <= start:
        raise ValueError(f"Invalid slice {raw}")
    return slice(start, end)


def load_env_name_from_hdf5(hdf5_path: str) -> str:
    with h5py.File(hdf5_path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
    return env_args["env_name"].split("_")[0]


def infer_pos_slices(env_name: str, object_obs_dim: int) -> tuple[str, str]:
    env_name = str(env_name)
    if env_name == "Stack":
        if object_obs_dim < 10:
            raise ValueError(
                f"Stack object obs dim is too small for auto slice inference: {object_obs_dim}"
            )
        return "0:3", "7:10"
    raise ValueError(
        f"Auto position slice inference is not configured for env={env_name}. "
        "Pass --object-pos-slice and --target-pos-slice explicitly."
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-hdf5", required=True)
    parser.add_argument("--low-dim-hdf5", required=True)
    parser.add_argument("--depth-hdf5", required=True)
    parser.add_argument("--output-zarr", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--workspace-margin-xy", type=float, default=0.25)
    parser.add_argument("--workspace-margin-z", type=float, default=0.12)
    parser.add_argument("--mask-object-name", default="cubeA")
    parser.add_argument("--mask-target-name", default="cubeB")
    parser.add_argument("--mask-radius", type=float, default=0.045)
    parser.add_argument("--mask-dilation-iters", type=int, default=2)
    parser.add_argument("--object-pos-slice", default="auto")
    parser.add_argument("--target-pos-slice", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--control-steps",
        type=int,
        default=1,
        help="Internal robosuite control repeats used for replay_h1 calibration.",
    )
    return parser.parse_args()


def build_reset_state(demo_group, frame_idx: int) -> dict:
    reset_state = {
        "states": demo_group["states"][frame_idx],
        "model": demo_group.attrs["model_file"],
    }
    if "ep_meta" in demo_group.attrs:
        reset_state["ep_meta"] = demo_group.attrs["ep_meta"]
    return reset_state


def build_replay_h1_delta(
    env: Robosuite3DEnv,
    demo_group,
    exec_action: np.ndarray,
    forward_delta: np.ndarray,
    demo_key: str | None = None,
) -> np.ndarray:
    replay_h1_delta = np.zeros_like(forward_delta, dtype=np.float32)
    total_frames = len(exec_action)
    for frame_idx in range(total_frames):
        obs_before = env.reset_to(build_reset_state(demo_group, frame_idx))
        before = np.asarray(obs_before["agent_pos"], dtype=np.float32)
        obs_after, _, _, _ = env.step(exec_action[frame_idx])
        after = np.asarray(obs_after["agent_pos"], dtype=np.float32)

        replay_h1_delta[frame_idx, :6] = after[:6] - before[:6]
        replay_h1_delta[frame_idx, 6] = exec_action[frame_idx, 6]

        if (frame_idx + 1) % 100 == 0 or (frame_idx + 1) == total_frames:
            name = demo_key or "episode"
            cprint(
                f"[replay_h1] {name}: {frame_idx + 1}/{total_frames}",
                "cyan",
                flush=True,
            )

    return replay_h1_delta.astype(np.float32)


def create_dataset(group, key: str, data: np.ndarray, compressor) -> None:
    data = np.asarray(data)
    if data.ndim == 1:
        chunks = (min(100, data.shape[0]),)
    elif data.ndim == 2:
        chunks = (min(100, data.shape[0]), data.shape[1])
    elif data.ndim == 3:
        chunks = (min(100, data.shape[0]), data.shape[1], data.shape[2])
    else:
        raise ValueError(f"Unsupported ndim={data.ndim} for key={key}")

    group.create_dataset(
        key,
        data=data,
        chunks=chunks,
        dtype=str(data.dtype),
        overwrite=True,
        compressor=compressor,
    )


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    output_zarr = Path(args.output_zarr).expanduser()
    ensure_parent(output_zarr)
    sam_mask_root = output_zarr.parents[2] / "sam_mask"

    if args.control_steps <= 0:
        raise ValueError("--control-steps must be positive")

    robosuite_wrapper.N_CONTROL_STEPS = int(args.control_steps)
    env = Robosuite3DEnv(
        source_demo_path=str(Path(args.demo_hdf5).expanduser()),
        render=False,
        support_osc_control=False,
    )

    state_arrays_ls = []
    exec_action_arrays_ls = []
    motion_action_arrays_ls = []
    replay_h1_delta_arrays_ls = []
    forward_delta_arrays_ls = []
    point_cloud_arrays_ls = []
    episode_ends = []
    count = 0

    try:
        with h5py.File(args.demo_hdf5, "r") as demo_f, \
             h5py.File(args.low_dim_hdf5, "r") as low_dim_f, \
             h5py.File(args.depth_hdf5, "r") as depth_f:
            demo_keys = sorted_demo_keys(demo_f["data"])
            low_dim_keys = sorted_demo_keys(low_dim_f["data"])
            depth_keys = sorted_demo_keys(depth_f["data"])

            if demo_keys != low_dim_keys or demo_keys != depth_keys:
                raise ValueError("demo / low_dim / depth files do not share the same episode keys")

            env_name = load_env_name_from_hdf5(args.demo_hdf5)
            first_key = low_dim_keys[0]
            first_object_obs_dim = int(
                low_dim_f["data"][first_key]["obs"]["object"].shape[-1]
            )
            object_pos_slice_raw = args.object_pos_slice
            target_pos_slice_raw = args.target_pos_slice
            if object_pos_slice_raw == "auto" or target_pos_slice_raw == "auto":
                inferred_object_raw, inferred_target_raw = infer_pos_slices(
                    env_name=env_name,
                    object_obs_dim=first_object_obs_dim,
                )
                if object_pos_slice_raw == "auto":
                    object_pos_slice_raw = inferred_object_raw
                if target_pos_slice_raw == "auto":
                    target_pos_slice_raw = inferred_target_raw
            object_pos_slice = parse_slice(object_pos_slice_raw)
            target_pos_slice = parse_slice(target_pos_slice_raw)
            cprint(
                (
                    f"Resolved position slices for env={env_name}: "
                    f"object={object_pos_slice_raw}, target={target_pos_slice_raw}"
                ),
                "cyan",
                flush=True,
            )

            for episode_idx, demo_key in enumerate(demo_keys):
                cprint(f"Processing {demo_key}", "blue", flush=True)
                demo_group = demo_f["data"][demo_key]
                low_dim_group = low_dim_f["data"][demo_key]
                depth_group = depth_f["data"][demo_key]

                camera_info = load_camera_info(depth_group.attrs["camera_info"], args.camera_name)
                obs_group = low_dim_group["obs"]
                depth_obs = depth_group["obs"]

                state = build_state(obs_group)
                exec_action = build_exec_action(state, demo_group)
                forward_delta = build_forward_delta(state, obs_group, exec_action)
                replay_h1_delta = build_replay_h1_delta(
                    env=env,
                    demo_group=demo_group,
                    exec_action=exec_action,
                    forward_delta=forward_delta,
                    demo_key=demo_key,
                )
                motion_action = replay_h1_delta.copy()
                motion_action[:, 6] = exec_action[:, 6]

                object_obs = obs_group["object"][...].astype(np.float32)
                object_positions = object_obs[..., object_pos_slice].astype(np.float32)
                target_positions = object_obs[..., target_pos_slice].astype(np.float32)
                eef_positions = obs_group["robot0_eef_pos"][...].astype(np.float32)
                lower, upper = build_workspace_bounds(
                    object_positions=np.concatenate([object_positions, target_positions], axis=0),
                    eef_positions=eef_positions,
                    margin_xy=args.workspace_margin_xy,
                    margin_z=args.workspace_margin_z,
                )

                rgb_frames = depth_obs[f"{args.camera_name}_image"][...]
                depth_frames = depth_obs[f"{args.camera_name}_depth"][..., 0]

                sampled_frames = []
                first_dense_points = None
                for frame_idx in range(len(state)):
                    dense_points = reconstruct_point_cloud(
                        depth=depth_frames[frame_idx],
                        rgb=rgb_frames[frame_idx],
                        intrinsics=camera_info["intrinsics"],
                        extrinsics=camera_info["extrinsics"],
                    )
                    cropped_points = crop_workspace(dense_points, lower=lower, upper=upper)
                    sampled_points = sample_point_cloud(cropped_points, args.n_points, rng=rng)
                    sampled_frames.append(sampled_points)

                    if frame_idx == 0:
                        first_dense_points = cropped_points

                point_cloud = np.stack(sampled_frames, axis=0).astype(np.float32)

                save_mask_assets(
                    sam_mask_root=sam_mask_root,
                    source_name=args.source_name,
                    episode_idx=episode_idx,
                    first_rgb=rgb_frames[0],
                    first_points=first_dense_points,
                    first_object_center=object_positions[0],
                    camera_info=camera_info,
                    mask_object_name=args.mask_object_name,
                    mask_radius=args.mask_radius,
                    mask_dilation_iters=args.mask_dilation_iters,
                )
                save_mask_assets(
                    sam_mask_root=sam_mask_root,
                    source_name=args.source_name,
                    episode_idx=episode_idx,
                    first_rgb=rgb_frames[0],
                    first_points=first_dense_points,
                    first_object_center=target_positions[0],
                    camera_info=camera_info,
                    mask_object_name=args.mask_target_name,
                    mask_radius=args.mask_radius,
                    mask_dilation_iters=args.mask_dilation_iters,
                )

                state_arrays_ls.append(state)
                exec_action_arrays_ls.append(exec_action)
                motion_action_arrays_ls.append(motion_action)
                replay_h1_delta_arrays_ls.append(replay_h1_delta)
                forward_delta_arrays_ls.append(forward_delta)
                point_cloud_arrays_ls.append(point_cloud)

                count += len(state)
                episode_ends.append(count)
    finally:
        env.close()

    state_arrays = np.concatenate(state_arrays_ls, axis=0).astype(np.float32)
    exec_action_arrays = np.concatenate(exec_action_arrays_ls, axis=0).astype(np.float32)
    motion_action_arrays = np.concatenate(motion_action_arrays_ls, axis=0).astype(np.float32)
    replay_h1_delta_arrays = np.concatenate(replay_h1_delta_arrays_ls, axis=0).astype(np.float32)
    forward_delta_arrays = np.concatenate(forward_delta_arrays_ls, axis=0).astype(np.float32)
    point_cloud_arrays = np.concatenate(point_cloud_arrays_ls, axis=0).astype(np.float32)
    episode_ends_arrays = np.asarray(episode_ends, dtype=np.int64)

    pos_scale, neg_scale, pos_count, neg_count, recommended_scale = compute_exec_scale_stats(
        exec_action=exec_action_arrays,
        motion_action=motion_action_arrays,
    )

    zarr_root = zarr.group(str(output_zarr), overwrite=True)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    create_dataset(zarr_data, "state", state_arrays, compressor)
    create_dataset(zarr_data, "agent_pos", state_arrays, compressor)
    create_dataset(zarr_data, "action", exec_action_arrays, compressor)
    create_dataset(zarr_data, "raw_action", exec_action_arrays, compressor)
    create_dataset(zarr_data, "motion_action", motion_action_arrays, compressor)
    create_dataset(zarr_data, "replay_h1_delta", replay_h1_delta_arrays, compressor)
    create_dataset(zarr_data, "forward_delta", forward_delta_arrays, compressor)
    create_dataset(zarr_data, "motion_delta", motion_action_arrays, compressor)
    create_dataset(zarr_data, "point_cloud", point_cloud_arrays, compressor)
    create_dataset(zarr_meta, "episode_ends", episode_ends_arrays, compressor)

    zarr_meta.attrs["source_name"] = str(args.source_name)
    zarr_meta.attrs["camera_name"] = str(args.camera_name)
    zarr_meta.attrs["state_semantics"] = "eef_pose_in_robosuite_action_frame_plus_gripper_gap"
    zarr_meta.attrs["action_semantics"] = "executable_controller_action_for_replay"
    zarr_meta.attrs["motion_action_semantics"] = (
        "replay_calibrated_local_step_delta_from_reset_to_state_t_then_step_action_t"
    )
    zarr_meta.attrs["forward_delta_alignment"] = "forward_delta[t] corresponds to obs_t -> obs_t+1"
    zarr_meta.attrs["replay_h1_alignment"] = "replay_h1_delta[t] corresponds to reset_to(obs_t) then step(action_t)"
    zarr_meta.attrs["replay_h1_control_steps"] = int(args.control_steps)
    zarr_meta.attrs["exec_motion_xyz_positive_scale_median"] = pos_scale.tolist()
    zarr_meta.attrs["exec_motion_xyz_negative_scale_median"] = neg_scale.tolist()
    zarr_meta.attrs["exec_motion_xyz_positive_count"] = pos_count.tolist()
    zarr_meta.attrs["exec_motion_xyz_negative_count"] = neg_count.tolist()
    zarr_meta.attrs["exec_motion_xyz_scale_recommended"] = recommended_scale.tolist()
    zarr_meta.attrs["object_mask_name"] = str(args.mask_object_name)
    zarr_meta.attrs["target_mask_name"] = str(args.mask_target_name)
    zarr_meta.attrs["object_pos_slice"] = str(object_pos_slice_raw)
    zarr_meta.attrs["target_pos_slice"] = str(target_pos_slice_raw)
    zarr_meta.attrs["task_stage_semantics"] = "two_phase_motion1_skill1_motion2_skill2"

    cprint("-" * 50, "cyan")
    cprint(f"Saved zarr file to {output_zarr}", "green", flush=True)
    cprint(f"state shape: {state_arrays.shape}, range: [{state_arrays.min()}, {state_arrays.max()}]", "green", flush=True)
    cprint(f"action shape: {exec_action_arrays.shape}, range: [{exec_action_arrays.min()}, {exec_action_arrays.max()}]", "green", flush=True)
    cprint(
        f"motion_action shape: {motion_action_arrays.shape}, range: [{motion_action_arrays.min()}, {motion_action_arrays.max()}]",
        "green",
        flush=True,
    )
    cprint(
        f"point_cloud shape: {point_cloud_arrays.shape}, xyz range: [{point_cloud_arrays[..., :3].min()}, {point_cloud_arrays[..., :3].max()}], "
        f"rgb range: [{point_cloud_arrays[..., 3:].min()}, {point_cloud_arrays[..., 3:].max()}]",
        "green",
        flush=True,
    )
    summarize_xyz("exec action", exec_action_arrays)
    summarize_xyz("motion_action", motion_action_arrays)
    summarize_xyz("replay_h1_delta", replay_h1_delta_arrays)
    summarize_xyz("forward_delta", forward_delta_arrays)
    cprint(
        (
            "exec controller -> replay_h1 xyz scale medians: "
            f"pos={np.round(pos_scale, 6).tolist()} "
            f"neg={np.round(neg_scale, 6).tolist()} "
            f"counts_pos={pos_count.tolist()} "
            f"counts_neg={neg_count.tolist()}"
        ),
        "cyan",
        flush=True,
    )
    cprint(
        f"recommended motion_exec_pulse_scale_xyz={np.round(recommended_scale, 6).tolist()}",
        "cyan",
        flush=True,
    )


if __name__ == "__main__":
    main()
