"""
Convert robomimic / robosuite HDF5 exports into a DemoGen source zarr dataset
with replay-calibrated local motion semantics.

Compared with convert_robomimic_hdf5_to_zarr_exec_motion.py:
- data/action remains the executable controller action for replay
- data/forward_delta keeps the collected obs_t -> obs_t+1 delta
- data/replay_h1_delta stores the local delta produced by:
  reset_to(state_t) -> step(action_t) with a fixed control repeat count
- data/motion_action defaults to replay_h1_delta so downstream generators can
  consume a controller-faithful motion baseline without changing their code
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import zarr
from termcolor import cprint

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
    parser.add_argument("--mask-object-name", default="cube")
    parser.add_argument("--mask-radius", type=float, default=0.045)
    parser.add_argument("--mask-dilation-iters", type=int, default=2)
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
) -> np.ndarray:
    replay_h1_delta = np.zeros_like(forward_delta, dtype=np.float32)
    for frame_idx in range(len(exec_action)):
        obs_before = env.reset_to(build_reset_state(demo_group, frame_idx))
        before = np.asarray(obs_before["agent_pos"], dtype=np.float32)
        obs_after, _, _, _ = env.step(exec_action[frame_idx])
        after = np.asarray(obs_after["agent_pos"], dtype=np.float32)

        replay_h1_delta[frame_idx, :6] = after[:6] - before[:6]
        replay_h1_delta[frame_idx, 6] = exec_action[frame_idx, 6]

    return replay_h1_delta.astype(np.float32)


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

            for episode_idx, demo_key in enumerate(demo_keys):
                cprint(f"Processing {demo_key}", "blue")
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
                )
                motion_action = replay_h1_delta.copy()
                motion_action[:, 6] = exec_action[:, 6]

                object_positions = obs_group["object"][..., :3].astype(np.float32)
                eef_positions = obs_group["robot0_eef_pos"][...].astype(np.float32)
                lower, upper = build_workspace_bounds(
                    object_positions=object_positions,
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

    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=(100, state_arrays.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "agent_pos",
        data=state_arrays,
        chunks=(100, state_arrays.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=exec_action_arrays,
        chunks=(100, exec_action_arrays.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "raw_action",
        data=exec_action_arrays,
        chunks=(100, exec_action_arrays.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "motion_action",
        data=motion_action_arrays,
        chunks=(100, motion_action_arrays.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "replay_h1_delta",
        data=replay_h1_delta_arrays,
        chunks=(100, replay_h1_delta_arrays.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "forward_delta",
        data=forward_delta_arrays,
        chunks=(100, forward_delta_arrays.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "motion_delta",
        data=motion_action_arrays,
        chunks=(100, motion_action_arrays.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "point_cloud",
        data=point_cloud_arrays,
        chunks=(100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )

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

    cprint("-" * 50, "cyan")
    cprint(f"Saved zarr file to {output_zarr}", "green")
    cprint(f"state shape: {state_arrays.shape}, range: [{state_arrays.min()}, {state_arrays.max()}]", "green")
    cprint(f"action shape: {exec_action_arrays.shape}, range: [{exec_action_arrays.min()}, {exec_action_arrays.max()}]", "green")
    cprint(
        f"motion_action shape: {motion_action_arrays.shape}, range: [{motion_action_arrays.min()}, {motion_action_arrays.max()}]",
        "green",
    )
    cprint(
        f"point_cloud shape: {point_cloud_arrays.shape}, xyz range: [{point_cloud_arrays[..., :3].min()}, {point_cloud_arrays[..., :3].max()}], "
        f"rgb range: [{point_cloud_arrays[..., 3:].min()}, {point_cloud_arrays[..., 3:].max()}]",
        "green",
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
    )
    cprint(
        f"recommended motion_exec_pulse_scale_xyz={np.round(recommended_scale, 6).tolist()}",
        "cyan",
    )


if __name__ == "__main__":
    main()
