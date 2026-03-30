"""
Convert robomimic / robosuite HDF5 exports into a DemoGen source zarr dataset
with separate executable and geometric motion interfaces.

This variant uses a mixed motion_action interface chosen from replay-consistency
search:
- x / z axes: current raw action mapped through the identified full 3x3 4-step
  FIR response sum
- y axis: forward-aligned realized delta (forward_h1)

The executable controller action remains untouched in `data/action`.
"""

import h5py
import numpy as np
import zarr
from pathlib import Path
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
    parse_args,
    reconstruct_point_cloud,
    sample_point_cloud,
    save_mask_assets,
    sorted_demo_keys,
)


def fit_full_fir(
    exec_xyz: np.ndarray,
    motion_xyz: np.ndarray,
    episode_ends: np.ndarray,
    kernel_len: int,
) -> np.ndarray:
    x_rows = []
    y_rows = []
    start = 0
    for end in episode_ends:
        end = int(end)
        u = exec_xyz[start:end]
        y = motion_xyz[start:end]
        for t in range(len(u)):
            row = np.zeros((kernel_len, 3), dtype=np.float32)
            for lag in range(kernel_len):
                src_t = t - lag
                if src_t >= 0:
                    row[lag] = u[src_t]
            x_rows.append(row.reshape(-1))
            y_rows.append(y[t])
        start = end
    x = np.asarray(x_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.float32)
    kernel, *_ = np.linalg.lstsq(x, y, rcond=None)
    return kernel.reshape(kernel_len, 3, 3).astype(np.float32)


def build_motion_action(
    exec_action: np.ndarray,
    forward_delta: np.ndarray,
    full_fir_4: np.ndarray,
) -> np.ndarray:
    motion_action = np.zeros_like(exec_action, dtype=np.float32)
    full_sum = full_fir_4.sum(axis=0).astype(np.float32)
    mapped_xyz = exec_action[:, :3] @ full_sum
    motion_action[:, 0] = mapped_xyz[:, 0]
    motion_action[:, 1] = forward_delta[:, 1]
    motion_action[:, 2] = mapped_xyz[:, 2]
    motion_action[:, 6] = exec_action[:, 6]
    return motion_action.astype(np.float32)


def summarize_xyz(name, arr, threshold=1e-4):
    xyz = arr[:, :3]
    mean_abs = np.abs(xyz).mean(axis=0)
    nonzero = (np.abs(xyz) > threshold).mean(axis=0)
    cprint(
        (
            f"{name} xyz mean_abs="
            f"({mean_abs[0]:.6f}, {mean_abs[1]:.6f}, {mean_abs[2]:.6f}) "
            f"nonzero>{threshold:g}="
            f"({nonzero[0]:.4f}, {nonzero[1]:.4f}, {nonzero[2]:.4f})"
        ),
        "green",
    )


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    output_zarr = Path(args.output_zarr).expanduser()
    ensure_parent(output_zarr)

    sam_mask_root = output_zarr.parents[2] / "sam_mask"

    state_arrays_ls = []
    exec_action_arrays_ls = []
    forward_delta_arrays_ls = []
    point_cloud_arrays_ls = []
    episode_ends = []
    count = 0

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
            forward_delta_arrays_ls.append(forward_delta)
            point_cloud_arrays_ls.append(point_cloud)

            count += len(state)
            episode_ends.append(count)

    state_arrays = np.concatenate(state_arrays_ls, axis=0).astype(np.float32)
    exec_action_arrays = np.concatenate(exec_action_arrays_ls, axis=0).astype(np.float32)
    forward_delta_arrays = np.concatenate(forward_delta_arrays_ls, axis=0).astype(np.float32)
    point_cloud_arrays = np.concatenate(point_cloud_arrays_ls, axis=0).astype(np.float32)
    episode_ends_arrays = np.asarray(episode_ends, dtype=np.int64)

    full_fir_4 = fit_full_fir(
        exec_xyz=exec_action_arrays[:, :3],
        motion_xyz=forward_delta_arrays[:, :3],
        episode_ends=episode_ends_arrays,
        kernel_len=4,
    )
    motion_action_arrays = build_motion_action(
        exec_action=exec_action_arrays,
        forward_delta=forward_delta_arrays,
        full_fir_4=full_fir_4,
    )

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
    zarr_meta.attrs["motion_action_semantics"] = "xz_from_full_fir4_sum_current_exec_plus_y_from_forward_h1"
    zarr_meta.attrs["forward_delta_alignment"] = "forward_delta[t] corresponds to obs_t -> obs_t+1"
    zarr_meta.attrs["motion_full_fir_4"] = full_fir_4.round(8).tolist()
    zarr_meta.attrs["motion_full_fir_sum_4"] = full_fir_4.sum(axis=0).round(8).tolist()
    zarr_meta.attrs["exec_motion_xyz_positive_scale_median"] = pos_scale.round(8).tolist()
    zarr_meta.attrs["exec_motion_xyz_negative_scale_median"] = neg_scale.round(8).tolist()
    zarr_meta.attrs["exec_motion_xyz_positive_count"] = pos_count.tolist()
    zarr_meta.attrs["exec_motion_xyz_negative_count"] = neg_count.tolist()
    zarr_meta.attrs["exec_motion_xyz_scale_recommended"] = recommended_scale.round(8).tolist()

    summarize_xyz("exec_action", exec_action_arrays)
    summarize_xyz("motion_action", motion_action_arrays)
    summarize_xyz("forward_delta", forward_delta_arrays)

    cprint(f"Saved source zarr to {output_zarr}", "green")
    cprint(f"Full FIR4 response sum matrix: {full_fir_4.sum(axis=0).round(6).tolist()}", "green")


if __name__ == "__main__":
    main()
