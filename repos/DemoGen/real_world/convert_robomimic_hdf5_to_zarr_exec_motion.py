"""
Convert robomimic / robosuite HDF5 exports into a DemoGen source zarr dataset
with separate executable and geometric motion interfaces.

This converter is designed for tasks like Lift where the raw robosuite action
stored in the dataset is a sparse controller command rather than the geometric
motion that DemoGen expects for retargeting.

Produced keys:
- data/action: executable controller action for replay / env stepping
- data/motion_action: forward-aligned realized ee motion for DemoGen retargeting
- data/raw_action: alias of the original controller command for provenance
- data/forward_delta: forward-aligned state delta for debugging
"""

import argparse
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import zarr
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation as R
from termcolor import cprint

try:
    import fpsample
except ImportError:  # pragma: no cover - fallback for lighter envs
    fpsample = None


ACTION_FRAME_ROT_OFFSET = R.from_rotvec([0.0, 0.0, -1.5707]).as_matrix()


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
    return parser.parse_args()


def sorted_demo_keys(group):
    def demo_order(name):
        suffix = name.split("_")[-1]
        return int(suffix)

    return sorted(group.keys(), key=demo_order)


def load_camera_info(raw_attr, camera_name):
    if isinstance(raw_attr, bytes):
        raw_attr = raw_attr.decode("utf-8")
    camera_info = json.loads(raw_attr)
    if camera_name not in camera_info:
        raise KeyError(f"Camera '{camera_name}' not found in camera_info")
    info = camera_info[camera_name]
    intrinsics = np.asarray(info["intrinsics"], dtype=np.float32)
    extrinsics = np.asarray(info["extrinsics"], dtype=np.float32)
    return {
        "camera_name": camera_name,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
    }


def world_to_camera(extrinsics):
    rot = extrinsics[:3, :3]
    trans = extrinsics[:3, 3]
    world_to_cam = np.eye(4, dtype=np.float32)
    world_to_cam[:3, :3] = rot.T
    world_to_cam[:3, 3] = -rot.T @ trans
    return world_to_cam


def build_workspace_bounds(object_positions, eef_positions, margin_xy, margin_z):
    xyz = np.concatenate([object_positions, eef_positions], axis=0)
    lower = xyz.min(axis=0) - np.array([margin_xy, margin_xy, margin_z], dtype=np.float32)
    upper = xyz.max(axis=0) + np.array([margin_xy, margin_xy, margin_z], dtype=np.float32)
    return lower.astype(np.float32), upper.astype(np.float32)


def reconstruct_point_cloud(depth, rgb, intrinsics, extrinsics):
    depth = depth.astype(np.float32)
    rgb = rgb.astype(np.float32)
    height, width = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    uu, vv = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    z = depth.reshape(-1)
    valid = np.isfinite(z) & (z > 0)

    x = (uu.reshape(-1) - cx) * z / fx
    y = (vv.reshape(-1) - cy) * z / fy
    points_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)[valid]
    colors = rgb.reshape(-1, 3)[valid]

    points_world = (extrinsics @ points_cam.T).T[:, :3]
    return np.concatenate([points_world, colors], axis=1).astype(np.float32)


def crop_workspace(points, lower, upper):
    keep = np.all((points[:, :3] >= lower) & (points[:, :3] <= upper), axis=1)
    cropped = points[keep]
    return cropped if len(cropped) > 0 else points


def sample_point_cloud(points, n_points, rng):
    if len(points) == 0:
        return np.zeros((n_points, 6), dtype=np.float32)

    xyz = points[:, :3].astype(np.float32)
    if len(points) >= n_points:
        if fpsample is not None:
            try:
                indices = fpsample.bucket_fps_kdline_sampling(xyz, n_points, h=3)
            except Exception:
                indices = rng.choice(len(points), size=n_points, replace=False)
        else:
            indices = rng.choice(len(points), size=n_points, replace=False)
    else:
        extra = rng.choice(len(points), size=n_points - len(points), replace=True)
        indices = np.concatenate([np.arange(len(points)), extra], axis=0)

    return points[np.asarray(indices, dtype=np.int64)].astype(np.float32)


def quaternion_to_action_frame_rotvec(quat_xyzw):
    state_rot = R.from_quat(quat_xyzw)
    action_frame_rot = state_rot.as_matrix() @ ACTION_FRAME_ROT_OFFSET
    return R.from_matrix(action_frame_rot).as_rotvec().astype(np.float32)


def build_state(obs_group):
    eef_pos = obs_group["robot0_eef_pos"][...].astype(np.float32)
    eef_quat = obs_group["robot0_eef_quat"][...].astype(np.float32)
    rotvec = quaternion_to_action_frame_rotvec(eef_quat)
    gripper_qpos = obs_group["robot0_gripper_qpos"][...].astype(np.float32)
    gripper_gap = (gripper_qpos[:, 0] - gripper_qpos[:, 1])[:, None]
    return np.concatenate([eef_pos, rotvec, gripper_gap], axis=1).astype(np.float32)


def build_exec_action(state, demo_group):
    if "actions" in demo_group:
        action = demo_group["actions"][...].astype(np.float32)
        if action.shape[1] != state.shape[1]:
            raise ValueError(
                f"Expected action dim {state.shape[1]}, got {action.shape[1]}"
            )
        if "action_dict" in demo_group and "gripper" in demo_group["action_dict"]:
            action[:, 6:7] = demo_group["action_dict"]["gripper"][...].astype(np.float32)
        return action

    action = np.zeros_like(state, dtype=np.float32)
    if "action_dict" in demo_group and "gripper" in demo_group["action_dict"]:
        action[:, 6:7] = demo_group["action_dict"]["gripper"][...].astype(np.float32)
    else:
        action[:, 6:7] = state[:, 6:7]
    return action.astype(np.float32)


def build_forward_delta(state, obs_group, exec_action):
    forward_delta = np.zeros_like(state, dtype=np.float32)
    if len(state) > 1:
        forward_delta[:-1, :3] = state[1:, :3] - state[:-1, :3]

        eef_quat = obs_group["robot0_eef_quat"][...].astype(np.float32)
        eef_rot = R.from_quat(eef_quat)
        rot_delta = (eef_rot[1:] * eef_rot[:-1].inv()).as_rotvec().astype(np.float32)
        forward_delta[:-1, 3:6] = rot_delta

    forward_delta[:, 6] = exec_action[:, 6]
    return forward_delta.astype(np.float32)


def build_motion_action(forward_delta, exec_action):
    motion_action = forward_delta.copy().astype(np.float32)
    motion_action[:, 6] = exec_action[:, 6]
    return motion_action.astype(np.float32)


def project_world_points(points_xyz, intrinsics, extrinsics, image_size):
    width, height = image_size
    cam_points = (world_to_camera(extrinsics) @ np.concatenate(
        [points_xyz, np.ones((len(points_xyz), 1), dtype=np.float32)], axis=1
    ).T).T[:, :3]

    valid = cam_points[:, 2] > 1e-6
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.int32), valid

    cam_points = cam_points[valid]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    px = np.floor((cam_points[:, 0] * fx / cam_points[:, 2]) + cx).astype(np.int32)
    py = np.floor((cam_points[:, 1] * fy / cam_points[:, 2]) + cy).astype(np.int32)

    in_frame = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    pixels = np.stack([px[in_frame], py[in_frame]], axis=1)

    full_valid = np.zeros(len(valid), dtype=bool)
    full_valid[np.where(valid)[0][in_frame]] = True
    return pixels, full_valid


def generate_object_mask(points_world, object_center, intrinsics, extrinsics, image_shape, radius, dilation_iters):
    height, width = image_shape
    distances = np.linalg.norm(points_world[:, :3] - object_center[None, :], axis=1)
    object_points = points_world[distances <= radius]

    mask = np.zeros((height, width), dtype=bool)
    if len(object_points) > 0:
        pixels, _ = project_world_points(
            object_points[:, :3],
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_size=(width, height),
        )
        if len(pixels) > 0:
            mask[pixels[:, 1], pixels[:, 0]] = True

    if dilation_iters > 0 and np.any(mask):
        mask = binary_dilation(mask, iterations=dilation_iters)

    return (mask.astype(np.uint8) * 255)


def save_mask_assets(
    sam_mask_root,
    source_name,
    episode_idx,
    first_rgb,
    first_points,
    first_object_center,
    camera_info,
    mask_object_name,
    mask_radius,
    mask_dilation_iters,
):
    episode_dir = Path(sam_mask_root) / source_name / str(episode_idx)
    episode_dir.mkdir(parents=True, exist_ok=True)

    imageio.imwrite(episode_dir / "source.jpg", first_rgb.astype(np.uint8))

    camera_info_payload = {
        "camera_name": camera_info["camera_name"],
        "intrinsics": camera_info["intrinsics"].tolist(),
        "extrinsics": camera_info["extrinsics"].tolist(),
        "image_size": [int(first_rgb.shape[1]), int(first_rgb.shape[0])],
    }
    with open(episode_dir / "camera_info.json", "w", encoding="utf-8") as f:
        json.dump(camera_info_payload, f, indent=2)

    mask = generate_object_mask(
        points_world=first_points,
        object_center=first_object_center.astype(np.float32),
        intrinsics=camera_info["intrinsics"],
        extrinsics=camera_info["extrinsics"],
        image_shape=first_rgb.shape[:2],
        radius=mask_radius,
        dilation_iters=mask_dilation_iters,
    )
    imageio.imwrite(episode_dir / f"{mask_object_name}.jpg", mask)


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


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


def compute_exec_scale_stats(exec_action, motion_action, action_threshold=1e-4):
    pos_scale = np.zeros(3, dtype=np.float32)
    neg_scale = np.zeros(3, dtype=np.float32)
    pos_count = np.zeros(3, dtype=np.int64)
    neg_count = np.zeros(3, dtype=np.int64)
    recommended_scale = np.zeros(3, dtype=np.float32)

    for axis in range(3):
        exec_axis = exec_action[:, axis]
        motion_axis = motion_action[:, axis]
        nonzero_mask = np.abs(exec_axis) > action_threshold

        pos_mask = exec_axis > action_threshold
        neg_mask = exec_axis < -action_threshold

        if np.any(pos_mask):
            ratios = motion_axis[pos_mask] / exec_axis[pos_mask]
            pos_scale[axis] = np.median(ratios).astype(np.float32)
            pos_count[axis] = int(np.sum(pos_mask))
        if np.any(neg_mask):
            ratios = np.abs(motion_axis[neg_mask] / exec_axis[neg_mask])
            neg_scale[axis] = np.median(ratios).astype(np.float32)
            neg_count[axis] = int(np.sum(neg_mask))
        if np.any(nonzero_mask):
            ratios = np.abs(motion_axis[nonzero_mask] / exec_axis[nonzero_mask])
            recommended_scale[axis] = np.median(ratios).astype(np.float32)

    return pos_scale, neg_scale, pos_count, neg_count, recommended_scale


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    output_zarr = Path(args.output_zarr).expanduser()
    ensure_parent(output_zarr)

    sam_mask_root = output_zarr.parents[2] / "sam_mask"

    state_arrays_ls = []
    exec_action_arrays_ls = []
    motion_action_arrays_ls = []
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
            motion_action = build_motion_action(forward_delta, exec_action)

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
            forward_delta_arrays_ls.append(forward_delta)
            point_cloud_arrays_ls.append(point_cloud)

            count += len(state)
            episode_ends.append(count)

    state_arrays = np.concatenate(state_arrays_ls, axis=0).astype(np.float32)
    exec_action_arrays = np.concatenate(exec_action_arrays_ls, axis=0).astype(np.float32)
    motion_action_arrays = np.concatenate(motion_action_arrays_ls, axis=0).astype(np.float32)
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
    zarr_meta.attrs["motion_action_semantics"] = "forward_aligned_realized_ee_motion_for_demogen"
    zarr_meta.attrs["forward_delta_alignment"] = "motion_action[t] corresponds to obs_t -> obs_t+1"
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
    summarize_xyz("forward_delta", forward_delta_arrays)
    cprint(
        (
            "exec controller -> realized xyz scale medians: "
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
