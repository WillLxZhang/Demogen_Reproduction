"""
Lightweight replay-calibrated convert.

Compared with convert_robomimic_hdf5_to_zarr_exec_replay_h1.py:
- keeps the original heavy version intact
- reloads XML model only once per episode
- per-frame calibration only does set_state -> low-dim obs -> step -> low-dim obs
- never builds camera images / point clouds from the replay env
- supports calibrating only a prefix window for speed
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import robosuite
import zarr
from termcolor import cprint

from robosuite.controllers import load_composite_controller_config

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

from diffusion_policies.env.robosuite.robosuite_wrapper import (
    Robosuite3DEnv,
    load_env_metadata_from_dataset,
    sanitize_model_xml,
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
    parser.add_argument("--mask-object-name", default="cube")
    parser.add_argument("--mask-radius", type=float, default=0.045)
    parser.add_argument("--mask-dilation-iters", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control-steps", type=int, default=1)
    parser.add_argument(
        "--replay-prefix-frames",
        type=int,
        default=None,
        help="Only calibrate the first N frames of each episode. Tail falls back to forward_delta.",
    )
    parser.add_argument(
        "--replay-prefix-frames-per-episode",
        default=None,
        help=(
            "Optional per-episode replay prefix frames. Accepts comma-separated integers "
            "or a JSON list such as '[216, 367, 189, 343]'. Overrides --replay-prefix-frames."
        ),
    )
    parser.add_argument(
        "--episode-limit",
        type=int,
        default=None,
        help="Optional debug limit on number of episodes to convert.",
    )
    return parser.parse_args()


def parse_int_sequence(raw: str) -> list[int]:
    raw = str(raw).strip()
    if not raw:
        return []
    if raw.startswith("["):
        values = json.loads(raw)
    else:
        values = [part.strip() for part in raw.split(",") if part.strip()]
    return [int(value) for value in values]


def resolve_replay_prefix_frames(
    scalar_value: int | None,
    per_episode_value: str | None,
    num_episodes: int,
) -> list[int | None]:
    if per_episode_value is not None:
        resolved = parse_int_sequence(per_episode_value)
        if len(resolved) != num_episodes:
            raise ValueError(
                "Length mismatch for --replay-prefix-frames-per-episode: "
                f"expected {num_episodes}, got {len(resolved)}"
            )
        if any(value <= 0 for value in resolved):
            raise ValueError("--replay-prefix-frames-per-episode values must be positive")
        return resolved

    return [scalar_value] * num_episodes


class RobosuiteRobotStateEnv:
    def __init__(self, source_demo_path: str, control_steps: int):
        env_meta = load_env_metadata_from_dataset(source_demo_path)
        controller_config = None
        try:
            controller_config = load_composite_controller_config(
                controller="WHOLE_BODY_MINK_IK",
                robot="Panda",
            )
        except AssertionError:
            controller_config = None

        self._is_v1 = (robosuite.__version__.split(".")[0] == "1")
        self._robosuite_minor = int(robosuite.__version__.split(".")[1])
        self.control_steps = int(control_steps)

        env_kwargs = deepcopy(env_meta["env_kwargs"])
        env_kwargs["env_name"] = env_meta["env_name"].split("_")[0]
        if controller_config is not None:
            env_kwargs["controller_configs"] = controller_config

        env_kwargs.update(
            has_renderer=False,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=False,
        )

        self.env = robosuite.make(**env_kwargs)

    @staticmethod
    def _extract_agent_pos(obs_dict) -> np.ndarray:
        pos = np.asarray(obs_dict["robot0_eef_pos"], dtype=np.float32)
        rot = Robosuite3DEnv._convert_state_quat_to_action_rotvec(
            np.asarray(obs_dict["robot0_eef_quat"], dtype=np.float32)
        ).astype(np.float32)
        gripper_qpos = np.asarray(obs_dict["robot0_gripper_qpos"], dtype=np.float32)
        gripper = np.asarray([gripper_qpos[0] - gripper_qpos[1]], dtype=np.float32)
        return np.concatenate([pos, rot, gripper], axis=0).astype(np.float32)

    def _get_lowdim_obs(self):
        if self._is_v1:
            return self.env._get_observations(force_update=True)
        return self.env._get_observation()

    def load_episode_model(self, model_xml, ep_meta=None):
        if ep_meta is None:
            ep_meta = {}
        if self.is_v15_or_higher:
            self.env.set_ep_meta(ep_meta)
        self.env.reset()

        if self._robosuite_minor <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(model_xml)
        else:
            xml = self.env.edit_model_xml(model_xml)
        xml = sanitize_model_xml(xml)
        self.env.reset_from_xml_string(xml)
        self.env.sim.reset()

    def set_state(self, state_flat):
        self.env.sim.set_state_from_flattened(state_flat)
        self.env.sim.forward()

    def get_agent_pos(self):
        return self._extract_agent_pos(self._get_lowdim_obs())

    def step(self, action):
        obs_dict = None
        reward = 0.0
        done = False
        info = {}
        for _ in range(self.control_steps):
            obs_dict, reward, done, info = self.env.step(action)
        return self._extract_agent_pos(obs_dict), reward, done, info

    def close(self):
        self.env.close()

    @property
    def is_v15_or_higher(self):
        main_version = int(robosuite.__version__.split(".")[0])
        sub_version = int(robosuite.__version__.split(".")[1])
        return (main_version > 1) or (main_version == 1 and sub_version >= 5)


def build_replay_h1_delta_light(
    env: RobosuiteRobotStateEnv,
    demo_group,
    exec_action: np.ndarray,
    forward_delta: np.ndarray,
    replay_prefix_frames: int | None,
) -> tuple[np.ndarray, int]:
    replay_h1_delta = forward_delta.copy().astype(np.float32)
    calibrated_len = len(exec_action)
    if replay_prefix_frames is not None:
        calibrated_len = min(int(replay_prefix_frames), len(exec_action))

    ep_meta = None
    if "ep_meta" in demo_group.attrs:
        ep_meta = demo_group.attrs["ep_meta"]
        if isinstance(ep_meta, bytes):
            ep_meta = ep_meta.decode("utf-8")
        import json

        ep_meta = json.loads(ep_meta)

    env.load_episode_model(
        model_xml=demo_group.attrs["model_file"],
        ep_meta=ep_meta,
    )

    states = demo_group["states"][...]
    for frame_idx in range(calibrated_len):
        env.set_state(states[frame_idx])
        before = env.get_agent_pos()
        after, _, _, _ = env.step(exec_action[frame_idx])
        replay_h1_delta[frame_idx, :6] = after[:6] - before[:6]
        replay_h1_delta[frame_idx, 6] = exec_action[frame_idx, 6]

    replay_h1_delta[:, 6] = exec_action[:, 6]
    return replay_h1_delta.astype(np.float32), int(calibrated_len)


def main():
    args = parse_args()
    if args.control_steps <= 0:
        raise ValueError("--control-steps must be positive")
    if args.replay_prefix_frames is not None and args.replay_prefix_frames <= 0:
        raise ValueError("--replay-prefix-frames must be positive when provided")
    if args.episode_limit is not None and args.episode_limit <= 0:
        raise ValueError("--episode-limit must be positive when provided")

    rng = np.random.default_rng(args.seed)
    output_zarr = Path(args.output_zarr).expanduser()
    ensure_parent(output_zarr)
    sam_mask_root = output_zarr.parents[2] / "sam_mask"

    env = RobosuiteRobotStateEnv(
        source_demo_path=str(Path(args.demo_hdf5).expanduser()),
        control_steps=int(args.control_steps),
    )

    state_arrays_ls = []
    exec_action_arrays_ls = []
    motion_action_arrays_ls = []
    replay_h1_delta_arrays_ls = []
    forward_delta_arrays_ls = []
    point_cloud_arrays_ls = []
    episode_ends = []
    replay_h1_calibrated_frame_count = []
    calibrated_exec_segments = []
    calibrated_motion_segments = []
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

            if args.episode_limit is not None:
                demo_keys = demo_keys[: args.episode_limit]

            replay_prefix_frames_per_episode = resolve_replay_prefix_frames(
                scalar_value=args.replay_prefix_frames,
                per_episode_value=args.replay_prefix_frames_per_episode,
                num_episodes=len(demo_keys),
            )

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
                replay_h1_delta, calibrated_len = build_replay_h1_delta_light(
                    env=env,
                    demo_group=demo_group,
                    exec_action=exec_action,
                    forward_delta=forward_delta,
                    replay_prefix_frames=replay_prefix_frames_per_episode[episode_idx],
                )
                motion_action = replay_h1_delta.copy()

                if calibrated_len > 0:
                    calibrated_exec_segments.append(exec_action[:calibrated_len, :])
                    calibrated_motion_segments.append(replay_h1_delta[:calibrated_len, :])
                replay_h1_calibrated_frame_count.append(int(calibrated_len))

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

    if not calibrated_exec_segments:
        raise ValueError("No calibrated replay_h1 frames were collected")

    calibrated_exec = np.concatenate(calibrated_exec_segments, axis=0).astype(np.float32)
    calibrated_motion = np.concatenate(calibrated_motion_segments, axis=0).astype(np.float32)
    pos_scale, neg_scale, pos_count, neg_count, recommended_scale = compute_exec_scale_stats(
        exec_action=calibrated_exec,
        motion_action=calibrated_motion,
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
        "replay_h1_delta_for_calibrated_prefix_and_forward_delta_for_uncalibrated_tail"
    )
    zarr_meta.attrs["forward_delta_alignment"] = "forward_delta[t] corresponds to obs_t -> obs_t+1"
    zarr_meta.attrs["replay_h1_alignment"] = "replay_h1_delta[t] corresponds to set_state(obs_t) then step(action_t)"
    zarr_meta.attrs["replay_h1_control_steps"] = int(args.control_steps)
    zarr_meta.attrs["replay_h1_prefix_frames"] = (
        None if args.replay_prefix_frames is None else int(args.replay_prefix_frames)
    )
    if args.replay_prefix_frames_per_episode is not None:
        zarr_meta.attrs["replay_h1_prefix_frames_per_episode"] = replay_prefix_frames_per_episode
    zarr_meta.attrs["replay_h1_calibrated_frame_count_per_episode"] = replay_h1_calibrated_frame_count
    zarr_meta.attrs["exec_motion_xyz_positive_scale_median"] = pos_scale.tolist()
    zarr_meta.attrs["exec_motion_xyz_negative_scale_median"] = neg_scale.tolist()
    zarr_meta.attrs["exec_motion_xyz_positive_count"] = pos_count.tolist()
    zarr_meta.attrs["exec_motion_xyz_negative_count"] = neg_count.tolist()
    zarr_meta.attrs["exec_motion_xyz_scale_recommended"] = recommended_scale.tolist()

    cprint("-" * 50, "cyan")
    cprint(f"Saved zarr file to {output_zarr}", "green")
    summarize_xyz("exec action", exec_action_arrays)
    summarize_xyz("motion_action", motion_action_arrays)
    summarize_xyz("replay_h1_delta", replay_h1_delta_arrays)
    summarize_xyz("forward_delta", forward_delta_arrays)
    if args.replay_prefix_frames_per_episode is not None:
        cprint(
            f"replay_h1_prefix_frames_per_episode={replay_prefix_frames_per_episode}",
            "cyan",
        )
    cprint(f"replay_h1_calibrated_frame_count_per_episode={replay_h1_calibrated_frame_count}", "cyan")
    cprint(
        f"recommended motion_exec_pulse_scale_xyz={np.round(recommended_scale, 6).tolist()}",
        "cyan",
    )


if __name__ == "__main__":
    main()
