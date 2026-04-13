from __future__ import annotations

"""
Build a copied source zarr for two-phase tasks.

Compared with the one-stage replay_h1 schedule fork:
- rewrite motion_action on [0, skill1) and [motion2, skill2)
- keep skill segments copied from the parent source zarr
- calibrate each motion segment total with replay_h1 semantics
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import zarr
from termcolor import cprint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-zarr", required=True)
    parser.add_argument("--output-zarr", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--skill1-frame", type=int, required=True)
    parser.add_argument("--motion2-frame", type=int, required=True)
    parser.add_argument("--skill2-frame", type=int, required=True)
    parser.add_argument("--use-linear-interpolation", action="store_true")
    parser.add_argument("--z-step-size", type=float, default=0.015)
    parser.add_argument("--copy-sam-mask", action="store_true")
    parser.add_argument(
        "--motion-source-key",
        default="replay_h1_delta",
        help="Key whose segment prefix sum is used to calibrate the schedule total.",
    )
    return parser.parse_args()


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


def build_original_one_stage_schedule_from_total(
    total_xyz: np.ndarray,
    segment_len: int,
    use_linear_interpolation: bool,
    z_step_size: float,
):
    total_xyz = np.asarray(total_xyz, dtype=np.float32)
    segment_len = int(segment_len)
    if segment_len <= 0:
        return np.zeros((0, 3), dtype=np.float32), {
            "segment_len": 0,
            "xy_stage_frame": 0,
            "z_step_num": 0,
            "schedule_total_xyz": [0.0, 0.0, 0.0],
        }

    if use_linear_interpolation:
        step_action = (total_xyz / float(segment_len)).astype(np.float32)
        schedule = np.repeat(step_action[None, :], segment_len, axis=0).astype(np.float32)
        meta = {
            "segment_len": int(segment_len),
            "xy_stage_frame": int(segment_len),
            "z_step_num": 0,
            "schedule_total_xyz": total_xyz.tolist(),
        }
        return schedule, meta

    xy_stage_frame = int(segment_len)
    step_actions = []
    z_action = float(total_xyz[2])
    xy_action = total_xyz[:2].astype(np.float32)

    if not np.isclose(z_action, 0.0):
        z_action = float(np.sign(z_action) * round(abs(z_action), 3))
        z_step_num = int(abs(z_action) / float(z_step_size))
        for _ in range(z_step_num):
            step_actions.append(np.array([0.0, 0.0, np.sign(z_action) * z_step_size], dtype=np.float32))
            xy_stage_frame -= 1
    else:
        z_step_num = 0

    if xy_stage_frame > 0:
        xy_step = (xy_action / float(xy_stage_frame)).astype(np.float32)
        for _ in range(xy_stage_frame):
            step_actions.append(np.array([xy_step[0], xy_step[1], 0.0], dtype=np.float32))

    step_actions = step_actions[::-1]
    if len(step_actions) < segment_len:
        pad = [np.zeros(3, dtype=np.float32) for _ in range(segment_len - len(step_actions))]
        step_actions.extend(pad)
    schedule = np.asarray(step_actions[:segment_len], dtype=np.float32)

    meta = {
        "segment_len": int(segment_len),
        "xy_stage_frame": int(max(0, xy_stage_frame)),
        "z_step_num": int(z_step_num),
        "schedule_total_xyz": total_xyz.tolist(),
    }
    return schedule, meta


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


def clamp_segment(start: int, end: int, ep_start: int, ep_end: int) -> tuple[int, int]:
    start = max(ep_start, start)
    end = min(ep_end, end)
    return start, max(start, end)


def main():
    args = parse_args()
    input_zarr = Path(args.input_zarr).expanduser().resolve()
    output_zarr = Path(args.output_zarr).expanduser().resolve()
    output_zarr.parent.mkdir(parents=True, exist_ok=True)

    if args.motion2_frame < args.skill1_frame:
        raise ValueError("motion2-frame must be >= skill1-frame")
    if args.skill2_frame < args.motion2_frame:
        raise ValueError("skill2-frame must be >= motion2-frame")

    input_root = zarr.open(str(input_zarr), mode="r")
    input_data = input_root["data"]
    input_meta = input_root["meta"]

    if args.motion_source_key not in input_data:
        raise KeyError(
            f"motion_source_key={args.motion_source_key} not found in input zarr. "
            f"available_keys={list(input_data.keys())}"
        )

    state = np.asarray(input_data["state"], dtype=np.float32)
    action = np.asarray(input_data["action"], dtype=np.float32)
    point_cloud = np.asarray(input_data["point_cloud"], dtype=np.float32)
    raw_action = (
        np.asarray(input_data["raw_action"], dtype=np.float32)
        if "raw_action" in input_data
        else action.copy()
    )
    parent_motion_action = (
        np.asarray(input_data["motion_action"], dtype=np.float32)
        if "motion_action" in input_data
        else action.copy()
    )
    motion_source = np.asarray(input_data[args.motion_source_key], dtype=np.float32)
    forward_delta = (
        np.asarray(input_data["forward_delta"], dtype=np.float32)
        if "forward_delta" in input_data
        else None
    )
    replay_h1_delta = (
        np.asarray(input_data["replay_h1_delta"], dtype=np.float32)
        if "replay_h1_delta" in input_data
        else None
    )
    episode_ends = np.asarray(input_meta["episode_ends"], dtype=np.int64)

    new_motion = parent_motion_action.copy().astype(np.float32)
    schedule_skill1 = []
    schedule_motion2 = []
    schedule_skill2 = []
    motion1_xy_stage = []
    motion1_z_steps = []
    motion1_total_xyz_ls = []
    motion2_xy_stage = []
    motion2_z_steps = []
    motion2_total_xyz_ls = []

    ep_start = 0
    for ep_end in episode_ends:
        ep_end = int(ep_end)
        m1_start, m1_end = clamp_segment(ep_start, ep_start + args.skill1_frame, ep_start, ep_end)
        m2_start, m2_end = clamp_segment(ep_start + args.motion2_frame, ep_start + args.skill2_frame, ep_start, ep_end)

        m1_len = m1_end - m1_start
        m2_len = m2_end - m2_start

        total_xyz_1 = motion_source[m1_start:m1_end, :3].sum(axis=0).astype(np.float32)
        total_xyz_2 = motion_source[m2_start:m2_end, :3].sum(axis=0).astype(np.float32)

        schedule_xyz_1, meta_1 = build_original_one_stage_schedule_from_total(
            total_xyz=total_xyz_1,
            segment_len=m1_len,
            use_linear_interpolation=bool(args.use_linear_interpolation),
            z_step_size=float(args.z_step_size),
        )
        schedule_xyz_2, meta_2 = build_original_one_stage_schedule_from_total(
            total_xyz=total_xyz_2,
            segment_len=m2_len,
            use_linear_interpolation=bool(args.use_linear_interpolation),
            z_step_size=float(args.z_step_size),
        )

        if m1_len > 0:
            new_motion[m1_start:m1_end, :3] = schedule_xyz_1
            new_motion[m1_start:m1_end, 6] = action[m1_start:m1_end, 6]
        if m2_len > 0:
            new_motion[m2_start:m2_end, :3] = schedule_xyz_2
            new_motion[m2_start:m2_end, 6] = action[m2_start:m2_end, 6]

        schedule_skill1.append(int(m1_end - ep_start))
        schedule_motion2.append(int(max(0, m2_start - ep_start)))
        schedule_skill2.append(int(max(0, m2_end - ep_start)))
        motion1_xy_stage.append(int(meta_1["xy_stage_frame"]))
        motion1_z_steps.append(int(meta_1["z_step_num"]))
        motion1_total_xyz_ls.append([float(x) for x in meta_1["schedule_total_xyz"]])
        motion2_xy_stage.append(int(meta_2["xy_stage_frame"]))
        motion2_z_steps.append(int(meta_2["z_step_num"]))
        motion2_total_xyz_ls.append([float(x) for x in meta_2["schedule_total_xyz"]])
        ep_start = ep_end

    if output_zarr.exists():
        shutil.rmtree(output_zarr)

    out_root = zarr.group(str(output_zarr), overwrite=True)
    out_data = out_root.create_group("data")
    out_meta = out_root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    create_dataset(out_data, "state", state, compressor)
    create_dataset(
        out_data,
        "agent_pos",
        np.asarray(input_data["agent_pos"], dtype=np.float32) if "agent_pos" in input_data else state,
        compressor,
    )
    create_dataset(out_data, "action", action, compressor)
    create_dataset(out_data, "raw_action", raw_action, compressor)
    create_dataset(out_data, "motion_action", new_motion, compressor)
    create_dataset(out_data, "motion_delta", new_motion, compressor)
    if forward_delta is not None:
        create_dataset(out_data, "forward_delta", forward_delta, compressor)
    if replay_h1_delta is not None:
        create_dataset(out_data, "replay_h1_delta", replay_h1_delta, compressor)
    create_dataset(out_data, "point_cloud", point_cloud, compressor)
    create_dataset(out_meta, "episode_ends", episode_ends, compressor)

    for key, value in dict(input_meta.attrs).items():
        out_meta.attrs[key] = value
    out_meta.attrs["source_name"] = str(args.source_name)
    out_meta.attrs["source_parent_zarr"] = str(input_zarr)
    out_meta.attrs["motion_action_semantics"] = (
        "two_phase_pre_skill1_and_motion2_original_schedule_from_replay_calibrated_segment_sums_"
        "post_segments_parent_motion_action"
    )
    out_meta.attrs["skill1_frame_for_schedule"] = int(args.skill1_frame)
    out_meta.attrs["motion2_frame_for_schedule"] = int(args.motion2_frame)
    out_meta.attrs["skill2_frame_for_schedule"] = int(args.skill2_frame)
    out_meta.attrs["use_linear_interpolation_for_schedule"] = bool(args.use_linear_interpolation)
    out_meta.attrs["z_step_size_for_schedule"] = float(args.z_step_size)
    out_meta.attrs["schedule_total_xyz_source_key"] = str(args.motion_source_key)
    out_meta.attrs["schedule_skill1_per_episode"] = json.dumps(schedule_skill1)
    out_meta.attrs["schedule_motion2_per_episode"] = json.dumps(schedule_motion2)
    out_meta.attrs["schedule_skill2_per_episode"] = json.dumps(schedule_skill2)
    out_meta.attrs["motion1_xy_stage_frame_per_episode"] = json.dumps(motion1_xy_stage)
    out_meta.attrs["motion1_z_step_num_per_episode"] = json.dumps(motion1_z_steps)
    out_meta.attrs["motion1_total_xyz_per_episode"] = json.dumps(motion1_total_xyz_ls)
    out_meta.attrs["motion2_xy_stage_frame_per_episode"] = json.dumps(motion2_xy_stage)
    out_meta.attrs["motion2_z_step_num_per_episode"] = json.dumps(motion2_z_steps)
    out_meta.attrs["motion2_total_xyz_per_episode"] = json.dumps(motion2_total_xyz_ls)

    if args.copy_sam_mask:
        data_root = input_zarr.parents[2]
        input_source_name = input_zarr.stem
        input_mask_dir = data_root / "sam_mask" / input_source_name
        output_mask_dir = data_root / "sam_mask" / args.source_name
        if input_mask_dir.exists():
            shutil.copytree(input_mask_dir, output_mask_dir, dirs_exist_ok=True)
            cprint(f"Copied sam_mask from {input_mask_dir} to {output_mask_dir}", "green")
        else:
            cprint(f"[WARN] Input sam_mask dir not found: {input_mask_dir}", "yellow")

    summarize_xyz("new_motion_action", new_motion)
    summarize_xyz("parent_motion_action", parent_motion_action)
    summarize_xyz(args.motion_source_key, motion_source)
    if replay_h1_delta is not None:
        summarize_xyz("replay_h1_delta", replay_h1_delta)
    if forward_delta is not None:
        summarize_xyz("forward_delta", forward_delta)
    cprint(f"Saved two-phase schedule source zarr to {output_zarr}", "green")


if __name__ == "__main__":
    main()
