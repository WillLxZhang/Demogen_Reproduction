from __future__ import annotations

"""
Build a copied source zarr whose pre-skill motion_action matches the baseline
zero-translation schedule implied by the original one-stage DemoGen logic.

Why this exists:
- original DemoGen does not consume observed framewise motion directly
- during motion generation it synthesizes a hand-crafted schedule and compares
  it against the source motion interface
- for Lift, we still want to keep `data/action` as the raw executable replay
  interface, but make `data/motion_action` look like the schedule DemoGen
  actually subtracts against
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
    parser.add_argument("--use-linear-interpolation", action="store_true")
    parser.add_argument("--z-step-size", type=float, default=0.015)
    parser.add_argument("--copy-sam-mask", action="store_true")
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


def build_original_one_stage_schedule(
    state: np.ndarray,
    skill1_frame: int,
    use_linear_interpolation: bool,
    z_step_size: float,
):
    state = np.asarray(state, dtype=np.float32)
    skill1_frame = min(int(skill1_frame), len(state))
    if skill1_frame <= 0:
        return np.zeros((0, 3), dtype=np.float32), {
            "skill1_frame": 0,
            "xy_stage_frame": 0,
            "z_step_num": 0,
        }

    start_pos = state[0, :3].astype(np.float32).copy()
    end_pos = state[skill1_frame - 1, :3].astype(np.float32).copy()

    if use_linear_interpolation:
        step_action = ((end_pos - start_pos) / float(skill1_frame)).astype(np.float32)
        schedule = np.repeat(step_action[None, :], skill1_frame, axis=0).astype(np.float32)
        meta = {
            "skill1_frame": int(skill1_frame),
            "xy_stage_frame": int(skill1_frame),
            "z_step_num": 0,
        }
        return schedule, meta

    xy_stage_frame = int(skill1_frame)
    step_actions = []
    z_action = float(end_pos[2] - start_pos[2])
    xy_action = (end_pos[:2] - start_pos[:2]).astype(np.float32)

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
    if len(step_actions) < skill1_frame:
        pad = [np.zeros(3, dtype=np.float32) for _ in range(skill1_frame - len(step_actions))]
        step_actions.extend(pad)
    schedule = np.asarray(step_actions[:skill1_frame], dtype=np.float32)

    meta = {
        "skill1_frame": int(skill1_frame),
        "xy_stage_frame": int(max(0, xy_stage_frame)),
        "z_step_num": int(z_step_num),
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


def main():
    args = parse_args()
    input_zarr = Path(args.input_zarr).expanduser().resolve()
    output_zarr = Path(args.output_zarr).expanduser().resolve()
    output_zarr.parent.mkdir(parents=True, exist_ok=True)

    input_root = zarr.open(str(input_zarr), mode="r")
    input_data = input_root["data"]
    input_meta = input_root["meta"]

    state = np.asarray(input_data["state"], dtype=np.float32)
    action = np.asarray(input_data["action"], dtype=np.float32)
    point_cloud = np.asarray(input_data["point_cloud"], dtype=np.float32)
    raw_action = (
        np.asarray(input_data["raw_action"], dtype=np.float32)
        if "raw_action" in input_data
        else action.copy()
    )
    motion_action = (
        np.asarray(input_data["motion_action"], dtype=np.float32)
        if "motion_action" in input_data
        else action.copy()
    )
    forward_delta = (
        np.asarray(input_data["forward_delta"], dtype=np.float32)
        if "forward_delta" in input_data
        else None
    )
    episode_ends = np.asarray(input_meta["episode_ends"], dtype=np.int64)

    new_motion = motion_action.copy().astype(np.float32)
    schedule_skill1 = []
    schedule_xy_stage = []
    schedule_z_steps = []

    start = 0
    for end in episode_ends:
        end = int(end)
        ep_skill1 = min(args.skill1_frame, end - start)
        schedule_xyz, meta = build_original_one_stage_schedule(
            state=state[start:end],
            skill1_frame=ep_skill1,
            use_linear_interpolation=bool(args.use_linear_interpolation),
            z_step_size=float(args.z_step_size),
        )
        new_motion[start : start + ep_skill1, :3] = schedule_xyz
        new_motion[start : start + ep_skill1, 6] = action[start : start + ep_skill1, 6]
        schedule_skill1.append(int(meta["skill1_frame"]))
        schedule_xy_stage.append(int(meta["xy_stage_frame"]))
        schedule_z_steps.append(int(meta["z_step_num"]))
        start = end

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
    create_dataset(out_data, "point_cloud", point_cloud, compressor)
    create_dataset(out_meta, "episode_ends", episode_ends, compressor)

    for key, value in dict(input_meta.attrs).items():
        out_meta.attrs[key] = value
    out_meta.attrs["source_name"] = str(args.source_name)
    out_meta.attrs["source_parent_zarr"] = str(input_zarr)
    out_meta.attrs["motion_action_semantics"] = (
        "pre_skill1_original_one_stage_zero_translation_schedule_post_skill1_parent_motion_action"
    )
    out_meta.attrs["skill1_frame_for_schedule"] = int(args.skill1_frame)
    out_meta.attrs["use_linear_interpolation_for_schedule"] = bool(args.use_linear_interpolation)
    out_meta.attrs["z_step_size_for_schedule"] = float(args.z_step_size)
    out_meta.attrs["schedule_skill1_per_episode"] = json.dumps(schedule_skill1)
    out_meta.attrs["schedule_xy_stage_frame_per_episode"] = json.dumps(schedule_xy_stage)
    out_meta.attrs["schedule_z_step_num_per_episode"] = json.dumps(schedule_z_steps)

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
    summarize_xyz("parent_motion_action", motion_action)
    summarize_xyz("raw_action", raw_action)
    cprint(f"schedule_skill1_per_episode={schedule_skill1}", "green")
    cprint(f"schedule_xy_stage_frame_per_episode={schedule_xy_stage}", "green")
    cprint(f"schedule_z_step_num_per_episode={schedule_z_steps}", "green")


if __name__ == "__main__":
    main()
