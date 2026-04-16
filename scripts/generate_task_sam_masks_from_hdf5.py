#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import h5py


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_WORLD_ROOT = REPO_ROOT / "repos" / "DemoGen" / "real_world"
DIFFUSION_POLICIES_ROOT = REPO_ROOT / "repos" / "DemoGen" / "diffusion_policies"
for root in (REAL_WORLD_ROOT, DIFFUSION_POLICIES_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from convert_robomimic_hdf5_to_zarr_exec_motion import (  # noqa: E402
    load_camera_info,
    reconstruct_point_cloud,
    save_mask_assets,
    sorted_demo_keys,
)
from diffusion_policies.env.robosuite.robosuite_wrapper import Robosuite3DEnv  # noqa: E402

from relalign_task_spec import capture_body_xyz, load_relalign_env_name  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate task-aware SAM-style binary masks from a raw / converted robosuite "
            "demo.hdf5 plus depth.hdf5. Useful for two-stage tasks whose target mask is "
            "not auto-exported by the source zarr converter."
        )
    )
    parser.add_argument("--demo-hdf5", required=True, help="Path to selected demo.hdf5")
    parser.add_argument("--depth-hdf5", required=True, help="Path to matching depth.hdf5")
    parser.add_argument("--source-name", required=True, help="Source bundle name under data/sam_mask/")
    parser.add_argument(
        "--output-data-root",
        default=str(REPO_ROOT / "data"),
        help="Root that contains sam_mask/, typically the repo data directory.",
    )
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument(
        "--episodes",
        default="all",
        help="Comma-separated episode indices / demo keys, or 'all'.",
    )
    parser.add_argument("--mask-object-name", default=None, help="Mask filename stem for the moving object")
    parser.add_argument("--mask-target-name", default=None, help="Mask filename stem for the target object")
    parser.add_argument("--mask-radius", type=float, default=0.045)
    parser.add_argument("--mask-dilation-iters", type=int, default=2)
    return parser.parse_args()


def numeric_suffix(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    if match:
        return int(match.group(1)), name
    return math.inf, name


def parse_episode_selection(raw: str, demo_keys: list[str]) -> list[tuple[int, str]]:
    if raw.strip().lower() == "all":
        return list(enumerate(demo_keys))

    selected: list[tuple[int, str]] = []
    for token in [part.strip() for part in raw.split(",") if part.strip()]:
        if token in demo_keys:
            selected.append((demo_keys.index(token), token))
            continue
        idx = int(token)
        if idx < 0 or idx >= len(demo_keys):
            raise IndexError(f"episode {idx} out of range for demos={demo_keys}")
        selected.append((idx, demo_keys[idx]))
    if not selected:
        raise ValueError("No episodes selected")
    return selected


def build_reset_state(ep_group: h5py.Group) -> dict:
    state = {
        "states": ep_group["states"][0],
        "model": ep_group.attrs["model_file"],
    }
    if "ep_meta" in ep_group.attrs:
        state["ep_meta"] = ep_group.attrs["ep_meta"]
    return state


def main() -> None:
    args = parse_args()
    demo_hdf5 = Path(args.demo_hdf5).expanduser().resolve()
    depth_hdf5 = Path(args.depth_hdf5).expanduser().resolve()
    output_data_root = Path(args.output_data_root).expanduser().resolve()
    sam_mask_root = output_data_root / "sam_mask"

    if args.mask_object_name is None and args.mask_target_name is None:
        raise ValueError("At least one of --mask-object-name or --mask-target-name must be provided")

    env_name = load_relalign_env_name(demo_hdf5)
    env = Robosuite3DEnv(str(demo_hdf5), render=False)

    try:
        with h5py.File(demo_hdf5, "r") as demo_f, h5py.File(depth_hdf5, "r") as depth_f:
            demo_keys = sorted(sorted_demo_keys(demo_f["data"]), key=numeric_suffix)
            depth_keys = sorted(sorted_demo_keys(depth_f["data"]), key=numeric_suffix)
            if demo_keys != depth_keys:
                raise ValueError("demo.hdf5 and depth.hdf5 do not share the same episode keys")

            selected = parse_episode_selection(args.episodes, demo_keys)
            for episode_idx, demo_key in selected:
                demo_group = demo_f["data"][demo_key]
                depth_group = depth_f["data"][demo_key]
                depth_obs = depth_group["obs"]

                camera_info = load_camera_info(depth_group.attrs["camera_info"], args.camera_name)
                first_rgb = depth_obs[f"{args.camera_name}_image"][0]
                first_depth = depth_obs[f"{args.camera_name}_depth"][0, ..., 0]
                first_points = reconstruct_point_cloud(
                    depth=first_depth,
                    rgb=first_rgb,
                    intrinsics=camera_info["intrinsics"],
                    extrinsics=camera_info["extrinsics"],
                )

                env.reset_to(build_reset_state(demo_group))
                payload = {
                    "episode_idx": int(episode_idx),
                    "demo_key": demo_key,
                    "env_name": env_name,
                }

                if args.mask_object_name is not None:
                    object_center = capture_body_xyz(env, env_name, "object")
                    save_mask_assets(
                        sam_mask_root=sam_mask_root,
                        source_name=args.source_name,
                        episode_idx=episode_idx,
                        first_rgb=first_rgb,
                        first_points=first_points,
                        first_object_center=object_center,
                        camera_info=camera_info,
                        mask_object_name=args.mask_object_name,
                        mask_radius=args.mask_radius,
                        mask_dilation_iters=args.mask_dilation_iters,
                    )
                    payload["object_center"] = [float(v) for v in object_center]

                if args.mask_target_name is not None:
                    target_center = capture_body_xyz(env, env_name, "target")
                    save_mask_assets(
                        sam_mask_root=sam_mask_root,
                        source_name=args.source_name,
                        episode_idx=episode_idx,
                        first_rgb=first_rgb,
                        first_points=first_points,
                        first_object_center=target_center,
                        camera_info=camera_info,
                        mask_object_name=args.mask_target_name,
                        mask_radius=args.mask_radius,
                        mask_dilation_iters=args.mask_dilation_iters,
                    )
                    payload["target_center"] = [float(v) for v in target_center]

                print(json.dumps(payload, ensure_ascii=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
