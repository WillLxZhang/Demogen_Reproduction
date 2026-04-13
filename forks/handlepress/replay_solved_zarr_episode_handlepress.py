#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio
import numpy as np
import zarr


REPRO_ROOT = Path(__file__).resolve().parents[2]
DEMOGEN_ROOT = REPRO_ROOT / "repos" / "DemoGen"
DEMOGEN_PKG_ROOT = DEMOGEN_ROOT / "demo_generation"
DIFFUSION_POLICIES_ROOT = DEMOGEN_ROOT / "diffusion_policies"
SCRIPTS_ROOT = REPRO_ROOT / "scripts"

for path in [DEMOGEN_PKG_ROOT, DIFFUSION_POLICIES_ROOT, SCRIPTS_ROOT]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from demo_generation.handlepress_robosuite_wrapper import HandlePressRobosuite3DEnv
from diffusion_policies.common.replay_buffer import ReplayBuffer
import diffusion_policies.env.robosuite.robosuite_wrapper as robosuite_wrapper
from replay_zarr_episode import ensure_parent, load_reset_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one solved HandlePress zarr episode into a video."
    )
    parser.add_argument("--zarr", required=True, help="Path to solved HandlePress zarr")
    parser.add_argument("--source-demo", required=True, help="Path to source demo.hdf5")
    parser.add_argument("--episode", type=int, required=True, help="Solved episode index")
    parser.add_argument("--output-video", required=True, help="Video output path")
    parser.add_argument("--render-size", type=int, default=256, help="Rendered frame size")
    parser.add_argument("--fps", type=int, default=20, help="Video FPS")
    parser.add_argument("--camera", default="agentview", help="Camera name")
    parser.add_argument("--control-steps", type=int, default=1, help="Internal control repeats per action")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on replay steps")
    return parser.parse_args()


def load_episode_meta(root, episode_idx: int) -> dict:
    meta = root["meta"]
    return {
        "source_episode_idx": int(meta["source_episode_idx"][episode_idx]),
        "object_translation": np.asarray(meta["object_translation"][episode_idx], dtype=np.float32),
        "motion_frame_count": int(meta["motion_frame_count"][episode_idx]),
    }


def main() -> None:
    args = parse_args()
    zarr_path = Path(args.zarr).expanduser().resolve()
    source_demo_path = Path(args.source_demo).expanduser().resolve()
    output_video = Path(args.output_video).expanduser().resolve()
    if args.control_steps <= 0:
        raise ValueError("--control-steps must be positive")

    replay_buffer = ReplayBuffer.copy_from_path(
        str(zarr_path),
        keys=["action", "agent_pos"],
    )
    if args.episode < 0 or args.episode >= replay_buffer.n_episodes:
        raise IndexError(
            f"episode {args.episode} out of range for dataset with {replay_buffer.n_episodes} episodes"
        )

    root = zarr.open(str(zarr_path), mode="r")
    episode_meta = load_episode_meta(root, args.episode)
    episode = replay_buffer.get_episode(args.episode, copy=True)
    actions = np.asarray(episode["action"], dtype=np.float32)
    if args.max_steps is not None:
        actions = actions[: args.max_steps]

    reset_state = load_reset_state(source_demo_path, episode_meta["source_episode_idx"])
    reset_state["object_translation"] = np.asarray(
        episode_meta["object_translation"], dtype=np.float32
    )

    robosuite_wrapper.N_CONTROL_STEPS = int(args.control_steps)
    env = HandlePressRobosuite3DEnv(
        str(source_demo_path),
        render=False,
        cam_width=args.render_size,
        cam_height=args.render_size,
        render_cam=args.camera,
    )
    env.reset()
    env.reset_to(reset_state)

    ensure_parent(output_video)
    frames = [
        env.render(
            mode="rgb_array",
            height=args.render_size,
            width=args.render_size,
            camera_name=args.camera,
        )
    ]
    success_trace: list[bool] = []

    for action in actions:
        env.step(action)
        success_trace.append(bool(env.check_success()))
        frames.append(
            env.render(
                mode="rgb_array",
                height=args.render_size,
                width=args.render_size,
                camera_name=args.camera,
            )
        )

    with imageio.get_writer(output_video, fps=args.fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    summary = {
        "zarr": str(zarr_path),
        "source_demo": str(source_demo_path),
        "episode": int(args.episode),
        "source_episode_idx": int(episode_meta["source_episode_idx"]),
        "n_actions": int(len(actions)),
        "motion_frame_count": int(episode_meta["motion_frame_count"]),
        "control_steps": int(args.control_steps),
        "object_translation": episode_meta["object_translation"].round(6).tolist(),
        "success_any": bool(any(success_trace)),
        "output_video": str(output_video),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
