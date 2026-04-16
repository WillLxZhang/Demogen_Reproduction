#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio
import numpy as np
import zarr


REPO_ROOT = Path(__file__).resolve().parents[1] / "repos" / "DemoGen"
DIFFUSION_POLICIES_ROOT = REPO_ROOT / "diffusion_policies"
if str(DIFFUSION_POLICIES_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICIES_ROOT))

from diffusion_policies.common.replay_buffer import ReplayBuffer
import diffusion_policies.env.robosuite.robosuite_wrapper as robosuite_wrapper
from diffusion_policies.env.robosuite.robosuite_wrapper import Robosuite3DEnv

from relalign_task_spec import apply_translation_to_reset_state, split_translation
from replay_zarr_episode import (
    ensure_parent,
    infer_source_episode,
    list_demo_keys,
    load_env_name,
    load_generated_episode_meta,
    load_reset_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one episode from a two-phase generated Zarr dataset in robosuite."
    )
    parser.add_argument("--zarr", required=True, help="Path to generated .zarr dataset")
    parser.add_argument("--source-demo", required=True, help="Path to source demo.hdf5")
    parser.add_argument("--episode", type=int, default=0, help="Generated episode index to replay")
    parser.add_argument("--source-episode", type=int, default=None, help="Optional source episode index")
    parser.add_argument("--output-video", required=True, help="Where to save the replay video")
    parser.add_argument("--render-size", type=int, default=256, help="Rendered video size")
    parser.add_argument("--fps", type=int, default=20, help="Video FPS")
    parser.add_argument("--control-steps", type=int, default=1, help="Internal control repeats")
    parser.add_argument("--camera", default=None, help="Render camera name")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on replay steps")
    parser.add_argument(
        "--translate-object",
        choices=["auto", "on", "off"],
        default="auto",
        help="Whether to apply stored object / target translation to the reset state.",
    )
    return parser.parse_args()

def resolve_translation(reset_state: dict, env_name: str, translation: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    object_translation, target_translation = split_translation(translation)
    if object_translation is None and target_translation is None:
        return None, None

    translated = apply_translation_to_reset_state(
        reset_state=reset_state,
        env_name=env_name,
        object_translation=object_translation,
        target_translation=target_translation,
    )
    reset_state["states"] = translated["states"]
    return object_translation, target_translation


def main() -> None:
    args = parse_args()
    zarr_path = Path(args.zarr).expanduser().resolve()
    source_demo_path = Path(args.source_demo).expanduser().resolve()
    output_video = Path(args.output_video).expanduser().resolve()
    if args.control_steps <= 0:
        raise ValueError("--control-steps must be positive")

    replay_buffer = ReplayBuffer.copy_from_path(
        str(zarr_path),
        keys=["agent_pos", "action", "point_cloud"],
    )

    n_episodes = replay_buffer.n_episodes
    if args.episode < 0 or args.episode >= n_episodes:
        raise IndexError(f"episode {args.episode} out of range, dataset has {n_episodes} episodes")

    episode_meta = load_generated_episode_meta(zarr_path, args.episode)
    demo_keys = list_demo_keys(source_demo_path)
    if args.source_episode is None:
        source_episode_idx = episode_meta.get("source_episode_idx")
        if source_episode_idx is None:
            source_episode_idx = infer_source_episode(args.episode, n_episodes, len(demo_keys))
    else:
        source_episode_idx = args.source_episode

    robosuite_wrapper.N_CONTROL_STEPS = args.control_steps

    reset_state = load_reset_state(source_demo_path, source_episode_idx)
    env_name = load_env_name(source_demo_path)
    applied_object_translation = None
    applied_target_translation = None
    if args.translate_object != "off":
        translation = episode_meta.get("object_translation")
        if translation is not None:
            applied_object_translation, applied_target_translation = resolve_translation(
                reset_state=reset_state,
                env_name=env_name,
                translation=translation,
            )
        elif args.translate_object == "on":
            raise FileNotFoundError(
                "Generated zarr meta is missing object_translation, cannot translate reset state."
            )

    episode = replay_buffer.get_episode(args.episode, copy=True)
    actions = np.asarray(episode["action"], dtype=np.float32)
    if args.max_steps is not None:
        actions = actions[: args.max_steps]

    env = Robosuite3DEnv(
        str(source_demo_path),
        render=False,
        cam_width=args.render_size,
        cam_height=args.render_size,
        render_cam=args.camera,
    )
    env.reset()
    env.reset_to(reset_state)

    ensure_parent(output_video)
    frames = [env.render(mode="rgb_array", height=args.render_size, width=args.render_size, camera_name=args.camera)]
    observed_agent_pos = []
    successes = []

    for action in actions:
        obs, _, _, _ = env.step(action)
        observed_agent_pos.append(obs["agent_pos"])
        successes.append(bool(env.check_success()))
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

    observed_agent_pos = np.asarray(observed_agent_pos, dtype=np.float32)
    pos = observed_agent_pos[:, :3] if len(observed_agent_pos) else np.zeros((0, 3), dtype=np.float32)
    summary = {
        "zarr": str(zarr_path),
        "source_demo": str(source_demo_path),
        "env_name": env_name,
        "episode": int(args.episode),
        "source_episode": int(source_episode_idx),
        "n_actions": int(len(actions)),
        "control_steps": int(args.control_steps),
        "object_translation": None if applied_object_translation is None else applied_object_translation.round(6).tolist(),
        "target_translation": None if applied_target_translation is None else applied_target_translation.round(6).tolist(),
        "video": str(output_video),
        "success_any": bool(any(successes)),
        "pos_min": pos.min(axis=0).round(6).tolist() if len(pos) else None,
        "pos_max": pos.max(axis=0).round(6).tolist() if len(pos) else None,
        "pos_span": (pos.max(axis=0) - pos.min(axis=0)).round(6).tolist() if len(pos) else None,
        "first_obs_agent_pos": observed_agent_pos[0].round(6).tolist() if len(observed_agent_pos) else None,
        "last_obs_agent_pos": observed_agent_pos[-1].round(6).tolist() if len(observed_agent_pos) else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
