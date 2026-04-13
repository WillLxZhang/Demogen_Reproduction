#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import h5py
import imageio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1] / "repos" / "DemoGen"
DIFFUSION_POLICIES_ROOT = REPO_ROOT / "diffusion_policies"
if str(DIFFUSION_POLICIES_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICIES_ROOT))

import diffusion_policies.env.robosuite.robosuite_wrapper as robosuite_wrapper
from diffusion_policies.env.robosuite.robosuite_wrapper import Robosuite3DEnv

from replay_zarr_episode import ensure_parent, load_env_name, load_reset_state
from replay_zarr_episode_stack_twophase import TASK_OBJECT_STATE_INDICES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one episode from an exported two-phase low_dim HDF5 in robosuite, "
            "including Stack-style object + target translations."
        )
    )
    parser.add_argument("--dataset", required=True, help="Path to exported low_dim.hdf5")
    parser.add_argument("--source-demo", required=True, help="Path to source demo.hdf5")
    parser.add_argument("--demo-key", default=None, help="Episode key such as demo_3")
    parser.add_argument("--episode", type=int, default=None, help="Episode index if demo-key is not given")
    parser.add_argument("--output-video", required=True, help="Where to save replay video")
    parser.add_argument("--render-size", type=int, default=256, help="Rendered image size")
    parser.add_argument("--fps", type=int, default=20, help="Video FPS")
    parser.add_argument("--control-steps", type=int, default=1, help="Internal control repeats")
    parser.add_argument("--camera", default=None, help="Render camera name")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on replay steps")
    return parser.parse_args()


def numeric_suffix(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    if match:
        return int(match.group(1)), name
    return math.inf, name


def sorted_keys(group: h5py.Group) -> list[str]:
    return sorted(list(group.keys()), key=numeric_suffix)


def split_translation(raw) -> tuple[np.ndarray | None, np.ndarray | None]:
    if raw is None:
        return None, None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        raw = json.loads(raw)
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.shape == (3,):
        return arr, None
    if arr.shape == (6,):
        return arr[:3], arr[3:6]
    raise ValueError(f"object_translation attr must have shape (3,) or (6,), got {arr.shape}")


def resolve_demo_key(data_group: h5py.Group, demo_key: str | None, episode: int | None) -> tuple[str, int]:
    keys = sorted_keys(data_group)
    if not keys:
        raise ValueError("Dataset has no demos under /data")

    if demo_key is not None:
        if demo_key not in data_group:
            raise KeyError(f"demo_key={demo_key} not found. available={keys}")
        return demo_key, keys.index(demo_key)

    ep_idx = 0 if episode is None else int(episode)
    if ep_idx < 0 or ep_idx >= len(keys):
        raise IndexError(f"episode {ep_idx} out of range for dataset with {len(keys)} demos")
    return keys[ep_idx], ep_idx


def load_lowdim_episode(dataset_path: Path, demo_key: str | None, episode: int | None):
    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{dataset_path} missing /data group")
        data_group = f["data"]
        resolved_key, resolved_idx = resolve_demo_key(data_group, demo_key, episode)
        ep = data_group[resolved_key]
        if "actions" not in ep:
            raise KeyError(f"{resolved_key} missing actions")
        actions = np.asarray(ep["actions"][()], dtype=np.float32)
        source_episode_idx = ep.attrs.get("source_episode_idx", None)
        if source_episode_idx is None:
            raise KeyError(f"{resolved_key} missing attr source_episode_idx")
        object_translation, target_translation = split_translation(ep.attrs.get("object_translation", None))
        model_file = ep.attrs.get("model_file", None)
        return {
            "demo_key": resolved_key,
            "episode_idx": int(resolved_idx),
            "actions": actions,
            "source_episode_idx": int(source_episode_idx),
            "object_translation": object_translation,
            "target_translation": target_translation,
            "model_file": model_file,
        }


def apply_translation_to_reset_state(
    reset_state: dict,
    env_name: str,
    object_translation: np.ndarray | None,
    target_translation: np.ndarray | None,
) -> None:
    indices_cfg = TASK_OBJECT_STATE_INDICES.get(env_name)
    if indices_cfg is None:
        raise ValueError(f"No object-state indices configured for env {env_name}")

    if object_translation is None and target_translation is None:
        return

    reset_state["states"] = np.asarray(reset_state["states"], dtype=np.float64).copy()

    if object_translation is not None and indices_cfg.get("object") is not None:
        idx = np.asarray(indices_cfg["object"], dtype=np.int64)
        reset_state["states"][idx] += object_translation[: len(idx)]

    if target_translation is not None and indices_cfg.get("target") is not None:
        idx = np.asarray(indices_cfg["target"], dtype=np.int64)
        reset_state["states"][idx] += target_translation[: len(idx)]


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    source_demo_path = Path(args.source_demo).expanduser().resolve()
    output_video = Path(args.output_video).expanduser().resolve()

    if args.control_steps <= 0:
        raise ValueError("--control-steps must be positive")

    episode = load_lowdim_episode(dataset_path, args.demo_key, args.episode)
    actions = episode["actions"]
    if args.max_steps is not None:
        actions = actions[: args.max_steps]

    source_episode_idx = int(episode["source_episode_idx"])
    env_name = load_env_name(source_demo_path)

    robosuite_wrapper.N_CONTROL_STEPS = args.control_steps
    reset_state = load_reset_state(source_demo_path, source_episode_idx)
    apply_translation_to_reset_state(
        reset_state=reset_state,
        env_name=env_name,
        object_translation=episode["object_translation"],
        target_translation=episode["target_translation"],
    )

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
    frames = [
        env.render(
            mode="rgb_array",
            height=args.render_size,
            width=args.render_size,
            camera_name=args.camera,
        )
    ]
    observed_agent_pos = []
    successes = []

    for action in actions:
        obs, _, _, _ = env.step(action)
        observed_agent_pos.append(np.asarray(obs["agent_pos"], dtype=np.float32))
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
        "dataset": str(dataset_path),
        "source_demo": str(source_demo_path),
        "env_name": env_name,
        "demo_key": episode["demo_key"],
        "episode_idx": int(episode["episode_idx"]),
        "source_episode": int(source_episode_idx),
        "n_actions": int(len(actions)),
        "control_steps": int(args.control_steps),
        "object_translation": (
            episode["object_translation"].round(6).tolist()
            if episode["object_translation"] is not None
            else None
        ),
        "target_translation": (
            episode["target_translation"].round(6).tolist()
            if episode["target_translation"] is not None
            else None
        ),
        "video": str(output_video),
        "success_any": bool(any(successes)),
        "pos_min": pos.min(axis=0).round(6).tolist() if len(pos) else None,
        "pos_max": pos.max(axis=0).round(6).tolist() if len(pos) else None,
        "last_obs_agent_pos": observed_agent_pos[-1].round(6).tolist() if len(observed_agent_pos) else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
