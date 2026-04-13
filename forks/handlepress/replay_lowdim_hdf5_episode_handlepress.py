#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import h5py
import imageio
import numpy as np

import diffusion_policies.env.robosuite.robosuite_wrapper as robosuite_wrapper


REPRO_ROOT = Path(__file__).resolve().parents[2]
DEMOGEN_ROOT = REPRO_ROOT / "repos" / "DemoGen"
DEMOGEN_PKG_ROOT = DEMOGEN_ROOT / "demo_generation"
SCRIPTS_ROOT = REPRO_ROOT / "scripts"

for path in [DEMOGEN_PKG_ROOT, SCRIPTS_ROOT]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from demo_generation.handlepress_robosuite_wrapper import HandlePressRobosuite3DEnv
from replay_zarr_episode import ensure_parent, load_reset_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one exported HandlePress low-dim episode into a video."
    )
    parser.add_argument("--dataset", required=True, help="Path to exported replayobs lowdim hdf5")
    parser.add_argument("--source-demo", required=True, help="Path to source demo.hdf5")
    parser.add_argument("--demo-key", default=None, help="Episode key such as demo_92")
    parser.add_argument("--episode", type=int, default=None, help="Episode index if demo-key is omitted")
    parser.add_argument("--random", action="store_true", help="Pick a random episode from the dataset")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for --random")
    parser.add_argument("--output-video", required=True, help="Video output path")
    parser.add_argument("--render-size", type=int, default=256, help="Rendered frame size")
    parser.add_argument("--fps", type=int, default=20, help="Video FPS")
    parser.add_argument("--camera", default="agentview", help="Camera name")
    parser.add_argument("--control-steps", type=int, default=1, help="Internal control repeats per action")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on replay steps")
    return parser.parse_args()


def numeric_suffix(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    if match:
        return int(match.group(1)), name
    return math.inf, name


def sorted_demo_keys(group: h5py.Group) -> list[str]:
    return sorted(list(group.keys()), key=numeric_suffix)


def parse_translation_attr(raw) -> np.ndarray | None:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        raw = json.loads(raw)
    arr = np.asarray(raw, dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"object_translation attr must be shape (3,), got {arr.shape}")
    return arr


def resolve_demo_key(
    keys: list[str],
    demo_key: str | None,
    episode: int | None,
    use_random: bool,
    seed: int | None,
) -> tuple[str, int]:
    if not keys:
        raise ValueError("Dataset has no demos under /data")

    if use_random:
        rng = random.Random(seed)
        chosen_key = rng.choice(keys)
        return chosen_key, keys.index(chosen_key)

    if demo_key is not None:
        if demo_key not in keys:
            raise KeyError(f"demo_key={demo_key} not found. available={keys}")
        return demo_key, keys.index(demo_key)

    ep_idx = 0 if episode is None else int(episode)
    if ep_idx < 0 or ep_idx >= len(keys):
        raise IndexError(f"episode {ep_idx} out of range for dataset with {len(keys)} demos")
    return keys[ep_idx], ep_idx


def load_episode(
    dataset_path: Path,
    demo_key: str | None,
    episode: int | None,
    use_random: bool,
    seed: int | None,
) -> dict:
    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{dataset_path} missing /data group")
        data_group = f["data"]
        keys = sorted_demo_keys(data_group)
        resolved_key, resolved_idx = resolve_demo_key(keys, demo_key, episode, use_random, seed)
        ep = data_group[resolved_key]
        actions = np.asarray(ep["actions"][()], dtype=np.float32)
        source_episode_idx = int(ep.attrs["source_episode_idx"])
        object_translation = parse_translation_attr(ep.attrs.get("object_translation"))
        return {
            "demo_key": resolved_key,
            "episode_idx": int(resolved_idx),
            "actions": actions,
            "source_episode_idx": source_episode_idx,
            "object_translation": object_translation,
        }


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    source_demo_path = Path(args.source_demo).expanduser().resolve()
    output_video = Path(args.output_video).expanduser().resolve()
    if args.control_steps <= 0:
        raise ValueError("--control-steps must be positive")

    episode = load_episode(
        dataset_path=dataset_path,
        demo_key=args.demo_key,
        episode=args.episode,
        use_random=bool(args.random),
        seed=args.seed,
    )

    actions = episode["actions"]
    if args.max_steps is not None:
        actions = actions[: args.max_steps]

    reset_state = load_reset_state(source_demo_path, int(episode["source_episode_idx"]))
    if episode["object_translation"] is not None:
        reset_state["object_translation"] = np.asarray(
            episode["object_translation"], dtype=np.float32
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
        "dataset": str(dataset_path),
        "source_demo": str(source_demo_path),
        "demo_key": episode["demo_key"],
        "episode_idx": int(episode["episode_idx"]),
        "source_episode_idx": int(episode["source_episode_idx"]),
        "n_actions": int(len(actions)),
        "control_steps": int(args.control_steps),
        "object_translation": (
            np.asarray(episode["object_translation"]).round(6).tolist()
            if episode["object_translation"] is not None
            else None
        ),
        "success_any": bool(any(success_trace)),
        "output_video": str(output_video),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
