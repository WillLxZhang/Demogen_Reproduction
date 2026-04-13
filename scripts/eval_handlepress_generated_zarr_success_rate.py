#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMOGEN_ROOT = REPO_ROOT / "repos" / "DemoGen" / "demo_generation"
DIFFUSION_POLICIES_ROOT = REPO_ROOT / "repos" / "DemoGen" / "diffusion_policies"
if str(DEMOGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMOGEN_ROOT))
if str(DIFFUSION_POLICIES_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICIES_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from replay_zarr_episode import (
    ReplayBuffer,
    infer_source_episode,
    list_demo_keys,
    load_generated_episode_meta,
    load_reset_state,
)

import diffusion_policies.env.robosuite.robosuite_wrapper as robosuite_wrapper
from demo_generation.handlepress_robosuite_wrapper import HandlePressRobosuite3DEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a HandlePress generated zarr dataset and report success rate."
    )
    parser.add_argument("--zarr", required=True, help="Path to generated .zarr dataset")
    parser.add_argument("--source-demo", required=True, help="Path to source demo.hdf5")
    parser.add_argument(
        "--episodes",
        default="all",
        help="Comma-separated episode indices, or 'all' to evaluate the full dataset.",
    )
    parser.add_argument("--control-steps", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def parse_episode_list(raw: str, n_episodes: int) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(n_episodes))
    episodes = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= n_episodes:
            raise IndexError(f"episode {idx} out of range for dataset with {n_episodes} episodes")
        episodes.append(idx)
    if not episodes:
        raise ValueError("No episodes selected")
    return episodes


def main() -> None:
    args = parse_args()
    zarr_path = Path(args.zarr).expanduser().resolve()
    source_demo_path = Path(args.source_demo).expanduser().resolve()

    replay_buffer = ReplayBuffer.copy_from_path(str(zarr_path), keys=["action"])
    n_episodes = replay_buffer.n_episodes
    episode_indices = parse_episode_list(args.episodes, n_episodes)
    demo_keys = list_demo_keys(source_demo_path)

    robosuite_wrapper.N_CONTROL_STEPS = int(args.control_steps)
    env = HandlePressRobosuite3DEnv(str(source_demo_path), render=False)

    rows = []
    print("episode | source | success | translation")

    try:
        for episode_idx in episode_indices:
            episode = replay_buffer.get_episode(episode_idx, copy=True)
            episode_meta = load_generated_episode_meta(zarr_path, episode_idx)

            source_episode_idx = episode_meta.get("source_episode_idx")
            if source_episode_idx is None:
                source_episode_idx = infer_source_episode(episode_idx, n_episodes, len(demo_keys))

            reset_state = load_reset_state(source_demo_path, int(source_episode_idx))
            object_translation = episode_meta.get("object_translation")
            if object_translation is not None:
                reset_state["object_translation"] = np.asarray(object_translation, dtype=np.float32).copy()

            env.reset_to(reset_state)

            ep_success = False
            actions = np.asarray(episode["action"], dtype=np.float32)
            for action in actions:
                _, _, _, _ = env.step(action)
                ep_success = bool(env.check_success()) or ep_success

            row = {
                "episode": int(episode_idx),
                "source_episode": int(source_episode_idx),
                "n_actions": int(len(actions)),
                "success": bool(ep_success),
                "object_translation": np.asarray(object_translation, dtype=np.float32).round(6).tolist()
                if object_translation is not None
                else None,
            }
            rows.append(row)
            print(
                f"{row['episode']:03d} | {row['source_episode']:02d} | {int(row['success'])} | "
                f"{row['object_translation']}"
            )
    finally:
        env.close()

    success_count = int(sum(row["success"] for row in rows))
    result = {
        "zarr": str(zarr_path),
        "source_demo": str(source_demo_path),
        "control_steps": int(args.control_steps),
        "n_checked": int(len(rows)),
        "success_count": success_count,
        "success_rate": round(success_count / max(1, len(rows)), 6),
        "episodes": rows,
    }

    print(json.dumps(result, indent=2))

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
