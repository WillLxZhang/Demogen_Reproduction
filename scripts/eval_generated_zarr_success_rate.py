#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from replay_zarr_episode import (
    TASK_OBJECT_STATE_INDICES,
    ReplayBuffer,
    Robosuite3DEnv,
    infer_object_translation_from_low_dim,
    infer_source_episode,
    list_demo_keys,
    load_env_name,
    load_generated_episode_meta,
    load_reset_state,
    resolve_source_low_dim_path,
    robosuite_wrapper,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a generated DemoGen zarr dataset in robosuite and report "
            "episode-level success rate."
        )
    )
    parser.add_argument("--zarr", required=True, help="Path to generated .zarr dataset")
    parser.add_argument("--source-demo", required=True, help="Path to source demo.hdf5")
    parser.add_argument(
        "--episodes",
        default="all",
        help="Comma-separated episode indices, or 'all' to evaluate the full dataset.",
    )
    parser.add_argument(
        "--control-steps",
        type=int,
        default=1,
        help="How many internal robosuite control repeats to apply per action during replay.",
    )
    parser.add_argument(
        "--translate-object",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Whether to translate the replayed object to match the generated episode. "
            "'auto' uses stored meta when available and otherwise falls back to Lift-specific "
            "agent_pos inference from low_dim.hdf5."
        ),
    )
    parser.add_argument(
        "--source-low-dim",
        default=None,
        help=(
            "Optional source low_dim.hdf5 used to infer generated object translation when the "
            "zarr does not store object_translation meta. Defaults to a sibling low_dim.hdf5 "
            "next to --source-demo when present."
        ),
    )
    parser.add_argument(
        "--motion-frame",
        type=int,
        default=90,
        help=(
            "Motion / pre-grasp frame count used only for fallback translation inference "
            "from generated agent_pos versus source low_dim observations."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the aggregate success report as JSON.",
    )
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
    source_low_dim_path = resolve_source_low_dim_path(source_demo_path, args.source_low_dim)

    replay_buffer = ReplayBuffer.copy_from_path(str(zarr_path), keys=["agent_pos", "action"])
    n_episodes = replay_buffer.n_episodes
    episode_indices = parse_episode_list(args.episodes, n_episodes)
    demo_keys = list_demo_keys(source_demo_path)
    env_name = load_env_name(source_demo_path)
    object_state_indices = TASK_OBJECT_STATE_INDICES.get(env_name)

    robosuite_wrapper.N_CONTROL_STEPS = args.control_steps

    rows = []
    print("episode | source | success | translation_source | translation")

    for episode_idx in episode_indices:
        episode = replay_buffer.get_episode(episode_idx, copy=True)
        episode_meta = load_generated_episode_meta(zarr_path, episode_idx)

        source_episode_idx = episode_meta.get("source_episode_idx")
        if source_episode_idx is None:
            source_episode_idx = infer_source_episode(episode_idx, n_episodes, len(demo_keys))

        reset_state = load_reset_state(source_demo_path, source_episode_idx)

        applied_object_translation = None
        object_translation_source = None
        if args.translate_object != "off":
            if object_state_indices is None:
                if args.translate_object == "on":
                    raise ValueError(f"Object translation replay is not configured for task {env_name}")
            else:
                object_translation = episode_meta.get("object_translation")
                if object_translation is not None:
                    object_translation_source = "zarr_meta"
                else:
                    if source_low_dim_path is not None and source_low_dim_path.exists():
                        motion_frame = int(episode_meta.get("motion_frame_count", args.motion_frame))
                        object_translation = infer_object_translation_from_low_dim(
                            source_low_dim_path=source_low_dim_path,
                            source_episode_idx=source_episode_idx,
                            generated_episode=episode,
                            motion_frame=motion_frame,
                        )
                        object_translation_source = "low_dim_inference"
                    elif args.translate_object == "on":
                        raise FileNotFoundError(
                            "Could not determine object translation: zarr meta is missing and no low_dim.hdf5 "
                            "was found. Pass --source-low-dim or regenerate the dataset with saved meta."
                        )
                    else:
                        object_translation = None

                if object_translation is not None:
                    reset_state["states"] = np.asarray(reset_state["states"], dtype=np.float64).copy()
                    reset_state["states"][object_state_indices] += object_translation[: len(object_state_indices)]
                    applied_object_translation = np.asarray(object_translation, dtype=np.float32)

        env = Robosuite3DEnv(str(source_demo_path), render=False)
        env.reset()
        env.reset_to(reset_state)

        ep_success = False
        actions = np.asarray(episode["action"], dtype=np.float32)
        for action in actions:
            _, _, _, _ = env.step(action)
            ep_success = env.check_success() or ep_success

        env.close()

        row = {
            "episode": int(episode_idx),
            "source_episode": int(source_episode_idx),
            "n_actions": int(len(actions)),
            "success": bool(ep_success),
            "object_translation": applied_object_translation.round(6).tolist()
            if applied_object_translation is not None
            else None,
            "object_translation_source": object_translation_source,
        }
        rows.append(row)
        print(
            f"{row['episode']:03d} | {row['source_episode']:02d} | {int(row['success'])} | "
            f"{row['object_translation_source']} | {row['object_translation']}"
        )

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
