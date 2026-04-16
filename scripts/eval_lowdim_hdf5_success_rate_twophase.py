#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lowdim_hdf5_twophase_gate_utils import (
    build_translated_reset_state,
    infer_prepended_source_demo_count,
    load_dataset_demo_keys,
    load_episode,
    make_replay_env,
    parse_episode_selection,
    replay_success_episode,
    resolve_source_demo_path,
    rounded_list,
)
from replay_zarr_episode import load_env_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay episodes from an exported two-phase replayobs low_dim HDF5 "
            "and report episode-level success rate."
        )
    )
    parser.add_argument("--dataset", required=True, help="Path to exported replayobs low_dim.hdf5")
    parser.add_argument(
        "--source-demo",
        default=None,
        help=(
            "Path to the source low_dim / demo HDF5 used to reconstruct reset states. "
            "Defaults to data.attrs['source_low_dim_hdf5'] recorded in --dataset."
        ),
    )
    parser.add_argument(
        "--episodes",
        default="generated",
        help="Which demos to check: generated, source, all, or a comma-separated list of episode indices.",
    )
    parser.add_argument(
        "--control-steps",
        type=int,
        default=1,
        help="Internal robosuite control repeats to use during replay.",
    )
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=None,
        help="Optional gate threshold. Exit nonzero when success_rate falls below this value.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to save the full JSON report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    source_demo_path = resolve_source_demo_path(dataset_path, args.source_demo)
    env_name = load_env_name(source_demo_path)

    demo_keys = load_dataset_demo_keys(dataset_path)
    source_prefix_count = infer_prepended_source_demo_count(dataset_path, source_demo_path)
    episode_indices = parse_episode_selection(args.episodes, len(demo_keys), source_prefix_count)

    env = make_replay_env(source_demo_path, args.control_steps)
    rows = []
    try:
        print("episode | demo_key | source | success | object_translation | target_translation")
        for episode_idx in episode_indices:
            episode = load_episode(dataset_path, episode_idx)
            reset_state = build_translated_reset_state(
                source_demo_path=source_demo_path,
                env_name=env_name,
                source_episode_idx=episode.source_episode_idx,
                object_translation=episode.object_translation,
                target_translation=episode.target_translation,
            )
            success = replay_success_episode(
                env=env,
                reset_state=reset_state,
                actions=episode.actions,
            )

            row = {
                "episode": int(episode.episode_idx),
                "demo_key": episode.demo_key,
                "source_episode": int(episode.source_episode_idx),
                "n_actions": int(len(episode.actions)),
                "success": bool(success),
                "object_translation": rounded_list(episode.object_translation),
                "target_translation": rounded_list(episode.target_translation),
            }
            rows.append(row)
            print(
                f"{row['episode']:>7} | "
                f"{row['demo_key']:<8} | "
                f"{row['source_episode']:>6} | "
                f"{int(row['success'])} | "
                f"{row['object_translation']} | "
                f"{row['target_translation']}"
            )
    finally:
        env.close()

    success_count = int(sum(row["success"] for row in rows))
    payload = {
        "dataset": str(dataset_path),
        "source_demo": str(source_demo_path),
        "env_name": env_name,
        "control_steps": int(args.control_steps),
        "episodes_selector": args.episodes,
        "detected_source_demo_prefix_count": int(source_prefix_count),
        "n_checked": int(len(rows)),
        "success_count": success_count,
        "success_rate": round(success_count / max(1, len(rows)), 6),
        "episodes": rows,
    }

    print("\n" + json.dumps(payload, indent=2))

    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.min_success_rate is not None and payload["success_rate"] < float(args.min_success_rate):
        sys.exit(2)


if __name__ == "__main__":
    main()
