#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
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
            "Validate whether a generated DemoGen zarr remains self-consistent "
            "with actual robosuite replay before training on it."
        )
    )
    parser.add_argument("--zarr", required=True)
    parser.add_argument("--source-demo", required=True)
    parser.add_argument(
        "--episodes",
        default="all",
        help="Comma-separated episode indices, or 'all' to scan the full dataset.",
    )
    parser.add_argument(
        "--compare-steps",
        type=int,
        default=None,
        help=(
            "How many steps to compare. Defaults to each episode's motion_frame_count "
            "meta when present, otherwise the full episode length."
        ),
    )
    parser.add_argument("--control-steps", type=int, default=1)
    parser.add_argument(
        "--translate-object",
        choices=["auto", "on", "off"],
        default="auto",
        help="Whether to apply stored / inferred object translation during replay.",
    )
    parser.add_argument(
        "--source-low-dim",
        default=None,
        help="Optional low_dim.hdf5 path used for fallback object translation inference.",
    )
    parser.add_argument(
        "--motion-frame",
        type=int,
        default=90,
        help="Fallback motion frame count when generated meta is missing.",
    )
    parser.add_argument(
        "--rmse-threshold",
        type=float,
        default=0.02,
        help="Episode passes when position RMSE norm is at most this value.",
    )
    parser.add_argument(
        "--final-threshold",
        type=float,
        default=0.03,
        help="Episode passes when final aligned position error norm is at most this value.",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit nonzero when any checked episode fails the thresholds.",
    )
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


def resolve_object_translation(
    source_demo_path: Path,
    source_low_dim_path: Path | None,
    source_episode_idx: int,
    generated_episode: dict,
    episode_meta: dict,
    translate_object: str,
    motion_frame_default: int,
) -> tuple[np.ndarray | None, str | None]:
    if translate_object == "off":
        return None, None

    env_name = load_env_name(source_demo_path)
    object_state_indices = TASK_OBJECT_STATE_INDICES.get(env_name)
    if object_state_indices is None:
        if translate_object == "on":
            raise ValueError(f"Object translation replay is not configured for task {env_name}")
        return None, None

    object_translation = episode_meta.get("object_translation")
    if object_translation is not None:
        return np.asarray(object_translation, dtype=np.float32), "zarr_meta"

    if source_low_dim_path is not None and source_low_dim_path.exists():
        motion_frame = int(episode_meta.get("motion_frame_count", motion_frame_default))
        object_translation = infer_object_translation_from_low_dim(
            source_low_dim_path=source_low_dim_path,
            source_episode_idx=source_episode_idx,
            generated_episode=generated_episode,
            motion_frame=motion_frame,
        )
        return np.asarray(object_translation, dtype=np.float32), "low_dim_inference"

    if translate_object == "on":
        raise FileNotFoundError(
            "Could not determine object translation: zarr meta is missing and no low_dim.hdf5 "
            "was found. Pass --source-low-dim or regenerate the dataset with saved meta."
        )
    return None, None


def replay_aligned_agent_pos(
    source_demo_path: Path,
    source_episode_idx: int,
    actions: np.ndarray,
    object_translation: np.ndarray | None,
    control_steps: int,
) -> np.ndarray:
    reset_state = load_reset_state(source_demo_path, source_episode_idx)
    if object_translation is not None:
        env_name = load_env_name(source_demo_path)
        object_state_indices = TASK_OBJECT_STATE_INDICES.get(env_name)
        if object_state_indices is None:
            raise KeyError(f"No object state indices configured for env={env_name}")
        reset_state["states"][object_state_indices] += object_translation[: len(object_state_indices)]

    robosuite_wrapper.N_CONTROL_STEPS = control_steps
    env = Robosuite3DEnv(str(source_demo_path), render=False)
    obs = env.reset_to(reset_state)
    hist = [np.asarray(obs["agent_pos"], dtype=np.float32).copy()]
    for action in actions:
        obs, _, _, _ = env.step(action)
        hist.append(np.asarray(obs["agent_pos"], dtype=np.float32).copy())
    env.close()
    return np.asarray(hist[:-1], dtype=np.float32)


def summarize_episode(
    episode_idx: int,
    source_episode_idx: int,
    compare_steps: int,
    stored_agent_pos: np.ndarray,
    replay_agent_pos: np.ndarray,
    object_translation: np.ndarray | None,
    object_translation_source: str | None,
    rmse_threshold: float,
    final_threshold: float,
) -> dict:
    err = replay_agent_pos[:, :3] - stored_agent_pos[:, :3]
    rmse_xyz = np.sqrt(np.mean(err ** 2, axis=0))
    mae_xyz = np.mean(np.abs(err), axis=0)
    max_abs_xyz = np.max(np.abs(err), axis=0)
    final_err_xyz = err[-1]

    rmse_norm = float(np.linalg.norm(rmse_xyz))
    final_err_norm = float(np.linalg.norm(final_err_xyz))
    passed = (rmse_norm <= rmse_threshold) and (final_err_norm <= final_threshold)

    return {
        "episode": int(episode_idx),
        "source_episode": int(source_episode_idx),
        "compare_steps": int(compare_steps),
        "rmse_xyz": rmse_xyz.round(6).tolist(),
        "mae_xyz": mae_xyz.round(6).tolist(),
        "max_abs_xyz": max_abs_xyz.round(6).tolist(),
        "final_err_xyz": final_err_xyz.round(6).tolist(),
        "rmse_norm": round(rmse_norm, 6),
        "final_err_norm": round(final_err_norm, 6),
        "stored_start": stored_agent_pos[0].round(6).tolist(),
        "stored_end": stored_agent_pos[-1].round(6).tolist(),
        "replay_start": replay_agent_pos[0].round(6).tolist(),
        "replay_end": replay_agent_pos[-1].round(6).tolist(),
        "object_translation": object_translation.round(6).tolist() if object_translation is not None else None,
        "object_translation_source": object_translation_source,
        "pass": bool(passed),
    }


def main() -> None:
    args = parse_args()
    zarr_path = Path(args.zarr).expanduser().resolve()
    source_demo_path = Path(args.source_demo).expanduser().resolve()
    source_low_dim_path = resolve_source_low_dim_path(source_demo_path, args.source_low_dim)

    replay_buffer = ReplayBuffer.copy_from_path(
        str(zarr_path),
        keys=["agent_pos", "action"],
    )
    n_episodes = replay_buffer.n_episodes
    episode_indices = parse_episode_list(args.episodes, n_episodes)
    demo_keys = list_demo_keys(source_demo_path)

    episode_summaries = []
    print(
        "episode | source | steps | rmse_norm | final_err_norm | pass | translation"
    )

    for episode_idx in episode_indices:
        episode = replay_buffer.get_episode(episode_idx, copy=True)
        episode_meta = load_generated_episode_meta(zarr_path, episode_idx)

        source_episode_idx = episode_meta.get("source_episode_idx")
        if source_episode_idx is None:
            source_episode_idx = infer_source_episode(episode_idx, n_episodes, len(demo_keys))

        stored_agent_pos = np.asarray(episode["agent_pos"], dtype=np.float32)
        actions = np.asarray(episode["action"], dtype=np.float32)

        compare_steps = args.compare_steps
        if compare_steps is None:
            compare_steps = int(episode_meta.get("motion_frame_count", len(actions)))
        compare_steps = max(1, min(compare_steps, len(actions), len(stored_agent_pos)))

        object_translation, object_translation_source = resolve_object_translation(
            source_demo_path=source_demo_path,
            source_low_dim_path=source_low_dim_path,
            source_episode_idx=source_episode_idx,
            generated_episode=episode,
            episode_meta=episode_meta,
            translate_object=args.translate_object,
            motion_frame_default=args.motion_frame,
        )

        replay_agent_pos = replay_aligned_agent_pos(
            source_demo_path=source_demo_path,
            source_episode_idx=source_episode_idx,
            actions=actions[:compare_steps],
            object_translation=object_translation,
            control_steps=args.control_steps,
        )

        summary = summarize_episode(
            episode_idx=episode_idx,
            source_episode_idx=source_episode_idx,
            compare_steps=compare_steps,
            stored_agent_pos=stored_agent_pos[:compare_steps],
            replay_agent_pos=replay_agent_pos[:compare_steps],
            object_translation=object_translation,
            object_translation_source=object_translation_source,
            rmse_threshold=args.rmse_threshold,
            final_threshold=args.final_threshold,
        )
        episode_summaries.append(summary)

        translation_label = (
            summary["object_translation"]
            if summary["object_translation"] is not None
            else None
        )
        print(
            f"{summary['episode']:>7} | "
            f"{summary['source_episode']:>6} | "
            f"{summary['compare_steps']:>5} | "
            f"{summary['rmse_norm']:.6f} | "
            f"{summary['final_err_norm']:.6f} | "
            f"{str(summary['pass']):>4} | "
            f"{translation_label}"
        )

    rmse_norms = np.asarray([row["rmse_norm"] for row in episode_summaries], dtype=np.float32)
    final_norms = np.asarray([row["final_err_norm"] for row in episode_summaries], dtype=np.float32)
    pass_count = int(sum(row["pass"] for row in episode_summaries))
    aggregate = {
        "n_checked": int(len(episode_summaries)),
        "n_pass": pass_count,
        "pass_rate": round(pass_count / max(1, len(episode_summaries)), 6),
        "rmse_norm_mean": round(float(rmse_norms.mean()), 6),
        "rmse_norm_max": round(float(rmse_norms.max()), 6),
        "final_err_norm_mean": round(float(final_norms.mean()), 6),
        "final_err_norm_max": round(float(final_norms.max()), 6),
        "rmse_threshold": float(args.rmse_threshold),
        "final_threshold": float(args.final_threshold),
    }

    print(
        "\n"
        + json.dumps(
            {
                "zarr": str(zarr_path),
                "source_demo": str(source_demo_path),
                "control_steps": int(args.control_steps),
                "aggregate": aggregate,
            },
            indent=2,
        )
    )

    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "zarr": str(zarr_path),
            "source_demo": str(source_demo_path),
            "control_steps": int(args.control_steps),
            "episodes": episode_summaries,
            "aggregate": aggregate,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.require_pass and pass_count != len(episode_summaries):
        sys.exit(2)


if __name__ == "__main__":
    main()
