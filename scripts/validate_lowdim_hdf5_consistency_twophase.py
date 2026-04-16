#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from lowdim_hdf5_twophase_gate_utils import (
    build_translated_reset_state,
    compute_error_summary,
    extract_stored_object_positions,
    infer_prepended_source_demo_count,
    load_dataset_demo_keys,
    load_episode,
    make_replay_env,
    parse_episode_selection,
    replay_observation_episode,
    resolve_source_demo_path,
    rounded_list,
)
from replay_zarr_episode import load_env_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate whether an exported two-phase replayobs low_dim HDF5 "
            "remains self-consistent with robosuite replay."
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
        "--compare-steps",
        type=int,
        default=None,
        help="Optional cap on how many steps to compare per episode. Defaults to the full stored episode.",
    )
    parser.add_argument(
        "--control-steps",
        type=int,
        default=1,
        help="Internal robosuite control repeats to use during replay.",
    )
    parser.add_argument("--eef-rmse-threshold", type=float, default=0.02)
    parser.add_argument("--eef-final-threshold", type=float, default=0.03)
    parser.add_argument("--object-rmse-threshold", type=float, default=0.02)
    parser.add_argument("--object-final-threshold", type=float, default=0.03)
    parser.add_argument("--target-rmse-threshold", type=float, default=0.02)
    parser.add_argument("--target-final-threshold", type=float, default=0.03)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit nonzero when any checked episode fails the configured thresholds.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to save the full JSON report.")
    return parser.parse_args()


def summarize_metric(metric: dict | None) -> dict | None:
    if metric is None:
        return None
    return {
        "rmse": rounded_list(metric["rmse"]),
        "mae": rounded_list(metric["mae"]),
        "max_abs": rounded_list(metric["max_abs"]),
        "final_err": rounded_list(metric["final_err"]),
        "rmse_norm": round(float(metric["rmse_norm"]), 6),
        "final_err_norm": round(float(metric["final_err_norm"]), 6),
    }


def aggregate_norms(rows: list[dict], key: str) -> dict | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": round(float(arr.mean()), 6),
        "max": round(float(arr.max()), 6),
    }


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
        print(
            "episode | demo_key | source | steps | "
            "eef_rmse | object_rmse | target_rmse | pass"
        )
        for episode_idx in episode_indices:
            episode = load_episode(dataset_path, episode_idx)

            compare_steps = min(
                len(episode.actions),
                len(episode.stored_obs["robot0_eef_pos"]),
                len(episode.stored_obs["robot0_eef_quat"]),
                len(episode.stored_obs["robot0_gripper_qpos"]),
                len(episode.stored_obs["object"]),
            )
            if args.compare_steps is not None:
                compare_steps = min(compare_steps, int(args.compare_steps))
            compare_steps = max(1, int(compare_steps))

            reset_state = build_translated_reset_state(
                source_demo_path=source_demo_path,
                env_name=env_name,
                source_episode_idx=episode.source_episode_idx,
                object_translation=episode.object_translation,
                target_translation=episode.target_translation,
            )
            replay_obs = replay_observation_episode(
                env=env,
                env_name=env_name,
                reset_state=reset_state,
                actions=episode.actions[:compare_steps],
            )

            stored_eef_pos = episode.stored_obs["robot0_eef_pos"][:compare_steps]
            replay_eef_pos = np.asarray(replay_obs["robot0_eef_pos"], dtype=np.float32)[:compare_steps]
            eef_metric = compute_error_summary(stored_eef_pos, replay_eef_pos)

            stored_eef_quat = episode.stored_obs["robot0_eef_quat"][:compare_steps]
            replay_eef_quat = np.asarray(replay_obs["robot0_eef_quat"], dtype=np.float32)[:compare_steps]
            eef_quat_metric = compute_error_summary(stored_eef_quat, replay_eef_quat)

            stored_gripper = episode.stored_obs["robot0_gripper_qpos"][:compare_steps]
            replay_gripper = np.asarray(replay_obs["robot0_gripper_qpos"], dtype=np.float32)[:compare_steps]
            gripper_metric = compute_error_summary(stored_gripper, replay_gripper)

            stored_object_obs = episode.stored_obs["object"][:compare_steps]
            replay_object_obs = np.asarray(replay_obs["object"], dtype=np.float32)[:compare_steps]
            object_obs_metric = compute_error_summary(stored_object_obs, replay_object_obs)

            stored_object_pos, stored_target_pos = extract_stored_object_positions(
                env_name,
                stored_object_obs,
            )
            replay_object_pos = np.asarray(replay_obs["object_pos"], dtype=np.float32)[:compare_steps]
            object_pos_metric = (
                compute_error_summary(stored_object_pos, replay_object_pos)
                if stored_object_pos is not None
                else None
            )

            replay_target_pos_raw = replay_obs["target_pos"]
            target_pos_metric = None
            if stored_target_pos is not None and replay_target_pos_raw is not None:
                replay_target_pos = np.asarray(replay_target_pos_raw, dtype=np.float32)[:compare_steps]
                target_pos_metric = compute_error_summary(stored_target_pos, replay_target_pos)

            passed = True
            passed = passed and (eef_metric["rmse_norm"] <= args.eef_rmse_threshold)
            passed = passed and (eef_metric["final_err_norm"] <= args.eef_final_threshold)
            if object_pos_metric is not None:
                passed = passed and (object_pos_metric["rmse_norm"] <= args.object_rmse_threshold)
                passed = passed and (object_pos_metric["final_err_norm"] <= args.object_final_threshold)
            if target_pos_metric is not None:
                passed = passed and (target_pos_metric["rmse_norm"] <= args.target_rmse_threshold)
                passed = passed and (target_pos_metric["final_err_norm"] <= args.target_final_threshold)

            row = {
                "episode": int(episode.episode_idx),
                "demo_key": episode.demo_key,
                "source_episode": int(episode.source_episode_idx),
                "compare_steps": int(compare_steps),
                "object_translation": rounded_list(episode.object_translation),
                "target_translation": rounded_list(episode.target_translation),
                "success_any": bool(replay_obs["success_any"]),
                "eef_pos": summarize_metric(eef_metric),
                "eef_quat": summarize_metric(eef_quat_metric),
                "gripper_qpos": summarize_metric(gripper_metric),
                "object_obs": summarize_metric(object_obs_metric),
                "object_pos": summarize_metric(object_pos_metric),
                "target_pos": summarize_metric(target_pos_metric),
                "eef_rmse_norm": round(float(eef_metric["rmse_norm"]), 6),
                "eef_final_err_norm": round(float(eef_metric["final_err_norm"]), 6),
                "object_rmse_norm": None
                if object_pos_metric is None
                else round(float(object_pos_metric["rmse_norm"]), 6),
                "object_final_err_norm": None
                if object_pos_metric is None
                else round(float(object_pos_metric["final_err_norm"]), 6),
                "target_rmse_norm": None
                if target_pos_metric is None
                else round(float(target_pos_metric["rmse_norm"]), 6),
                "target_final_err_norm": None
                if target_pos_metric is None
                else round(float(target_pos_metric["final_err_norm"]), 6),
                "pass": bool(passed),
            }
            rows.append(row)

            object_rmse_label = (
                "-"
                if row["object_rmse_norm"] is None
                else f"{row['object_rmse_norm']:.6f}"
            )
            target_rmse_label = (
                "-"
                if row["target_rmse_norm"] is None
                else f"{row['target_rmse_norm']:.6f}"
            )

            print(
                f"{row['episode']:>7} | "
                f"{row['demo_key']:<8} | "
                f"{row['source_episode']:>6} | "
                f"{row['compare_steps']:>5} | "
                f"{row['eef_rmse_norm']:.6f} | "
                f"{object_rmse_label} | "
                f"{target_rmse_label} | "
                f"{str(row['pass']):>4}"
            )
    finally:
        env.close()

    pass_count = int(sum(row["pass"] for row in rows))
    aggregate = {
        "n_checked": int(len(rows)),
        "n_pass": pass_count,
        "pass_rate": round(pass_count / max(1, len(rows)), 6),
        "eef_rmse_norm": aggregate_norms(rows, "eef_rmse_norm"),
        "eef_final_err_norm": aggregate_norms(rows, "eef_final_err_norm"),
        "object_rmse_norm": aggregate_norms(rows, "object_rmse_norm"),
        "object_final_err_norm": aggregate_norms(rows, "object_final_err_norm"),
        "target_rmse_norm": aggregate_norms(rows, "target_rmse_norm"),
        "target_final_err_norm": aggregate_norms(rows, "target_final_err_norm"),
        "thresholds": {
            "eef_rmse": float(args.eef_rmse_threshold),
            "eef_final": float(args.eef_final_threshold),
            "object_rmse": float(args.object_rmse_threshold),
            "object_final": float(args.object_final_threshold),
            "target_rmse": float(args.target_rmse_threshold),
            "target_final": float(args.target_final_threshold),
        },
    }
    payload = {
        "dataset": str(dataset_path),
        "source_demo": str(source_demo_path),
        "env_name": env_name,
        "control_steps": int(args.control_steps),
        "episodes_selector": args.episodes,
        "detected_source_demo_prefix_count": int(source_prefix_count),
        "aggregate": aggregate,
        "episodes": rows,
    }

    print("\n" + json.dumps(payload, indent=2))

    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.require_pass and pass_count != len(rows):
        sys.exit(2)


if __name__ == "__main__":
    main()
