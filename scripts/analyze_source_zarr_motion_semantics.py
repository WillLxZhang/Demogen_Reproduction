#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze source-zarr motion semantics before running DemoGen. "
            "This is a source-level gate that compares motion_action against "
            "replay_h1_delta / forward_delta on the pre-skill prefix."
        )
    )
    parser.add_argument("--zarr", required=True, help="Path to source zarr")
    parser.add_argument("--skill1-frame", type=int, required=True, help="Pre-skill prefix length")
    parser.add_argument("--output-json", default=None, help="Optional JSON report path")
    return parser.parse_args()


def safe_mean(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def safe_max(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.max(arr))


def summarize_pair(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> dict:
    diff = a - b
    return {
        "compare": f"{name_a}_vs_{name_b}",
        "frame_rmse_xyz_mean": np.sqrt(np.mean(diff[:, :3] ** 2, axis=0)).round(8).tolist(),
        "frame_l1_xyz_mean": np.mean(np.abs(diff[:, :3]), axis=0).round(8).tolist(),
        "prefix_sum_xyz_a": np.sum(a[:, :3], axis=0).round(8).tolist(),
        "prefix_sum_xyz_b": np.sum(b[:, :3], axis=0).round(8).tolist(),
        "prefix_sum_xyz_diff": np.sum(diff[:, :3], axis=0).round(8).tolist(),
        "sum_abs_xyz_a": np.sum(np.abs(a[:, :3]), axis=0).round(8).tolist(),
        "sum_abs_xyz_b": np.sum(np.abs(b[:, :3]), axis=0).round(8).tolist(),
    }


def main() -> None:
    args = parse_args()
    zarr_path = Path(args.zarr).expanduser().resolve()
    root = zarr.open(str(zarr_path), mode="r")
    data = root["data"]
    meta = root["meta"]

    skill1_frame = int(args.skill1_frame)
    episode_ends = np.asarray(meta["episode_ends"], dtype=np.int64)
    motion_action = np.asarray(data["motion_action"], dtype=np.float32)
    replay_h1_delta = np.asarray(data["replay_h1_delta"], dtype=np.float32) if "replay_h1_delta" in data else None
    forward_delta = np.asarray(data["forward_delta"], dtype=np.float32) if "forward_delta" in data else None

    per_episode = []
    start = 0
    for episode_idx, end in enumerate(episode_ends):
        end = int(end)
        ep_len = end - start
        prefix = min(skill1_frame, ep_len)
        record = {
            "episode_idx": int(episode_idx),
            "episode_len": int(ep_len),
            "prefix_len": int(prefix),
        }
        motion_prefix = motion_action[start : start + prefix]
        record["motion_action_prefix_sum_xyz"] = np.sum(motion_prefix[:, :3], axis=0).round(8).tolist()
        record["motion_action_sum_abs_xyz"] = np.sum(np.abs(motion_prefix[:, :3]), axis=0).round(8).tolist()

        if replay_h1_delta is not None:
            replay_prefix = replay_h1_delta[start : start + prefix]
            record["motion_vs_replay_h1"] = summarize_pair(
                "motion_action", motion_prefix, "replay_h1_delta", replay_prefix
            )
        if forward_delta is not None:
            forward_prefix = forward_delta[start : start + prefix]
            record["motion_vs_forward_delta"] = summarize_pair(
                "motion_action", motion_prefix, "forward_delta", forward_prefix
            )
        per_episode.append(record)
        start = end

    def collect_axis(metric_path: list[str], axis: int) -> np.ndarray:
        vals = []
        for ep in per_episode:
            cur = ep
            ok = True
            for key in metric_path:
                if key not in cur:
                    ok = False
                    break
                cur = cur[key]
            if ok:
                vals.append(float(cur[axis]))
        return np.asarray(vals, dtype=np.float32)

    summary = {
        "zarr": str(zarr_path),
        "skill1_frame": skill1_frame,
        "source_name": meta.attrs.get("source_name", None),
        "motion_action_semantics": meta.attrs.get("motion_action_semantics", None),
        "replay_h1_prefix_frames": meta.attrs.get("replay_h1_prefix_frames", None),
        "n_episodes": len(per_episode),
    }

    if replay_h1_delta is not None:
        summary["motion_vs_replay_h1_prefix_sum_diff_xyz_mean"] = [
            safe_mean(collect_axis(["motion_vs_replay_h1", "prefix_sum_xyz_diff"], axis))
            for axis in range(3)
        ]
        summary["motion_vs_replay_h1_prefix_sum_diff_xyz_max_abs"] = [
            safe_max(np.abs(collect_axis(["motion_vs_replay_h1", "prefix_sum_xyz_diff"], axis)))
            for axis in range(3)
        ]
        summary["motion_vs_replay_h1_frame_rmse_xyz_mean"] = [
            safe_mean(collect_axis(["motion_vs_replay_h1", "frame_rmse_xyz_mean"], axis))
            for axis in range(3)
        ]

    if forward_delta is not None:
        summary["motion_vs_forward_delta_prefix_sum_diff_xyz_mean"] = [
            safe_mean(collect_axis(["motion_vs_forward_delta", "prefix_sum_xyz_diff"], axis))
            for axis in range(3)
        ]
        summary["motion_vs_forward_delta_prefix_sum_diff_xyz_max_abs"] = [
            safe_max(np.abs(collect_axis(["motion_vs_forward_delta", "prefix_sum_xyz_diff"], axis)))
            for axis in range(3)
        ]
        summary["motion_vs_forward_delta_frame_rmse_xyz_mean"] = [
            safe_mean(collect_axis(["motion_vs_forward_delta", "frame_rmse_xyz_mean"], axis))
            for axis in range(3)
        ]

    report = {
        "summary": summary,
        "per_episode": per_episode,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
