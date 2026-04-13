#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMOGEN_ROOT = REPO_ROOT / "repos" / "DemoGen" / "demo_generation"
DIFFUSION_POLICIES_ROOT = REPO_ROOT / "repos" / "DemoGen" / "diffusion_policies"
if str(DEMOGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMOGEN_ROOT))
if str(DIFFUSION_POLICIES_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICIES_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_lift_solved_prefix_zarr import (
    build_replayed_point_clouds,
    build_summary,
    replay_full_episode,
)
from solve_lift_prefix_xyz_actions import (
    build_desired_prefix_states,
    instantiate_generator,
    load_cfg,
    solve_prefix_actions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            REPO_ROOT
            / "repos"
            / "DemoGen"
            / "demo_generation"
            / "demo_generation"
            / "config"
            / "lift_0_v36_replayh1_light_schedule_phasecopy_execconsistent_deferconflict.yaml"
        ),
    )
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "repos" / "DemoGen" / "data"),
    )
    parser.add_argument(
        "--source-demo",
        default=str(REPO_ROOT / "data" / "raw" / "lift_0" / "1774702988_8036063" / "demo.hdf5"),
    )
    parser.add_argument("--template-zarr", required=True)
    parser.add_argument(
        "--episodes",
        default="all",
        help="Comma-separated generated episode indices, or 'all'.",
    )
    parser.add_argument("--solve-steps", type=int, default=None)
    parser.add_argument("--control-steps", type=int, default=1)
    parser.add_argument("--action-deviation-weight", type=float, default=1e-4)
    parser.add_argument("--output-zarr", required=True)
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
            raise IndexError(f"episode {idx} out of range for {n_episodes} episodes")
        episodes.append(idx)
    if not episodes:
        raise ValueError("No episodes selected")
    return episodes


def load_template_meta(template_zarr: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = zarr.open(str(template_zarr), mode="r")
    meta = root["meta"]
    required = ["source_episode_idx", "object_translation", "motion_frame_count"]
    missing = [k for k in required if k not in meta]
    if missing:
        raise KeyError(f"Template zarr missing required meta keys: {missing}")
    source_episode_idx = np.asarray(meta["source_episode_idx"][:], dtype=np.int64)
    object_translation = np.asarray(meta["object_translation"][:], dtype=np.float32)
    motion_frame_count = np.asarray(meta["motion_frame_count"][:], dtype=np.int64)
    return source_episode_idx, object_translation, motion_frame_count


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    data_root = Path(args.data_root).resolve()
    source_demo_path = Path(args.source_demo).resolve()
    template_zarr = Path(args.template_zarr).resolve()
    output_zarr = Path(args.output_zarr).resolve()
    output_json = Path(args.output_json).resolve() if args.output_json else None

    cfg = load_cfg(config_path, data_root)
    generator = instantiate_generator(cfg)
    template_source_idx, template_translation, template_motion_frames = load_template_meta(
        template_zarr
    )

    selected = parse_episode_list(args.episodes, len(template_source_idx))
    generated_episodes = []
    summaries = []

    for gen_ep_idx in selected:
        source_episode_idx = int(template_source_idx[gen_ep_idx])
        translation = np.asarray(template_translation[gen_ep_idx], dtype=np.float32)
        motion_frame_count = int(template_motion_frames[gen_ep_idx])
        solve_steps = int(args.solve_steps) if args.solve_steps is not None else motion_frame_count
        if solve_steps > motion_frame_count:
            raise ValueError(
                f"solve_steps={solve_steps} exceeds template motion_frame_count={motion_frame_count} "
                f"for generated episode {gen_ep_idx}"
            )

        source_demo = generator.replay_buffer.get_episode(source_episode_idx)
        _, source_actions, desired_obs_xyz, _ = build_desired_prefix_states(
            generator=generator,
            episode_idx=source_episode_idx,
            translation=translation,
            solve_steps=solve_steps,
        )
        solved_actions, observed_obs_xyz, _ = solve_prefix_actions(
            source_demo_path=source_demo_path,
            source_episode_idx=source_episode_idx,
            source_actions=source_actions,
            desired_obs_xyz=desired_obs_xyz,
            translation=translation,
            control_steps=args.control_steps,
            action_deviation_weight=args.action_deviation_weight,
        )

        replay_states, replay_actions = replay_full_episode(
            source_demo_path=source_demo_path,
            source_demo=source_demo,
            episode_idx=source_episode_idx,
            translation=translation,
            solved_prefix_actions=solved_actions,
            control_steps=args.control_steps,
        )
        replay_pcds = build_replayed_point_clouds(
            generator=generator,
            source_demo=source_demo,
            episode_idx=source_episode_idx,
            replay_states=replay_states,
            object_translation=translation,
            motion_frame_count=solve_steps,
        )
        generated_episodes.append(
            {
                "state": replay_states,
                "action": replay_actions,
                "point_cloud": replay_pcds,
                "_source_episode_idx": source_episode_idx,
                "_object_translation": translation,
                "_motion_frame_count": solve_steps,
            }
        )

        summary = build_summary(
            observed_obs_xyz=observed_obs_xyz,
            desired_obs_xyz=desired_obs_xyz,
            source_actions=source_actions,
            solved_actions=solved_actions,
            solve_steps=solve_steps,
        )
        summary.update(
            {
                "generated_episode_idx": gen_ep_idx,
                "source_episode_idx": source_episode_idx,
                "translation": [float(v) for v in translation],
                "motion_frame_count": motion_frame_count,
            }
        )
        summaries.append(summary)
        print(
            json.dumps(
                {
                    "generated_episode_idx": gen_ep_idx,
                    "source_episode_idx": source_episode_idx,
                    "translation": [float(v) for v in translation],
                    "final_err_norm": summary["final_err_norm"],
                    "changed_steps": len(summary["changed_steps"]),
                },
                ensure_ascii=False,
            )
        )

    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    generator.save_episodes(generated_episodes, str(output_zarr))

    aggregate = {
        "n_episodes": len(summaries),
        "selected_episodes": selected,
        "rmse_norm_mean": float(np.mean([s["rmse_norm"] for s in summaries])),
        "rmse_norm_max": float(np.max([s["rmse_norm"] for s in summaries])),
        "final_err_norm_mean": float(np.mean([s["final_err_norm"] for s in summaries])),
        "final_err_norm_max": float(np.max([s["final_err_norm"] for s in summaries])),
    }
    result = {
        "config": str(config_path),
        "source_demo": str(source_demo_path),
        "template_zarr": str(template_zarr),
        "output_zarr": str(output_zarr),
        "aggregate": aggregate,
        "episodes": summaries,
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
