#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_lift_solved_prefix_zarr import (
    build_replayed_point_clouds,
    build_summary,
    replay_full_episode,
)
from solve_lift_prefix_relalign_actions import (
    build_desired_relative_xyz,
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
            / "lift_0_v37_replayh1_light_schedule_phasecopy_replayconsistent.yaml"
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
    parser.add_argument("--relative-tail-steps", type=int, default=40)
    parser.add_argument("--relative-cost-weight", type=float, default=4.0)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Episode-level parallel workers. This solve is CPU-bound; larger values use more CPU cores.",
    )
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


def solve_single_generated_episode(
    *,
    config_path: str,
    data_root: str,
    source_demo_path: str,
    gen_ep_idx: int,
    source_episode_idx: int,
    translation: list[float],
    motion_frame_count: int,
    solve_steps_override: int | None,
    control_steps: int,
    action_deviation_weight: float,
    relative_tail_steps: int,
    relative_cost_weight: float,
):
    config_path = Path(config_path).resolve()
    data_root = Path(data_root).resolve()
    source_demo_path = Path(source_demo_path).resolve()
    translation = np.asarray(translation, dtype=np.float32)

    cfg = load_cfg(config_path, data_root)
    generator = instantiate_generator(cfg)
    solve_steps = int(solve_steps_override) if solve_steps_override is not None else int(motion_frame_count)
    if solve_steps > int(motion_frame_count):
        raise ValueError(
            f"solve_steps={solve_steps} exceeds template motion_frame_count={motion_frame_count} "
            f"for generated episode {gen_ep_idx}"
        )

    source_demo = generator.replay_buffer.get_episode(int(source_episode_idx))
    _, source_actions, desired_obs_xyz, _ = build_desired_prefix_states(
        generator=generator,
        episode_idx=int(source_episode_idx),
        translation=translation,
        solve_steps=solve_steps,
    )
    desired_rel_xyz = build_desired_relative_xyz(
        source_demo_path=source_demo_path,
        source_episode_idx=int(source_episode_idx),
        source_actions=source_actions,
        control_steps=int(control_steps),
    )
    solved_actions, observed_obs_xyz, observed_rel_xyz, _ = solve_prefix_actions(
        source_demo_path=source_demo_path,
        source_episode_idx=int(source_episode_idx),
        source_actions=source_actions,
        desired_obs_xyz=desired_obs_xyz,
        desired_rel_xyz=desired_rel_xyz,
        translation=translation,
        control_steps=int(control_steps),
        action_deviation_weight=float(action_deviation_weight),
        relative_tail_steps=int(relative_tail_steps),
        relative_cost_weight=float(relative_cost_weight),
    )

    replay_states, replay_actions = replay_full_episode(
        source_demo_path=source_demo_path,
        source_demo=source_demo,
        episode_idx=int(source_episode_idx),
        translation=translation,
        solved_prefix_actions=solved_actions,
        control_steps=int(control_steps),
    )
    replay_pcds = build_replayed_point_clouds(
        generator=generator,
        source_demo=source_demo,
        episode_idx=int(source_episode_idx),
        replay_states=replay_states,
        object_translation=translation,
        motion_frame_count=solve_steps,
    )
    generated_episode = {
        "state": replay_states,
        "action": replay_actions,
        "point_cloud": replay_pcds,
        "_source_episode_idx": int(source_episode_idx),
        "_object_translation": translation,
        "_motion_frame_count": int(solve_steps),
    }

    summary = build_summary(
        observed_obs_xyz=observed_obs_xyz,
        desired_obs_xyz=desired_obs_xyz,
        source_actions=source_actions,
        solved_actions=solved_actions,
        solve_steps=solve_steps,
    )
    rel_err = observed_rel_xyz - desired_rel_xyz
    rel_rmse_xyz = np.sqrt(np.mean(rel_err[:-1] ** 2, axis=0))
    rel_final_err_xyz = rel_err[-1]
    summary.update(
        {
            "generated_episode_idx": int(gen_ep_idx),
            "source_episode_idx": int(source_episode_idx),
            "translation": [float(v) for v in translation],
            "motion_frame_count": int(motion_frame_count),
            "relative_tail_steps": int(relative_tail_steps),
            "relative_cost_weight": float(relative_cost_weight),
            "rel_rmse_xyz": [float(v) for v in rel_rmse_xyz],
            "rel_rmse_norm": float(np.linalg.norm(rel_rmse_xyz)),
            "rel_final_err_xyz": [float(v) for v in rel_final_err_xyz],
            "rel_final_err_norm": float(np.linalg.norm(rel_final_err_xyz)),
        }
    )
    return {
        "generated_episode_idx": int(gen_ep_idx),
        "generated_episode": generated_episode,
        "summary": summary,
    }


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
    jobs = []
    for gen_ep_idx in selected:
        jobs.append(
            {
                "config_path": str(config_path),
                "data_root": str(data_root),
                "source_demo_path": str(source_demo_path),
                "gen_ep_idx": int(gen_ep_idx),
                "source_episode_idx": int(template_source_idx[gen_ep_idx]),
                "translation": np.asarray(template_translation[gen_ep_idx], dtype=np.float32).tolist(),
                "motion_frame_count": int(template_motion_frames[gen_ep_idx]),
                "solve_steps_override": args.solve_steps,
                "control_steps": int(args.control_steps),
                "action_deviation_weight": float(args.action_deviation_weight),
                "relative_tail_steps": int(args.relative_tail_steps),
                "relative_cost_weight": float(args.relative_cost_weight),
            }
        )

    results_by_episode = {}

    if int(args.num_workers) <= 1:
        for job in jobs:
            result = solve_single_generated_episode(**job)
            results_by_episode[result["generated_episode_idx"]] = result
            summary = result["summary"]
            print(
                json.dumps(
                    {
                        "generated_episode_idx": result["generated_episode_idx"],
                        "source_episode_idx": summary["source_episode_idx"],
                        "translation": summary["translation"],
                        "final_err_norm": summary["final_err_norm"],
                        "rel_final_err_norm": summary["rel_final_err_norm"],
                        "changed_steps": len(summary["changed_steps"]),
                    },
                    ensure_ascii=False,
                )
            )
    else:
        max_workers = min(int(args.num_workers), len(jobs))
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            future_to_ep = {
                executor.submit(solve_single_generated_episode, **job): int(job["gen_ep_idx"])
                for job in jobs
            }
            completed = 0
            total = len(future_to_ep)
            for future in as_completed(future_to_ep):
                result = future.result()
                results_by_episode[result["generated_episode_idx"]] = result
                summary = result["summary"]
                completed += 1
                print(
                    json.dumps(
                        {
                            "progress": f"{completed}/{total}",
                            "generated_episode_idx": result["generated_episode_idx"],
                            "source_episode_idx": summary["source_episode_idx"],
                            "translation": summary["translation"],
                            "final_err_norm": summary["final_err_norm"],
                            "rel_final_err_norm": summary["rel_final_err_norm"],
                            "changed_steps": len(summary["changed_steps"]),
                        },
                        ensure_ascii=False,
                    )
                )

    generated_episodes = [results_by_episode[idx]["generated_episode"] for idx in selected]
    summaries = [results_by_episode[idx]["summary"] for idx in selected]

    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    generator.save_episodes(generated_episodes, str(output_zarr))

    aggregate = {
        "n_episodes": len(summaries),
        "selected_episodes": selected,
        "rmse_norm_mean": float(np.mean([s["rmse_norm"] for s in summaries])),
        "rmse_norm_max": float(np.max([s["rmse_norm"] for s in summaries])),
        "final_err_norm_mean": float(np.mean([s["final_err_norm"] for s in summaries])),
        "final_err_norm_max": float(np.max([s["final_err_norm"] for s in summaries])),
        "rel_final_err_norm_mean": float(np.mean([s["rel_final_err_norm"] for s in summaries])),
        "rel_final_err_norm_max": float(np.max([s["rel_final_err_norm"] for s in summaries])),
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
