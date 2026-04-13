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
DEMOGEN_ROOT = REPO_ROOT / "repos" / "DemoGen" / "demo_generation"
DIFFUSION_POLICIES_ROOT = REPO_ROOT / "repos" / "DemoGen" / "diffusion_policies"
if str(DEMOGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMOGEN_ROOT))
if str(DIFFUSION_POLICIES_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICIES_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from replay_zarr_episode import load_reset_state
from solve_lift_prefix_xyz_actions import (
    build_desired_prefix_states,
    candidate_xyzs,
    instantiate_generator,
    load_cfg,
)

import diffusion_policies.env.robosuite.robosuite_wrapper as robosuite_wrapper
from demo_generation.handlepress_robosuite_wrapper import HandlePressRobosuite3DEnv


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
            / "handlepress_0_v37_replayh1_light_schedule_phasecopy_replayconsistent.yaml"
        ),
    )
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "repos" / "DemoGen" / "data"),
    )
    parser.add_argument(
        "--source-demo",
        default=str(
            REPO_ROOT
            / "data"
            / "raw"
            / "handlepress_0"
            / "1776042489_1873188"
            / "demo.hdf5"
        ),
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
        help="Episode-level parallel workers.",
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


def _capture_handle_pos(env: HandlePressRobosuite3DEnv) -> np.ndarray:
    return np.asarray(env.env._handle_xpos, dtype=np.float32).copy()


def replay_source_reference(
    source_demo_path: Path,
    source_episode_idx: int,
    source_actions: np.ndarray,
    control_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    robosuite_wrapper.N_CONTROL_STEPS = int(control_steps)
    env = HandlePressRobosuite3DEnv(str(source_demo_path), render=False)
    reset_state = load_reset_state(source_demo_path, source_episode_idx)
    obs = env.reset_to(reset_state)

    eef_hist = [np.asarray(obs["agent_pos"][:3], dtype=np.float32).copy()]
    handle_hist = [_capture_handle_pos(env)]

    try:
        for action in np.asarray(source_actions, dtype=np.float32):
            obs, _, _, _ = env.step(action)
            eef_hist.append(np.asarray(obs["agent_pos"][:3], dtype=np.float32).copy())
            handle_hist.append(_capture_handle_pos(env))
    finally:
        env.close()

    return np.asarray(eef_hist, dtype=np.float32), np.asarray(handle_hist, dtype=np.float32)


def build_desired_relative_xyz(
    source_demo_path: Path,
    source_episode_idx: int,
    source_actions: np.ndarray,
    control_steps: int,
) -> np.ndarray:
    source_eef_xyz, source_handle_xyz = replay_source_reference(
        source_demo_path=source_demo_path,
        source_episode_idx=source_episode_idx,
        source_actions=source_actions,
        control_steps=control_steps,
    )
    return np.asarray(source_eef_xyz - source_handle_xyz, dtype=np.float32)


def _relative_weight_at_step(
    step_idx: int,
    solve_steps: int,
    relative_tail_steps: int,
    relative_cost_weight: float,
) -> float:
    if relative_tail_steps <= 0 or relative_cost_weight <= 0:
        return 0.0
    tail_start = max(0, int(solve_steps) - int(relative_tail_steps))
    if step_idx < tail_start:
        return 0.0
    span = max(1, int(solve_steps) - tail_start)
    alpha = float(step_idx - tail_start + 1) / float(span)
    return float(relative_cost_weight) * alpha


def solve_prefix_actions(
    source_demo_path: Path,
    source_episode_idx: int,
    source_actions: np.ndarray,
    desired_obs_xyz: np.ndarray,
    desired_rel_xyz: np.ndarray,
    translation: np.ndarray,
    control_steps: int,
    action_deviation_weight: float,
    relative_tail_steps: int,
    relative_cost_weight: float,
):
    robosuite_wrapper.N_CONTROL_STEPS = int(control_steps)
    env = HandlePressRobosuite3DEnv(str(source_demo_path), render=False)
    reset_state = load_reset_state(source_demo_path, source_episode_idx)
    reset_state["object_translation"] = np.asarray(translation, dtype=np.float32).copy()
    obs = env.reset_to(reset_state)

    cands = candidate_xyzs()
    solved_actions = []
    observed_obs_xyz = [np.asarray(obs["agent_pos"][:3], dtype=np.float32).copy()]
    observed_rel_xyz = [
        np.asarray(observed_obs_xyz[-1] - _capture_handle_pos(env), dtype=np.float32)
    ]
    step_summaries = []

    try:
        for t in range(len(source_actions)):
            current_state = np.asarray(env.env.sim.get_state().flatten(), dtype=np.float64).copy()
            desired_next_xyz = np.asarray(desired_obs_xyz[t + 1], dtype=np.float32)
            desired_next_rel = np.asarray(desired_rel_xyz[t + 1], dtype=np.float32)
            source_action = np.asarray(source_actions[t], dtype=np.float32)
            step_rel_weight = _relative_weight_at_step(
                step_idx=t,
                solve_steps=len(source_actions),
                relative_tail_steps=relative_tail_steps,
                relative_cost_weight=relative_cost_weight,
            )

            best = None
            for cand_xyz in cands:
                cand_action = source_action.copy()
                cand_action[:3] = cand_xyz

                obs_candidate = env.reset_to({"states": current_state})
                if obs_candidate is None:
                    raise RuntimeError("env.reset_to(states=...) did not return observation")
                obs_next, _, _, _ = env.step(cand_action)
                next_xyz = np.asarray(obs_next["agent_pos"][:3], dtype=np.float32)
                next_rel = next_xyz - _capture_handle_pos(env)

                pos_err = next_xyz - desired_next_xyz
                pos_cost = float(np.dot(pos_err, pos_err))
                rel_err = next_rel - desired_next_rel
                rel_cost = float(np.dot(rel_err, rel_err))
                act_diff = cand_action[:3] - source_action[:3]
                act_cost = float(np.dot(act_diff, act_diff))
                total_cost = (
                    pos_cost
                    + step_rel_weight * rel_cost
                    + action_deviation_weight * act_cost
                )

                if best is None or total_cost < best["total_cost"]:
                    best = {
                        "cand_action": cand_action.copy(),
                        "total_cost": total_cost,
                    }

            env.reset_to({"states": current_state})
            obs_next, _, _, _ = env.step(best["cand_action"])
            next_xyz = np.asarray(obs_next["agent_pos"][:3], dtype=np.float32)
            next_rel = next_xyz - _capture_handle_pos(env)

            solved_actions.append(best["cand_action"].copy())
            observed_obs_xyz.append(next_xyz.copy())
            observed_rel_xyz.append(next_rel.copy())
            step_summaries.append(
                {
                    "step": int(t),
                    "source_action_xyz": [float(v) for v in source_action[:3]],
                    "solved_action_xyz": [float(v) for v in best["cand_action"][:3]],
                    "desired_next_xyz": [float(v) for v in desired_next_xyz],
                    "actual_next_xyz": [float(v) for v in next_xyz],
                    "next_err_xyz": [float(v) for v in (next_xyz - desired_next_xyz)],
                    "desired_next_rel_xyz": [float(v) for v in desired_next_rel],
                    "actual_next_rel_xyz": [float(v) for v in next_rel],
                    "next_rel_err_xyz": [float(v) for v in (next_rel - desired_next_rel)],
                    "relative_weight": float(step_rel_weight),
                    "total_cost": float(best["total_cost"]),
                }
            )
    finally:
        env.close()

    return (
        np.asarray(solved_actions, dtype=np.float32),
        np.asarray(observed_obs_xyz, dtype=np.float32),
        np.asarray(observed_rel_xyz, dtype=np.float32),
        step_summaries,
    )


def replay_full_episode(
    source_demo_path: Path,
    source_demo,
    episode_idx: int,
    translation: np.ndarray,
    solved_prefix_actions: np.ndarray,
    control_steps: int,
):
    robosuite_wrapper.N_CONTROL_STEPS = int(control_steps)
    env = HandlePressRobosuite3DEnv(str(source_demo_path), render=False)

    reset_state = load_reset_state(source_demo_path, episode_idx)
    reset_state["object_translation"] = np.asarray(translation, dtype=np.float32).copy()
    obs = env.reset_to(reset_state)

    source_actions = np.asarray(source_demo["action"], dtype=np.float32)
    traj_states = []
    traj_actions = []

    try:
        for step in range(len(source_actions)):
            traj_states.append(np.asarray(obs["agent_pos"], dtype=np.float32).copy())
            if step < len(solved_prefix_actions):
                action = np.asarray(solved_prefix_actions[step], dtype=np.float32).copy()
            else:
                action = np.asarray(source_actions[step], dtype=np.float32).copy()
            traj_actions.append(action)
            obs, _, _, _ = env.step(action)
    finally:
        env.close()

    return np.asarray(traj_states, dtype=np.float32), np.asarray(traj_actions, dtype=np.float32)


def build_replayed_point_clouds(
    generator,
    source_demo,
    episode_idx: int,
    replay_states: np.ndarray,
    object_translation: np.ndarray,
    motion_frame_count: int,
):
    source_states = np.asarray(source_demo["state"], dtype=np.float32)
    source_pcds = np.asarray(source_demo["point_cloud"], dtype=np.float32)

    first_obj_pcd = generator.get_objects_pcd_from_sam_mask(source_pcds[0], episode_idx, "object")
    obj_bbox = generator.pcd_bbox(first_obj_pcd)

    replay_pcds = []
    for frame_idx in range(len(replay_states)):
        source_pcd = source_pcds[frame_idx].copy()
        trans_sofar = np.asarray(replay_states[frame_idx, :3], dtype=np.float32) - np.asarray(
            source_states[frame_idx, :3], dtype=np.float32
        )

        if frame_idx < motion_frame_count:
            pcd_obj, pcd_robot = generator.pcd_divide(source_pcd, [obj_bbox])
            pcd_obj = generator.pcd_translate(pcd_obj, object_translation)
            pcd_robot = generator.pcd_translate(pcd_robot, trans_sofar)
            replay_pcd = np.concatenate([pcd_robot, pcd_obj], axis=0)
        else:
            replay_pcd = generator.pcd_translate(source_pcd, trans_sofar)

        replay_pcds.append(np.asarray(replay_pcd, dtype=np.float32))

    return replay_pcds


def build_summary(
    observed_obs_xyz: np.ndarray,
    desired_obs_xyz: np.ndarray,
    observed_rel_xyz: np.ndarray,
    desired_rel_xyz: np.ndarray,
    source_actions: np.ndarray,
    solved_actions: np.ndarray,
    solve_steps: int,
) -> dict:
    obs_err = observed_obs_xyz - desired_obs_xyz
    rmse_xyz = np.sqrt(np.mean(obs_err[:-1] ** 2, axis=0))
    final_err_xyz = obs_err[-1]

    rel_err = observed_rel_xyz - desired_rel_xyz
    rel_rmse_xyz = np.sqrt(np.mean(rel_err[:-1] ** 2, axis=0))
    rel_final_err_xyz = rel_err[-1]

    changed_steps = []
    for step in range(solve_steps):
        src = np.asarray(source_actions[step, :3], dtype=np.float32)
        sol = np.asarray(solved_actions[step, :3], dtype=np.float32)
        if np.any(np.abs(src - sol) > 1e-6):
            changed_steps.append(
                {
                    "step": int(step),
                    "source_action_xyz": [float(v) for v in src],
                    "solved_action_xyz": [float(v) for v in sol],
                }
            )

    return {
        "solve_steps": int(solve_steps),
        "rmse_xyz": [float(v) for v in rmse_xyz],
        "rmse_norm": float(np.linalg.norm(rmse_xyz)),
        "final_err_xyz": [float(v) for v in final_err_xyz],
        "final_err_norm": float(np.linalg.norm(final_err_xyz)),
        "rel_rmse_xyz": [float(v) for v in rel_rmse_xyz],
        "rel_rmse_norm": float(np.linalg.norm(rel_rmse_xyz)),
        "rel_final_err_xyz": [float(v) for v in rel_final_err_xyz],
        "rel_final_err_norm": float(np.linalg.norm(rel_final_err_xyz)),
        "source_action_pulse_count_xyz": [
            int(v) for v in (np.abs(source_actions[:solve_steps, :3]) > 0.5).sum(axis=0)
        ],
        "solved_action_pulse_count_xyz": [
            int(v) for v in (np.abs(solved_actions[:solve_steps, :3]) > 0.5).sum(axis=0)
        ],
        "changed_steps": changed_steps,
    }


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
    solved_actions, observed_obs_xyz, observed_rel_xyz, step_summaries = solve_prefix_actions(
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
        observed_rel_xyz=observed_rel_xyz,
        desired_rel_xyz=desired_rel_xyz,
        source_actions=source_actions,
        solved_actions=solved_actions,
        solve_steps=solve_steps,
    )
    summary.update(
        {
            "generated_episode_idx": int(gen_ep_idx),
            "source_episode_idx": int(source_episode_idx),
            "translation": [float(v) for v in translation],
            "motion_frame_count": int(motion_frame_count),
            "relative_tail_steps": int(relative_tail_steps),
            "relative_cost_weight": float(relative_cost_weight),
            "step_summaries": step_summaries,
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
