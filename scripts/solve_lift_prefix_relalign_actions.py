#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DIFFUSION_POLICIES_ROOT = REPO_ROOT / "repos" / "DemoGen" / "diffusion_policies"
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
if str(DIFFUSION_POLICIES_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICIES_ROOT))

from replay_zarr_episode import load_reset_state
import diffusion_policies.env.robosuite.robosuite_wrapper as robosuite_wrapper
from diffusion_policies.env.robosuite.robosuite_wrapper import Robosuite3DEnv

from solve_lift_prefix_xyz_actions import (
    TASK_OBJECT_STATE_INDICES,
    build_desired_prefix_states,
    candidate_xyzs,
    instantiate_generator,
    load_cfg,
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
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--translation", type=float, nargs=3, default=[0.035, 0.035, 0.0])
    parser.add_argument("--solve-steps", type=int, default=190)
    parser.add_argument("--control-steps", type=int, default=1)
    parser.add_argument("--action-deviation-weight", type=float, default=1e-4)
    parser.add_argument(
        "--relative-tail-steps",
        type=int,
        default=40,
        help="How many late prefix steps receive extra relative-pose alignment cost.",
    )
    parser.add_argument(
        "--relative-cost-weight",
        type=float,
        default=4.0,
        help="Weight for end-of-prefix eef-to-object relative pose alignment.",
    )
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "outputs" / "analysis" / "lift_prefix_relalign_solver_ep0.json"),
    )
    return parser.parse_args()


def _capture_object_pos(env: Robosuite3DEnv) -> np.ndarray:
    if not hasattr(env.env, "cube_body_id"):
        raise AttributeError(
            "Current rel-align solver expects env.env.cube_body_id. "
            "For other tasks, add a task-specific object body accessor."
        )
    return np.asarray(env.env.sim.data.body_xpos[env.env.cube_body_id][:3], dtype=np.float32).copy()


def replay_source_reference(
    source_demo_path: Path,
    source_episode_idx: int,
    source_actions: np.ndarray,
    control_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    robosuite_wrapper.N_CONTROL_STEPS = control_steps
    env = Robosuite3DEnv(str(source_demo_path), render=False)
    reset_state = load_reset_state(source_demo_path, source_episode_idx)
    obs = env.reset_to(reset_state)

    eef_hist = [np.asarray(obs["agent_pos"][:3], dtype=np.float32).copy()]
    obj_hist = [_capture_object_pos(env)]

    try:
        for action in np.asarray(source_actions, dtype=np.float32):
            obs, _, _, _ = env.step(action)
            eef_hist.append(np.asarray(obs["agent_pos"][:3], dtype=np.float32).copy())
            obj_hist.append(_capture_object_pos(env))
    finally:
        env.close()

    return np.asarray(eef_hist, dtype=np.float32), np.asarray(obj_hist, dtype=np.float32)


def build_desired_relative_xyz(
    source_demo_path: Path,
    source_episode_idx: int,
    source_actions: np.ndarray,
    control_steps: int,
) -> np.ndarray:
    source_eef_xyz, source_obj_xyz = replay_source_reference(
        source_demo_path=source_demo_path,
        source_episode_idx=source_episode_idx,
        source_actions=source_actions,
        control_steps=control_steps,
    )
    return np.asarray(source_eef_xyz - source_obj_xyz, dtype=np.float32)


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
    robosuite_wrapper.N_CONTROL_STEPS = control_steps
    env = Robosuite3DEnv(str(source_demo_path), render=False)
    reset_state = load_reset_state(source_demo_path, source_episode_idx)
    reset_state["states"] = np.asarray(reset_state["states"], dtype=np.float64).copy()
    reset_state["states"][TASK_OBJECT_STATE_INDICES["Lift"]] += np.asarray(
        translation[:3], dtype=np.float64
    )
    obs = env.reset_to(reset_state)

    cands = candidate_xyzs()
    solved_actions = []
    observed_obs_xyz = [np.asarray(obs["agent_pos"][:3], dtype=np.float32).copy()]
    observed_rel_xyz = [
        np.asarray(observed_obs_xyz[-1] - _capture_object_pos(env), dtype=np.float32)
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
                next_obj_xyz = _capture_object_pos(env)
                next_rel = next_xyz - next_obj_xyz

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
                        "next_xyz": next_xyz.copy(),
                        "next_rel": next_rel.copy(),
                        "pos_err": pos_err.copy(),
                        "rel_err": rel_err.copy(),
                        "pos_cost": pos_cost,
                        "rel_cost": rel_cost,
                        "act_cost": act_cost,
                        "relative_weight": step_rel_weight,
                        "total_cost": total_cost,
                    }

            env.reset_to({"states": current_state})
            obs_next, _, _, _ = env.step(best["cand_action"])
            next_xyz = np.asarray(obs_next["agent_pos"][:3], dtype=np.float32)
            next_rel = next_xyz - _capture_object_pos(env)

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

    solved_actions = np.asarray(solved_actions, dtype=np.float32)
    observed_obs_xyz = np.asarray(observed_obs_xyz, dtype=np.float32)
    observed_rel_xyz = np.asarray(observed_rel_xyz, dtype=np.float32)
    return solved_actions, observed_obs_xyz, observed_rel_xyz, step_summaries


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    data_root = Path(args.data_root).resolve()
    source_demo_path = Path(args.source_demo).resolve()
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    translation = np.asarray(args.translation, dtype=np.float32)
    cfg = load_cfg(config_path, data_root)
    generator = instantiate_generator(cfg)
    _, source_actions, desired_obs_xyz, _ = build_desired_prefix_states(
        generator=generator,
        episode_idx=args.episode_idx,
        translation=translation,
        solve_steps=args.solve_steps,
    )
    desired_rel_xyz = build_desired_relative_xyz(
        source_demo_path=source_demo_path,
        source_episode_idx=args.episode_idx,
        source_actions=source_actions,
        control_steps=args.control_steps,
    )

    solved_actions, observed_obs_xyz, observed_rel_xyz, step_summaries = solve_prefix_actions(
        source_demo_path=source_demo_path,
        source_episode_idx=args.episode_idx,
        source_actions=source_actions,
        desired_obs_xyz=desired_obs_xyz,
        desired_rel_xyz=desired_rel_xyz,
        translation=translation,
        control_steps=args.control_steps,
        action_deviation_weight=args.action_deviation_weight,
        relative_tail_steps=args.relative_tail_steps,
        relative_cost_weight=args.relative_cost_weight,
    )

    obs_err = observed_obs_xyz - desired_obs_xyz
    rmse_xyz = np.sqrt(np.mean(obs_err[:-1] ** 2, axis=0))
    final_err_xyz = obs_err[-1]
    rel_err = observed_rel_xyz - desired_rel_xyz
    rel_rmse_xyz = np.sqrt(np.mean(rel_err[:-1] ** 2, axis=0))
    rel_final_err_xyz = rel_err[-1]

    result = {
        "config": str(config_path),
        "source_demo": str(source_demo_path),
        "episode_idx": int(args.episode_idx),
        "solve_steps": int(args.solve_steps),
        "translation": [float(v) for v in translation],
        "control_steps": int(args.control_steps),
        "action_deviation_weight": float(args.action_deviation_weight),
        "relative_tail_steps": int(args.relative_tail_steps),
        "relative_cost_weight": float(args.relative_cost_weight),
        "rmse_xyz": [float(v) for v in rmse_xyz],
        "rmse_norm": float(np.linalg.norm(rmse_xyz)),
        "final_err_xyz": [float(v) for v in final_err_xyz],
        "final_err_norm": float(np.linalg.norm(final_err_xyz)),
        "rel_rmse_xyz": [float(v) for v in rel_rmse_xyz],
        "rel_rmse_norm": float(np.linalg.norm(rel_rmse_xyz)),
        "rel_final_err_xyz": [float(v) for v in rel_final_err_xyz],
        "rel_final_err_norm": float(np.linalg.norm(rel_final_err_xyz)),
        "source_action_pulse_count_xyz": [
            int(v) for v in (np.abs(source_actions[:, :3]) > 0.5).sum(axis=0)
        ],
        "solved_action_pulse_count_xyz": [
            int(v) for v in (np.abs(solved_actions[:, :3]) > 0.5).sum(axis=0)
        ],
        "step_summaries": step_summaries,
    }

    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved report to {output_json}")


if __name__ == "__main__":
    main()
