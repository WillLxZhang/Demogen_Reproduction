#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
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

import diffusion_policies.env.robosuite.robosuite_wrapper as robosuite_wrapper
from diffusion_policies.env.robosuite.robosuite_wrapper import Robosuite3DEnv

from replay_zarr_episode import load_reset_state
from solve_lift_prefix_xyz_actions import instantiate_generator, load_cfg


TASK_OBJECT_STATE_INDICES = {
    "Stack": {
        "object": np.array([10, 11, 12], dtype=np.int64),
        "target": np.array([17, 18, 19], dtype=np.int64),
    },
}


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
            / "stack_cube_0_v5_replayh1_twophase_noschedule.yaml"
        ),
    )
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "repos" / "DemoGen" / "data"),
    )
    parser.add_argument(
        "--source-demo",
        default=str(
            REPO_ROOT / "data" / "raw" / "stack_cube_0" / "1775663680_9007828" / "demo.hdf5"
        ),
    )
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--translate-object", type=float, nargs=3, default=[0.03, 0.03, 0.0])
    parser.add_argument("--translate-target", type=float, nargs=3, default=[0.03, 0.03, 0.0])
    parser.add_argument("--solve-steps", type=int, default=None)
    parser.add_argument("--control-steps", type=int, default=1)
    parser.add_argument("--action-deviation-weight", type=float, default=1e-4)
    parser.add_argument("--relative-tail-steps", type=int, default=40)
    parser.add_argument("--relative-cost-weight", type=float, default=4.0)
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "outputs" / "analysis" / "stack_motion1_relalign_solver_ep0.json"),
    )
    return parser.parse_args()


def split_translation(raw: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.shape == (3,):
        return arr, np.zeros(3, dtype=np.float32)
    if arr.shape == (6,):
        return arr[:3], arr[3:6]
    raise ValueError(f"Expected translation shape (3,) or (6,), got {arr.shape}")


def infer_twophase_frames(generator, source_demo, episode_idx: int) -> tuple[int, int, int]:
    if getattr(generator, "use_manual_parsing_frames", False):
        return (
            int(generator.parsing_frames["skill-1"]),
            int(generator.parsing_frames["motion-2"]),
            int(generator.parsing_frames["skill-2"]),
        )
    ee_poses = np.asarray(source_demo["state"][:, :3], dtype=np.float32)
    return tuple(
        int(v) for v in generator.parse_frames_two_stage(source_demo["point_cloud"], episode_idx, ee_poses)
    )


def build_desired_motion1_states(
    generator,
    episode_idx: int,
    object_translation: np.ndarray,
    solve_steps: int,
    skill1_frame: int | None = None,
):
    source_demo = generator.replay_buffer.get_episode(episode_idx)
    inferred_skill1_frame, _, _ = infer_twophase_frames(generator, source_demo, episode_idx)
    skill1_frame = inferred_skill1_frame if skill1_frame is None else int(skill1_frame)
    if solve_steps > skill1_frame:
        raise ValueError(f"solve_steps={solve_steps} exceeds skill1_frame={skill1_frame}")

    source_state_xyz = np.asarray(source_demo["state"][:, :3], dtype=np.float32)
    if source_state_xyz.shape[0] < solve_steps + 1:
        raise ValueError(
            f"source episode len={source_state_xyz.shape[0]} too short for solve_steps={solve_steps}"
        )

    increments = generator._build_segment_translation_increments(
        source_demo=source_demo,
        start_frame=0,
        end_frame=skill1_frame,
        translation=np.asarray(object_translation, dtype=np.float32),
    )[:solve_steps]
    cum_before = np.zeros((solve_steps + 1, 3), dtype=np.float32)
    for t in range(solve_steps):
        cum_before[t + 1] = cum_before[t] + increments[t]
    desired_obs_xyz = source_state_xyz[: solve_steps + 1] + cum_before
    return source_demo, desired_obs_xyz, increments, skill1_frame


def candidate_xyzs():
    return np.asarray(list(itertools.product([-1.0, 0.0, 1.0], repeat=3)), dtype=np.float32)


def _capture_stack_object_pos(env: Robosuite3DEnv, which: str) -> np.ndarray:
    attr = {"object": "cubeA_body_id", "target": "cubeB_body_id"}.get(which)
    if attr is None:
        raise ValueError(f"Unsupported stack object selector: {which}")
    if not hasattr(env.env, attr):
        raise AttributeError(
            f"Current rel-align solver expects env.env.{attr}. "
            "For other tasks, add a task-specific object body accessor."
        )
    body_id = getattr(env.env, attr)
    return np.asarray(env.env.sim.data.body_xpos[body_id][:3], dtype=np.float32).copy()


def replay_source_reference(
    source_demo_path: Path,
    source_episode_idx: int,
    source_actions: np.ndarray,
    control_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    robosuite_wrapper.N_CONTROL_STEPS = control_steps
    env = Robosuite3DEnv(str(source_demo_path), render=False)
    reset_state = load_reset_state(source_demo_path, source_episode_idx)
    obs = env.reset_to(reset_state)

    eef_hist = [np.asarray(obs["agent_pos"][:3], dtype=np.float32).copy()]
    obj_hist = [_capture_stack_object_pos(env, "object")]
    tar_hist = [_capture_stack_object_pos(env, "target")]

    try:
        for action in np.asarray(source_actions, dtype=np.float32):
            obs, _, _, _ = env.step(action)
            eef_hist.append(np.asarray(obs["agent_pos"][:3], dtype=np.float32).copy())
            obj_hist.append(_capture_stack_object_pos(env, "object"))
            tar_hist.append(_capture_stack_object_pos(env, "target"))
    finally:
        env.close()

    return (
        np.asarray(eef_hist, dtype=np.float32),
        np.asarray(obj_hist, dtype=np.float32),
        np.asarray(tar_hist, dtype=np.float32),
    )


def build_desired_relative_xyz(
    source_demo_path: Path,
    source_episode_idx: int,
    source_actions: np.ndarray,
    control_steps: int,
) -> np.ndarray:
    source_eef_xyz, source_obj_xyz, _ = replay_source_reference(
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


def solve_motion1_actions(
    source_demo_path: Path,
    source_episode_idx: int,
    base_actions: np.ndarray,
    desired_obs_xyz: np.ndarray,
    desired_rel_xyz: np.ndarray,
    object_translation: np.ndarray,
    target_translation: np.ndarray,
    control_steps: int,
    action_deviation_weight: float,
    relative_tail_steps: int,
    relative_cost_weight: float,
):
    robosuite_wrapper.N_CONTROL_STEPS = control_steps
    env = Robosuite3DEnv(str(source_demo_path), render=False)
    reset_state = load_reset_state(source_demo_path, source_episode_idx)
    reset_state["states"] = np.asarray(reset_state["states"], dtype=np.float64).copy()
    reset_state["states"][TASK_OBJECT_STATE_INDICES["Stack"]["object"]] += np.asarray(
        object_translation[:3], dtype=np.float64
    )
    reset_state["states"][TASK_OBJECT_STATE_INDICES["Stack"]["target"]] += np.asarray(
        target_translation[:3], dtype=np.float64
    )
    obs = env.reset_to(reset_state)

    cands = candidate_xyzs()
    solved_actions = []
    observed_obs_xyz = [np.asarray(obs["agent_pos"][:3], dtype=np.float32).copy()]
    observed_rel_xyz = [
        np.asarray(observed_obs_xyz[-1] - _capture_stack_object_pos(env, "object"), dtype=np.float32)
    ]
    step_summaries = []

    try:
        for t in range(len(base_actions)):
            current_state = np.asarray(env.env.sim.get_state().flatten(), dtype=np.float64).copy()
            desired_next_xyz = np.asarray(desired_obs_xyz[t + 1], dtype=np.float32)
            desired_next_rel = np.asarray(desired_rel_xyz[t + 1], dtype=np.float32)
            base_action = np.asarray(base_actions[t], dtype=np.float32)
            step_rel_weight = _relative_weight_at_step(
                step_idx=t,
                solve_steps=len(base_actions),
                relative_tail_steps=relative_tail_steps,
                relative_cost_weight=relative_cost_weight,
            )

            best = None
            for cand_xyz in cands:
                cand_action = base_action.copy()
                cand_action[:3] = cand_xyz

                obs_candidate = env.reset_to({"states": current_state})
                if obs_candidate is None:
                    raise RuntimeError("env.reset_to(states=...) did not return observation")
                obs_next, _, _, _ = env.step(cand_action)
                next_xyz = np.asarray(obs_next["agent_pos"][:3], dtype=np.float32)
                next_obj_xyz = _capture_stack_object_pos(env, "object")
                next_rel = next_xyz - next_obj_xyz

                pos_err = next_xyz - desired_next_xyz
                pos_cost = float(np.dot(pos_err, pos_err))
                rel_err = next_rel - desired_next_rel
                rel_cost = float(np.dot(rel_err, rel_err))
                act_diff = cand_action[:3] - base_action[:3]
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
            next_rel = next_xyz - _capture_stack_object_pos(env, "object")

            solved_actions.append(best["cand_action"].copy())
            observed_obs_xyz.append(next_xyz.copy())
            observed_rel_xyz.append(next_rel.copy())
            step_summaries.append(
                {
                    "step": int(t),
                    "base_action_xyz": [float(v) for v in base_action[:3]],
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


def build_summary(
    observed_obs_xyz: np.ndarray,
    desired_obs_xyz: np.ndarray,
    observed_rel_xyz: np.ndarray,
    desired_rel_xyz: np.ndarray,
    base_actions: np.ndarray,
    solved_actions: np.ndarray,
    solve_steps: int,
    relative_tail_steps: int,
) -> dict:
    obs_err = observed_obs_xyz - desired_obs_xyz
    rmse_xyz = np.sqrt(np.mean(obs_err[:-1] ** 2, axis=0))
    final_err_xyz = obs_err[-1]

    rel_err = observed_rel_xyz - desired_rel_xyz
    rel_rmse_xyz = np.sqrt(np.mean(rel_err[:-1] ** 2, axis=0))
    rel_final_err_xyz = rel_err[-1]

    tail_start = max(0, int(solve_steps) - int(relative_tail_steps))
    rel_tail = rel_err[tail_start:]
    rel_tail_rmse_xyz = (
        np.sqrt(np.mean(rel_tail**2, axis=0)) if len(rel_tail) else np.zeros(3, dtype=np.float32)
    )

    changed_steps = []
    for step in range(solve_steps):
        base = np.asarray(base_actions[step, :3], dtype=np.float32)
        sol = np.asarray(solved_actions[step, :3], dtype=np.float32)
        if np.any(np.abs(base - sol) > 1e-6):
            changed_steps.append(
                {
                    "step": int(step),
                    "base_action_xyz": [float(v) for v in base],
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
        "rel_tail_start_step": int(tail_start),
        "rel_tail_rmse_xyz": [float(v) for v in rel_tail_rmse_xyz],
        "rel_tail_rmse_norm": float(np.linalg.norm(rel_tail_rmse_xyz)),
        "base_action_pulse_count_xyz": [
            int(v) for v in (np.abs(base_actions[:solve_steps, :3]) > 0.5).sum(axis=0)
        ],
        "solved_action_pulse_count_xyz": [
            int(v) for v in (np.abs(solved_actions[:solve_steps, :3]) > 0.5).sum(axis=0)
        ],
        "changed_steps": changed_steps,
    }


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    data_root = Path(args.data_root).resolve()
    source_demo_path = Path(args.source_demo).resolve()
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    object_translation = np.asarray(args.translate_object, dtype=np.float32)
    target_translation = np.asarray(args.translate_target, dtype=np.float32)

    cfg = load_cfg(config_path, data_root)
    cfg.source_demo_hdf5 = str(source_demo_path)
    generator = instantiate_generator(cfg)
    source_demo = generator.replay_buffer.get_episode(args.episode_idx)
    skill1_frame, _, _ = infer_twophase_frames(generator, source_demo, args.episode_idx)
    solve_steps = int(args.solve_steps) if args.solve_steps is not None else int(skill1_frame)

    _, desired_obs_xyz, _, _ = build_desired_motion1_states(
        generator=generator,
        episode_idx=args.episode_idx,
        object_translation=object_translation,
        solve_steps=solve_steps,
        skill1_frame=skill1_frame,
    )
    source_prefix_actions = np.asarray(source_demo["action"][:solve_steps], dtype=np.float32)
    desired_rel_xyz = build_desired_relative_xyz(
        source_demo_path=source_demo_path,
        source_episode_idx=args.episode_idx,
        source_actions=source_prefix_actions,
        control_steps=args.control_steps,
    )
    solved_actions, observed_obs_xyz, observed_rel_xyz, step_summaries = solve_motion1_actions(
        source_demo_path=source_demo_path,
        source_episode_idx=args.episode_idx,
        base_actions=source_prefix_actions,
        desired_obs_xyz=desired_obs_xyz,
        desired_rel_xyz=desired_rel_xyz,
        object_translation=object_translation,
        target_translation=target_translation,
        control_steps=args.control_steps,
        action_deviation_weight=args.action_deviation_weight,
        relative_tail_steps=args.relative_tail_steps,
        relative_cost_weight=args.relative_cost_weight,
    )

    result = build_summary(
        observed_obs_xyz=observed_obs_xyz,
        desired_obs_xyz=desired_obs_xyz,
        observed_rel_xyz=observed_rel_xyz,
        desired_rel_xyz=desired_rel_xyz,
        base_actions=source_prefix_actions,
        solved_actions=solved_actions,
        solve_steps=solve_steps,
        relative_tail_steps=args.relative_tail_steps,
    )
    result.update(
        {
            "config": str(config_path),
            "source_demo": str(source_demo_path),
            "episode_idx": int(args.episode_idx),
            "skill1_frame": int(skill1_frame),
            "control_steps": int(args.control_steps),
            "action_deviation_weight": float(args.action_deviation_weight),
            "relative_tail_steps": int(args.relative_tail_steps),
            "relative_cost_weight": float(args.relative_cost_weight),
            "object_translation": [float(v) for v in object_translation],
            "target_translation": [float(v) for v in target_translation],
            "desired_last_xyz": [float(v) for v in desired_obs_xyz[-1]],
            "observed_last_xyz": [float(v) for v in observed_obs_xyz[-1]],
            "desired_last_rel_xyz": [float(v) for v in desired_rel_xyz[-1]],
            "observed_last_rel_xyz": [float(v) for v in observed_rel_xyz[-1]],
            "step_summaries": step_summaries,
        }
    )

    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
