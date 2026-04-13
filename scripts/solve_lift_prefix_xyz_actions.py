#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import hydra

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
import diffusion_policies.env.robosuite.robosuite_wrapper as robosuite_wrapper
from diffusion_policies.env.robosuite.robosuite_wrapper import Robosuite3DEnv


TASK_OBJECT_STATE_INDICES = {
    "Lift": np.array([10, 11, 12], dtype=np.int64),
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
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--translation", type=float, nargs=3, default=[0.035, 0.035, 0.0])
    parser.add_argument("--solve-steps", type=int, default=190)
    parser.add_argument("--control-steps", type=int, default=1)
    parser.add_argument("--action-deviation-weight", type=float, default=1e-4)
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "outputs" / "analysis" / "lift_prefix_xyz_solver_ep0.json"),
    )
    return parser.parse_args()


def load_cfg(config_path: Path, data_root: Path):
    cfg = OmegaConf.load(config_path)
    cfg.data_root = str(data_root)
    OmegaConf.resolve(cfg)
    return cfg


def instantiate_generator(cfg):
    cls = hydra.utils.get_class(cfg._target_)
    return cls(cfg)


def build_desired_prefix_states(generator, episode_idx: int, translation: np.ndarray, solve_steps: int):
    source_demo = generator.replay_buffer.get_episode(episode_idx)
    if hasattr(generator, "resolve_parsing_frame"):
        skill1_frame = int(generator.resolve_parsing_frame("skill-1", episode_idx))
    else:
        skill1_frame = int(generator.parsing_frames["skill-1"])
    if solve_steps > skill1_frame:
        raise ValueError(f"solve_steps={solve_steps} exceeds skill1_frame={skill1_frame}")
    source_state_xyz = np.asarray(source_demo["state"][:, :3], dtype=np.float32)
    if source_state_xyz.shape[0] < solve_steps + 1:
        raise ValueError(
            f"source episode len={source_state_xyz.shape[0]} too short for solve_steps={solve_steps}"
        )
    increments = generator._build_translation_increments(
        source_demo=source_demo,
        skill_1_frame=skill1_frame,
        object_translation=np.asarray(translation, dtype=np.float32),
    )[:solve_steps]
    cum_before = np.zeros((solve_steps + 1, 3), dtype=np.float32)
    for t in range(solve_steps):
        cum_before[t + 1] = cum_before[t] + increments[t]
    desired_obs_xyz = source_state_xyz[: solve_steps + 1] + cum_before
    source_actions = np.asarray(source_demo["action"][:solve_steps], dtype=np.float32)
    return source_demo, source_actions, desired_obs_xyz, increments


def candidate_xyzs():
    return np.asarray(list(itertools.product([-1.0, 0.0, 1.0], repeat=3)), dtype=np.float32)


def solve_prefix_actions(
    source_demo_path: Path,
    source_episode_idx: int,
    source_actions: np.ndarray,
    desired_obs_xyz: np.ndarray,
    translation: np.ndarray,
    control_steps: int,
    action_deviation_weight: float,
):
    robosuite_wrapper.N_CONTROL_STEPS = control_steps
    env = Robosuite3DEnv(str(source_demo_path), render=False)
    reset_state = load_reset_state(source_demo_path, source_episode_idx)
    reset_state["states"] = np.asarray(reset_state["states"], dtype=np.float64).copy()
    reset_state["states"][TASK_OBJECT_STATE_INDICES["Lift"]] += np.asarray(translation[:3], dtype=np.float64)
    obs = env.reset_to(reset_state)

    cands = candidate_xyzs()
    solved_actions = []
    observed_obs_xyz = [np.asarray(obs["agent_pos"][:3], dtype=np.float32).copy()]
    step_summaries = []

    try:
        for t in range(len(source_actions)):
            current_state = np.asarray(env.env.sim.get_state().flatten(), dtype=np.float64).copy()
            desired_next_xyz = np.asarray(desired_obs_xyz[t + 1], dtype=np.float32)
            source_action = np.asarray(source_actions[t], dtype=np.float32)

            best = None
            for cand_xyz in cands:
                cand_action = source_action.copy()
                cand_action[:3] = cand_xyz

                obs_candidate = env.reset_to({"states": current_state})
                if obs_candidate is None:
                    raise RuntimeError("env.reset_to(states=...) did not return observation")
                obs_next, _, _, _ = env.step(cand_action)
                next_xyz = np.asarray(obs_next["agent_pos"][:3], dtype=np.float32)

                pos_err = next_xyz - desired_next_xyz
                pos_cost = float(np.dot(pos_err, pos_err))
                act_diff = cand_action[:3] - source_action[:3]
                act_cost = float(np.dot(act_diff, act_diff))
                total_cost = pos_cost + action_deviation_weight * act_cost

                if best is None or total_cost < best["total_cost"]:
                    best = {
                        "cand_action": cand_action.copy(),
                        "next_xyz": next_xyz.copy(),
                        "pos_err": pos_err.copy(),
                        "pos_cost": pos_cost,
                        "act_cost": act_cost,
                        "total_cost": total_cost,
                    }

            env.reset_to({"states": current_state})
            obs_next, _, _, _ = env.step(best["cand_action"])
            next_xyz = np.asarray(obs_next["agent_pos"][:3], dtype=np.float32)

            solved_actions.append(best["cand_action"].copy())
            observed_obs_xyz.append(next_xyz.copy())
            step_summaries.append(
                {
                    "step": int(t),
                    "source_action_xyz": [float(v) for v in source_action[:3]],
                    "solved_action_xyz": [float(v) for v in best["cand_action"][:3]],
                    "desired_next_xyz": [float(v) for v in desired_next_xyz],
                    "actual_next_xyz": [float(v) for v in next_xyz],
                    "next_err_xyz": [float(v) for v in (next_xyz - desired_next_xyz)],
                    "total_cost": float(best["total_cost"]),
                }
            )
    finally:
        env.close()

    solved_actions = np.asarray(solved_actions, dtype=np.float32)
    observed_obs_xyz = np.asarray(observed_obs_xyz, dtype=np.float32)
    return solved_actions, observed_obs_xyz, step_summaries


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

    solved_actions, observed_obs_xyz, step_summaries = solve_prefix_actions(
        source_demo_path=source_demo_path,
        source_episode_idx=args.episode_idx,
        source_actions=source_actions,
        desired_obs_xyz=desired_obs_xyz,
        translation=translation,
        control_steps=args.control_steps,
        action_deviation_weight=args.action_deviation_weight,
    )

    obs_err = observed_obs_xyz - desired_obs_xyz
    rmse_xyz = np.sqrt(np.mean(obs_err[:-1] ** 2, axis=0))
    final_err_xyz = obs_err[-1]

    result = {
        "config": str(config_path),
        "source_demo": str(source_demo_path),
        "episode_idx": int(args.episode_idx),
        "solve_steps": int(args.solve_steps),
        "translation": [float(v) for v in translation],
        "control_steps": int(args.control_steps),
        "action_deviation_weight": float(args.action_deviation_weight),
        "rmse_xyz": [float(v) for v in rmse_xyz],
        "rmse_norm": float(np.linalg.norm(rmse_xyz)),
        "final_err_xyz": [float(v) for v in final_err_xyz],
        "final_err_norm": float(np.linalg.norm(final_err_xyz)),
        "source_action_pulse_count_xyz": [
            int(v) for v in (np.abs(source_actions[:, :3]) > 0.5).sum(axis=0)
        ],
        "solved_action_pulse_count_xyz": [
            int(v) for v in (np.abs(solved_actions[:, :3]) > 0.5).sum(axis=0)
        ],
        "source_action_sum_xyz": [float(v) for v in source_actions[:, :3].sum(axis=0)],
        "solved_action_sum_xyz": [float(v) for v in solved_actions[:, :3].sum(axis=0)],
        "desired_last_xyz": [float(v) for v in desired_obs_xyz[-1]],
        "observed_last_xyz": [float(v) for v in observed_obs_xyz[-1]],
        "step_summaries": step_summaries,
    }

    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved report to {output_json}")


if __name__ == "__main__":
    main()
