#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from replay_zarr_episode import load_reset_state
from solve_lift_prefix_xyz_actions import (
    TASK_OBJECT_STATE_INDICES,
    build_desired_prefix_states,
    instantiate_generator,
    load_cfg,
    solve_prefix_actions,
)

import diffusion_policies.env.robosuite.robosuite_wrapper as robosuite_wrapper
from diffusion_policies.env.robosuite.robosuite_wrapper import Robosuite3DEnv


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
    parser.add_argument("--translation", type=float, nargs=3, required=True)
    parser.add_argument("--solve-steps", type=int, default=None)
    parser.add_argument("--control-steps", type=int, default=1)
    parser.add_argument("--action-deviation-weight", type=float, default=1e-4)
    parser.add_argument("--output-zarr", required=True)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def infer_skill1_frame(generator, source_demo, episode_idx: int) -> int:
    if getattr(generator, "use_manual_parsing_frames", False):
        if hasattr(generator, "resolve_parsing_frame"):
            return int(generator.resolve_parsing_frame("skill-1", episode_idx))
        return int(generator.parsing_frames["skill-1"])
    ee_poses = np.asarray(source_demo["state"][:, :3], dtype=np.float32)
    return int(generator.parse_frames_one_stage(source_demo["point_cloud"], episode_idx, ee_poses))


def replay_full_episode(
    source_demo_path: Path,
    source_demo,
    episode_idx: int,
    translation: np.ndarray,
    solved_prefix_actions: np.ndarray,
    control_steps: int,
):
    robosuite_wrapper.N_CONTROL_STEPS = control_steps
    env = Robosuite3DEnv(str(source_demo_path), render=False)

    reset_state = load_reset_state(source_demo_path, episode_idx)
    reset_state["states"] = np.asarray(reset_state["states"], dtype=np.float64).copy()
    reset_state["states"][TASK_OBJECT_STATE_INDICES["Lift"]] += np.asarray(
        translation[:3], dtype=np.float64
    )
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
    source_actions: np.ndarray,
    solved_actions: np.ndarray,
    solve_steps: int,
) -> dict:
    obs_err = observed_obs_xyz - desired_obs_xyz
    rmse_xyz = np.sqrt(np.mean(obs_err[:-1] ** 2, axis=0))
    final_err_xyz = obs_err[-1]

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
        "source_action_pulse_count_xyz": [
            int(v) for v in (np.abs(source_actions[:solve_steps, :3]) > 0.5).sum(axis=0)
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
    output_zarr = Path(args.output_zarr).resolve()
    output_json = Path(args.output_json).resolve() if args.output_json else None

    translation = np.asarray(args.translation, dtype=np.float32)
    cfg = load_cfg(config_path, data_root)
    generator = instantiate_generator(cfg)
    source_demo = generator.replay_buffer.get_episode(args.episode_idx)

    skill1_frame = infer_skill1_frame(generator, source_demo, args.episode_idx)
    solve_steps = int(args.solve_steps) if args.solve_steps is not None else skill1_frame
    if solve_steps > skill1_frame:
        raise ValueError(f"solve_steps={solve_steps} exceeds skill1_frame={skill1_frame}")

    _, source_actions, desired_obs_xyz, _ = build_desired_prefix_states(
        generator=generator,
        episode_idx=args.episode_idx,
        translation=translation,
        solve_steps=solve_steps,
    )

    solved_actions, observed_obs_xyz, _ = solve_prefix_actions(
        source_demo_path=source_demo_path,
        source_episode_idx=args.episode_idx,
        source_actions=source_actions,
        desired_obs_xyz=desired_obs_xyz,
        translation=translation,
        control_steps=args.control_steps,
        action_deviation_weight=args.action_deviation_weight,
    )

    replay_states, replay_actions = replay_full_episode(
        source_demo_path=source_demo_path,
        source_demo=source_demo,
        episode_idx=args.episode_idx,
        translation=translation,
        solved_prefix_actions=solved_actions,
        control_steps=args.control_steps,
    )

    replay_pcds = build_replayed_point_clouds(
        generator=generator,
        source_demo=source_demo,
        episode_idx=args.episode_idx,
        replay_states=replay_states,
        object_translation=translation,
        motion_frame_count=solve_steps,
    )

    generated_episode = {
        "state": replay_states,
        "action": replay_actions,
        "point_cloud": replay_pcds,
        "_source_episode_idx": int(args.episode_idx),
        "_object_translation": np.asarray(translation, dtype=np.float32),
        "_motion_frame_count": int(solve_steps),
    }
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    generator.save_episodes([generated_episode], str(output_zarr))

    summary = build_summary(
        observed_obs_xyz=observed_obs_xyz,
        desired_obs_xyz=desired_obs_xyz,
        source_actions=source_actions,
        solved_actions=solved_actions,
        solve_steps=solve_steps,
    )
    summary.update(
        {
            "config": str(config_path),
            "source_demo": str(source_demo_path),
            "episode_idx": int(args.episode_idx),
            "translation": [float(v) for v in translation],
            "output_zarr": str(output_zarr),
            "skill1_frame": int(skill1_frame),
        }
    )

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
