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

import diffusion_policies.env.robosuite.robosuite_wrapper as robosuite_wrapper
from diffusion_policies.env.robosuite.robosuite_wrapper import Robosuite3DEnv
from diffusion_policies.common.replay_buffer import ReplayBuffer

from export_stack_solved_from_template_zarr_relalign import (
    build_replayed_point_clouds,
    load_template_meta,
    parse_episode_list,
    replay_full_episode,
)
from replay_zarr_episode import load_reset_state
from solve_stack_motion1_relalign_actions import (
    TASK_OBJECT_STATE_INDICES,
    build_desired_motion1_states,
    build_desired_relative_xyz,
    build_summary,
    infer_twophase_frames,
    instantiate_generator,
    load_cfg,
    replay_source_reference,
    solve_motion1_actions,
    split_translation,
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
    parser.add_argument("--template-zarr", required=True)
    parser.add_argument(
        "--episodes",
        default="all",
        help="Comma-separated generated episode indices, or 'all'.",
    )
    parser.add_argument("--motion1-solve-steps", type=int, default=None)
    parser.add_argument("--motion2-solve-steps", type=int, default=None)
    parser.add_argument("--control-steps", type=int, default=1)
    parser.add_argument("--action-deviation-weight", type=float, default=1e-4)
    parser.add_argument("--motion1-relative-tail-steps", type=int, default=40)
    parser.add_argument("--motion1-relative-cost-weight", type=float, default=4.0)
    parser.add_argument("--motion2-relative-tail-steps", type=int, default=40)
    parser.add_argument("--motion2-relative-cost-weight", type=float, default=4.0)
    parser.add_argument("--output-zarr", required=True)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def build_desired_motion2_states(
    generator,
    source_episode_idx: int,
    object_translation: np.ndarray,
    target_translation: np.ndarray,
    motion2_frame: int,
    solve_steps: int,
):
    source_demo = generator.replay_buffer.get_episode(source_episode_idx)
    skill1_frame, inferred_motion2_frame, skill2_frame = infer_twophase_frames(
        generator,
        source_demo,
        source_episode_idx,
    )
    if motion2_frame != inferred_motion2_frame:
        raise ValueError(
            f"motion2_frame={motion2_frame} disagrees with inferred motion2_frame={inferred_motion2_frame}"
        )
    segment_len = int(skill2_frame) - int(motion2_frame)
    if solve_steps > segment_len:
        raise ValueError(f"solve_steps={solve_steps} exceeds motion2 segment length={segment_len}")

    translation_delta = np.asarray(target_translation, dtype=np.float32) - np.asarray(
        object_translation, dtype=np.float32
    )
    increments = generator._build_segment_translation_increments(
        source_demo=source_demo,
        start_frame=motion2_frame,
        end_frame=skill2_frame,
        translation=translation_delta,
    )[:solve_steps]
    cum_before = np.zeros((solve_steps + 1, 3), dtype=np.float32)
    for t in range(solve_steps):
        cum_before[t + 1] = cum_before[t] + increments[t]
    current_object_offset = np.asarray(object_translation, dtype=np.float32)[None, :] + cum_before

    source_state_xyz = np.asarray(
        source_demo["state"][motion2_frame : motion2_frame + solve_steps + 1, :3],
        dtype=np.float32,
    )
    desired_obs_xyz = source_state_xyz + current_object_offset
    return desired_obs_xyz, current_object_offset, skill1_frame, skill2_frame


def build_desired_object_target_relative_xyz(
    source_demo_path: Path,
    source_episode_idx: int,
    source_actions: np.ndarray,
    motion2_frame: int,
    solve_steps: int,
    current_object_offset: np.ndarray,
    target_translation: np.ndarray,
    control_steps: int,
) -> np.ndarray:
    _, source_obj_xyz, source_tar_xyz = replay_source_reference(
        source_demo_path=source_demo_path,
        source_episode_idx=source_episode_idx,
        source_actions=source_actions,
        control_steps=control_steps,
    )
    source_rel = np.asarray(source_obj_xyz - source_tar_xyz, dtype=np.float32)
    segment_source_rel = np.asarray(
        source_rel[motion2_frame : motion2_frame + solve_steps + 1],
        dtype=np.float32,
    )
    return segment_source_rel + current_object_offset - np.asarray(
        target_translation,
        dtype=np.float32,
    )[None, :]


def _capture_object_target_rel(env: Robosuite3DEnv) -> np.ndarray:
    cubeA = np.asarray(env.env.sim.data.body_xpos[env.env.cubeA_body_id][:3], dtype=np.float32)
    cubeB = np.asarray(env.env.sim.data.body_xpos[env.env.cubeB_body_id][:3], dtype=np.float32)
    return cubeA - cubeB


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


def candidate_xyzs():
    return np.asarray(
        [[x, y, z] for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0) for z in (-1.0, 0.0, 1.0)],
        dtype=np.float32,
    )


def solve_motion2_actions(
    source_demo_path: Path,
    source_episode_idx: int,
    object_translation: np.ndarray,
    target_translation: np.ndarray,
    prefix_actions: np.ndarray,
    base_actions: np.ndarray,
    desired_obs_xyz: np.ndarray,
    desired_obj_target_rel_xyz: np.ndarray,
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
        object_translation[:3],
        dtype=np.float64,
    )
    reset_state["states"][TASK_OBJECT_STATE_INDICES["Stack"]["target"]] += np.asarray(
        target_translation[:3],
        dtype=np.float64,
    )
    obs = env.reset_to(reset_state)

    try:
        for action in np.asarray(prefix_actions, dtype=np.float32):
            obs, _, _, _ = env.step(action)

        cands = candidate_xyzs()
        solved_actions = []
        observed_obs_xyz = [np.asarray(obs["agent_pos"][:3], dtype=np.float32).copy()]
        observed_rel_xyz = [_capture_object_target_rel(env)]
        step_summaries = []

        for t in range(len(base_actions)):
            current_state = np.asarray(env.env.sim.get_state().flatten(), dtype=np.float64).copy()
            desired_next_xyz = np.asarray(desired_obs_xyz[t + 1], dtype=np.float32)
            desired_next_rel = np.asarray(desired_obj_target_rel_xyz[t + 1], dtype=np.float32)
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
                next_rel = _capture_object_target_rel(env)

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
                        "total_cost": total_cost,
                    }

            env.reset_to({"states": current_state})
            obs_next, _, _, _ = env.step(best["cand_action"])
            next_xyz = np.asarray(obs_next["agent_pos"][:3], dtype=np.float32)
            next_rel = _capture_object_target_rel(env)

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
                    "desired_next_object_target_rel_xyz": [float(v) for v in desired_next_rel],
                    "actual_next_object_target_rel_xyz": [float(v) for v in next_rel],
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


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    data_root = Path(args.data_root).resolve()
    source_demo_path = Path(args.source_demo).resolve()
    template_zarr = Path(args.template_zarr).resolve()
    output_zarr = Path(args.output_zarr).resolve()
    output_json = Path(args.output_json).resolve() if args.output_json else None

    cfg = load_cfg(config_path, data_root)
    cfg.source_demo_hdf5 = str(source_demo_path)
    generator = instantiate_generator(cfg)
    template_meta = load_template_meta(template_zarr)
    template_buffer = ReplayBuffer.copy_from_path(
        str(template_zarr),
        keys=["agent_pos", "action", "point_cloud"],
    )

    selected = parse_episode_list(args.episodes, template_buffer.n_episodes)
    generated_episodes = []
    summaries = []

    for gen_ep_idx in selected:
        source_episode_idx = int(template_meta["source_episode_idx"][gen_ep_idx])
        translation = np.asarray(template_meta["object_translation"][gen_ep_idx], dtype=np.float32)
        object_translation, target_translation = split_translation(translation)
        motion_frame_count = int(template_meta["motion_frame_count"][gen_ep_idx])

        source_demo = generator.replay_buffer.get_episode(source_episode_idx)
        inferred_skill1_frame, inferred_motion2_frame, inferred_skill2_frame = infer_twophase_frames(
            generator,
            source_demo,
            source_episode_idx,
        )
        motion2_frame = int(
            template_meta["motion_2_frame"][gen_ep_idx]
            if "motion_2_frame" in template_meta
            else inferred_motion2_frame
        )
        skill2_frame = int(
            template_meta["skill_2_frame"][gen_ep_idx]
            if "skill_2_frame" in template_meta
            else inferred_skill2_frame
        )
        if motion_frame_count != inferred_skill1_frame:
            raise ValueError(
                f"Template motion_frame_count={motion_frame_count} disagrees with source skill1_frame={inferred_skill1_frame}"
            )

        motion1_solve_steps = (
            int(args.motion1_solve_steps)
            if args.motion1_solve_steps is not None
            else motion_frame_count
        )
        _, desired_motion1_obs_xyz, _, _ = build_desired_motion1_states(
            generator=generator,
            episode_idx=source_episode_idx,
            object_translation=object_translation,
            solve_steps=motion1_solve_steps,
            skill1_frame=motion_frame_count,
        )
        source_prefix_actions = np.asarray(source_demo["action"][:motion1_solve_steps], dtype=np.float32)
        desired_motion1_rel_xyz = build_desired_relative_xyz(
            source_demo_path=source_demo_path,
            source_episode_idx=source_episode_idx,
            source_actions=source_prefix_actions,
            control_steps=args.control_steps,
        )
        template_episode = template_buffer.get_episode(gen_ep_idx, copy=True)
        template_actions = np.asarray(template_episode["action"], dtype=np.float32)
        base_motion1_actions = template_actions[:motion1_solve_steps].copy()
        solved_motion1_actions, observed_motion1_obs_xyz, observed_motion1_rel_xyz, _ = solve_motion1_actions(
            source_demo_path=source_demo_path,
            source_episode_idx=source_episode_idx,
            base_actions=base_motion1_actions,
            desired_obs_xyz=desired_motion1_obs_xyz,
            desired_rel_xyz=desired_motion1_rel_xyz,
            object_translation=object_translation,
            target_translation=target_translation,
            control_steps=args.control_steps,
            action_deviation_weight=args.action_deviation_weight,
            relative_tail_steps=args.motion1_relative_tail_steps,
            relative_cost_weight=args.motion1_relative_cost_weight,
        )

        full_actions = template_actions.copy()
        full_actions[:motion1_solve_steps] = solved_motion1_actions

        max_motion2_solve_steps = int(skill2_frame) - int(motion2_frame)
        motion2_solve_steps = (
            int(args.motion2_solve_steps)
            if args.motion2_solve_steps is not None
            else max_motion2_solve_steps
        )
        desired_motion2_obs_xyz, current_object_offset, _, _ = build_desired_motion2_states(
            generator=generator,
            source_episode_idx=source_episode_idx,
            object_translation=object_translation,
            target_translation=target_translation,
            motion2_frame=motion2_frame,
            solve_steps=motion2_solve_steps,
        )
        source_actions_full = np.asarray(source_demo["action"], dtype=np.float32)
        desired_motion2_rel_xyz = build_desired_object_target_relative_xyz(
            source_demo_path=source_demo_path,
            source_episode_idx=source_episode_idx,
            source_actions=source_actions_full,
            motion2_frame=motion2_frame,
            solve_steps=motion2_solve_steps,
            current_object_offset=current_object_offset,
            target_translation=target_translation,
            control_steps=args.control_steps,
        )
        prefix_actions_to_motion2 = full_actions[:motion2_frame].copy()
        base_motion2_actions = template_actions[motion2_frame : motion2_frame + motion2_solve_steps].copy()
        solved_motion2_actions, observed_motion2_obs_xyz, observed_motion2_rel_xyz, _ = solve_motion2_actions(
            source_demo_path=source_demo_path,
            source_episode_idx=source_episode_idx,
            object_translation=object_translation,
            target_translation=target_translation,
            prefix_actions=prefix_actions_to_motion2,
            base_actions=base_motion2_actions,
            desired_obs_xyz=desired_motion2_obs_xyz,
            desired_obj_target_rel_xyz=desired_motion2_rel_xyz,
            control_steps=args.control_steps,
            action_deviation_weight=args.action_deviation_weight,
            relative_tail_steps=args.motion2_relative_tail_steps,
            relative_cost_weight=args.motion2_relative_cost_weight,
        )
        full_actions[motion2_frame : motion2_frame + motion2_solve_steps] = solved_motion2_actions

        replay_states, replay_actions = replay_full_episode(
            source_demo_path=source_demo_path,
            source_episode_idx=source_episode_idx,
            full_actions=full_actions,
            object_translation=object_translation,
            target_translation=target_translation,
            control_steps=args.control_steps,
        )
        replay_pcds = build_replayed_point_clouds(
            generator=generator,
            source_demo=source_demo,
            episode_idx=source_episode_idx,
            replay_states=replay_states,
            object_translation=object_translation,
            target_translation=target_translation,
            skill1_frame=motion_frame_count,
            motion2_frame=motion2_frame,
            skill2_frame=skill2_frame,
        )

        generated_episodes.append(
            {
                "state": replay_states,
                "action": replay_actions,
                "point_cloud": replay_pcds,
                "_source_episode_idx": source_episode_idx,
                "_object_translation": np.asarray(translation, dtype=np.float32),
                "_motion_frame_count": int(motion_frame_count),
                "_motion_2_frame": int(motion2_frame),
                "_skill_2_frame": int(skill2_frame),
            }
        )

        motion1_summary = build_summary(
            observed_obs_xyz=observed_motion1_obs_xyz,
            desired_obs_xyz=desired_motion1_obs_xyz,
            observed_rel_xyz=observed_motion1_rel_xyz,
            desired_rel_xyz=desired_motion1_rel_xyz,
            base_actions=base_motion1_actions,
            solved_actions=solved_motion1_actions,
            solve_steps=motion1_solve_steps,
            relative_tail_steps=args.motion1_relative_tail_steps,
        )
        motion2_summary = build_summary(
            observed_obs_xyz=observed_motion2_obs_xyz,
            desired_obs_xyz=desired_motion2_obs_xyz,
            observed_rel_xyz=observed_motion2_rel_xyz,
            desired_rel_xyz=desired_motion2_rel_xyz,
            base_actions=base_motion2_actions,
            solved_actions=solved_motion2_actions,
            solve_steps=motion2_solve_steps,
            relative_tail_steps=args.motion2_relative_tail_steps,
        )
        summary = {
            "generated_episode_idx": gen_ep_idx,
            "source_episode_idx": source_episode_idx,
            "translation": [float(v) for v in translation],
            "motion_frame_count": motion_frame_count,
            "motion_2_frame": int(motion2_frame),
            "skill_2_frame": int(skill2_frame),
            "motion1": motion1_summary,
            "motion2": motion2_summary,
        }
        summaries.append(summary)
        print(
            json.dumps(
                {
                    "generated_episode_idx": gen_ep_idx,
                    "motion1_rel_final_err_norm": motion1_summary["rel_final_err_norm"],
                    "motion2_rel_final_err_norm": motion2_summary["rel_final_err_norm"],
                    "motion1_changed_steps": len(motion1_summary["changed_steps"]),
                    "motion2_changed_steps": len(motion2_summary["changed_steps"]),
                },
                ensure_ascii=False,
            )
        )

    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    generator.save_episodes(generated_episodes, str(output_zarr))

    motion1_rel = [s["motion1"]["rel_final_err_norm"] for s in summaries]
    motion2_rel = [s["motion2"]["rel_final_err_norm"] for s in summaries]
    result = {
        "config": str(config_path),
        "source_demo": str(source_demo_path),
        "template_zarr": str(template_zarr),
        "output_zarr": str(output_zarr),
        "aggregate": {
            "n_episodes": len(summaries),
            "selected_episodes": selected,
            "motion1_rel_final_err_norm_mean": float(np.mean(motion1_rel)) if motion1_rel else 0.0,
            "motion1_rel_final_err_norm_max": float(np.max(motion1_rel)) if motion1_rel else 0.0,
            "motion2_rel_final_err_norm_mean": float(np.mean(motion2_rel)) if motion2_rel else 0.0,
            "motion2_rel_final_err_norm_max": float(np.max(motion2_rel)) if motion2_rel else 0.0,
        },
        "episodes": summaries,
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
