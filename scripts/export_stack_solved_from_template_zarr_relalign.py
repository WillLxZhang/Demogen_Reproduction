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

from diffusion_policies.common.replay_buffer import ReplayBuffer
import diffusion_policies.env.robosuite.robosuite_wrapper as robosuite_wrapper
from diffusion_policies.env.robosuite.robosuite_wrapper import Robosuite3DEnv

from replay_zarr_episode import load_reset_state
from solve_stack_motion1_relalign_actions import (
    TASK_OBJECT_STATE_INDICES,
    build_desired_motion1_states,
    build_desired_relative_xyz,
    build_summary,
    infer_twophase_frames,
    instantiate_generator,
    load_cfg,
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
    parser.add_argument("--solve-steps", type=int, default=None)
    parser.add_argument("--control-steps", type=int, default=1)
    parser.add_argument("--action-deviation-weight", type=float, default=1e-4)
    parser.add_argument("--relative-tail-steps", type=int, default=40)
    parser.add_argument("--relative-cost-weight", type=float, default=4.0)
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


def load_template_meta(template_zarr: Path) -> dict[str, np.ndarray]:
    root = zarr.open(str(template_zarr), mode="r")
    meta = root["meta"]
    required = ["source_episode_idx", "object_translation", "motion_frame_count"]
    missing = [k for k in required if k not in meta]
    if missing:
        raise KeyError(f"Template zarr missing required meta keys: {missing}")

    result = {
        "source_episode_idx": np.asarray(meta["source_episode_idx"][:], dtype=np.int64),
        "object_translation": np.asarray(meta["object_translation"][:], dtype=np.float32),
        "motion_frame_count": np.asarray(meta["motion_frame_count"][:], dtype=np.int64),
    }
    if "motion_2_frame" in meta:
        result["motion_2_frame"] = np.asarray(meta["motion_2_frame"][:], dtype=np.int64)
    if "skill_2_frame" in meta:
        result["skill_2_frame"] = np.asarray(meta["skill_2_frame"][:], dtype=np.int64)
    return result


def replay_full_episode(
    source_demo_path: Path,
    source_episode_idx: int,
    full_actions: np.ndarray,
    object_translation: np.ndarray,
    target_translation: np.ndarray,
    control_steps: int,
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

    traj_states = []
    traj_actions = []

    try:
        for action in np.asarray(full_actions, dtype=np.float32):
            traj_states.append(np.asarray(obs["agent_pos"], dtype=np.float32).copy())
            traj_actions.append(np.asarray(action, dtype=np.float32).copy())
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
    target_translation: np.ndarray,
    skill1_frame: int,
    motion2_frame: int,
    skill2_frame: int,
):
    source_states = np.asarray(source_demo["state"], dtype=np.float32)
    source_pcds = np.asarray(source_demo["point_cloud"], dtype=np.float32)

    first_obj_pcd = generator.get_objects_pcd_from_sam_mask(source_pcds[0], episode_idx, "object")
    first_tar_pcd = generator.get_objects_pcd_from_sam_mask(source_pcds[0], episode_idx, "target")
    obj_bbox = generator.pcd_bbox(first_obj_pcd)
    tar_bbox = generator.pcd_bbox(first_tar_pcd)

    replay_pcds = []
    for frame_idx in range(len(replay_states)):
        if frame_idx < skill1_frame:
            stage = "motion1"
        elif frame_idx < motion2_frame:
            stage = "skill1"
        elif frame_idx < skill2_frame:
            stage = "motion2"
        else:
            stage = "skill2"

        source_pcd = source_pcds[frame_idx].copy()
        robot_trans_vec = np.asarray(replay_states[frame_idx, :3], dtype=np.float32) - np.asarray(
            source_states[frame_idx, :3], dtype=np.float32
        )
        replay_pcd = generator._render_stage_point_cloud(
            source_pcd=source_pcd,
            obj_bbox=obj_bbox,
            tar_bbox=tar_bbox,
            obj_trans_vec=np.asarray(object_translation, dtype=np.float32),
            tar_trans_vec=np.asarray(target_translation, dtype=np.float32),
            robot_trans_vec=np.asarray(robot_trans_vec, dtype=np.float32),
            stage=stage,
        )
        replay_pcds.append(np.asarray(replay_pcd, dtype=np.float32))

    return replay_pcds


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
        solve_steps = int(args.solve_steps) if args.solve_steps is not None else motion_frame_count
        if solve_steps > motion_frame_count:
            raise ValueError(
                f"solve_steps={solve_steps} exceeds template motion_frame_count={motion_frame_count} "
                f"for generated episode {gen_ep_idx}"
            )

        source_demo = generator.replay_buffer.get_episode(source_episode_idx)
        inferred_skill1_frame, inferred_motion2_frame, inferred_skill2_frame = infer_twophase_frames(
            generator,
            source_demo,
            source_episode_idx,
        )
        skill1_frame = motion_frame_count
        motion2_frame = int(
            template_meta.get("motion_2_frame", np.asarray([inferred_motion2_frame]))[gen_ep_idx]
            if "motion_2_frame" in template_meta
            else inferred_motion2_frame
        )
        skill2_frame = int(
            template_meta.get("skill_2_frame", np.asarray([inferred_skill2_frame]))[gen_ep_idx]
            if "skill_2_frame" in template_meta
            else inferred_skill2_frame
        )
        if skill1_frame != inferred_skill1_frame:
            raise ValueError(
                f"Template motion_frame_count={skill1_frame} disagrees with source skill1_frame={inferred_skill1_frame} "
                f"for generated episode {gen_ep_idx}"
            )

        _, desired_obs_xyz, _, _ = build_desired_motion1_states(
            generator=generator,
            episode_idx=source_episode_idx,
            object_translation=object_translation,
            solve_steps=solve_steps,
            skill1_frame=skill1_frame,
        )

        source_prefix_actions = np.asarray(source_demo["action"][:solve_steps], dtype=np.float32)
        desired_rel_xyz = build_desired_relative_xyz(
            source_demo_path=source_demo_path,
            source_episode_idx=source_episode_idx,
            source_actions=source_prefix_actions,
            control_steps=args.control_steps,
        )

        template_episode = template_buffer.get_episode(gen_ep_idx, copy=True)
        template_actions = np.asarray(template_episode["action"], dtype=np.float32)
        base_actions = template_actions[:solve_steps].copy()

        solved_actions, observed_obs_xyz, observed_rel_xyz, _ = solve_motion1_actions(
            source_demo_path=source_demo_path,
            source_episode_idx=source_episode_idx,
            base_actions=base_actions,
            desired_obs_xyz=desired_obs_xyz,
            desired_rel_xyz=desired_rel_xyz,
            object_translation=object_translation,
            target_translation=target_translation,
            control_steps=args.control_steps,
            action_deviation_weight=args.action_deviation_weight,
            relative_tail_steps=args.relative_tail_steps,
            relative_cost_weight=args.relative_cost_weight,
        )

        full_actions = template_actions.copy()
        full_actions[:solve_steps] = solved_actions

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
            skill1_frame=skill1_frame,
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
                "_motion_frame_count": int(skill1_frame),
                "_motion_2_frame": int(motion2_frame),
                "_skill_2_frame": int(skill2_frame),
            }
        )

        summary = build_summary(
            observed_obs_xyz=observed_obs_xyz,
            desired_obs_xyz=desired_obs_xyz,
            observed_rel_xyz=observed_rel_xyz,
            desired_rel_xyz=desired_rel_xyz,
            base_actions=base_actions,
            solved_actions=solved_actions,
            solve_steps=solve_steps,
            relative_tail_steps=args.relative_tail_steps,
        )
        summary.update(
            {
                "generated_episode_idx": gen_ep_idx,
                "source_episode_idx": source_episode_idx,
                "translation": [float(v) for v in translation],
                "motion_frame_count": motion_frame_count,
                "motion_2_frame": int(motion2_frame),
                "skill_2_frame": int(skill2_frame),
                "relative_tail_steps": int(args.relative_tail_steps),
                "relative_cost_weight": float(args.relative_cost_weight),
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
                    "rel_final_err_norm": summary["rel_final_err_norm"],
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
        "rmse_norm_mean": float(np.mean([s["rmse_norm"] for s in summaries])) if summaries else 0.0,
        "rmse_norm_max": float(np.max([s["rmse_norm"] for s in summaries])) if summaries else 0.0,
        "final_err_norm_mean": float(np.mean([s["final_err_norm"] for s in summaries])) if summaries else 0.0,
        "final_err_norm_max": float(np.max([s["final_err_norm"] for s in summaries])) if summaries else 0.0,
        "rel_final_err_norm_mean": float(np.mean([s["rel_final_err_norm"] for s in summaries]))
        if summaries
        else 0.0,
        "rel_final_err_norm_max": float(np.max([s["rel_final_err_norm"] for s in summaries]))
        if summaries
        else 0.0,
        "rel_tail_rmse_norm_mean": float(np.mean([s["rel_tail_rmse_norm"] for s in summaries]))
        if summaries
        else 0.0,
        "rel_tail_rmse_norm_max": float(np.max([s["rel_tail_rmse_norm"] for s in summaries]))
        if summaries
        else 0.0,
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
