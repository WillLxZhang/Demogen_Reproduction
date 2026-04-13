#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import hydra
import zarr


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMOGEN_ROOT = REPO_ROOT / "repos" / "DemoGen" / "demo_generation"
if str(DEMOGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMOGEN_ROOT))


def _load_cfg(config_path: Path, data_root: Path):
    cfg = OmegaConf.load(config_path)
    cfg.data_root = str(data_root)
    OmegaConf.resolve(cfg)
    return cfg


def _instantiate_generator(cfg):
    cls = hydra.utils.get_class(cfg._target_)
    return cls(cfg)


def _get_skill1_frame(generator, source_demo, episode_idx: int) -> int:
    if generator.use_manual_parsing_frames:
        return int(generator.parsing_frames["skill-1"])
    pcds = source_demo["point_cloud"]
    ee_poses = source_demo["state"][:, :3]
    return int(generator.parse_frames_one_stage(pcds, episode_idx, ee_poses))


def _episode_slice(episode_ends: np.ndarray, episode_idx: int) -> slice:
    start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end = int(episode_ends[episode_idx])
    return slice(start, end)


def _collect_calibration_arrays(generator, zarr_root, episode_ends: np.ndarray):
    action_chunks = []
    replay_chunks = []
    skill1_frames = []
    for episode_idx in range(generator.n_source_episodes):
        source_demo = generator.replay_buffer.get_episode(episode_idx)
        skill1_frame = _get_skill1_frame(generator, source_demo, episode_idx)
        ep_slice = _episode_slice(episode_ends, episode_idx)
        action_chunks.append(
            np.asarray(source_demo["action"][:skill1_frame, :3], dtype=np.float32)
        )
        replay_chunks.append(
            np.asarray(
                zarr_root["data"]["replay_h1_delta"][ep_slice][:skill1_frame, :3],
                dtype=np.float32,
            )
        )
        skill1_frames.append(int(skill1_frame))
    return (
        np.concatenate(action_chunks, axis=0),
        np.concatenate(replay_chunks, axis=0),
        skill1_frames,
    )


def _pulse_scale_stats(action_xyz: np.ndarray, replay_xyz: np.ndarray):
    axes = "xyz"
    stats = {}
    signed_scales = []
    for axis, name in enumerate(axes):
        mask = np.abs(action_xyz[:, axis]) > 0.5
        raw = replay_xyz[mask]
        if not np.any(mask):
            stats[name] = {
                "pulse_count": 0,
                "mean_signed_realized": 0.0,
                "median_signed_realized": 0.0,
                "p90_signed_realized": 0.0,
                "mean_signed_cross_axis": [0.0, 0.0, 0.0],
            }
            signed_scales.append(0.0)
            continue
        sign = np.sign(action_xyz[mask, axis : axis + 1])
        signed = raw * sign
        axis_signed = signed[:, axis]
        stats[name] = {
            "pulse_count": int(mask.sum()),
            "mean_signed_realized": float(np.mean(axis_signed)),
            "median_signed_realized": float(np.median(axis_signed)),
            "p90_signed_realized": float(np.percentile(axis_signed, 90)),
            "mean_signed_cross_axis": [float(v) for v in np.mean(signed, axis=0)],
        }
        signed_scales.append(float(np.median(axis_signed)))
    return stats, np.asarray(signed_scales, dtype=np.float32)


def _simulate_phase_copy(generator, source_demo, skill1_frame: int, object_translation: np.ndarray):
    action_xyz = np.asarray(source_demo["action"][:skill1_frame, :3], dtype=np.float32)
    motion_xyz = np.asarray(
        source_demo[generator.motion_action_key][:skill1_frame, :3], dtype=np.float32
    )
    increments = generator._build_translation_increments(
        source_demo=source_demo,
        skill_1_frame=skill1_frame,
        object_translation=object_translation,
    )

    pulse_scale = np.asarray(generator.motion_exec_pulse_scale_xyz, dtype=np.float32)
    correction_scale = np.asarray(generator.translation_correction_scale_xyz, dtype=np.float32)
    state = generator._init_motion_exec_state()

    correction_pulses = []
    delivered_extra_pulses = []
    clip_events = []
    same_sign_conflicts = []
    opposite_sign_conflicts = []
    desired_scaled_extra = []

    for frame_idx in range(skill1_frame):
        source_exec_xyz = action_xyz[frame_idx]
        scaled_extra = increments[frame_idx] * correction_scale
        correction_xyz = generator._encode_motion_exec_xyz(
            scaled_extra,
            motion_exec_state=state,
        )
        final_exec_xyz = np.clip(source_exec_xyz + correction_xyz, -1.0, 1.0)
        delivered_extra = final_exec_xyz - source_exec_xyz

        correction_pulses.append(correction_xyz)
        delivered_extra_pulses.append(delivered_extra)
        desired_scaled_extra.append(scaled_extra)

        clip_mask = np.abs(source_exec_xyz + correction_xyz) > 1.0 + 1e-6
        same_sign_mask = (
            (np.abs(source_exec_xyz) > 1e-6)
            & (np.abs(correction_xyz) > 1e-6)
            & (np.sign(source_exec_xyz) == np.sign(correction_xyz))
        )
        opposite_sign_mask = (
            (np.abs(source_exec_xyz) > 1e-6)
            & (np.abs(correction_xyz) > 1e-6)
            & (np.sign(source_exec_xyz) != np.sign(correction_xyz))
        )
        clip_events.append(clip_mask.astype(np.int32))
        same_sign_conflicts.append(same_sign_mask.astype(np.int32))
        opposite_sign_conflicts.append(opposite_sign_mask.astype(np.int32))

    correction_pulses = np.asarray(correction_pulses, dtype=np.float32)
    delivered_extra_pulses = np.asarray(delivered_extra_pulses, dtype=np.float32)
    desired_scaled_extra = np.asarray(desired_scaled_extra, dtype=np.float32)
    clip_events = np.asarray(clip_events, dtype=np.int32)
    same_sign_conflicts = np.asarray(same_sign_conflicts, dtype=np.int32)
    opposite_sign_conflicts = np.asarray(opposite_sign_conflicts, dtype=np.int32)
    delivered_extra_sum_cfg_scale = (delivered_extra_pulses * pulse_scale).sum(axis=0)
    requested_translation_sum = increments.sum(axis=0)
    state_update_semantics = getattr(generator, "phase_copy_state_update_semantics", "requested_extra")
    if state_update_semantics == "delivered_exec_extra":
        stored_state_extra_sum_cfg_scale = delivered_extra_sum_cfg_scale
    else:
        stored_state_extra_sum_cfg_scale = requested_translation_sum

    return {
        "skill1_frame": int(skill1_frame),
        "object_translation": [float(v) for v in object_translation],
        "requested_translation_sum": [float(v) for v in requested_translation_sum],
        "desired_scaled_extra_sum": [float(v) for v in desired_scaled_extra.sum(axis=0)],
        "requested_translation_per_axis_abs_sum": [
            float(v) for v in np.abs(increments).sum(axis=0)
        ],
        "state_update_semantics": str(state_update_semantics),
        "source_motion_sum": [float(v) for v in motion_xyz.sum(axis=0)],
        "source_exec_pulse_count": [int(v) for v in (np.abs(action_xyz) > 0.5).sum(axis=0)],
        "correction_pulse_sum": [float(v) for v in correction_pulses.sum(axis=0)],
        "correction_pulse_count": [int(v) for v in (np.abs(correction_pulses) > 0.5).sum(axis=0)],
        "delivered_extra_pulse_sum": [float(v) for v in delivered_extra_pulses.sum(axis=0)],
        "delivered_extra_pulse_count": [
            int(v) for v in (np.abs(delivered_extra_pulses) > 0.5).sum(axis=0)
        ],
        "clip_event_count": [int(v) for v in clip_events.sum(axis=0)],
        "same_sign_conflict_count": [int(v) for v in same_sign_conflicts.sum(axis=0)],
        "opposite_sign_conflict_count": [int(v) for v in opposite_sign_conflicts.sum(axis=0)],
        "encoder_implied_extra_sum_cfg_scale": [
            float(v) for v in (correction_pulses * pulse_scale).sum(axis=0)
        ],
        "delivered_extra_sum_cfg_scale": [
            float(v) for v in delivered_extra_sum_cfg_scale
        ],
        "stored_state_extra_sum_cfg_scale": [
            float(v) for v in stored_state_extra_sum_cfg_scale
        ],
        "state_action_extra_gap_cfg_scale": [
            float(v)
            for v in (
                stored_state_extra_sum_cfg_scale - delivered_extra_sum_cfg_scale
            )
        ],
        "lost_extra_sum_cfg_scale_due_to_clip": [
            float(v) for v in ((correction_pulses - delivered_extra_pulses) * pulse_scale).sum(axis=0)
        ],
        "residual_xyz_after": [float(v) for v in state["residual_xyz"]],
    }


def main():
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
            / "lift_0_v31_replayh1_light_schedule_phasecopy_statedelta_halfcorr.yaml"
        ),
    )
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "repos" / "DemoGen" / "data"),
    )
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--translation", type=float, nargs=3, default=[0.035, 0.035, 0.0])
    parser.add_argument(
        "--output",
        default=str(
            REPO_ROOT / "outputs" / "analysis" / "motion_exec_pulse_semantics_v31_ep0.json"
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    data_root = Path(args.data_root).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = _load_cfg(config_path, data_root)
    generator = _instantiate_generator(cfg)
    source_zarr = data_root / "datasets" / "source" / f"{cfg.source_name}.zarr"
    zarr_root = zarr.open(str(source_zarr), mode="r")
    if "replay_h1_delta" not in zarr_root["data"]:
        raise ValueError("Source zarr must contain replay_h1_delta for pulse analysis.")

    episode_ends = np.asarray(zarr_root["meta"]["episode_ends"], dtype=np.int64)
    calibration_action_xyz, calibration_replay_h1_xyz, calibration_skill1_frames = (
        _collect_calibration_arrays(generator, zarr_root, episode_ends)
    )
    pulse_stats, empirical_median_scale = _pulse_scale_stats(
        calibration_action_xyz,
        calibration_replay_h1_xyz,
    )

    source_demo = generator.replay_buffer.get_episode(args.episode_idx)
    skill1_frame = _get_skill1_frame(generator, source_demo, args.episode_idx)
    phase_copy_stats = _simulate_phase_copy(
        generator=generator,
        source_demo=source_demo,
        skill1_frame=skill1_frame,
        object_translation=np.asarray(args.translation, dtype=np.float32),
    )
    phase_copy_stats["approx_delivered_extra_sum_empirical_median_scale"] = [
        float(v)
        for v in (
            np.asarray(phase_copy_stats["delivered_extra_pulse_sum"], dtype=np.float32)
            * empirical_median_scale
        )
    ]
    phase_copy_stats["approx_lost_extra_sum_empirical_median_scale_due_to_clip"] = [
        float(v)
        for v in (
            (
                np.asarray(phase_copy_stats["correction_pulse_sum"], dtype=np.float32)
                - np.asarray(phase_copy_stats["delivered_extra_pulse_sum"], dtype=np.float32)
            )
            * empirical_median_scale
        )
    ]

    report = {
        "config": str(config_path),
        "data_root": str(data_root),
        "episode_idx": int(args.episode_idx),
        "skill1_frame": int(skill1_frame),
        "calibration_episode_count": int(generator.n_source_episodes),
        "calibration_skill1_frames": calibration_skill1_frames,
        "config_motion_exec_pulse_scale_xyz": [
            float(v) for v in np.asarray(generator.motion_exec_pulse_scale_xyz, dtype=np.float32)
        ],
        "empirical_pulse_scale_median_xyz": [float(v) for v in empirical_median_scale],
        "empirical_pulse_scale_mean_xyz": [
            float(pulse_stats[axis]["mean_signed_realized"]) for axis in "xyz"
        ],
        "empirical_over_config_ratio_median_xyz": [
            float(empirical_median_scale[i] / generator.motion_exec_pulse_scale_xyz[i])
            if abs(generator.motion_exec_pulse_scale_xyz[i]) > 1e-12
            else 0.0
            for i in range(3)
        ],
        "pulse_stats": pulse_stats,
        "phase_copy_simulation": phase_copy_stats,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()
