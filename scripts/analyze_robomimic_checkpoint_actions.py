#!/usr/bin/env python3
"""
Analyze rollout action statistics for one or more robomimic checkpoints.

This is meant for comparing checkpoints on questions like:
- does a later model reduce large erratic actions?
- is the action sequence smoother or still jittery?
- did gripper behavior become more stable?

Example:
    conda run -n robomimic python scripts/analyze_robomimic_checkpoint_actions.py \
      --checkpoints outputs/robomimic/diffusion_policy_demogen/run/20260328145323/models/model_epoch_300.pth \
      --n-rollouts 10 \
      --horizon 400 \
      --output-json outputs/analysis/dp_action_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import RolloutPolicy
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths to compare")
    parser.add_argument("--n-rollouts", type=int, default=10, help="Number of rollouts per checkpoint")
    parser.add_argument("--horizon", type=int, default=400, help="Maximum rollout horizon")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for rollouts")
    parser.add_argument("--output-json", default=None, help="Optional output json path")
    return parser.parse_args()


def _to_action_vector(act: Any) -> np.ndarray:
    """
    Normalize a robomimic policy action into a flat [Da] numpy vector.

    In practice, rollouts may yield actions shaped like:
    - [Da]
    - [1, Da]
    - torch tensors or numpy arrays
    """
    act_np = np.asarray(act, dtype=np.float32)
    if act_np.ndim == 0:
        raise ValueError(f"Unexpected scalar action: {act_np!r}")
    if act_np.ndim == 1:
        return act_np
    if act_np.ndim == 2 and act_np.shape[0] == 1:
        return act_np[0]
    return act_np.reshape(-1)


def rollout_once(policy: RolloutPolicy, env: EnvBase | EnvWrapper, horizon: int) -> tuple[dict[str, float], np.ndarray]:
    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)

    actions = []
    total_reward = 0.0
    success = False

    for step_i in range(horizon):
        act = policy(ob=obs)
        act_np = _to_action_vector(act)
        actions.append(act_np)

        next_obs, reward, done, _ = env.step(act)
        total_reward += float(reward)
        success = bool(env.is_success()["task"])

        if done or success:
            break
        obs = next_obs

    stats = {
        "Return": float(total_reward),
        "Horizon": float(step_i + 1),
        "Success_Rate": float(success),
    }
    return stats, np.asarray(actions, dtype=np.float32)


def summarize_action_stats(action_sequences: list[np.ndarray], rollout_stats: list[dict[str, float]]) -> dict[str, Any]:
    if not action_sequences:
        return {}

    valid_action_sequences = [seq for seq in action_sequences if seq.size > 0]
    if not valid_action_sequences:
        return {
            "num_rollouts": len(action_sequences),
            "num_action_steps": 0,
            "success_rate_mean": float(np.mean([s["Success_Rate"] for s in rollout_stats])) if rollout_stats else 0.0,
            "return_mean": float(np.mean([s["Return"] for s in rollout_stats])) if rollout_stats else 0.0,
            "horizon_mean": float(np.mean([s["Horizon"] for s in rollout_stats])) if rollout_stats else 0.0,
            "empty_action_sequences": True,
        }

    action_dim = valid_action_sequences[0].shape[-1]
    normalized_sequences = []
    for seq in valid_action_sequences:
        if seq.ndim == 1:
            seq = seq.reshape(1, -1)
        elif seq.ndim != 2:
            seq = seq.reshape(-1, action_dim)
        normalized_sequences.append(seq)

    all_actions = np.concatenate(normalized_sequences, axis=0)
    diffs = [np.diff(seq, axis=0) for seq in normalized_sequences if len(seq) >= 2]
    diff2 = [np.diff(seq, n=2, axis=0) for seq in normalized_sequences if len(seq) >= 3]

    all_diffs = np.concatenate(diffs, axis=0) if diffs else np.zeros((0, all_actions.shape[1]), dtype=np.float32)
    all_diff2 = np.concatenate(diff2, axis=0) if diff2 else np.zeros((0, all_actions.shape[1]), dtype=np.float32)

    gripper = all_actions[:, -1]
    gripper_sign = np.sign(gripper)
    gripper_switches = float(np.sum(np.abs(np.diff(gripper_sign)) > 0)) if len(gripper_sign) >= 2 else 0.0

    stats = {
        "num_rollouts": len(action_sequences),
        "num_action_steps": int(all_actions.shape[0]),
        "success_rate_mean": float(np.mean([s["Success_Rate"] for s in rollout_stats])),
        "return_mean": float(np.mean([s["Return"] for s in rollout_stats])),
        "horizon_mean": float(np.mean([s["Horizon"] for s in rollout_stats])),
        "action_abs_mean": np.mean(np.abs(all_actions), axis=0).tolist(),
        "action_std": np.std(all_actions, axis=0).tolist(),
        "action_saturation_ratio": float(np.mean(np.abs(all_actions) > 0.95)),
        "action_norm_mean": float(np.mean(np.linalg.norm(all_actions, axis=1))),
        "action_delta_norm_mean": float(np.mean(np.linalg.norm(all_diffs, axis=1))) if len(all_diffs) else 0.0,
        "action_delta_norm_p95": float(np.percentile(np.linalg.norm(all_diffs, axis=1), 95)) if len(all_diffs) else 0.0,
        "action_jerk_norm_mean": float(np.mean(np.linalg.norm(all_diff2, axis=1))) if len(all_diff2) else 0.0,
        "xyz_abs_mean": float(np.mean(np.linalg.norm(all_actions[:, :3], axis=1))),
        "rot_abs_mean": float(np.mean(np.linalg.norm(all_actions[:, 3:6], axis=1))),
        "gripper_abs_mean": float(np.mean(np.abs(gripper))),
        "gripper_std": float(np.std(gripper)),
        "gripper_switches_total": gripper_switches,
        "gripper_switches_per_rollout": float(gripper_switches / max(len(action_sequences), 1)),
    }
    return stats


def analyze_checkpoint(ckpt_path: Path, n_rollouts: int, horizon: int, seed: int) -> dict[str, Any]:
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=str(ckpt_path), device=device, verbose=False)
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        render=False,
        render_offscreen=False,
        verbose=False,
    )

    np.random.seed(seed)
    torch.manual_seed(seed)

    rollout_stats = []
    action_sequences = []
    for _ in range(n_rollouts):
        stats, actions = rollout_once(policy=policy, env=env, horizon=horizon)
        rollout_stats.append(stats)
        action_sequences.append(actions)

    return {
        "checkpoint": str(ckpt_path),
        "summary": summarize_action_stats(action_sequences=action_sequences, rollout_stats=rollout_stats),
        "rollouts": rollout_stats,
    }


def main() -> None:
    args = parse_args()
    checkpoints = [Path(p).expanduser().resolve() for p in args.checkpoints]
    for ckpt in checkpoints:
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    reports = [
        analyze_checkpoint(
            ckpt_path=ckpt,
            n_rollouts=args.n_rollouts,
            horizon=args.horizon,
            seed=args.seed,
        )
        for ckpt in checkpoints
    ]

    text = json.dumps(reports, indent=2)
    print(text)

    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")


if __name__ == "__main__":
    main()
