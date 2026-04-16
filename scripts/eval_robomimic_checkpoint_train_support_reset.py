#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import imageio
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "repos" / "robomimic") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "repos" / "robomimic"))

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import RolloutPolicy
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper


TASK_STATE_SPECS = {
    "Stack": {
        "object": np.array([10, 11, 12], dtype=np.int64),
        "target": np.array([17, 18, 19], dtype=np.int64),
    },
    "NutAssemblyRound": {
        "object": np.array([17, 18, 19], dtype=np.int64),
        "target": None,
    },
    "NutAssemblySquare": {
        "object": np.array([10, 11, 12], dtype=np.int64),
        "target": None,
    },
}


@dataclass(frozen=True)
class ResetCandidate:
    demo_key: str
    source_episode: int
    object_translation: tuple[float, float, float]
    target_translation: tuple[float, float, float]


def load_reset_state(source_demo_path: Path, source_episode_idx: int) -> dict:
    with h5py.File(source_demo_path, "r") as f:
        demos = list(f["data"].keys())
        if source_episode_idx < 0 or source_episode_idx >= len(demos):
            raise IndexError(
                f"source episode index {source_episode_idx} out of range for demos={demos}"
            )
        ep = demos[source_episode_idx]
        group = f[f"data/{ep}"]
        state = {
            "states": group["states"][0],
            "model": group.attrs["model_file"],
        }
        if "ep_meta" in group.attrs:
            state["ep_meta"] = group.attrs["ep_meta"]
        return state


def get_task_spec(env_name: str) -> dict[str, np.ndarray | None]:
    if env_name not in TASK_STATE_SPECS:
        raise KeyError(
            f"No reset-state translation spec registered for env={env_name}. "
            "Add it to TASK_STATE_SPECS in eval_robomimic_checkpoint_train_support_reset.py."
        )
    return TASK_STATE_SPECS[env_name]


def apply_translation_to_reset_state(
    reset_state: dict,
    env_name: str,
    object_translation: np.ndarray | None,
    target_translation: np.ndarray | None,
) -> dict:
    spec = get_task_spec(env_name)
    translated = dict(reset_state)
    translated["states"] = np.asarray(reset_state["states"], dtype=np.float64).copy()

    object_indices = spec["object"]
    if object_translation is not None and object_indices is not None:
        idx = np.asarray(object_indices, dtype=np.int64)
        translated["states"][idx] += np.asarray(object_translation[: len(idx)], dtype=np.float64)

    target_indices = spec["target"]
    if target_translation is not None and target_indices is not None:
        idx = np.asarray(target_indices, dtype=np.int64)
        translated["states"][idx] += np.asarray(target_translation[: len(idx)], dtype=np.float64)

    return translated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a robomimic checkpoint from resets sampled from the training "
            "dataset support, keeping source orientation fixed and sampling only "
            "the recorded training translations."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-demo", required=True)
    parser.add_argument(
        "--train-dataset-hdf5",
        required=True,
        help="Exported robomimic training HDF5 with source_episode_idx and object_translation attrs.",
    )
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=800)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--video-skip", type=int, default=5)
    parser.add_argument("--camera-names", nargs="+", default=["agentview"])
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--dataset-obs", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--sample-mode",
        choices=["support", "uniform_range"],
        default="support",
        help=(
            "support: sample from recorded training support points; "
            "uniform_range: sample uniformly inside the training translation bounds."
        ),
    )
    parser.add_argument(
        "--force-source-episode",
        type=int,
        default=None,
        help=(
            "Optional fixed source episode for all rollouts. Useful when you want "
            "to keep orientation / scene identity identical across interpolation tests."
        ),
    )
    parser.add_argument(
        "--exclude-zero-translation",
        action="store_true",
        help="Skip the zero-translation support point when sampling candidates.",
    )
    parser.add_argument(
        "--sample-with-replacement",
        action="store_true",
        help="Sample reset candidates with replacement. Default uses unique candidates when possible.",
    )
    parser.add_argument(
        "--no-dedupe-support",
        action="store_true",
        help="Do not deduplicate identical (source_episode, object_translation, target_translation) support points.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def parse_translation_attr(raw_value) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(raw_value, (bytes, str)):
        raw_value = ast.literal_eval(raw_value.decode("utf-8") if isinstance(raw_value, bytes) else raw_value)
    arr = np.asarray(raw_value, dtype=np.float32).reshape(-1)
    if arr.size == 6:
        return arr[:3].copy(), arr[3:6].copy()
    if arr.size == 3:
        return arr[:3].copy(), np.zeros(3, dtype=np.float32)
    raise ValueError(f"Unsupported object_translation attr shape {arr.shape} values={arr}")


def load_training_reset_candidates(
    train_dataset_hdf5: Path,
    *,
    dedupe_support: bool,
    exclude_zero_translation: bool,
) -> list[ResetCandidate]:
    candidates: list[ResetCandidate] = []
    seen: set[tuple[float, ...]] = set()
    with h5py.File(train_dataset_hdf5, "r") as f:
        for demo_key in sorted(f["data"].keys()):
            group = f[f"data/{demo_key}"]
            if "source_episode_idx" not in group.attrs or "object_translation" not in group.attrs:
                continue
            source_episode = int(np.asarray(group.attrs["source_episode_idx"]).reshape(()))
            object_translation, target_translation = parse_translation_attr(group.attrs["object_translation"])
            if exclude_zero_translation and np.allclose(object_translation, 0.0) and np.allclose(target_translation, 0.0):
                continue
            support_key = (
                float(source_episode),
                *np.round(object_translation, 6).tolist(),
                *np.round(target_translation, 6).tolist(),
            )
            if dedupe_support and support_key in seen:
                continue
            seen.add(support_key)
            candidates.append(
                ResetCandidate(
                    demo_key=demo_key,
                    source_episode=source_episode,
                    object_translation=tuple(float(x) for x in object_translation.tolist()),
                    target_translation=tuple(float(x) for x in target_translation.tolist()),
                )
            )
    if not candidates:
        raise ValueError(f"No reset candidates found in {train_dataset_hdf5}")
    return candidates


def extract_eef_pos(obs: dict) -> np.ndarray | None:
    value = obs.get("robot0_eef_pos", None)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        return arr.copy()
    return arr.reshape(-1, arr.shape[-1])[-1].copy()


def compute_translation_bounds(
    candidates: list[ResetCandidate],
) -> dict[str, np.ndarray]:
    object_translations = np.asarray([c.object_translation for c in candidates], dtype=np.float32)
    target_translations = np.asarray([c.target_translation for c in candidates], dtype=np.float32)
    return {
        "object_min": object_translations.min(axis=0),
        "object_max": object_translations.max(axis=0),
        "target_min": target_translations.min(axis=0),
        "target_max": target_translations.max(axis=0),
    }


def rollout_from_custom_reset(
    *,
    policy: RolloutPolicy,
    env: EnvBase | EnvWrapper,
    reset_state: dict,
    horizon: int,
    render: bool = False,
    video_writer=None,
    video_skip: int = 5,
    return_obs: bool = False,
    camera_names: list[str] | None = None,
) -> tuple[dict, dict, dict]:
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    camera_names = camera_names or ["agentview"]
    policy.start_episode()
    env.reset()
    obs = env.reset_to(reset_state)
    state_dict = env.get_state()

    success = False
    video_count = 0
    total_reward = 0.0
    steps_taken = 0
    traj = dict(
        actions=[],
        rewards=[],
        dones=[],
        states=[],
        initial_state_dict=deepcopy(reset_state),
    )
    if return_obs:
        traj.update(dict(obs=[], next_obs=[]))

    actions_hist: list[np.ndarray] = []
    eef_hist: list[np.ndarray] = []
    eef_pos = extract_eef_pos(obs)
    if eef_pos is not None:
        eef_hist.append(eef_pos)

    try:
        for step_i in range(int(horizon)):
            action = policy(ob=obs)
            next_obs, reward, done, _ = env.step(action)

            total_reward += float(reward)
            success = bool(env.is_success()["task"])
            steps_taken = step_i + 1

            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % max(1, int(video_skip)) == 0:
                    frames = []
                    for camera_name in camera_names:
                        frames.append(
                            env.render(
                                mode="rgb_array",
                                height=512,
                                width=512,
                                camera_name=camera_name,
                            )
                        )
                    video_writer.append_data(np.concatenate(frames, axis=1))
                video_count += 1

            traj["actions"].append(action)
            traj["rewards"].append(reward)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                traj["obs"].append(obs)
                traj["next_obs"].append(next_obs)

            action_np = np.asarray(action, dtype=np.float32)
            actions_hist.append(action_np.copy())
            eef_pos = extract_eef_pos(next_obs)
            if eef_pos is not None:
                eef_hist.append(eef_pos)

            if done or success:
                break

            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as exc:
        print(f"WARNING: got rollout exception {exc}")

    stats = dict(Return=total_reward, Horizon=steps_taken, Success_Rate=float(success))

    if return_obs:
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    for key in traj:
        if key == "initial_state_dict":
            continue
        if isinstance(traj[key], dict):
            for subkey in traj[key]:
                traj[key][subkey] = np.asarray(traj[key][subkey])
        else:
            traj[key] = np.asarray(traj[key])

    actions_np = np.asarray(actions_hist, dtype=np.float32)
    eef_np = np.asarray(eef_hist, dtype=np.float32) if eef_hist else None
    diagnostics = {
        "steps": int(actions_np.shape[0]),
        "success": bool(success),
        "return": float(total_reward),
        "action_xyz_meanabs": np.mean(np.abs(actions_np[:, :3]), axis=0).round(6).tolist()
        if actions_np.size
        else None,
        "action_rot_meanabs": np.mean(np.abs(actions_np[:, 3:6]), axis=0).round(6).tolist()
        if actions_np.size
        else None,
        "action_grip_meanabs": np.mean(np.abs(actions_np[:, 6:]), axis=0).round(6).tolist()
        if actions_np.size
        else None,
        "eef_span": (eef_np.max(axis=0) - eef_np.min(axis=0)).round(6).tolist() if eef_np is not None else None,
        "eef_start": eef_np[0].round(6).tolist() if eef_np is not None else None,
        "eef_end": eef_np[-1].round(6).tolist() if eef_np is not None else None,
        "first10_xyz": actions_np[:10, :3].round(6).tolist() if actions_np.size else [],
    }
    return stats, traj, diagnostics


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    source_demo = Path(args.source_demo).expanduser().resolve()
    train_dataset_hdf5 = Path(args.train_dataset_hdf5).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not source_demo.exists():
        raise FileNotFoundError(f"source demo not found: {source_demo}")
    if not train_dataset_hdf5.exists():
        raise FileNotFoundError(f"train dataset hdf5 not found: {train_dataset_hdf5}")

    write_video = args.video_path is not None
    write_dataset = args.dataset_path is not None
    if args.render and write_video:
        raise ValueError("render and video recording cannot both be enabled")
    if args.render and len(args.camera_names) != 1:
        raise ValueError("on-screen rendering only supports exactly one camera")

    candidates = load_training_reset_candidates(
        train_dataset_hdf5=train_dataset_hdf5,
        dedupe_support=not bool(args.no_dedupe_support),
        exclude_zero_translation=bool(args.exclude_zero_translation),
    )
    translation_bounds = compute_translation_bounds(candidates)
    unique_source_episodes = sorted({int(c.source_episode) for c in candidates})
    fixed_source_episode = (
        int(args.force_source_episode)
        if args.force_source_episode is not None
        else int(unique_source_episodes[0])
    )

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=str(checkpoint),
        device=device,
        verbose=False,
    )
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=None,
        render=bool(args.render),
        render_offscreen=write_video,
        verbose=False,
    )
    env_name = getattr(env, "name", None) or ckpt_dict.get("env_metadata", {}).get("env_name")
    get_task_spec(env_name)

    sample_rng = np.random.default_rng(int(args.seed))
    replace = False
    sampled_indices = None
    if args.sample_mode == "support":
        replace = bool(args.sample_with_replacement) or int(args.n_rollouts) > len(candidates)
        sampled_indices = sample_rng.choice(len(candidates), size=int(args.n_rollouts), replace=replace)

    video_writer = None
    if write_video:
        video_path = Path(args.video_path).expanduser().resolve()
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(str(video_path), fps=20)

    data_writer = None
    data_grp = None
    total_samples = 0
    if write_dataset:
        dataset_path = Path(args.dataset_path).expanduser().resolve()
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        data_writer = h5py.File(dataset_path, "w")
        data_grp = data_writer.create_group("data")

    rollout_stats = []
    diagnostics_per_rollout = []
    sampled_candidates = []
    try:
        for rollout_idx in range(int(args.n_rollouts)):
            rollout_seed = int(args.seed) + rollout_idx
            np.random.seed(rollout_seed)
            torch.manual_seed(rollout_seed)

            if args.sample_mode == "support":
                assert sampled_indices is not None
                candidate_idx = int(sampled_indices.tolist()[rollout_idx])
                candidate = candidates[candidate_idx]
                source_episode = (
                    fixed_source_episode
                    if args.force_source_episode is not None
                    else int(candidate.source_episode)
                )
                object_translation = np.asarray(candidate.object_translation, dtype=np.float32)
                target_translation = np.asarray(candidate.target_translation, dtype=np.float32)
                sampled_entry = asdict(candidate)
                sampled_entry["sampling_mode"] = "support"
                sampled_entry["candidate_index"] = candidate_idx
            else:
                candidate_idx = None
                source_episode = fixed_source_episode
                object_translation = sample_rng.uniform(
                    low=translation_bounds["object_min"],
                    high=translation_bounds["object_max"],
                ).astype(np.float32)
                target_translation = sample_rng.uniform(
                    low=translation_bounds["target_min"],
                    high=translation_bounds["target_max"],
                ).astype(np.float32)
                sampled_entry = {
                    "demo_key": None,
                    "source_episode": int(source_episode),
                    "object_translation": object_translation.round(6).tolist(),
                    "target_translation": target_translation.round(6).tolist(),
                    "sampling_mode": "uniform_range",
                    "candidate_index": None,
                }
            sampled_candidates.append(sampled_entry)

            reset_state = load_reset_state(source_demo, int(source_episode))
            reset_state = apply_translation_to_reset_state(
                reset_state=reset_state,
                env_name=env_name,
                object_translation=object_translation,
                target_translation=target_translation,
            )

            stats, traj, diagnostics = rollout_from_custom_reset(
                policy=policy,
                env=env,
                reset_state=reset_state,
                horizon=int(args.horizon),
                render=bool(args.render),
                video_writer=video_writer,
                video_skip=int(args.video_skip),
                return_obs=bool(write_dataset and args.dataset_obs),
                camera_names=list(args.camera_names),
            )
            diagnostics["rollout_index"] = rollout_idx
            diagnostics["seed"] = rollout_seed
            diagnostics["candidate_index"] = candidate_idx
            diagnostics["sampled_demo_key"] = sampled_entry["demo_key"]
            diagnostics["source_episode"] = int(source_episode)
            diagnostics["object_translation"] = object_translation.round(6).tolist()
            diagnostics["target_translation"] = target_translation.round(6).tolist()
            diagnostics["sampling_mode"] = str(args.sample_mode)
            rollout_stats.append(stats)
            diagnostics_per_rollout.append(diagnostics)

            if data_grp is not None:
                ep_data_grp = data_grp.create_group(f"demo_{rollout_idx}")
                ep_data_grp.create_dataset("actions", data=np.asarray(traj["actions"]))
                ep_data_grp.create_dataset("states", data=np.asarray(traj["states"]))
                ep_data_grp.create_dataset("rewards", data=np.asarray(traj["rewards"]))
                ep_data_grp.create_dataset("dones", data=np.asarray(traj["dones"]))
                if args.dataset_obs:
                    for key in traj["obs"]:
                        ep_data_grp.create_dataset(f"obs/{key}", data=np.asarray(traj["obs"][key]))
                        ep_data_grp.create_dataset(f"next_obs/{key}", data=np.asarray(traj["next_obs"][key]))
                if "model" in traj["initial_state_dict"]:
                    ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]
                if "ep_meta" in traj["initial_state_dict"]:
                    ep_data_grp.attrs["ep_meta"] = traj["initial_state_dict"]["ep_meta"]
                ep_data_grp.attrs["num_samples"] = int(traj["actions"].shape[0])
                ep_data_grp.attrs["source_episode_idx"] = int(source_episode)
                ep_data_grp.attrs["object_translation"] = np.concatenate(
                    [object_translation, target_translation], axis=0
                ).astype(np.float32)
                ep_data_grp.attrs["sampled_demo_key"] = sampled_entry["demo_key"] or ""
                total_samples += int(traj["actions"].shape[0])
    finally:
        if video_writer is not None:
            video_writer.close()
        if data_writer is not None and data_grp is not None:
            data_grp.attrs["total"] = total_samples
            data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)
            data_writer.close()

    rollout_stats_dict = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = {
        key: float(np.mean(rollout_stats_dict[key]))
        for key in rollout_stats_dict
    }
    avg_rollout_stats["Num_Success"] = float(np.sum(rollout_stats_dict["Success_Rate"]))

    support_summary = {
        "sample_mode": str(args.sample_mode),
        "candidate_count": len(candidates),
        "unique_source_episodes": unique_source_episodes,
        "fixed_source_episode": int(fixed_source_episode),
        "dedupe_support": not bool(args.no_dedupe_support),
        "exclude_zero_translation": bool(args.exclude_zero_translation),
        "sample_with_replacement": replace,
        "object_translation_min": translation_bounds["object_min"].round(6).tolist(),
        "object_translation_max": translation_bounds["object_max"].round(6).tolist(),
        "target_translation_min": translation_bounds["target_min"].round(6).tolist(),
        "target_translation_max": translation_bounds["target_max"].round(6).tolist(),
    }
    payload = {
        "checkpoint": str(checkpoint),
        "source_demo": str(source_demo),
        "train_dataset_hdf5": str(train_dataset_hdf5),
        "n_rollouts": int(args.n_rollouts),
        "horizon": int(args.horizon),
        "seed": int(args.seed),
        "support_summary": support_summary,
        "avg_rollout_stats": avg_rollout_stats,
        "sampled_candidates": sampled_candidates,
        "per_rollout": diagnostics_per_rollout,
    }

    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))
    print(json.dumps(payload, indent=2))

    if args.json_output:
        json_output = Path(args.json_output).expanduser().resolve()
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
