#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1] / "repos" / "DemoGen"
DIFFUSION_POLICIES_ROOT = REPO_ROOT / "diffusion_policies"
if str(DIFFUSION_POLICIES_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICIES_ROOT))

import diffusion_policies.env.robosuite.robosuite_wrapper as robosuite_wrapper
from diffusion_policies.env.robosuite.robosuite_wrapper import Robosuite3DEnv

from relalign_task_spec import (
    apply_translation_to_reset_state as apply_relalign_translation,
    capture_body_xyz,
    split_translation,
    zero_translation_for_env,
)
from replay_zarr_episode import load_env_name, load_reset_state


OBS_OBJECT_POSITION_SLICES: dict[str, dict[str, tuple[int, int] | None]] = {
    "Stack": {
        "object": (0, 3),
        "target": (7, 10),
    },
    "NutAssemblyRound": {
        "object": (7, 10),
        "target": None,
    },
    "NutAssemblySquare": {
        "object": (7, 10),
        "target": None,
    },
}


@dataclass(frozen=True)
class StoredLowDimEpisode:
    demo_key: str
    episode_idx: int
    source_episode_idx: int
    actions: np.ndarray
    stored_obs: dict[str, np.ndarray]
    object_translation: np.ndarray | None
    target_translation: np.ndarray | None
    model_file: str | None


def numeric_suffix(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    if match:
        return int(match.group(1)), name
    return math.inf, name


def sorted_keys(group: h5py.Group) -> list[str]:
    return sorted(list(group.keys()), key=numeric_suffix)


def decode_attr(raw):
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return raw


def split_translation_attr(raw) -> tuple[np.ndarray | None, np.ndarray | None]:
    if raw is None:
        return None, None
    raw = decode_attr(raw)
    if isinstance(raw, str):
        raw = json.loads(raw)
    return split_translation(raw)


def resolve_source_demo_path(dataset_path: Path, source_demo_arg: str | None) -> Path:
    if source_demo_arg is not None:
        resolved = Path(source_demo_arg).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"source demo not found: {resolved}")
        return resolved

    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{dataset_path} missing /data group")
        raw = f["data"].attrs.get("source_low_dim_hdf5", None)

    if raw is None:
        raise ValueError(
            f"{dataset_path} does not store data.attrs['source_low_dim_hdf5']; pass --source-demo explicitly."
        )

    resolved = Path(str(decode_attr(raw))).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"source demo path recorded in {dataset_path} does not exist: {resolved}"
        )
    return resolved


def load_dataset_demo_keys(dataset_path: Path) -> list[str]:
    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{dataset_path} missing /data group")
        return sorted_keys(f["data"])


def _episode_matches_source_prefix(
    exported_ep: h5py.Group,
    source_ep: h5py.Group,
    source_idx: int,
    zero_translation: np.ndarray,
) -> bool:
    source_episode_idx = exported_ep.attrs.get("source_episode_idx", None)
    if source_episode_idx is None or int(source_episode_idx) != int(source_idx):
        return False

    object_translation, target_translation = split_translation_attr(
        exported_ep.attrs.get("object_translation", None)
    )
    combined_translation = zero_translation.copy()
    if object_translation is not None:
        combined_translation[: len(object_translation)] = object_translation
    if target_translation is not None:
        start = 3
        combined_translation[start : start + len(target_translation)] = target_translation
    if not np.allclose(combined_translation, zero_translation, atol=1e-6):
        return False

    exported_actions = np.asarray(exported_ep["actions"][()], dtype=np.float32)
    source_actions = np.asarray(source_ep["actions"][()], dtype=np.float32)
    if exported_actions.shape != source_actions.shape or not np.allclose(
        exported_actions, source_actions, atol=1e-6
    ):
        return False

    for obs_key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"):
        if f"obs/{obs_key}" not in exported_ep or f"obs/{obs_key}" not in source_ep:
            return False
        exported_obs = np.asarray(exported_ep[f"obs/{obs_key}"][()], dtype=np.float32)
        source_obs = np.asarray(source_ep[f"obs/{obs_key}"][()], dtype=np.float32)
        if exported_obs.shape != source_obs.shape or not np.allclose(exported_obs, source_obs, atol=1e-6):
            return False

    return True


def infer_prepended_source_demo_count(dataset_path: Path, source_demo_path: Path) -> int:
    env_name = load_env_name(source_demo_path)
    zero_translation = zero_translation_for_env(env_name)

    with h5py.File(dataset_path, "r") as exported_f, h5py.File(source_demo_path, "r") as source_f:
        exported_keys = sorted_keys(exported_f["data"])
        source_keys = sorted_keys(source_f["data"])
        matched = 0
        for idx, (exported_key, source_key) in enumerate(zip(exported_keys, source_keys)):
            if _episode_matches_source_prefix(
                exported_ep=exported_f["data"][exported_key],
                source_ep=source_f["data"][source_key],
                source_idx=idx,
                zero_translation=zero_translation,
            ):
                matched += 1
            else:
                break
        return matched


def parse_episode_selection(raw: str, n_episodes: int, source_prefix_count: int) -> list[int]:
    value = raw.strip().lower()
    if value == "all":
        return list(range(n_episodes))
    if value == "generated":
        return list(range(source_prefix_count, n_episodes))
    if value == "source":
        return list(range(source_prefix_count))

    selected = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= n_episodes:
            raise IndexError(f"episode {idx} out of range for dataset with {n_episodes} demos")
        selected.append(idx)
    if not selected:
        raise ValueError("No episodes selected")
    return selected


def load_episode(dataset_path: Path, episode_idx: int) -> StoredLowDimEpisode:
    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{dataset_path} missing /data group")
        data_group = f["data"]
        keys = sorted_keys(data_group)
        if episode_idx < 0 or episode_idx >= len(keys):
            raise IndexError(f"episode {episode_idx} out of range for dataset with {len(keys)} demos")

        demo_key = keys[episode_idx]
        ep = data_group[demo_key]
        source_episode_idx = ep.attrs.get("source_episode_idx", None)
        if source_episode_idx is None:
            raise KeyError(f"{demo_key} missing attr source_episode_idx")

        actions = np.asarray(ep["actions"][()], dtype=np.float32)
        stored_obs = {}
        for obs_key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"):
            path = f"obs/{obs_key}"
            if path not in ep:
                raise KeyError(f"{demo_key} missing {path}")
            stored_obs[obs_key] = np.asarray(ep[path][()], dtype=np.float32)

        object_translation, target_translation = split_translation_attr(
            ep.attrs.get("object_translation", None)
        )
        model_file = decode_attr(ep.attrs.get("model_file", None))

        return StoredLowDimEpisode(
            demo_key=demo_key,
            episode_idx=int(episode_idx),
            source_episode_idx=int(source_episode_idx),
            actions=actions,
            stored_obs=stored_obs,
            object_translation=object_translation,
            target_translation=target_translation,
            model_file=model_file,
        )


def build_translated_reset_state(
    source_demo_path: Path,
    env_name: str,
    source_episode_idx: int,
    object_translation: np.ndarray | None,
    target_translation: np.ndarray | None,
) -> dict:
    reset_state = load_reset_state(source_demo_path, source_episode_idx)
    translated = apply_relalign_translation(
        reset_state=reset_state,
        env_name=env_name,
        object_translation=object_translation,
        target_translation=target_translation,
    )
    reset_state["states"] = translated["states"]
    return reset_state


def make_replay_env(source_demo_path: Path, control_steps: int) -> Robosuite3DEnv:
    if control_steps <= 0:
        raise ValueError("--control-steps must be positive")
    robosuite_wrapper.N_CONTROL_STEPS = int(control_steps)
    env = Robosuite3DEnv(str(source_demo_path), render=False)
    env.reset()
    return env


def replay_success_episode(
    env: Robosuite3DEnv,
    reset_state: dict,
    actions: np.ndarray,
) -> bool:
    env.reset_to(reset_state)
    success_any = False
    for action in np.asarray(actions, dtype=np.float32):
        env.step(action)
        success_any = bool(env.check_success()) or success_any
    return success_any


def replay_observation_episode(
    env: Robosuite3DEnv,
    env_name: str,
    reset_state: dict,
    actions: np.ndarray,
) -> dict[str, np.ndarray | bool | None]:
    env.reset_to(reset_state)

    eef_pos_seq = []
    eef_quat_seq = []
    gripper_qpos_seq = []
    object_obs_seq = []
    object_pos_seq = []
    target_pos_seq = []
    success_any = False

    capture_target = OBS_OBJECT_POSITION_SLICES.get(env_name, {}).get("target", None) is not None

    for action in np.asarray(actions, dtype=np.float32):
        raw_obs = env.get_observation()
        eef_pos_seq.append(np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32))
        eef_quat_seq.append(np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float32))
        gripper_qpos_seq.append(np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32))
        object_obs_seq.append(np.asarray(raw_obs["object"], dtype=np.float32))
        object_pos_seq.append(capture_body_xyz(env, env_name, "object"))
        if capture_target:
            target_pos_seq.append(capture_body_xyz(env, env_name, "target"))

        env.step(action)
        success_any = bool(env.check_success()) or success_any

    result: dict[str, np.ndarray | bool | None] = {
        "robot0_eef_pos": np.asarray(eef_pos_seq, dtype=np.float32),
        "robot0_eef_quat": np.asarray(eef_quat_seq, dtype=np.float32),
        "robot0_gripper_qpos": np.asarray(gripper_qpos_seq, dtype=np.float32),
        "object": np.asarray(object_obs_seq, dtype=np.float32),
        "object_pos": np.asarray(object_pos_seq, dtype=np.float32),
        "target_pos": np.asarray(target_pos_seq, dtype=np.float32) if target_pos_seq else None,
        "success_any": bool(success_any),
    }
    return result


def extract_stored_object_positions(env_name: str, stored_object_obs: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    slices = OBS_OBJECT_POSITION_SLICES.get(env_name, {})
    object_slice = slices.get("object", None)
    target_slice = slices.get("target", None)

    object_pos = None
    target_pos = None
    if object_slice is not None:
        object_pos = np.asarray(
            stored_object_obs[:, object_slice[0] : object_slice[1]],
            dtype=np.float32,
        )
    if target_slice is not None:
        target_pos = np.asarray(
            stored_object_obs[:, target_slice[0] : target_slice[1]],
            dtype=np.float32,
        )
    return object_pos, target_pos


def compute_error_summary(stored: np.ndarray, replayed: np.ndarray) -> dict:
    if stored.shape != replayed.shape:
        raise ValueError(f"Shape mismatch: stored={stored.shape}, replayed={replayed.shape}")

    err = np.asarray(replayed - stored, dtype=np.float32)
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    mae = np.mean(np.abs(err), axis=0)
    max_abs = np.max(np.abs(err), axis=0)
    final_err = err[-1]

    return {
        "rmse": rmse,
        "mae": mae,
        "max_abs": max_abs,
        "final_err": final_err,
        "rmse_norm": float(np.linalg.norm(rmse)),
        "final_err_norm": float(np.linalg.norm(final_err)),
    }


def rounded_list(value: np.ndarray | None) -> list[float] | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32).round(6).tolist()
