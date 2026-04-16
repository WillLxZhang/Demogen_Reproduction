#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1] / "repos" / "DemoGen"
DIFFUSION_POLICIES_ROOT = REPO_ROOT / "diffusion_policies"

import sys

if str(DIFFUSION_POLICIES_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICIES_ROOT))

from diffusion_policies.env.robosuite.dataset_meta import load_env_name_from_dataset


@dataclass(frozen=True)
class TaskObjectSpec:
    object_state_indices: np.ndarray | None
    target_state_indices: np.ndarray | None
    object_kind: str
    object_ref: str
    target_kind: str | None
    target_ref: str | None


TASK_OBJECT_SPECS: dict[str, TaskObjectSpec] = {
    "Stack": TaskObjectSpec(
        object_state_indices=np.array([10, 11, 12], dtype=np.int64),
        target_state_indices=np.array([17, 18, 19], dtype=np.int64),
        object_kind="attr",
        object_ref="cubeA_body_id",
        target_kind="attr",
        target_ref="cubeB_body_id",
    ),
    "NutAssemblyRound": TaskObjectSpec(
        object_state_indices=np.array([17, 18, 19], dtype=np.int64),
        target_state_indices=None,
        object_kind="obj_body_id",
        object_ref="RoundNut",
        target_kind="attr",
        target_ref="peg2_body_id",
    ),
    "NutAssemblySquare": TaskObjectSpec(
        object_state_indices=np.array([10, 11, 12], dtype=np.int64),
        target_state_indices=None,
        object_kind="obj_body_id",
        object_ref="SquareNut",
        target_kind="attr",
        target_ref="peg1_body_id",
    ),
}


def get_task_spec(env_name: str) -> TaskObjectSpec:
    if env_name not in TASK_OBJECT_SPECS:
        raise KeyError(
            f"No relalign task spec registered for env={env_name}. "
            "Add it to scripts/relalign_task_spec.py."
        )
    return TASK_OBJECT_SPECS[env_name]


def load_relalign_env_name(dataset_path: str | Path) -> str:
    return load_env_name_from_dataset(dataset_path)


def zero_translation_for_env(env_name: str) -> np.ndarray:
    spec = get_task_spec(env_name)
    if spec.target_state_indices is None:
        return np.zeros(3, dtype=np.float32)
    return np.zeros(6, dtype=np.float32)


def split_translation(raw: np.ndarray | list[float] | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if raw is None:
        return None, None
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.shape == (3,):
        return arr, None
    if arr.shape == (6,):
        return arr[:3], arr[3:6]
    raise ValueError(f"Expected translation shape (3,) or (6,), got {arr.shape}")


def apply_translation_to_reset_state(
    reset_state: dict,
    env_name: str,
    object_translation: np.ndarray | None,
    target_translation: np.ndarray | None,
) -> dict:
    spec = get_task_spec(env_name)
    translated = dict(reset_state)
    translated["states"] = np.asarray(reset_state["states"], dtype=np.float64).copy()

    if object_translation is not None and spec.object_state_indices is not None:
        idx = np.asarray(spec.object_state_indices, dtype=np.int64)
        translated["states"][idx] += np.asarray(object_translation[: len(idx)], dtype=np.float64)

    if target_translation is not None and spec.target_state_indices is not None:
        idx = np.asarray(spec.target_state_indices, dtype=np.int64)
        translated["states"][idx] += np.asarray(target_translation[: len(idx)], dtype=np.float64)

    return translated


def _resolve_body_id(raw_env, kind: str, ref: str | None) -> int:
    if ref is None:
        raise ValueError("Body reference is missing")
    if kind == "attr":
        if not hasattr(raw_env, ref):
            raise AttributeError(f"Expected env to have attribute {ref}")
        return int(getattr(raw_env, ref))
    if kind == "obj_body_id":
        if not hasattr(raw_env, "obj_body_id"):
            raise AttributeError("Expected env to have obj_body_id mapping")
        return int(raw_env.obj_body_id[ref])
    raise ValueError(f"Unsupported body resolver kind={kind}")


def capture_body_xyz(env, env_name: str, role: str) -> np.ndarray:
    spec = get_task_spec(env_name)
    if role == "object":
        body_id = _resolve_body_id(env.env, spec.object_kind, spec.object_ref)
    elif role == "target":
        if spec.target_kind is None or spec.target_ref is None:
            raise ValueError(f"Task env={env_name} has no target body configured")
        body_id = _resolve_body_id(env.env, spec.target_kind, spec.target_ref)
    else:
        raise ValueError(f"Unsupported role={role}")
    return np.asarray(env.env.sim.data.body_xpos[body_id][:3], dtype=np.float32).copy()
