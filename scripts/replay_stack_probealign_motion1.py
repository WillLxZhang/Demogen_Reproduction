#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import imageio
import numpy as np
import zarr


REPO_ROOT = Path(__file__).resolve().parents[1]
DIFFUSION_POLICIES_ROOT = REPO_ROOT / "repos" / "DemoGen" / "diffusion_policies"
if str(DIFFUSION_POLICIES_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICIES_ROOT))

import diffusion_policies.env.robosuite.robosuite_wrapper as robosuite_wrapper
from diffusion_policies.env.robosuite.robosuite_wrapper import Robosuite3DEnv


OBJECT_STATE_INDICES = np.array([10, 11, 12], dtype=np.int64)
TARGET_STATE_INDICES = np.array([17, 18, 19], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a one-off Stack-Cube trajectory that uses probe-align on the "
            "motion1 prefix and source actions afterwards."
        )
    )
    parser.add_argument(
        "--source-zarr",
        default=str(
            REPO_ROOT
            / "repos"
            / "DemoGen"
            / "data"
            / "datasets"
            / "source"
            / "stack_cube_0_v1_replayh1_twophase_source.zarr"
        ),
    )
    parser.add_argument(
        "--source-demo",
        default=str(
            REPO_ROOT / "data" / "raw" / "stack_cube_0" / "1775663680_9007828" / "demo.hdf5"
        ),
    )
    parser.add_argument(
        "--low-dim",
        default=str(
            REPO_ROOT / "data" / "raw" / "stack_cube_0" / "1775663680_9007828" / "low_dim.hdf5"
        ),
    )
    parser.add_argument(
        "--output-video",
        default=str(REPO_ROOT / "outputs" / "replay" / "stack_cube_probealign_motion1_ep0.mp4"),
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--source-episode", type=int, default=0)
    parser.add_argument("--skill1-frame", type=int, default=380)
    parser.add_argument("--probe-axes", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--translate-object", type=float, nargs=3, default=[0.03, 0.03, 0.0])
    parser.add_argument("--translate-target", type=float, nargs=3, default=[0.03, 0.03, 0.0])
    parser.add_argument("--control-steps", type=int, default=1)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def list_demo_keys(path: Path) -> list[str]:
    with h5py.File(path, "r") as f:
        return list(f["data"].keys())


def load_reset_state(path: Path, episode_idx: int) -> dict:
    with h5py.File(path, "r") as f:
        demo_keys = list_demo_keys(path)
        ep = demo_keys[episode_idx]
        group = f[f"data/{ep}"]
        state = {
            "states": group["states"][0],
            "model": group.attrs["model_file"],
        }
        if "ep_meta" in group.attrs:
            state["ep_meta"] = group.attrs["ep_meta"]
        return state


def capture_current_reset_state(env: Robosuite3DEnv, reset_template: dict) -> dict:
    state = {
        "states": np.array(env.env.sim.get_state().flatten(), dtype=np.float64),
        "model": reset_template["model"],
    }
    if "ep_meta" in reset_template:
        state["ep_meta"] = reset_template["ep_meta"]
    return state


def choose_probe_aligned_action(
    env: Robosuite3DEnv,
    probe_env: Robosuite3DEnv,
    reset_template: dict,
    source_exec_action: np.ndarray,
    desired_next_pos: np.ndarray,
    probe_axes: list[int],
) -> np.ndarray:
    probe_reset_state = capture_current_reset_state(env, reset_template)
    probe_env.reset_to(probe_reset_state)
    base_obs, _, _, _ = probe_env.step(source_exec_action.copy())
    base_next_pos = np.asarray(base_obs["agent_pos"][:3], dtype=np.float32)

    responses: dict[tuple[int, int], np.ndarray] = {}
    allowed_per_axis: list[tuple[int, ...]] = []
    for axis in probe_axes:
        if abs(float(source_exec_action[axis])) > 1e-6:
            allowed_per_axis.append((0,))
            continue
        allowed_per_axis.append((-1, 0, 1))
        for pulse in (-1, 1):
            probe_action = source_exec_action.copy()
            probe_action[axis] = np.clip(probe_action[axis] + pulse, -1.0, 1.0)
            probe_env.reset_to(probe_reset_state)
            obs, _, _, _ = probe_env.step(probe_action)
            responses[(axis, pulse)] = (
                np.asarray(obs["agent_pos"][:3], dtype=np.float32) - base_next_pos
            )

    best_extra = np.zeros(3, dtype=np.float32)
    best_err = float(np.linalg.norm(desired_next_pos - base_next_pos))
    for pulses in np.array(np.meshgrid(*allowed_per_axis)).T.reshape(-1, len(allowed_per_axis)):
        predicted = base_next_pos.copy()
        extra = np.zeros(3, dtype=np.float32)
        for axis, pulse in zip(probe_axes, pulses.tolist()):
            if pulse == 0:
                continue
            predicted = predicted + responses[(axis, pulse)]
            extra[axis] = pulse
        err = float(np.linalg.norm(desired_next_pos - predicted))
        if err < best_err:
            best_err = err
            best_extra = extra

    action = source_exec_action.copy()
    action[:3] = np.clip(action[:3] + best_extra, -1.0, 1.0)
    return action.astype(np.float32)


def main() -> None:
    args = parse_args()
    source_zarr = Path(args.source_zarr).expanduser().resolve()
    source_demo = Path(args.source_demo).expanduser().resolve()
    low_dim = Path(args.low_dim).expanduser().resolve()
    output_video = Path(args.output_video).expanduser().resolve()
    output_json = None if args.output_json is None else Path(args.output_json).expanduser().resolve()

    ensure_parent(output_video)
    if output_json is not None:
        ensure_parent(output_json)

    source_root = zarr.open(str(source_zarr), mode="r")
    source_state = np.asarray(source_root["data"]["state"], dtype=np.float32)
    source_action = np.asarray(source_root["data"]["action"], dtype=np.float32)
    n_steps = source_state.shape[0] if args.max_steps is None else min(args.max_steps, source_state.shape[0])

    object_translation = np.asarray(args.translate_object, dtype=np.float32)
    target_translation = np.asarray(args.translate_target, dtype=np.float32)

    reset_state = load_reset_state(source_demo, args.source_episode)
    reset_state["states"] = np.asarray(reset_state["states"], dtype=np.float64).copy()
    reset_state["states"][OBJECT_STATE_INDICES] += object_translation.astype(np.float64)
    reset_state["states"][TARGET_STATE_INDICES] += target_translation.astype(np.float64)

    robosuite_wrapper.N_CONTROL_STEPS = int(args.control_steps)
    env = Robosuite3DEnv(
        str(source_demo),
        render=False,
        cam_width=args.render_size,
        cam_height=args.render_size,
        render_cam=args.camera,
    )
    probe_env = Robosuite3DEnv(
        str(source_demo),
        render=False,
        cam_width=args.render_size,
        cam_height=args.render_size,
        render_cam=args.camera,
    )

    obs = env.reset_to(reset_state)
    probe_env.reset_to(reset_state)

    frames = [
        env.render(
            mode="rgb_array",
            height=args.render_size,
            width=args.render_size,
            camera_name=args.camera,
        )
    ]
    realized_actions = []
    realized_agent_pos = [np.asarray(obs["agent_pos"], dtype=np.float32).copy()]
    successes = []

    for frame_idx in range(n_steps):
        source_exec_action = np.asarray(source_action[frame_idx], dtype=np.float32)
        if frame_idx < int(args.skill1_frame):
            target_frame = min(frame_idx + 1, int(args.skill1_frame) - 1)
            desired_next_pos = (
                np.asarray(source_state[target_frame, :3], dtype=np.float32) + object_translation
            )
            action = choose_probe_aligned_action(
                env=env,
                probe_env=probe_env,
                reset_template=reset_state,
                source_exec_action=source_exec_action,
                desired_next_pos=desired_next_pos,
                probe_axes=list(args.probe_axes),
            )
        else:
            action = source_exec_action.copy()

        realized_actions.append(action.copy())
        obs, _, _, _ = env.step(action)
        realized_agent_pos.append(np.asarray(obs["agent_pos"], dtype=np.float32).copy())
        successes.append(bool(env.check_success()))
        frames.append(
            env.render(
                mode="rgb_array",
                height=args.render_size,
                width=args.render_size,
                camera_name=args.camera,
            )
        )

    with imageio.get_writer(output_video, fps=args.fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    realized_agent_pos = np.asarray(realized_agent_pos, dtype=np.float32)
    realized_actions = np.asarray(realized_actions, dtype=np.float32)

    with h5py.File(low_dim, "r") as f:
        demo = f["data"][sorted(f["data"].keys())[args.source_episode]]
        obj = demo["obs"]["object"][:, 0:3].astype(np.float32) + object_translation[None, :]
        src_obj = demo["obs"]["object"][:, 0:3].astype(np.float32)
        src_eef = demo["obs"]["robot0_eef_pos"][:].astype(np.float32)

    rel_379 = realized_agent_pos[380, :3] - obj[379]
    src_rel_379 = src_eef[379] - src_obj[379]
    rel_delta_379 = rel_379 - src_rel_379

    summary = {
        "source_zarr": str(source_zarr),
        "source_demo": str(source_demo),
        "output_video": str(output_video),
        "steps": int(n_steps),
        "skill1_frame": int(args.skill1_frame),
        "probe_axes": [int(v) for v in args.probe_axes],
        "object_translation": object_translation.round(6).tolist(),
        "target_translation": target_translation.round(6).tolist(),
        "success_any": bool(any(successes)),
        "success_last": bool(successes[-1]) if successes else False,
        "frame379_rel": rel_379.round(6).tolist(),
        "frame379_source_rel": src_rel_379.round(6).tolist(),
        "frame379_rel_delta": rel_delta_379.round(6).tolist(),
        "last_agent_pos": realized_agent_pos[-1].round(6).tolist(),
        "n_realized_actions": int(len(realized_actions)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if output_json is not None:
        output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    env.close()
    probe_env.close()


if __name__ == "__main__":
    main()
