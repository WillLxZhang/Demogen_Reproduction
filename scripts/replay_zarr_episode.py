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


REPO_ROOT = Path(__file__).resolve().parents[1] / "repos" / "DemoGen"
DIFFUSION_POLICIES_ROOT = REPO_ROOT / "diffusion_policies"
if str(DIFFUSION_POLICIES_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICIES_ROOT))

from diffusion_policies.common.replay_buffer import ReplayBuffer
from diffusion_policies.env.robosuite.dataset_meta import load_env_name_from_dataset
import diffusion_policies.env.robosuite.robosuite_wrapper as robosuite_wrapper
from diffusion_policies.env.robosuite.robosuite_wrapper import Robosuite3DEnv


TASK_OBJECT_STATE_INDICES = {
    "Lift": np.array([10, 11, 12], dtype=np.int64),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one episode from a generated Zarr dataset in robosuite."
    )
    parser.add_argument("--zarr", required=True, help="Path to generated .zarr dataset")
    parser.add_argument("--source-demo", required=True, help="Path to source demo.hdf5 used to build the environment")
    parser.add_argument("--episode", type=int, default=0, help="Generated episode index to replay")
    parser.add_argument(
        "--source-episode",
        type=int,
        default=None,
        help="Source episode index in demo.hdf5. Defaults to mapping generated episode blocks evenly onto source episodes.",
    )
    parser.add_argument("--output-video", required=True, help="Where to save the replay video")
    parser.add_argument("--render-size", type=int, default=256, help="Rendered video size")
    parser.add_argument("--fps", type=int, default=20, help="Video FPS")
    parser.add_argument(
        "--control-steps",
        type=int,
        default=1,
        help="How many internal robosuite control repeats to apply per action during replay.",
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="Render camera name. Defaults to robosuite wrapper default for the task.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on replayed actions for quick debugging.",
    )
    parser.add_argument(
        "--translate-object",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Whether to translate the replayed object to match the generated episode. "
            "'auto' uses stored meta when available and otherwise falls back to Lift-specific "
            "agent_pos inference from low_dim.hdf5."
        ),
    )
    parser.add_argument(
        "--source-low-dim",
        default=None,
        help=(
            "Optional source low_dim.hdf5 used to infer generated object translation when the "
            "zarr does not store object_translation meta. Defaults to a sibling low_dim.hdf5 "
            "next to --source-demo when present."
        ),
    )
    parser.add_argument(
        "--motion-frame",
        type=int,
        default=90,
        help=(
            "Motion / pre-grasp frame count used only for fallback translation inference "
            "from generated agent_pos versus source low_dim observations."
        ),
    )
    return parser.parse_args()


def list_demo_keys(source_demo_path: Path) -> list[str]:
    with h5py.File(source_demo_path, "r") as f:
        return list(f["data"].keys())


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


def load_env_name(source_demo_path: Path) -> str:
    return load_env_name_from_dataset(source_demo_path)


def load_generated_episode_meta(zarr_path: Path, episode_idx: int) -> dict:
    root = zarr.open(str(zarr_path), mode="r")
    meta = root.get("meta", None)
    if meta is None:
        return {}

    result = {}
    if "source_episode_idx" in meta:
        result["source_episode_idx"] = int(meta["source_episode_idx"][episode_idx])
    if "object_translation" in meta:
        result["object_translation"] = np.asarray(meta["object_translation"][episode_idx], dtype=np.float32)
    if "motion_frame_count" in meta:
        result["motion_frame_count"] = int(meta["motion_frame_count"][episode_idx])
    return result


def resolve_source_low_dim_path(source_demo_path: Path, source_low_dim_arg: str | None) -> Path | None:
    if source_low_dim_arg is not None:
        return Path(source_low_dim_arg).expanduser().resolve()
    candidate = source_demo_path.with_name("low_dim.hdf5")
    if candidate.exists():
        return candidate
    return None


def infer_object_translation_from_low_dim(
    source_low_dim_path: Path,
    source_episode_idx: int,
    generated_episode: dict,
    motion_frame: int,
) -> np.ndarray:
    if motion_frame <= 0:
        raise ValueError("--motion-frame must be positive for object translation inference")

    with h5py.File(source_low_dim_path, "r") as f:
        demos = list(f["data"].keys())
        if source_episode_idx < 0 or source_episode_idx >= len(demos):
            raise IndexError(
                f"source episode index {source_episode_idx} out of range for low_dim demos={demos}"
            )
        ep = demos[source_episode_idx]
        source_agent_pos = f[f"data/{ep}/obs/robot0_eef_pos"][...].astype(np.float32)

    generated_agent_pos = np.asarray(generated_episode["agent_pos"], dtype=np.float32)
    frame_idx = motion_frame - 1
    if frame_idx >= len(generated_agent_pos) or frame_idx >= len(source_agent_pos):
        raise IndexError(
            f"motion frame {motion_frame} exceeds generated/source episode length "
            f"({len(generated_agent_pos)}, {len(source_agent_pos)})"
        )

    translation = generated_agent_pos[frame_idx, :3] - source_agent_pos[frame_idx, :3]
    return np.asarray(translation, dtype=np.float32)


def infer_source_episode(generated_episode_idx: int, n_generated: int, n_source: int) -> int:
    if n_source <= 0:
        raise ValueError("n_source must be positive")
    if n_source == 1:
        return 0
    if n_generated <= n_source:
        return min(n_source - 1, generated_episode_idx)
    block = max(1, n_generated // n_source)
    return min(n_source - 1, generated_episode_idx // block)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    zarr_path = Path(args.zarr).expanduser().resolve()
    source_demo_path = Path(args.source_demo).expanduser().resolve()
    output_video = Path(args.output_video).expanduser().resolve()
    if args.control_steps <= 0:
        raise ValueError("--control-steps must be positive")

    replay_buffer = ReplayBuffer.copy_from_path(
        str(zarr_path),
        keys=["agent_pos", "action", "point_cloud"],
    )

    n_episodes = replay_buffer.n_episodes
    if args.episode < 0 or args.episode >= n_episodes:
        raise IndexError(f"episode {args.episode} out of range, dataset has {n_episodes} episodes")

    episode_meta = load_generated_episode_meta(zarr_path, args.episode)
    demo_keys = list_demo_keys(source_demo_path)
    if args.source_episode is None:
        source_episode_idx = episode_meta.get("source_episode_idx")
        if source_episode_idx is None:
            source_episode_idx = infer_source_episode(args.episode, n_episodes, len(demo_keys))
    else:
        source_episode_idx = args.source_episode

    # Make replay faithful to the stored action sequence. The main training /
    # eval stack currently uses a larger repeat count, but for debugging we
    # want one environment step per action by default.
    robosuite_wrapper.N_CONTROL_STEPS = args.control_steps

    reset_state = load_reset_state(source_demo_path, source_episode_idx)
    episode = replay_buffer.get_episode(args.episode, copy=True)
    actions = np.asarray(episode["action"], dtype=np.float32)
    if args.max_steps is not None:
        actions = actions[: args.max_steps]

    applied_object_translation = None
    object_translation_source = None
    if args.translate_object != "off":
        env_name = load_env_name(source_demo_path)
        object_state_indices = TASK_OBJECT_STATE_INDICES.get(env_name)
        if object_state_indices is not None:
            object_translation = episode_meta.get("object_translation")
            if object_translation is not None:
                object_translation_source = "zarr_meta"
            else:
                source_low_dim_path = resolve_source_low_dim_path(source_demo_path, args.source_low_dim)
                if source_low_dim_path is not None and source_low_dim_path.exists():
                    motion_frame = int(episode_meta.get("motion_frame_count", args.motion_frame))
                    object_translation = infer_object_translation_from_low_dim(
                        source_low_dim_path=source_low_dim_path,
                        source_episode_idx=source_episode_idx,
                        generated_episode=episode,
                        motion_frame=motion_frame,
                    )
                    object_translation_source = "low_dim_inference"
                elif args.translate_object == "on":
                    raise FileNotFoundError(
                        "Could not determine object translation: zarr meta is missing and no low_dim.hdf5 "
                        "was found. Pass --source-low-dim or regenerate the dataset with saved meta."
                    )
                else:
                    object_translation = None

            if object_translation is not None:
                reset_state["states"] = np.asarray(reset_state["states"], dtype=np.float64).copy()
                reset_state["states"][object_state_indices] += object_translation[: len(object_state_indices)]
                applied_object_translation = np.asarray(object_translation, dtype=np.float32)
        elif args.translate_object == "on":
            raise ValueError(f"Object translation replay is not configured for task {env_name}")

    env = Robosuite3DEnv(
        str(source_demo_path),
        render=False,
        cam_width=args.render_size,
        cam_height=args.render_size,
        render_cam=args.camera,
    )
    env.reset()
    env.reset_to(reset_state)

    ensure_parent(output_video)
    frames = [env.render(mode="rgb_array", height=args.render_size, width=args.render_size, camera_name=args.camera)]
    observed_agent_pos = []
    successes = []

    for action in actions:
        obs, reward, done, info = env.step(action)
        observed_agent_pos.append(obs["agent_pos"])
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

    observed_agent_pos = np.asarray(observed_agent_pos, dtype=np.float32)
    pos = observed_agent_pos[:, :3] if len(observed_agent_pos) else np.zeros((0, 3), dtype=np.float32)
    summary = {
        "zarr": str(zarr_path),
        "source_demo": str(source_demo_path),
        "episode": int(args.episode),
        "source_episode": int(source_episode_idx),
        "n_actions": int(len(actions)),
        "control_steps": int(args.control_steps),
        "object_translation": applied_object_translation.round(6).tolist() if applied_object_translation is not None else None,
        "object_translation_source": object_translation_source,
        "video": str(output_video),
        "success_any": bool(any(successes)),
        "pos_min": pos.min(axis=0).round(6).tolist() if len(pos) else None,
        "pos_max": pos.max(axis=0).round(6).tolist() if len(pos) else None,
        "pos_span": (pos.max(axis=0) - pos.min(axis=0)).round(6).tolist() if len(pos) else None,
        "first_obs_agent_pos": observed_agent_pos[0].round(6).tolist() if len(observed_agent_pos) else None,
        "last_obs_agent_pos": observed_agent_pos[-1].round(6).tolist() if len(observed_agent_pos) else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
