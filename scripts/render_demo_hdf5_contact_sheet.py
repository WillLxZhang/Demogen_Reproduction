#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1] / "repos" / "DemoGen"
DIFFUSION_POLICIES_ROOT = REPO_ROOT / "diffusion_policies"
if str(DIFFUSION_POLICIES_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICIES_ROOT))

from diffusion_policies.env.robosuite.robosuite_wrapper import Robosuite3DEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a raw robosuite demo.hdf5 and save evenly sampled contact sheets "
            "for selected episodes."
        )
    )
    parser.add_argument("--demo-hdf5", required=True, help="Path to raw demo.hdf5")
    parser.add_argument("--output-dir", required=True, help="Directory to save contact sheets")
    parser.add_argument(
        "--episodes",
        default="all",
        help="Comma-separated episode indices, demo keys, or 'all'.",
    )
    parser.add_argument("--camera", default="agentview", help="Render camera name")
    parser.add_argument("--render-size", type=int, default=256, help="Square render size")
    parser.add_argument("--n-frames", type=int, default=20, help="Frames per contact sheet")
    parser.add_argument(
        "--frame-stop",
        type=int,
        default=None,
        help="Only sample from the first N frames of each episode.",
    )
    parser.add_argument("--cols", type=int, default=5, help="Columns in the contact sheet grid")
    return parser.parse_args()


def numeric_suffix(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    if match:
        return int(match.group(1)), name
    return math.inf, name


def sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    return sorted(list(data_group.keys()), key=numeric_suffix)


def parse_episode_selection(raw: str, demo_keys: list[str]) -> list[tuple[int, str]]:
    if raw.strip().lower() == "all":
        return list(enumerate(demo_keys))

    resolved: list[tuple[int, str]] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if token in demo_keys:
            resolved.append((demo_keys.index(token), token))
            continue
        idx = int(token)
        if idx < 0 or idx >= len(demo_keys):
            raise IndexError(f"episode {idx} out of range for demos={demo_keys}")
        resolved.append((idx, demo_keys[idx]))
    if not resolved:
        raise ValueError("No episodes selected")
    return resolved


def build_reset_state(ep_group: h5py.Group) -> dict:
    state = {
        "states": np.asarray(ep_group["states"][0], dtype=np.float64),
        "model": ep_group.attrs["model_file"],
    }
    if "ep_meta" in ep_group.attrs:
        state["ep_meta"] = ep_group.attrs["ep_meta"]
    return state


def compute_frame_indices(num_states: int, n_frames: int, frame_stop: int | None) -> np.ndarray:
    if num_states <= 0:
        raise ValueError("Episode has no states")
    usable = num_states if frame_stop is None else min(num_states, int(frame_stop))
    if usable <= 0:
        raise ValueError(f"usable frame count is invalid: {usable}")
    if n_frames <= 1:
        return np.asarray([0], dtype=np.int64)
    return np.linspace(0, usable - 1, num=n_frames, dtype=np.int64)


def render_episode_frames(
    env: Robosuite3DEnv,
    ep_group: h5py.Group,
    frame_indices: np.ndarray,
    camera_name: str,
    render_size: int,
) -> list[np.ndarray]:
    states = np.asarray(ep_group["states"], dtype=np.float64)
    reset_state = build_reset_state(ep_group)
    env.reset_to(reset_state)

    frames: list[np.ndarray] = []
    for idx in frame_indices:
        env.reset_to({"states": states[int(idx)]})
        frame = env.render(
            mode="rgb_array",
            height=render_size,
            width=render_size,
            camera_name=camera_name,
        )
        frames.append(np.asarray(frame, dtype=np.uint8))
    return frames


def save_contact_sheet(
    *,
    output_path: Path,
    frames: list[np.ndarray],
    frame_indices: np.ndarray,
    demo_key: str,
    total_states: int,
    usable_states: int,
    cols: int,
) -> None:
    cols = max(1, int(cols))
    rows = int(math.ceil(len(frames) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.0), dpi=150)
    axes_arr = np.asarray(axes, dtype=object).reshape(-1)

    for ax in axes_arr:
        ax.axis("off")

    for ax, frame, idx in zip(axes_arr, frames, frame_indices):
        ax.imshow(frame)
        ax.set_title(f"f{int(idx) + 1}", fontsize=9)
        ax.axis("off")

    fig.suptitle(
        f"{demo_key} | total={total_states} | sampled_from=1..{usable_states} | n={len(frames)}",
        fontsize=12,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    demo_hdf5 = Path(args.demo_hdf5).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    env = Robosuite3DEnv(
        str(demo_hdf5),
        render=False,
        cam_width=args.render_size,
        cam_height=args.render_size,
        render_cam=args.camera,
    )

    try:
        with h5py.File(demo_hdf5, "r") as f:
            demo_keys = sorted_demo_keys(f["data"])
            selected = parse_episode_selection(args.episodes, demo_keys)

            for episode_idx, demo_key in selected:
                ep_group = f["data"][demo_key]
                states = np.asarray(ep_group["states"], dtype=np.float64)
                frame_indices = compute_frame_indices(
                    num_states=len(states),
                    n_frames=args.n_frames,
                    frame_stop=args.frame_stop,
                )
                frames = render_episode_frames(
                    env=env,
                    ep_group=ep_group,
                    frame_indices=frame_indices,
                    camera_name=args.camera,
                    render_size=args.render_size,
                )
                output_path = output_dir / f"{demo_key}_contact_sheet.png"
                save_contact_sheet(
                    output_path=output_path,
                    frames=frames,
                    frame_indices=frame_indices,
                    demo_key=demo_key,
                    total_states=len(states),
                    usable_states=int(frame_indices[-1]) + 1,
                    cols=args.cols,
                )
                manifest[demo_key] = {
                    "episode_idx": int(episode_idx),
                    "total_states": int(len(states)),
                    "frame_indices": [int(v) for v in frame_indices],
                    "output_png": str(output_path),
                }
                print(
                    json.dumps(
                        {
                            "demo_key": demo_key,
                            "episode_idx": int(episode_idx),
                            "total_states": int(len(states)),
                            "frame_indices": [int(v) for v in frame_indices],
                            "output_png": str(output_path),
                        },
                        ensure_ascii=False,
                    )
                )
    finally:
        env.close()

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
