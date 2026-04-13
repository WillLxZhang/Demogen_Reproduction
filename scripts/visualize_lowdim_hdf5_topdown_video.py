#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a top-down XY video from a robomimic-style low_dim HDF5."
    )
    parser.add_argument("--dataset", required=True, help="Path to low_dim.hdf5")
    parser.add_argument("--output-video", required=True, help="Output mp4 path")
    parser.add_argument("--demo-key", default=None, help="Episode key such as demo_9")
    parser.add_argument(
        "--obs-group",
        default="obs",
        choices=["obs", "next_obs"],
        help="Observation group to read.",
    )
    parser.add_argument("--fps", type=int, default=20, help="Video fps")
    parser.add_argument("--stride", type=int, default=1, help="Use every N-th frame")
    parser.add_argument("--dpi", type=int, default=140, help="Figure dpi")
    parser.add_argument(
        "--tail-length",
        type=int,
        default=80,
        help="How many past frames to keep in the highlighted trail. Use <=0 for full trail.",
    )
    return parser.parse_args()


def numeric_suffix(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    if match:
        return int(match.group(1)), name
    return math.inf, name


def sorted_keys(group: h5py.Group) -> list[str]:
    return sorted(list(group.keys()), key=numeric_suffix)


def pick_demo_key(data_group: h5py.Group, demo_key: str | None) -> str:
    keys = sorted_keys(data_group)
    if not keys:
        raise ValueError("Dataset has no demos under /data")
    if demo_key is not None:
        if demo_key not in data_group:
            raise KeyError(f"demo_key={demo_key} not found. available={keys}")
        return demo_key
    return keys[0]


def parse_translation_attr(raw) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
            if isinstance(value, list):
                return str([round(float(v), 4) for v in value])
        except Exception:
            return raw
    return str(raw)


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    output_path = Path(args.output_video).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{dataset_path} missing top-level /data group")
        demo_key = pick_demo_key(f["data"], args.demo_key)
        demo_group = f["data"][demo_key]
        if args.obs_group not in demo_group:
            raise KeyError(f"{demo_key} missing obs group {args.obs_group}")
        obs_group = demo_group[args.obs_group]
        if "robot0_eef_pos" not in obs_group or "object" not in obs_group:
            raise KeyError(
                f"{demo_key}/{args.obs_group} must contain robot0_eef_pos and object"
            )

        eef = np.asarray(obs_group["robot0_eef_pos"][...], dtype=np.float32)
        obj = np.asarray(obs_group["object"][...], dtype=np.float32)[:, :3]
        if eef.shape[0] != obj.shape[0]:
            raise ValueError(
                f"eef len {eef.shape[0]} != object len {obj.shape[0]} for {demo_key}"
            )

        translation_text = parse_translation_attr(demo_group.attrs.get("object_translation"))

    frames = np.arange(0, eef.shape[0], max(1, args.stride), dtype=int)
    all_xy = np.concatenate([eef[:, :2], obj[:, :2]], axis=0)
    xy_min = all_xy.min(axis=0)
    xy_max = all_xy.max(axis=0)
    span = np.maximum(xy_max - xy_min, 1e-3)
    pad = np.maximum(0.08 * span, 0.015)
    xlim = (float(xy_min[0] - pad[0]), float(xy_max[0] + pad[0]))
    ylim = (float(xy_min[1] - pad[1]), float(xy_max[1] + pad[1]))

    fig, ax = plt.subplots(figsize=(6, 6), dpi=args.dpi)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.plot(eef[:, 0], eef[:, 1], color="#4C78A8", alpha=0.18, linewidth=1.0)
    ax.plot(obj[:, 0], obj[:, 1], color="#F58518", alpha=0.18, linewidth=1.0)
    ax.scatter(eef[0, 0], eef[0, 1], color="#4C78A8", marker="o", s=36, alpha=0.5)
    ax.scatter(obj[0, 0], obj[0, 1], color="#F58518", marker="o", s=36, alpha=0.5)
    ax.scatter(eef[-1, 0], eef[-1, 1], color="#4C78A8", marker="x", s=42, alpha=0.7)
    ax.scatter(obj[-1, 0], obj[-1, 1], color="#F58518", marker="x", s=42, alpha=0.7)

    eef_trail, = ax.plot([], [], color="#4C78A8", linewidth=2.0, label="eef")
    obj_trail, = ax.plot([], [], color="#F58518", linewidth=2.0, label="object")
    eef_now, = ax.plot([], [], marker="o", color="#4C78A8", markersize=7, linestyle="None")
    obj_now, = ax.plot([], [], marker="s", color="#F58518", markersize=7, linestyle="None")
    title = ax.set_title("")
    ax.legend(loc="upper right")

    with imageio.get_writer(output_path, fps=args.fps) as writer:
        for frame_idx in frames:
            if args.tail_length <= 0:
                start = 0
            else:
                start = max(0, frame_idx - args.tail_length + 1)

            eef_trail.set_data(eef[start : frame_idx + 1, 0], eef[start : frame_idx + 1, 1])
            obj_trail.set_data(obj[start : frame_idx + 1, 0], obj[start : frame_idx + 1, 1])
            eef_now.set_data([eef[frame_idx, 0]], [eef[frame_idx, 1]])
            obj_now.set_data([obj[frame_idx, 0]], [obj[frame_idx, 1]])

            title_text = f"{demo_key} | frame {frame_idx + 1}/{eef.shape[0]}"
            if translation_text is not None:
                title_text += f" | trans {translation_text}"
            title.set_text(title_text)

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
            writer.append_data(frame)

    plt.close(fig)
    print(f"Saved topdown video to: {output_path}")


if __name__ == "__main__":
    main()
