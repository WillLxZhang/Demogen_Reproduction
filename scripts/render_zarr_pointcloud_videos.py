#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import imageio
import matplotlib
import numpy as np
import zarr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render point-cloud videos from a generated zarr dataset."
    )
    parser.add_argument("--zarr", required=True, help="Path to generated zarr")
    parser.add_argument(
        "--episodes",
        default="all",
        help="Comma-separated episode indices, or 'all'",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for mp4 outputs")
    parser.add_argument("--fps", type=int, default=20, help="Video fps")
    parser.add_argument("--elev", type=float, default=20.0, help="3D elevation")
    parser.add_argument("--azim", type=float, default=30.0, help="3D azimuth")
    parser.add_argument("--stride", type=int, default=1, help="Use every N-th frame")
    parser.add_argument(
        "--point-size",
        type=float,
        default=None,
        help="Optional matplotlib scatter size. Leave unset to match DemoGen's original default.",
    )
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


def iter_episode_bounds(episode_ends: np.ndarray):
    start = 0
    for end in episode_ends:
        end = int(end)
        yield start, end
        start = end


def point_cloud_to_video(point_clouds, output_file: Path, fps=20, elev=30, azim=45, point_size=None):
    fig = plt.figure(figsize=(8, 6), dpi=220)
    ax = fig.add_subplot(111, projection="3d")

    all_points = np.concatenate(point_clouds, axis=0)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    xyz_min = min_vals[:3]
    xyz_max = max_vals[:3]
    xyz_span = np.maximum(xyz_max - xyz_min, 1e-3)
    xyz_pad = np.maximum(xyz_span * 0.08, 0.02)
    plot_min = xyz_min - xyz_pad
    plot_max = xyz_max + xyz_pad

    output_file.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_file, fps=fps)

    for frame_idx, points in enumerate(point_clouds):
        ax.clear()
        color = np.clip(points[:, 3:] / 255.0, 0.0, 1.0)
        scatter_kwargs = {
            "c": color,
            "marker": ".",
        }
        if point_size is not None:
            scatter_kwargs["s"] = float(point_size)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], **scatter_kwargs)

        ax.set_box_aspect((plot_max - plot_min).tolist())
        ax.set_xlim(plot_min[0], plot_max[0])
        ax.set_ylim(plot_min[1], plot_max[1])
        ax.set_zlim(plot_min[2], plot_max[2])
        ax.tick_params(axis="both", which="major", labelsize=8)

        formatter = FormatStrFormatter("%.1f")
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.zaxis.set_major_formatter(formatter)

        ax.view_init(elev=elev, azim=azim)
        ax.text2D(
            0.05,
            0.95,
            f"Frame: {frame_idx}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.append_data(img)

    writer.close()
    plt.close(fig)


def main():
    args = parse_args()
    zarr_path = Path(args.zarr).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    root = zarr.open(str(zarr_path), mode="r")
    if "data" not in root or "meta" not in root:
        raise KeyError(f"Invalid zarr structure: {zarr_path}")
    data = root["data"]
    meta = root["meta"]
    if "point_cloud" not in data:
        raise KeyError(f"{zarr_path} missing data/point_cloud")
    if "episode_ends" not in meta:
        raise KeyError(f"{zarr_path} missing meta/episode_ends")

    point_cloud = data["point_cloud"]
    episode_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
    selected = parse_episode_list(args.episodes, len(episode_ends))
    bounds = list(iter_episode_bounds(episode_ends))

    for ep_idx in selected:
        start, end = bounds[ep_idx]
        frames = [np.asarray(p, dtype=np.float32) for p in point_cloud[start:end: max(1, args.stride)]]
        out_file = output_dir / f"episode_{ep_idx}.mp4"
        point_cloud_to_video(
            frames,
            out_file,
            fps=args.fps,
            elev=args.elev,
            azim=args.azim,
            point_size=args.point_size,
        )
        print(f"Saved point-cloud video: {out_file}")


if __name__ == "__main__":
    main()
