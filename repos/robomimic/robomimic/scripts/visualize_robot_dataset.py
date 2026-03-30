#!/usr/bin/env python3
"""
Offline visualization utility for robosuite / robomimic HDF5 datasets.

The script does not require a simulator window. It inspects the dataset
structure, summarizes trajectory lengths and actions, and saves plots for:

- trajectory lengths across demos
- action traces and action norms
- state heatmaps
- key low-dimensional observations
- image / depth contact sheets from obs or next_obs
- top-down XY paths for end-effector and object positions when available

Example:
    conda run -n robomimic python scripts/visualize_robot_dataset.py \
        --dataset data/raw/lift_keyboard/1774195240_3114173/demo.hdf5 \
        --output-dir outputs/dataset_viz/demo
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ACTION_LABELS_7 = [
    "dx",
    "dy",
    "dz",
    "drot_x",
    "drot_y",
    "drot_z",
    "gripper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to input HDF5 dataset")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save visualizations. Defaults to outputs/dataset_viz/<dataset_name>",
    )
    parser.add_argument(
        "--demo-key",
        default=None,
        help="Episode to visualize. Defaults to the first demo in numeric order.",
    )
    parser.add_argument(
        "--obs-group",
        default=None,
        choices=["obs", "next_obs"],
        help="Observation group to visualize. Defaults to obs, otherwise next_obs if available.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=12,
        help="Number of frames to sample for each contact sheet.",
    )
    parser.add_argument(
        "--max-low-dim-keys",
        type=int,
        default=8,
        help="Maximum number of low-dimensional observation keys to visualize.",
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=8,
        help="Maximum number of channels to plot for a low-dimensional key.",
    )
    parser.add_argument(
        "--export-videos",
        action="store_true",
        help="If set, export MP4 videos for image observations.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=20,
        help="FPS for exported videos.",
    )
    parser.add_argument(
        "--video-stride",
        type=int,
        default=1,
        help="Use every N-th frame when exporting videos.",
    )
    return parser.parse_args()


def numeric_suffix(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    if match:
        return int(match.group(1)), name
    return math.inf, name


def sorted_keys(group: h5py.Group) -> list[str]:
    return sorted(list(group.keys()), key=numeric_suffix)


def to_python(value: Any, max_str_len: int = 200) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    elif isinstance(value, np.generic):
        value = value.item()
    elif isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value.item()
        else:
            value = value.tolist()

    if isinstance(value, str) and len(value) > max_str_len:
        return value[: max_str_len - 3] + "..."
    return value


def infer_dataset_type(root: h5py.File, demo_group: h5py.Group) -> str:
    data_attrs = root["data"].attrs
    has_env_args = "env_args" in data_attrs
    has_env_info = "env_info" in data_attrs
    has_obs = "obs" in demo_group
    has_next_obs = "next_obs" in demo_group
    has_states = "states" in demo_group

    if has_env_info and has_states and not has_obs and not has_next_obs:
        return "robosuite_raw"
    if has_env_args and has_obs:
        return "robomimic"
    if has_env_info and (has_obs or has_next_obs):
        return "robosuite_or_hybrid_with_obs"
    return "generic_hdf5"


def collect_structure(group: h5py.Group, prefix: str = "") -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key in sorted_keys(group):
        obj = group[key]
        path = f"{prefix}{key}"
        if isinstance(obj, h5py.Dataset):
            items.append(
                {
                    "path": path,
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                }
            )
        else:
            items.extend(collect_structure(obj, prefix=f"{path}/"))
    return items


def find_first_length(group: h5py.Group) -> int | None:
    for key in sorted_keys(group):
        obj = group[key]
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 1:
            return int(obj.shape[0])
        if isinstance(obj, h5py.Group):
            length = find_first_length(obj)
            if length is not None:
                return length
    return None


def get_demo_length(demo_group: h5py.Group) -> int:
    if "actions" in demo_group and isinstance(demo_group["actions"], h5py.Dataset):
        return int(demo_group["actions"].shape[0])
    if "states" in demo_group and isinstance(demo_group["states"], h5py.Dataset):
        return int(demo_group["states"].shape[0])
    if "num_samples" in demo_group.attrs:
        return int(demo_group.attrs["num_samples"])
    length = find_first_length(demo_group)
    if length is None:
        raise ValueError("Unable to infer trajectory length from demo group")
    return length


def pick_obs_group(demo_group: h5py.Group, preferred: str | None) -> str | None:
    if preferred is not None and preferred in demo_group:
        return preferred
    if "obs" in demo_group:
        return "obs"
    if "next_obs" in demo_group:
        return "next_obs"
    return None


def is_image_array(dataset: h5py.Dataset) -> bool:
    return dataset.ndim == 4 and dataset.shape[-1] in (1, 3, 4)


def is_low_dim_array(dataset: h5py.Dataset) -> bool:
    return dataset.ndim == 2


def sample_indices(length: int, count: int) -> np.ndarray:
    count = max(1, min(length, count))
    return np.unique(np.linspace(0, length - 1, num=count, dtype=int))


def prepare_rgb_frame(frame: np.ndarray, frame_min: float | None = None, frame_max: float | None = None) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]

    if frame.ndim == 2:
        finite_mask = np.isfinite(frame)
        if not np.any(finite_mask):
            norm = np.zeros_like(frame, dtype=np.float32)
        else:
            lo = np.min(frame[finite_mask]) if frame_min is None else frame_min
            hi = np.max(frame[finite_mask]) if frame_max is None else frame_max
            scale = hi - lo
            if scale <= 1e-12:
                norm = np.zeros_like(frame, dtype=np.float32)
            else:
                norm = (frame.astype(np.float32) - lo) / scale
            norm = np.clip(norm, 0.0, 1.0)
        rgb = plt.get_cmap("viridis")(norm)[..., :3]
        return (rgb * 255).astype(np.uint8)

    if frame.dtype == np.uint8:
        if frame.shape[-1] == 4:
            return frame[..., :3]
        return frame

    frame = frame.astype(np.float32)
    max_value = np.nanmax(frame)
    if max_value > 1.0:
        frame = frame / 255.0
    frame = np.clip(frame, 0.0, 1.0)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return (frame * 255).astype(np.uint8)


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_text(text: str, path: Path) -> None:
    path.write_text(text, encoding="utf-8")


def plot_lengths(demo_keys: list[str], lengths: list[int], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(demo_keys) * 1.2), 4))
    ax.bar(np.arange(len(demo_keys)), lengths, color="#4C78A8")
    ax.set_title("Trajectory Lengths")
    ax.set_xlabel("Demo")
    ax.set_ylabel("Steps")
    ax.set_xticks(np.arange(len(demo_keys)))
    ax.set_xticklabels(demo_keys, rotation=45, ha="right")
    for idx, length in enumerate(lengths):
        ax.text(idx, length, str(length), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_action_summary(actions: np.ndarray, path: Path) -> dict[str, Any]:
    actions = np.asarray(actions, dtype=np.float32)
    num_steps = actions.shape[0]
    num_dims = actions.shape[1] if actions.ndim == 2 else 1
    labels = ACTION_LABELS_7 if num_dims == 7 else [f"a{i}" for i in range(num_dims)]
    x = np.arange(num_steps)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for dim in range(num_dims):
        axes[0].plot(x, actions[:, dim], linewidth=0.9, label=labels[dim])
    axes[0].set_title("Action Traces")
    axes[0].set_ylabel("Value")
    axes[0].grid(alpha=0.25)
    if num_dims <= 10:
        axes[0].legend(loc="upper right", ncol=2)

    summary: dict[str, Any] = {
        "shape": list(actions.shape),
        "per_dim_min": np.min(actions, axis=0).round(6).tolist(),
        "per_dim_max": np.max(actions, axis=0).round(6).tolist(),
        "per_dim_mean_abs": np.mean(np.abs(actions), axis=0).round(6).tolist(),
    }

    if num_dims >= 3:
        pos_norm = np.linalg.norm(actions[:, :3], axis=1)
        axes[1].plot(x, pos_norm, color="#54A24B", linewidth=1.0, label="||dpos||")
        summary["position_norm"] = {
            "mean": float(np.mean(pos_norm)),
            "max": float(np.max(pos_norm)),
            "fraction_lt_1e-4": float(np.mean(pos_norm < 1e-4)),
            "fraction_lt_1e-3": float(np.mean(pos_norm < 1e-3)),
            "fraction_lt_5e-3": float(np.mean(pos_norm < 5e-3)),
        }
    if num_dims >= 6:
        rot_norm = np.linalg.norm(actions[:, 3:6], axis=1)
        axes[1].plot(x, rot_norm, color="#E45756", linewidth=1.0, label="||drot||")
        summary["rotation_norm"] = {
            "mean": float(np.mean(rot_norm)),
            "max": float(np.max(rot_norm)),
            "fraction_lt_1e-4": float(np.mean(rot_norm < 1e-4)),
            "fraction_lt_1e-3": float(np.mean(rot_norm < 1e-3)),
            "fraction_lt_1e-2": float(np.mean(rot_norm < 1e-2)),
        }
    if num_dims >= 7:
        gripper = actions[:, 6]
        axes[1].plot(x, gripper, color="#B279A2", linewidth=1.0, label="gripper")
        summary["gripper"] = {
            "min": float(np.min(gripper)),
            "max": float(np.max(gripper)),
            "mean": float(np.mean(gripper)),
        }

    axes[1].set_title("Action Magnitudes")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Norm / Value")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return summary


def plot_state_heatmap(states: np.ndarray, path: Path) -> None:
    states = np.asarray(states, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(states.T, aspect="auto", interpolation="nearest", cmap="coolwarm")
    ax.set_title("States Heatmap")
    ax.set_xlabel("Step")
    ax.set_ylabel("State Dimension")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_low_dim(key: str, values: np.ndarray, path: Path, max_channels: int) -> None:
    values = np.asarray(values, dtype=np.float32)
    dims = values.shape[1]
    x = np.arange(values.shape[0])
    shown_dims = min(dims, max_channels)

    if dims <= max_channels:
        fig, axes = plt.subplots(shown_dims, 1, figsize=(14, max(2.5, shown_dims * 1.7)), sharex=True)
        if shown_dims == 1:
            axes = [axes]
        for dim in range(shown_dims):
            axes[dim].plot(x, values[:, dim], linewidth=0.9)
            axes[dim].set_ylabel(f"c{dim}")
            axes[dim].grid(alpha=0.25)
        axes[0].set_title(key)
        axes[-1].set_xlabel("Step")
    else:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        for dim in range(shown_dims):
            axes[0].plot(x, values[:, dim], linewidth=0.9, label=f"c{dim}")
        axes[0].set_title(f"{key} (first {shown_dims} channels)")
        axes[0].set_ylabel("Value")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="upper right", ncol=2)
        im = axes[1].imshow(values.T, aspect="auto", interpolation="nearest", cmap="coolwarm")
        axes[1].set_title(f"{key} heatmap (all {dims} channels)")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Channel")
        fig.colorbar(im, ax=axes[1], shrink=0.8)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_topdown(obs_group: h5py.Group, path: Path) -> bool:
    if "robot0_eef_pos" not in obs_group or "object" not in obs_group:
        return False

    eef = np.asarray(obs_group["robot0_eef_pos"][...], dtype=np.float32)
    obj = np.asarray(obs_group["object"][...], dtype=np.float32)
    if eef.ndim != 2 or eef.shape[1] < 2 or obj.ndim != 2 or obj.shape[1] < 2:
        return False

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(eef[:, 0], eef[:, 1], label="eef path", color="#4C78A8", linewidth=1.2)
    ax.scatter(eef[0, 0], eef[0, 1], color="#4C78A8", marker="o", s=50, label="eef start")
    ax.scatter(eef[-1, 0], eef[-1, 1], color="#4C78A8", marker="x", s=50, label="eef end")

    ax.plot(obj[:, 0], obj[:, 1], label="object path", color="#F58518", linewidth=1.2)
    ax.scatter(obj[0, 0], obj[0, 1], color="#F58518", marker="o", s=50, label="object start")
    ax.scatter(obj[-1, 0], obj[-1, 1], color="#F58518", marker="x", s=50, label="object end")

    ax.set_title("Top-Down XY Path")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def save_contact_sheet(key: str, frames: np.ndarray, path: Path, num_frames: int) -> None:
    frames = np.asarray(frames)
    indices = sample_indices(frames.shape[0], num_frames)
    is_depth = frames.ndim == 4 and frames.shape[-1] == 1
    frame_min = float(np.min(frames)) if is_depth else None
    frame_max = float(np.max(frames)) if is_depth else None

    cols = min(4, len(indices))
    rows = int(math.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.0))
    axes = np.array(axes).reshape(-1)

    for ax in axes[len(indices) :]:
        ax.axis("off")

    for ax, idx in zip(axes, indices):
        frame = prepare_rgb_frame(frames[idx], frame_min=frame_min, frame_max=frame_max)
        ax.imshow(frame)
        ax.set_title(f"{key} @ {idx}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def export_video(key: str, frames: np.ndarray, path: Path, fps: int, stride: int) -> None:
    frames = np.asarray(frames)
    is_depth = frames.ndim == 4 and frames.shape[-1] == 1
    frame_min = float(np.min(frames)) if is_depth else None
    frame_max = float(np.max(frames)) if is_depth else None

    writer = imageio.get_writer(path, fps=fps)
    try:
        for idx in range(0, frames.shape[0], max(1, stride)):
            writer.append_data(prepare_rgb_frame(frames[idx], frame_min=frame_min, frame_max=frame_max))
    finally:
        writer.close()


def build_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        f"Dataset: {summary['dataset_path']}",
        f"Dataset type: {summary['dataset_type']}",
        f"Trajectories: {summary['num_trajectories']}",
        f"Selected demo: {summary['selected_demo']}",
        f"Selected length: {summary['selected_length']}",
        f"Observation group: {summary['obs_group']}",
        f"Image keys: {', '.join(summary['image_keys']) if summary['image_keys'] else '(none)'}",
        f"Low-dim keys: {', '.join(summary['low_dim_keys']) if summary['low_dim_keys'] else '(none)'}",
        "",
        "Trajectory statistics:",
        json.dumps(summary["trajectory_stats"], indent=2, ensure_ascii=False),
        "",
    ]
    if "action_summary" in summary:
        lines.extend(
            [
                "Action summary:",
                json.dumps(summary["action_summary"], indent=2, ensure_ascii=False),
                "",
            ]
        )
    lines.extend(
        [
            "Top-level attrs:",
            json.dumps(summary["top_level_attrs"], indent=2, ensure_ascii=False),
            "",
            "Selected demo attrs:",
            json.dumps(summary["demo_attrs"], indent=2, ensure_ascii=False),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    if args.output_dir is None:
        output_dir = Path("outputs") / "dataset_viz" / dataset_path.stem
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{dataset_path} does not contain a top-level 'data' group")

        demo_keys = sorted_keys(f["data"])
        if not demo_keys:
            raise ValueError(f"{dataset_path} does not contain any demos under /data")

        selected_demo = args.demo_key or demo_keys[0]
        if selected_demo not in f["data"]:
            raise KeyError(f"Demo '{selected_demo}' not found. Available demos: {demo_keys}")

        lengths = [get_demo_length(f["data"][demo_key]) for demo_key in demo_keys]
        plot_lengths(demo_keys, lengths, output_dir / "trajectory_lengths.png")

        demo_group = f["data"][selected_demo]
        dataset_type = infer_dataset_type(f, demo_group)
        obs_group_name = pick_obs_group(demo_group, args.obs_group)

        structure = collect_structure(demo_group)
        top_level_attrs = {key: to_python(value) for key, value in f["data"].attrs.items()}
        demo_attrs = {key: to_python(value) for key, value in demo_group.attrs.items()}

        image_keys: list[str] = []
        low_dim_keys: list[str] = []

        if "actions" in demo_group:
            actions = np.asarray(demo_group["actions"][...], dtype=np.float32)
            action_summary = plot_action_summary(actions, output_dir / "actions.png")
        else:
            action_summary = None

        if "states" in demo_group:
            plot_state_heatmap(demo_group["states"][...], output_dir / "states_heatmap.png")

        if obs_group_name is not None:
            obs_group = demo_group[obs_group_name]
            for key in sorted_keys(obs_group):
                dataset = obs_group[key]
                if not isinstance(dataset, h5py.Dataset):
                    continue

                if is_image_array(dataset):
                    image_keys.append(key)
                    save_contact_sheet(
                        key=key,
                        frames=dataset[...],
                        path=output_dir / f"{obs_group_name}_{key}_contact_sheet.png",
                        num_frames=args.num_frames,
                    )
                    if args.export_videos:
                        export_video(
                            key=key,
                            frames=dataset[...],
                            path=output_dir / f"{obs_group_name}_{key}.mp4",
                            fps=args.video_fps,
                            stride=args.video_stride,
                        )
                elif is_low_dim_array(dataset) and len(low_dim_keys) < args.max_low_dim_keys:
                    low_dim_keys.append(key)
                    plot_low_dim(
                        key=key,
                        values=dataset[...],
                        path=output_dir / f"{obs_group_name}_{key}.png",
                        max_channels=args.max_channels,
                    )

            plot_topdown(obs_group, output_dir / f"{obs_group_name}_topdown_xy.png")

        summary = {
            "dataset_path": str(dataset_path),
            "dataset_type": dataset_type,
            "num_trajectories": len(demo_keys),
            "trajectory_lengths": {demo_key: int(length) for demo_key, length in zip(demo_keys, lengths)},
            "trajectory_stats": {
                "sum": int(np.sum(lengths)),
                "mean": float(np.mean(lengths)),
                "std": float(np.std(lengths)),
                "min": int(np.min(lengths)),
                "max": int(np.max(lengths)),
            },
            "selected_demo": selected_demo,
            "selected_length": int(get_demo_length(demo_group)),
            "obs_group": obs_group_name,
            "image_keys": image_keys,
            "low_dim_keys": low_dim_keys,
            "top_level_attrs": top_level_attrs,
            "demo_attrs": demo_attrs,
            "structure": structure,
        }
        if action_summary is not None:
            summary["action_summary"] = action_summary

        save_json(summary, output_dir / "summary.json")
        save_text(build_summary_text(summary), output_dir / "summary.txt")

    print(f"Saved visualizations to: {output_dir}")
    print(f"Summary: {output_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
