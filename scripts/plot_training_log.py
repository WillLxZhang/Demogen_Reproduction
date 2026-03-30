#!/usr/bin/env python3
"""
Plot training curves from DemoGen diffusion policy logs.json.txt.

Example:
    conda run -n demogen python scripts/plot_training_log.py \
        --log repos/DemoGen/data/ckpts/lift_test_9-dp3-seed0/logs.json.txt \
        --output-dir outputs/training_viz/lift_test_9
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to logs.json.txt")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save plots. Defaults to <log_dir>/viz",
    )
    parser.add_argument(
        "--max-step-points",
        type=int,
        default=2000,
        help="Maximum points to keep in step-level train loss plot.",
    )
    parser.add_argument(
        "--latest-run-only",
        action="store_true",
        help=(
            "Keep only the latest appended run segment after the last global_step reset. "
            "Useful when multiple resume attempts were appended into the same log."
        ),
    )
    parser.add_argument(
        "--min-line",
        type=int,
        default=None,
        help="Keep only rows at or after this 1-based line number in logs.json.txt.",
    )
    parser.add_argument(
        "--max-line",
        type=int,
        default=None,
        help="Keep only rows at or before this 1-based line number in logs.json.txt.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Ignore partial final lines from interrupted runs.
                continue
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def take_last_per_epoch(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    epoch_to_row: dict[int, dict[str, Any]] = {}
    for row in rows:
        epoch = row.get("epoch")
        if epoch is None:
            continue
        epoch_to_row[int(epoch)] = row
    return [epoch_to_row[k] for k in sorted(epoch_to_row.keys())]


def maybe_downsample(xs: np.ndarray, ys: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(xs) <= max_points:
        return xs, ys
    idx = np.linspace(0, len(xs) - 1, num=max_points, dtype=int)
    return xs[idx], ys[idx]


def keep_latest_run_segment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows

    last_reset_idx = 0
    prev_global_step: int | None = None
    for idx, row in enumerate(rows):
        global_step = row.get("global_step")
        if global_step is None:
            continue
        global_step = int(global_step)
        if prev_global_step is not None and global_step < prev_global_step:
            last_reset_idx = idx
        prev_global_step = global_step
    return rows[last_reset_idx:]


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    latest_run_only: bool,
    min_line: int | None,
    max_line: int | None,
) -> list[dict[str, Any]]:
    filtered = rows
    if latest_run_only:
        filtered = keep_latest_run_segment(filtered)

    if min_line is not None:
        filtered = [row for row in filtered if int(row.get("_line_number", 0)) >= min_line]

    if max_line is not None:
        filtered = [row for row in filtered if int(row.get("_line_number", 0)) <= max_line]

    return filtered


def plot_step_loss(rows: list[dict[str, Any]], output_path: Path, max_points: int) -> None:
    xs = []
    ys = []
    for row in rows:
        if "global_step" in row and "train_loss" in row:
            xs.append(row["global_step"])
            ys.append(row["train_loss"])
    if not xs:
        return
    x = np.asarray(xs, dtype=np.int64)
    y = np.asarray(ys, dtype=np.float32)
    x, y = maybe_downsample(x, y, max_points=max_points)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, y, linewidth=0.8, color="#4C78A8")
    ax.set_title("Train Loss vs Global Step")
    ax.set_xlabel("global_step")
    ax.set_ylabel("train_loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_epoch_metrics(epoch_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not epoch_rows:
        return

    epochs = np.asarray([int(row["epoch"]) for row in epoch_rows], dtype=np.int64)
    train_loss = np.asarray([float(row["train_loss"]) for row in epoch_rows], dtype=np.float32)
    val_loss = np.asarray(
        [float(row["val_loss"]) if "val_loss" in row else np.nan for row in epoch_rows],
        dtype=np.float32,
    )
    action_mse = np.asarray(
        [float(row["train_action_mse_error"]) if "train_action_mse_error" in row else np.nan for row in epoch_rows],
        dtype=np.float32,
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(epochs, train_loss, marker="o", linewidth=1.2, label="train_loss", color="#4C78A8")
    if np.any(np.isfinite(val_loss)):
        axes[0].plot(epochs, val_loss, marker="o", linewidth=1.2, label="val_loss", color="#E45756")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Epoch Metrics")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    if np.any(np.isfinite(action_mse)):
        axes[1].plot(
            epochs,
            action_mse,
            marker="o",
            linewidth=1.2,
            label="train_action_mse_error",
            color="#54A24B",
        )
        axes[1].legend()
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("action mse")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_summary(epoch_rows: list[dict[str, Any]], output_path: Path) -> None:
    lines = []
    if not epoch_rows:
        lines.append("No epoch-level rows found.")
        output_path.write_text("\n".join(lines) + "\n")
        return

    last = epoch_rows[-1]
    lines.append(f"last_epoch: {last.get('epoch')}")
    if "train_loss" in last:
        lines.append(f"last_train_loss: {float(last['train_loss']):.6f}")
    if "val_loss" in last:
        lines.append(f"last_val_loss: {float(last['val_loss']):.6f}")
    if "train_action_mse_error" in last:
        lines.append(f"last_train_action_mse_error: {float(last['train_action_mse_error']):.6f}")

    val_candidates = [row for row in epoch_rows if "val_loss" in row]
    if val_candidates:
        best = min(val_candidates, key=lambda row: float(row["val_loss"]))
        lines.append(f"best_val_epoch: {int(best['epoch'])}")
        lines.append(f"best_val_loss: {float(best['val_loss']):.6f}")

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    log_path = Path(args.log).expanduser().resolve()
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else log_path.parent / "viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(log_path)
    rows = filter_rows(
        rows,
        latest_run_only=args.latest_run_only,
        min_line=args.min_line,
        max_line=args.max_line,
    )
    epoch_rows = take_last_per_epoch(rows)

    plot_step_loss(rows, output_dir / "train_loss_steps.png", max_points=args.max_step_points)
    plot_epoch_metrics(epoch_rows, output_dir / "epoch_metrics.png")
    write_summary(epoch_rows, output_dir / "summary.txt")

    if rows:
        first_line = rows[0].get("_line_number")
        last_line = rows[-1].get("_line_number")
        print(
            f"Saved training visualizations to: {output_dir} "
            f"(filtered rows={len(rows)}, lines={first_line}-{last_line})"
        )
    else:
        print(f"Saved training visualizations to: {output_dir} (no rows after filtering)")


if __name__ == "__main__":
    main()
