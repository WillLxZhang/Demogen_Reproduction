#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
ROLLOUT_EPOCH_RE = re.compile(r"\bEpoch\s+(\d+)\s+Rollouts took\b")
SUCCESS_RATE_RE = re.compile(r'"Success_Rate"\s*:\s*([0-9.]+)')
CHECKPOINT_RE = re.compile(r"model_epoch_(\d+)\.pth")


@dataclass
class WatchState:
    log_offset: int = 0
    rollout_epoch: int = 0
    checkpoint_epoch: int = 0
    last_success_key: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch a robomimic training run and send desktop notifications on new progress."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Robomimic timestamped run directory that contains logs/ and models/.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--start-at",
        choices=["end", "beginning"],
        default="end",
        help="Watch only new progress from now, or replay existing history first.",
    )
    parser.add_argument(
        "--title",
        default="Robomimic Progress",
        help="Desktop notification title prefix.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print events only and skip notify-send.",
    )
    parser.add_argument(
        "--bell",
        action="store_true",
        help="Also print a terminal bell on each event.",
    )
    return parser.parse_args()


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def normalize_chunks(text: str) -> list[str]:
    return [part.strip() for part in strip_ansi(text).replace("\r", "\n").splitlines()]


def send_notification(title: str, body: str, *, dry_run: bool, bell: bool, critical: bool = False) -> None:
    timestamp = time.strftime("%F %T")
    print(f"[{timestamp}] {title}: {body}", flush=True)
    if bell:
        print("\a", end="", flush=True)
    if dry_run:
        return
    try:
        subprocess.run(
            [
                "notify-send",
                "-u",
                "critical" if critical else "normal",
                title,
                body,
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass


def existing_checkpoint_epochs(models_dir: Path) -> set[int]:
    epochs: set[int] = set()
    if not models_dir.exists():
        return epochs
    for path in models_dir.glob("model_epoch_*.pth"):
        match = CHECKPOINT_RE.search(path.name)
        if match:
            epochs.add(int(match.group(1)))
    return epochs


def process_log_text(
    text: str,
    state: WatchState,
    *,
    title: str,
    dry_run: bool,
    bell: bool,
) -> None:
    for raw_line in normalize_chunks(text):
        if not raw_line:
            continue

        match = ROLLOUT_EPOCH_RE.search(raw_line)
        if match:
            epoch = int(match.group(1))
            if epoch >= state.rollout_epoch:
                state.rollout_epoch = epoch

        match = SUCCESS_RATE_RE.search(raw_line)
        if match and state.rollout_epoch > 0:
            success_rate = match.group(1)
            key = f"{state.rollout_epoch}:{success_rate}"
            if key != state.last_success_key:
                state.last_success_key = key
                send_notification(
                    title,
                    f"Epoch {state.rollout_epoch} rollout Success_Rate={success_rate}",
                    dry_run=dry_run,
                    bell=bell,
                )

        if "save checkpoint to" in raw_line:
            match = CHECKPOINT_RE.search(raw_line)
            if match:
                epoch = int(match.group(1))
                if epoch > state.checkpoint_epoch:
                    state.checkpoint_epoch = epoch
                    send_notification(
                        title,
                        f"Saved checkpoint model_epoch_{epoch}.pth",
                        dry_run=dry_run,
                        bell=bell,
                    )


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    log_path = run_dir / "logs" / "log.txt"
    models_dir = run_dir / "models"

    state = WatchState()
    if args.start_at == "end":
        if log_path.exists():
            state.log_offset = log_path.stat().st_size
        existing_epochs = existing_checkpoint_epochs(models_dir)
        if existing_epochs:
            state.checkpoint_epoch = max(existing_epochs)

    stop = False

    def handle_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    print(f"[{time.strftime('%F %T')}] {args.title}: Watching {run_dir}", flush=True)

    seen_model_epochs = existing_checkpoint_epochs(models_dir)

    while not stop:
        if log_path.exists():
            current_size = log_path.stat().st_size
            if current_size < state.log_offset:
                state.log_offset = 0
            if current_size > state.log_offset:
                with log_path.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(state.log_offset)
                    text = f.read()
                    state.log_offset = f.tell()
                process_log_text(
                    text,
                    state,
                    title=args.title,
                    dry_run=args.dry_run,
                    bell=args.bell,
                )

        for epoch in sorted(existing_checkpoint_epochs(models_dir)):
            if epoch not in seen_model_epochs:
                seen_model_epochs.add(epoch)
                if epoch > state.checkpoint_epoch:
                    state.checkpoint_epoch = epoch
                    send_notification(
                        args.title,
                        f"Checkpoint file appeared: model_epoch_{epoch}.pth",
                        dry_run=args.dry_run,
                        bell=args.bell,
                    )

        time.sleep(max(0.2, float(args.poll_seconds)))

    print(f"[{time.strftime('%F %T')}] {args.title}: Watcher stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
