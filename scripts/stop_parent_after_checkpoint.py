#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for a checkpoint file, then stop a parent process without touching training."
    )
    parser.add_argument("--target-pid", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--train-pid", type=int, default=None)
    parser.add_argument("--poll-sec", type=float, default=15.0)
    parser.add_argument("--signal", default="TERM", choices=["TERM", "INT", "KILL"])
    return parser.parse_args()


def log_line(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%F %T')}] {message}\n")


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    log_path = Path(args.log).expanduser().resolve()
    sig = {
        "TERM": signal.SIGTERM,
        "INT": signal.SIGINT,
        "KILL": signal.SIGKILL,
    }[args.signal]

    log_line(
        log_path,
        (
            f"guard_start target_pid={args.target_pid} train_pid={args.train_pid} "
            f"checkpoint={checkpoint} signal={args.signal} poll_sec={args.poll_sec}"
        ),
    )

    while True:
        if not Path(f"/proc/{args.target_pid}").exists():
            log_line(log_path, "target_missing_exit")
            return 0

        if checkpoint.exists():
            log_line(log_path, f"checkpoint_detected sending {args.signal} to target_pid={args.target_pid}")
            try:
                os.kill(args.target_pid, sig)
            except ProcessLookupError:
                log_line(log_path, "target_already_gone")
                return 0

            time.sleep(5)
            if args.train_pid is not None:
                train_status_path = Path(f"/proc/{args.train_pid}/status")
                if train_status_path.exists():
                    status_lines = []
                    for line in train_status_path.read_text(encoding="utf-8").splitlines():
                        if line.startswith(("Pid:", "PPid:", "State:")):
                            status_lines.append(line)
                    log_line(log_path, "train_status_after_target_kill " + " | ".join(status_lines))
                else:
                    log_line(log_path, "train_pid_missing_after_target_kill")
            log_line(log_path, "guard_exit")
            return 0

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
