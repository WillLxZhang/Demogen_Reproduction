#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for a checkpoint, stop a parent automation process, optionally wait "
            "for a running training pid to exit, then launch the next command."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-pid", type=int, default=None)
    parser.add_argument("--wait-pid", type=int, default=None)
    parser.add_argument("--log", required=True)
    parser.add_argument("--poll-sec", type=float, default=15.0)
    parser.add_argument("--wait-timeout-sec", type=float, default=600.0)
    parser.add_argument("--stop-signal", choices=["TERM", "INT", "KILL"], default="TERM")
    parser.add_argument("--launch-cwd", default=None)
    parser.add_argument("--launch-log", default=None)
    parser.add_argument("launch_cmd", nargs=argparse.REMAINDER)
    return parser.parse_args()


def log_line(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%F %T')}] {message}\n")


def pid_exists(pid: int | None) -> bool:
    if pid is None:
        return False
    return Path(f"/proc/{pid}").exists()


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    log_path = Path(args.log).expanduser().resolve()
    launch_cwd = Path(args.launch_cwd).expanduser().resolve() if args.launch_cwd else None
    launch_log = Path(args.launch_log).expanduser().resolve() if args.launch_log else None
    launch_cmd = list(args.launch_cmd)
    if launch_cmd and launch_cmd[0] == "--":
        launch_cmd = launch_cmd[1:]

    sig = {
        "TERM": signal.SIGTERM,
        "INT": signal.SIGINT,
        "KILL": signal.SIGKILL,
    }[args.stop_signal]

    log_line(
        log_path,
        (
            f"chain_guard_start checkpoint={checkpoint} target_pid={args.target_pid} "
            f"wait_pid={args.wait_pid} stop_signal={args.stop_signal} "
            f"poll_sec={args.poll_sec} wait_timeout_sec={args.wait_timeout_sec}"
        ),
    )
    if launch_cmd:
        log_line(log_path, f"launch_cmd={' '.join(launch_cmd)}")

    while True:
        if checkpoint.exists():
            log_line(log_path, f"checkpoint_detected path={checkpoint}")
            break
        if args.target_pid is not None and not pid_exists(args.target_pid):
            log_line(log_path, f"target_missing_before_checkpoint target_pid={args.target_pid}")
            return 1
        time.sleep(args.poll_sec)

    if args.target_pid is None:
        log_line(log_path, "target_pid_not_set skip_stop")
    else:
        if pid_exists(args.target_pid):
            log_line(log_path, f"sending_{args.stop_signal} target_pid={args.target_pid}")
            try:
                os.kill(args.target_pid, sig)
            except ProcessLookupError:
                log_line(log_path, "target_already_gone")
        else:
            log_line(log_path, "target_already_missing_at_stop")

    if args.wait_pid is not None:
        deadline = time.time() + args.wait_timeout_sec
        while pid_exists(args.wait_pid) and time.time() < deadline:
            time.sleep(min(args.poll_sec, 5.0))
        if pid_exists(args.wait_pid):
            log_line(log_path, f"wait_pid_still_running wait_pid={args.wait_pid}")
        else:
            log_line(log_path, f"wait_pid_exited wait_pid={args.wait_pid}")

    if not launch_cmd:
        log_line(log_path, "no_launch_cmd guard_exit")
        return 0

    stdout_handle = None
    try:
        if launch_log is not None:
            launch_log.parent.mkdir(parents=True, exist_ok=True)
            stdout_handle = launch_log.open("a", encoding="utf-8")
            stdout_handle.write(f"[{time.strftime('%F %T')}] $ {' '.join(launch_cmd)}\n\n")
            stdout_handle.flush()

        proc = subprocess.Popen(
            launch_cmd,
            cwd=str(launch_cwd) if launch_cwd else None,
            stdout=stdout_handle if stdout_handle is not None else subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        log_line(log_path, f"launch_started pid={proc.pid}")
    finally:
        if stdout_handle is not None:
            stdout_handle.close()

    log_line(log_path, "chain_guard_exit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
