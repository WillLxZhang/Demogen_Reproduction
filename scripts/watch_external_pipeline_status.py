#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


def now_str() -> str:
    return time.strftime("%F %T")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def log_line(path: Path, message: str) -> None:
    line = f"[{now_str()}] {message}"
    print(line, flush=True)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_ps() -> list[dict[str, str]]:
    out = subprocess.check_output(["ps", "-eo", "pid=,etimes=,args="], text=True)
    rows: list[dict[str, str]] = []
    for raw in out.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        rows.append({"pid": parts[0], "etimes": parts[1], "args": parts[2]})
    return rows


def find_matching_procs(rows: list[dict[str, str]], external_root: str) -> list[dict[str, str]]:
    matched = []
    for row in rows:
        args = row["args"]
        if external_root in args and (
            "run_nutassemblyround_twostage_replayobs_train120_external.py" in args
            or "run_twostage_raw_hdf5_pipeline.py" in args
            or "gen_demo.py" in args
            or "export_stack_solved_from_template_zarr_relalign_twostage.py" in args
            or "train.py" in args
            or "eval_robomimic_checkpoint_multiseed.py" in args
            or "eval_robomimic_checkpoint_custom_reset.py" in args
            or "watch_robomimic_progress_notify.py" in args
        ):
            matched.append(row)
    return matched


def infer_current_stage(pipeline_log: Path) -> str:
    if not pipeline_log.exists():
        return "not_started"
    current = "unknown"
    for raw in pipeline_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "STEP START " in raw:
            current = raw.split("STEP START ", 1)[1].split(":", 1)[0].strip()
        elif "STEP DONE " in raw:
            done_name = raw.split("STEP DONE ", 1)[1].split(":", 1)[0].strip()
            if current == done_name:
                current = f"after_{done_name}"
        elif "TRAIN START:" in raw:
            current = "train"
        elif "PIPELINE DONE:" in raw:
            current = "done"
    return current


def pipeline_done(pipeline_log: Path) -> bool:
    if not pipeline_log.exists():
        return False
    return "PIPELINE DONE:" in pipeline_log.read_text(encoding="utf-8", errors="replace")


def latest_step_log(step_log_root: Path) -> Path | None:
    if not step_log_root.exists():
        return None
    files = [p for p in step_log_root.iterdir() if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def safe_du_size(path: Path) -> str:
    try:
        out = subprocess.check_output(["du", "-sh", str(path)], text=True)
        return out.split()[0]
    except Exception:
        return "unknown"


def query_gpu_snapshot() -> list[dict[str, Any]]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return []

    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "temp_c": float(parts[2]),
                "util_pct": float(parts[3]),
                "mem_used_mib": float(parts[4]),
                "mem_total_mib": float(parts[5]),
                "power_w": float(parts[6]),
            }
        )
    return rows


def format_gpu(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "gpu=unavailable"
    return "; ".join(
        (
            f"gpu{row['index']} temp={row['temp_c']:.1f}C util={row['util_pct']:.1f}% "
            f"mem={row['mem_used_mib']:.0f}/{row['mem_total_mib']:.0f}MiB power={row['power_w']:.1f}W"
        )
        for row in rows
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight watcher for a long-running external pipeline."
    )
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument(
        "--stall-minutes",
        type=float,
        default=30.0,
        help="Warn if the latest step log has not changed for this many minutes while the pipeline is still running.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    external_root = Path(args.external_root).expanduser().resolve()
    pipeline_log = external_root / "logs" / "pipeline.log"
    step_log_root = external_root / "logs" / "steps"
    watcher_log = external_root / "logs" / "pipeline_status_watcher.log"
    state_path = external_root / "logs" / "pipeline_status_state.json"
    lock_path = external_root / "logs" / "pipeline_status_watcher.lock"

    ensure_dir(lock_path.parent)
    lock_file = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(
            f"[{now_str()}] WATCHER ALREADY RUNNING for external_root={external_root}",
            flush=True,
        )
        return 2
    lock_file.write(f"{os.getpid()}\n")
    lock_file.flush()

    stop = False

    def handle_stop(signum, frame):
        nonlocal stop
        stop = True
        log_line(watcher_log, f"RECEIVED SIGNAL {signum}, stopping watcher")

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    last_stage = None
    last_step_log = None
    last_step_mtime = None
    last_warned_stall = None

    log_line(watcher_log, f"WATCHER START external_root={external_root}")

    while not stop:
        rows = read_ps()
        procs = find_matching_procs(rows, str(external_root))
        stage = infer_current_stage(pipeline_log)
        step_log = latest_step_log(step_log_root)
        step_log_name = step_log.name if step_log is not None else None
        step_log_size = step_log.stat().st_size if step_log is not None else None
        step_log_mtime = step_log.stat().st_mtime if step_log is not None else None
        root_size = safe_du_size(external_root)
        gpus = query_gpu_snapshot()

        if stage != last_stage:
            log_line(
                watcher_log,
                (
                    f"STAGE UPDATE stage={stage} "
                    f"active_pids={[row['pid'] for row in procs]} "
                    f"step_log={step_log_name}"
                ),
            )
            last_stage = stage

        if step_log_name != last_step_log or step_log_mtime != last_step_mtime:
            log_line(
                watcher_log,
                (
                    f"STEP LOG UPDATE file={step_log_name} size={step_log_size}B "
                    f"root_size={root_size} {format_gpu(gpus)}"
                ),
            )
            last_step_log = step_log_name
            last_step_mtime = step_log_mtime
            last_warned_stall = None
        elif step_log is not None and procs:
            idle_minutes = (time.time() - step_log_mtime) / 60.0
            if idle_minutes >= float(args.stall_minutes):
                if last_warned_stall is None or (time.time() - last_warned_stall) >= float(args.poll_seconds):
                    log_line(
                        watcher_log,
                        (
                            f"STALL WARN stage={stage} step_log={step_log_name} "
                            f"idle_minutes={idle_minutes:.1f} root_size={root_size} "
                            f"active_pids={[row['pid'] for row in procs]} {format_gpu(gpus)}"
                        ),
                    )
                    last_warned_stall = time.time()

        if procs:
            proc_desc = ", ".join(
                f"pid={row['pid']} etimes={row['etimes']}s" for row in procs[:6]
            )
        else:
            proc_desc = "none"
        log_line(
            watcher_log,
            f"HEARTBEAT stage={stage} procs={proc_desc} root_size={root_size} {format_gpu(gpus)}",
        )

        payload = {
            "updated_at": now_str(),
            "external_root": str(external_root),
            "watcher_pid": os.getpid(),
            "stage": stage,
            "pipeline_done": pipeline_done(pipeline_log),
            "active_processes": procs,
            "latest_step_log": step_log_name,
            "latest_step_log_size_bytes": step_log_size,
            "latest_step_log_mtime_epoch": step_log_mtime,
            "root_size": root_size,
            "gpu": gpus,
        }
        save_json(state_path, payload)

        if payload["pipeline_done"]:
            log_line(watcher_log, "PIPELINE DONE detected, watcher exiting")
            break

        time.sleep(max(5.0, float(args.poll_seconds)))

    save_json(
        state_path,
        {
            "updated_at": now_str(),
            "external_root": str(external_root),
            "watcher_pid": os.getpid(),
            "stage": infer_current_stage(pipeline_log),
            "pipeline_done": pipeline_done(pipeline_log),
            "stopped": True,
        },
    )
    log_line(watcher_log, "WATCHER EXIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
