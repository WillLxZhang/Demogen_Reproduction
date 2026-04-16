#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def now_str() -> str:
    return time.strftime("%F %T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Watch a Stack external pipeline run, keep the checkpoint guard alive, "
            "and auto-relaunch the pre-train stages once if the main pipeline dies unexpectedly."
        )
    )
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-pretrain-restarts", type=int, default=1)
    parser.add_argument("--short-threshold", type=float, default=0.5)
    parser.add_argument("--horizon", type=int, default=800)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def log_line(path: Path, message: str) -> None:
    line = f"[{now_str()}] {message}"
    print(line, flush=True)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def find_matching_procs(rows: list[dict[str, str]], token: str, external_root: str) -> list[dict[str, str]]:
    matched = []
    for row in rows:
        args = row["args"]
        if token in args and external_root in args:
            matched.append(row)
    return matched


def pipeline_done(pipeline_log: Path) -> bool:
    if not pipeline_log.exists():
        return False
    text = pipeline_log.read_text(encoding="utf-8", errors="replace")
    return "PIPELINE DONE:" in text


def train_started(pipeline_log: Path) -> bool:
    if not pipeline_log.exists():
        return False
    text = pipeline_log.read_text(encoding="utf-8", errors="replace")
    return "TRAIN START:" in text


def infer_current_stage(pipeline_log: Path) -> str:
    if not pipeline_log.exists():
        return "not_started"
    current = "unknown"
    for raw in pipeline_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "STEP START " in raw:
            current = raw.split("STEP START ", 1)[1].split(":", 1)[0].strip()
        elif "STEP DONE " in raw:
            done_name = raw.split("STEP DONE ", 1)[1].split(":", 1)[0].strip()
            if done_name == current:
                current = f"after_{done_name}"
        elif "TRAIN START:" in raw:
            current = "train"
        elif "PIPELINE DONE:" in raw:
            current = "done"
    return current


def start_process(cmd: list[str], log_path: Path, cwd: Path) -> int:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{now_str()}] $ {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        return proc.pid


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    external_root = Path(args.external_root).expanduser().resolve()
    manifest_path = external_root / "manifest.json"
    pipeline_log = external_root / "logs" / "pipeline.log"
    watchdog_log = external_root / "logs" / "pipeline_watchdog.log"
    state_path = external_root / "logs" / "pipeline_watchdog_state.json"
    relaunch_log = external_root / "logs" / "pipeline_relauncher.log"
    guard_log = external_root / "logs" / "checkpoint_eval_guard.stdout.log"

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    manifest = read_json(manifest_path)
    external_root_str = str(external_root)
    run_tag = external_root.name
    state = {
        "external_root": external_root_str,
        "run_tag": run_tag,
        "pretrain_restart_count": 0,
        "last_stage": None,
        "last_pipeline_pid": None,
        "last_guard_pid": None,
        "updated_at": now_str(),
    }
    if state_path.exists():
        try:
            state.update(read_json(state_path))
        except Exception:
            pass

    stop = False

    def handle_stop(signum, frame):
        nonlocal stop
        stop = True
        log_line(watchdog_log, f"RECEIVED SIGNAL {signum}, stopping watchdog")

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    log_line(
        watchdog_log,
        f"WATCHDOG START external_root={external_root_str} max_pretrain_restarts={args.max_pretrain_restarts}",
    )

    while not stop:
        rows = read_ps()
        pipeline_rows = find_matching_procs(
            rows,
            "scripts/run_stack_relalign_replayobs_train120_external.py",
            external_root_str,
        )
        guard_rows = find_matching_procs(
            rows,
            "scripts/conditional_eval_robomimic_checkpoints.py",
            external_root_str,
        )
        done = pipeline_done(pipeline_log)
        started_train = train_started(pipeline_log)
        stage = infer_current_stage(pipeline_log)

        if stage != state.get("last_stage"):
            log_line(
                watchdog_log,
                f"STAGE UPDATE stage={stage} pipeline_pids={[row['pid'] for row in pipeline_rows]} guard_pids={[row['pid'] for row in guard_rows]}",
            )
            state["last_stage"] = stage

        if not done and not guard_rows:
            guard_cmd = [
                sys.executable,
                str(repo_root / "scripts" / "conditional_eval_robomimic_checkpoints.py"),
                "--external-root",
                external_root_str,
                "--poll-seconds",
                "10",
                "--short-threshold",
                str(args.short_threshold),
                "--short-parallel-jobs",
                "1",
                "--full-parallel-jobs",
                "1",
                "--horizon",
                str(args.horizon),
                "--custom-reset-from-manifest",
            ]
            guard_pid = start_process(guard_cmd, guard_log, repo_root)
            log_line(watchdog_log, f"RELAUNCHED GUARD pid={guard_pid}")
            state["last_guard_pid"] = guard_pid

        if not done and not pipeline_rows:
            if not started_train and int(state.get("pretrain_restart_count", 0)) < int(args.max_pretrain_restarts):
                restart_cmd = [
                    sys.executable,
                    str(repo_root / "scripts" / "run_stack_relalign_replayobs_train120_external.py"),
                    "--external-root",
                    external_root_str,
                    "--run-tag",
                    run_tag,
                ]
                new_pid = start_process(restart_cmd, relaunch_log, repo_root)
                state["pretrain_restart_count"] = int(state.get("pretrain_restart_count", 0)) + 1
                state["last_pipeline_pid"] = new_pid
                log_line(
                    watchdog_log,
                    f"RELAUNCHED PIPELINE pid={new_pid} restart_count={state['pretrain_restart_count']}",
                )
            elif started_train:
                log_line(
                    watchdog_log,
                    "PIPELINE PROCESS MISSING AFTER TRAIN START; not auto-restarting to avoid retraining ambiguity",
                )
            else:
                log_line(
                    watchdog_log,
                    f"PIPELINE PROCESS MISSING; restart budget exhausted ({state.get('pretrain_restart_count', 0)})",
                )

        state["updated_at"] = now_str()
        if pipeline_rows:
            state["last_pipeline_pid"] = int(pipeline_rows[0]["pid"])
        if guard_rows:
            state["last_guard_pid"] = int(guard_rows[0]["pid"])
        save_json(state_path, state)

        if done:
            log_line(watchdog_log, "PIPELINE DONE detected, watchdog exiting")
            break

        time.sleep(max(5.0, float(args.poll_seconds)))

    state["updated_at"] = now_str()
    save_json(state_path, state)
    log_line(watchdog_log, "WATCHDOG EXIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
