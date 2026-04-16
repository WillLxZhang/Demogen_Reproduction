#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update_manifest(manifest_path: Path, **updates: Any) -> None:
    if manifest_path.exists():
        payload = load_json(manifest_path)
    else:
        payload = {}
    payload.update(updates)
    payload["updated_at"] = now_str()
    write_json(manifest_path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for a robomimic checkpoint, stop the current training service, "
            "resume training with a modified rollout count, and optionally run final eval."
        )
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--resume-config-out", required=True)
    parser.add_argument("--wait-checkpoint-epoch", type=int, required=True)
    parser.add_argument("--resume-rollout-n", type=int, required=True)
    parser.add_argument("--current-train-unit", required=True)
    parser.add_argument("--pipeline-log", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--resume-launcher-log", required=True)
    parser.add_argument("--resume-progress-log", required=True)
    parser.add_argument("--eval-log", required=True)
    parser.add_argument("--eval-checkpoint-epoch", type=int, default=120)
    parser.add_argument("--eval-rollouts", type=int, default=20)
    parser.add_argument("--eval-horizon", type=int, default=1200)
    parser.add_argument("--eval-parallel-jobs", type=int, default=3)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--robomimic-conda-env", default="robomimic")
    parser.add_argument("--conda-exe", default=None)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser.parse_args()


def run_capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def resolve_conda_exe(explicit: str | None) -> str:
    candidates = [
        explicit,
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
        "/home/willzhang/miniconda3/bin/conda",
        "/opt/conda/bin/conda",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate))
    raise FileNotFoundError(
        "could not locate conda executable; pass --conda-exe or set CONDA_EXE"
    )


def systemd_props(unit: str) -> dict[str, str]:
    result = run_capture(
        ["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "SubState", "-p", "MainPID"]
    )
    if result.returncode != 0:
        return {}
    props: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            props[key] = value
    return props


def stop_systemd_unit(unit: str, pipeline_log: Path) -> None:
    props = systemd_props(unit)
    state = props.get("ActiveState")
    if state in (None, "", "inactive", "failed"):
        log_line(pipeline_log, f"HANDOFF: systemd unit already inactive: {unit}")
        return
    log_line(pipeline_log, f"HANDOFF: stopping systemd unit {unit}")
    result = run_capture(["systemctl", "--user", "stop", unit])
    if result.returncode != 0:
        raise RuntimeError(f"failed to stop unit {unit}: {result.stderr.strip()}")


def wait_unit_inactive(unit: str, *, timeout_sec: float) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        props = systemd_props(unit)
        if props.get("ActiveState") in ("inactive", "failed", ""):
            return
        time.sleep(1.0)
    raise TimeoutError(f"timed out waiting for systemd unit {unit} to become inactive")


def find_epoch_checkpoint(models_dir: Path, epoch: int) -> Path | None:
    exact = models_dir / f"model_epoch_{epoch}.pth"
    if exact.exists():
        return exact
    matches = sorted(
        models_dir.glob(f"model_epoch_{epoch}*.pth"),
        key=lambda p: p.stat().st_mtime,
    )
    return matches[-1] if matches else None


def write_resume_config(base_config: Path, out_config: Path, rollout_n: int) -> None:
    payload = load_json(base_config)
    payload["experiment"]["rollout"]["n"] = int(rollout_n)
    ensure_dir(out_config.parent)
    out_config.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait()


def run_logged_process(
    *,
    cmd: list[str],
    cwd: Path,
    launcher_log: Path,
    progress_watcher_cmd: list[str] | None = None,
    progress_log: Path | None = None,
) -> None:
    ensure_dir(launcher_log.parent)
    launcher_file = launcher_log.open("w", encoding="utf-8")
    launcher_file.write(f"$ {shlex.join(cmd)}\n\n")
    launcher_file.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=launcher_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    watcher_proc: subprocess.Popen[str] | None = None
    watcher_file = None
    try:
        if progress_watcher_cmd is not None and progress_log is not None:
            ensure_dir(progress_log.parent)
            watcher_file = progress_log.open("w", encoding="utf-8")
            watcher_proc = subprocess.Popen(
                progress_watcher_cmd,
                cwd=str(cwd),
                stdout=watcher_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )

        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(f"process failed with returncode={return_code}: {shlex.join(cmd)}")
    finally:
        if watcher_proc is not None and watcher_proc.poll() is None:
            terminate_process_tree(watcher_proc)
        if watcher_file is not None:
            watcher_file.close()
        launcher_file.close()


def wait_for_checkpoint_or_failure(
    *,
    models_dir: Path,
    epoch: int,
    current_train_unit: str,
    pipeline_log: Path,
    poll_seconds: float,
) -> Path:
    while True:
        checkpoint = find_epoch_checkpoint(models_dir, epoch)
        if checkpoint is not None:
            return checkpoint

        props = systemd_props(current_train_unit)
        if props.get("ActiveState") in ("inactive", "failed", ""):
            raise RuntimeError(
                f"training unit {current_train_unit} exited before model_epoch_{epoch} checkpoint appeared"
            )

        log_line(
            pipeline_log,
            f"HANDOFF WAIT: waiting for model_epoch_{epoch} checkpoint in {models_dir}",
        )
        time.sleep(max(1.0, poll_seconds))


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    external_root = Path(args.external_root).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    dataset = Path(args.dataset).expanduser().resolve()
    base_config = Path(args.base_config).expanduser().resolve()
    resume_config_out = Path(args.resume_config_out).expanduser().resolve()
    pipeline_log = Path(args.pipeline_log).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve()
    resume_launcher_log = Path(args.resume_launcher_log).expanduser().resolve()
    resume_progress_log = Path(args.resume_progress_log).expanduser().resolve()
    eval_log = Path(args.eval_log).expanduser().resolve()
    models_dir = run_dir / "models"
    conda_exe = resolve_conda_exe(args.conda_exe)

    log_line(
        pipeline_log,
        (
            f"HANDOFF START: wait_epoch={args.wait_checkpoint_epoch} "
            f"resume_rollout_n={args.resume_rollout_n} run_dir={run_dir}"
        ),
    )
    update_manifest(
        manifest_path,
        status="running_handoff_after_epoch_checkpoint",
        handoff_run_dir=str(run_dir),
        handoff_wait_checkpoint_epoch=int(args.wait_checkpoint_epoch),
        handoff_resume_rollout_n=int(args.resume_rollout_n),
        handoff_resume_config=str(resume_config_out),
    )

    try:
        checkpoint = wait_for_checkpoint_or_failure(
            models_dir=models_dir,
            epoch=args.wait_checkpoint_epoch,
            current_train_unit=args.current_train_unit,
            pipeline_log=pipeline_log,
            poll_seconds=args.poll_seconds,
        )
        log_line(
            pipeline_log,
            f"HANDOFF READY: found checkpoint {checkpoint}",
        )

        stop_systemd_unit(args.current_train_unit, pipeline_log)
        wait_unit_inactive(args.current_train_unit, timeout_sec=120.0)
        log_line(
            pipeline_log,
            f"HANDOFF STOP DONE: stopped unit {args.current_train_unit}",
        )

        write_resume_config(base_config, resume_config_out, args.resume_rollout_n)
        log_line(
            pipeline_log,
            f"HANDOFF RESUME CONFIG READY: {resume_config_out}",
        )
        update_manifest(
            manifest_path,
            status="running_resume_after_epoch_checkpoint",
            handoff_checkpoint_path=str(checkpoint),
            handoff_checkpoint_epoch=int(args.wait_checkpoint_epoch),
        )

        train_cmd = [
            conda_exe,
            "run",
            "--no-capture-output",
            "-n",
            args.robomimic_conda_env,
            "python",
            str(repo_root / "repos" / "robomimic" / "robomimic" / "scripts" / "train.py"),
            "--config",
            str(resume_config_out),
            "--dataset",
            str(dataset),
            "--name",
            args.experiment_name,
            "--resume",
        ]
        progress_cmd = [
            conda_exe,
            "run",
            "--no-capture-output",
            "-n",
            args.robomimic_conda_env,
            "python",
            str(repo_root / "scripts" / "watch_robomimic_progress_notify.py"),
            "--run-dir",
            str(run_dir),
            "--start-at",
            "end",
            "--title",
            "Nut Resume",
            "--dry-run",
        ]
        log_line(pipeline_log, f"TRAIN RESUME START: {shlex.join(train_cmd)}")
        run_logged_process(
            cmd=train_cmd,
            cwd=repo_root,
            launcher_log=resume_launcher_log,
            progress_watcher_cmd=progress_cmd,
            progress_log=resume_progress_log,
        )
        log_line(pipeline_log, f"TRAIN RESUME DONE: run_dir={run_dir}")

        final_checkpoint = find_epoch_checkpoint(models_dir, args.eval_checkpoint_epoch)
        if final_checkpoint is None:
            raise FileNotFoundError(
                f"expected checkpoint for eval epoch {args.eval_checkpoint_epoch} not found under {models_dir}"
            )

        eval_stamp = time.strftime("%Y%m%d_%H%M%S")
        eval_output_dir = (
            run_dir
            / (
                f"eval_epoch{args.eval_checkpoint_epoch}_r{args.eval_rollouts}_"
                f"seeds{''.join(str(seed) for seed in args.eval_seeds)}_"
                f"resume_rollout{args.resume_rollout_n}_{eval_stamp}"
            )
        )
        eval_cmd = [
            conda_exe,
            "run",
            "--no-capture-output",
            "-n",
            args.robomimic_conda_env,
            "python",
            str(repo_root / "scripts" / "eval_robomimic_checkpoint_multiseed.py"),
            "--checkpoint",
            str(final_checkpoint),
            "--output-dir",
            str(eval_output_dir),
            "--n-rollouts",
            str(args.eval_rollouts),
            "--horizon",
            str(args.eval_horizon),
            "--parallel-jobs",
            str(args.eval_parallel_jobs),
            "--seeds",
            *[str(seed) for seed in args.eval_seeds],
            "--conda-env",
            args.robomimic_conda_env,
        ]
        eval_step_name = f"eval_epoch{args.eval_checkpoint_epoch}_multiseed_resume_rollout{args.resume_rollout_n}"
        log_line(pipeline_log, f"STEP START {eval_step_name}: {shlex.join(eval_cmd)}")
        run_logged_process(
            cmd=eval_cmd,
            cwd=repo_root,
            launcher_log=eval_log,
        )
        log_line(pipeline_log, f"STEP DONE {eval_step_name}: log={eval_log}")

        update_manifest(
            manifest_path,
            status="completed_resume_after_epoch_checkpoint",
            handoff_resume_config=str(resume_config_out),
            handoff_resume_checkpoint=str(checkpoint),
            checkpoint_epoch30=str(checkpoint),
            checkpoint_epoch120=str(final_checkpoint),
            eval_output_dir=str(eval_output_dir),
            error=None,
        )
        log_line(pipeline_log, f"PIPELINE DONE: resume_after_epoch_checkpoint report={eval_output_dir}")
        return 0
    except Exception as exc:
        update_manifest(
            manifest_path,
            status="failed_resume_after_epoch_checkpoint",
            error=str(exc),
        )
        log_line(pipeline_log, f"PIPELINE FAIL: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
