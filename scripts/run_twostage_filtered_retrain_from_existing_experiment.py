#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path

import h5py


def now_str() -> str:
    return time.strftime("%F %T")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def log_line(path: Path, message: str) -> None:
    line = f"[{now_str()}] {message}"
    print(line, flush=True)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


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


def numeric_suffix(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    if match:
        return int(match.group(1)), name
    return math.inf, name


def sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    return sorted(list(data_group.keys()), key=numeric_suffix)


def threshold_tag(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "neg").replace(".", "p")


def conda_exe_path() -> str:
    env_value = os.environ.get("CONDA_EXE")
    if env_value:
        return str(Path(env_value).expanduser().resolve())
    candidate = Path.home() / "miniconda3" / "bin" / "conda"
    if candidate.exists():
        return str(candidate.resolve())
    return "conda"


def parse_selected_episode_count(raw: str | None) -> int:
    if not raw:
        return 0
    return len([part.strip() for part in raw.split(",") if part.strip()])


def init_paths(output_root: Path) -> dict[str, Path]:
    paths = {
        "root": output_root,
        "inputs": output_root / "inputs",
        "logs": output_root / "logs",
        "step_logs": output_root / "logs" / "steps",
        "analysis": output_root / "outputs" / "analysis",
        "robomimic": output_root / "outputs" / "robomimic",
        "train_output_dir": output_root / "outputs" / "robomimic" / "diffusion_policy_demogen",
        "evals": output_root / "outputs" / "evals",
        "manifest": output_root / "manifest.json",
        "report": output_root / "report.md",
        "pipeline_log": output_root / "logs" / "pipeline.log",
        "gpu_log": output_root / "logs" / "gpu_health.log",
    }
    for key, path in paths.items():
        if key in {"manifest", "report", "pipeline_log", "gpu_log"}:
            continue
        ensure_dir(path)
    return paths


def query_gpu_snapshot() -> list[dict]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
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


def query_gpu_compute_processes() -> list[dict]:
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        rows.append(
            {
                "pid": int(parts[0]),
                "name": parts[1],
                "mem_mib": float(parts[2]),
            }
        )
    return rows


def log_gpu_health(
    *,
    gpu_log: Path,
    pipeline_log: Path,
    context: str,
    warn_temp_c: float | None = None,
) -> float | None:
    try:
        gpus = query_gpu_snapshot()
        compute = query_gpu_compute_processes()
    except Exception as exc:
        log_line(gpu_log, f"{context}: GPU_QUERY_FAIL error={exc}")
        log_line(pipeline_log, f"{context}: GPU_QUERY_FAIL error={exc}")
        return None

    gpu_desc = "; ".join(
        (
            f"gpu{row['index']} temp={row['temp_c']:.1f}C util={row['util_pct']:.1f}% "
            f"mem={row['mem_used_mib']:.0f}/{row['mem_total_mib']:.0f}MiB power={row['power_w']:.1f}W"
        )
        for row in gpus
    )
    compute_desc = (
        ", ".join(f"pid={row['pid']} name={row['name']} mem={row['mem_mib']:.0f}MiB" for row in compute)
        if compute
        else "none"
    )
    log_line(gpu_log, f"{context}: {gpu_desc} | compute={compute_desc}")
    max_temp = max((row["temp_c"] for row in gpus), default=None)
    if warn_temp_c is not None and max_temp is not None and max_temp >= warn_temp_c:
        log_line(
            pipeline_log,
            f"GPU TEMP WARN {context}: max_temp={max_temp:.1f}C threshold={warn_temp_c:.1f}C",
        )
    return max_temp


def wait_for_gpu_compute_idle(
    *,
    pipeline_log: Path,
    gpu_log: Path,
    context: str,
    warn_temp_c: float | None,
    poll_interval_sec: float = 10.0,
    required_idle_samples: int = 2,
) -> None:
    idle = 0
    while idle < required_idle_samples:
        max_temp = log_gpu_health(
            gpu_log=gpu_log,
            pipeline_log=pipeline_log,
            context=f"WAIT {context}",
            warn_temp_c=warn_temp_c,
        )
        try:
            compute = query_gpu_compute_processes()
        except Exception as exc:
            log_line(pipeline_log, f"GPU PROCESS QUERY FAIL before {context}: error={exc}")
            compute = []
        if not compute:
            idle += 1
            log_line(
                pipeline_log,
                f"GPU IDLE SAMPLE {idle}/{required_idle_samples} for {context} max_temp={max_temp}",
            )
        else:
            idle = 0
            log_line(pipeline_log, f"GPU BUSY before {context}; waiting for compute processes to exit")
        if idle < required_idle_samples:
            time.sleep(poll_interval_sec)


def run_step(
    name: str,
    cmd: list[str],
    *,
    cwd: Path,
    pipeline_log: Path,
    step_log: Path,
    gpu_log: Path | None = None,
    gpu_log_interval_sec: int = 30,
    gpu_warn_temp_c: float | None = None,
    gpu_stop_temp_c: float | None = None,
    wait_for_gpu_before: bool = False,
) -> None:
    ensure_dir(step_log.parent)
    if wait_for_gpu_before and gpu_log is not None:
        wait_for_gpu_compute_idle(
            pipeline_log=pipeline_log,
            gpu_log=gpu_log,
            context=name,
            warn_temp_c=gpu_warn_temp_c,
        )

    log_line(pipeline_log, f"STEP START {name}: {shlex.join(cmd)}")
    with step_log.open("w", encoding="utf-8") as f:
        f.write(f"$ {shlex.join(cmd)}\n\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        next_gpu_log = time.time()
        while proc.poll() is None:
            if gpu_log is not None and time.time() >= next_gpu_log:
                max_temp = log_gpu_health(
                    gpu_log=gpu_log,
                    pipeline_log=pipeline_log,
                    context=f"RUN {name}",
                    warn_temp_c=gpu_warn_temp_c,
                )
                if gpu_stop_temp_c is not None and max_temp is not None and max_temp >= gpu_stop_temp_c:
                    terminate_process_tree(proc)
                    raise RuntimeError(
                        f"{name} terminated because GPU temp {max_temp:.1f}C exceeded {gpu_stop_temp_c:.1f}C"
                    )
                next_gpu_log = time.time() + float(gpu_log_interval_sec)
            time.sleep(5.0)
        return_code = proc.wait()
        if return_code != 0:
            log_line(pipeline_log, f"STEP FAIL {name}: returncode={return_code} log={step_log}")
            raise RuntimeError(f"{name} failed with returncode={return_code}. See {step_log}")
    log_line(pipeline_log, f"STEP DONE {name}: log={step_log}")


def find_train_run_dir(run_root: Path) -> Path | None:
    if not run_root.exists():
        return None
    children = sorted([p for p in run_root.iterdir() if p.is_dir()])
    if not children:
        return None
    return children[-1]


def build_train_config(
    *,
    base_config_path: Path,
    output_path: Path,
    output_dir: Path,
    num_epochs: int,
    save_every_n_epochs: int,
    rollout_every_n_epochs: int,
    rollout_n: int,
    rollout_horizon: int,
    batch_size: int,
    experiment_name: str,
) -> None:
    cfg = load_json(base_config_path)
    cfg["experiment"]["name"] = experiment_name
    cfg["experiment"]["save"]["every_n_epochs"] = int(save_every_n_epochs)
    cfg["experiment"]["render_video"] = False
    cfg["experiment"]["rollout"]["enabled"] = True
    cfg["experiment"]["rollout"]["n"] = int(rollout_n)
    cfg["experiment"]["rollout"]["horizon"] = int(rollout_horizon)
    cfg["experiment"]["rollout"]["rate"] = int(rollout_every_n_epochs)
    cfg["train"]["output_dir"] = str(output_dir)
    cfg["train"]["num_epochs"] = int(num_epochs)
    cfg["train"]["batch_size"] = int(batch_size)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def launch_training_and_wait(
    *,
    conda_exe: str,
    repo_root: Path,
    pipeline_log: Path,
    gpu_log: Path,
    gpu_log_interval_sec: int,
    gpu_warn_temp_c: float | None,
    gpu_stop_temp_c: float | None,
    train_config: Path,
    dataset_path: Path,
    train_output_dir: Path,
    run_name: str,
    launcher_log: Path,
    watcher_title: str,
) -> Path:
    wait_for_gpu_compute_idle(
        pipeline_log=pipeline_log,
        gpu_log=gpu_log,
        context=f"training {run_name}",
        warn_temp_c=gpu_warn_temp_c,
    )

    ensure_dir(launcher_log.parent)
    train_cmd = [
        conda_exe,
        "run",
        "--no-capture-output",
        "-n",
        "robomimic",
        "python",
        str(repo_root / "repos" / "robomimic" / "robomimic" / "scripts" / "train.py"),
        "--config",
        str(train_config),
        "--dataset",
        str(dataset_path),
        "--name",
        run_name,
    ]
    log_line(pipeline_log, f"TRAIN START: {shlex.join(train_cmd)}")
    launcher_file = launcher_log.open("w", encoding="utf-8")
    launcher_file.write(f"$ {shlex.join(train_cmd)}\n\n")
    launcher_file.flush()

    proc = subprocess.Popen(
        train_cmd,
        cwd=str(repo_root),
        stdout=launcher_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    run_root = train_output_dir / run_name
    run_dir: Path | None = None
    watcher_proc: subprocess.Popen[str] | None = None
    watcher_file = None
    next_gpu_log = time.time()
    try:
        while proc.poll() is None:
            if run_dir is None:
                candidate = find_train_run_dir(run_root)
                if candidate is not None and (candidate / "logs").exists():
                    run_dir = candidate
                    watcher_log = candidate / "logs" / "progress_watcher.log"
                    watcher_cmd = [
                        conda_exe,
                        "run",
                        "--no-capture-output",
                        "-n",
                        "robomimic",
                        "python",
                        str(repo_root / "scripts" / "watch_robomimic_progress_notify.py"),
                        "--run-dir",
                        str(candidate),
                        "--start-at",
                        "beginning",
                        "--title",
                        watcher_title,
                        "--dry-run",
                    ]
                    watcher_file = watcher_log.open("w", encoding="utf-8")
                    watcher_proc = subprocess.Popen(
                        watcher_cmd,
                        cwd=str(repo_root),
                        stdout=watcher_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                        start_new_session=True,
                    )
                    log_line(
                        pipeline_log,
                        f"TRAIN RUN DIR READY: run_dir={candidate} watcher_log={watcher_log}",
                    )

            if time.time() >= next_gpu_log:
                max_temp = log_gpu_health(
                    gpu_log=gpu_log,
                    pipeline_log=pipeline_log,
                    context=f"TRAIN {run_name}",
                    warn_temp_c=gpu_warn_temp_c,
                )
                if gpu_stop_temp_c is not None and max_temp is not None and max_temp >= gpu_stop_temp_c:
                    terminate_process_tree(proc)
                    raise RuntimeError(
                        f"Training {run_name} terminated because GPU temp {max_temp:.1f}C exceeded "
                        f"{gpu_stop_temp_c:.1f}C"
                    )
                next_gpu_log = time.time() + float(gpu_log_interval_sec)
            time.sleep(5.0)

        return_code = proc.wait()
        if run_dir is None:
            run_dir = find_train_run_dir(run_root)
        if return_code != 0:
            raise RuntimeError(f"Training failed with returncode={return_code}. See {launcher_log}")
        if run_dir is None:
            raise FileNotFoundError(f"Could not locate run dir under {run_root}")
        log_line(pipeline_log, f"TRAIN DONE: run_dir={run_dir}")
        return run_dir
    finally:
        if watcher_proc is not None and watcher_proc.poll() is None:
            terminate_process_tree(watcher_proc)
        if watcher_file is not None:
            watcher_file.close()
        launcher_file.close()


def summarize_eval_results(eval_summary_path: Path) -> dict:
    payload = load_json(eval_summary_path)
    results = payload.get("results", {})
    completed = [
        row
        for row in results.values()
        if row.get("status") == "completed" and row.get("success_rate") is not None
    ]
    if not completed:
        return {
            "completed_seeds": 0,
            "mean_success_rate": None,
            "mean_avg_return": None,
            "mean_avg_horizon": None,
        }
    return {
        "completed_seeds": len(completed),
        "mean_success_rate": sum(float(row["success_rate"]) for row in completed) / len(completed),
        "mean_avg_return": sum(float(row["avg_return"]) for row in completed) / len(completed),
        "mean_avg_horizon": sum(float(row["avg_horizon"]) for row in completed) / len(completed),
    }


def update_manifest(manifest_path: Path, manifest: dict, **updates) -> dict:
    manifest.update(updates)
    manifest["updated_at"] = now_str()
    write_json(manifest_path, manifest)
    return manifest


def write_report(report_path: Path, manifest: dict) -> None:
    lines = [
        "# Twostage Filtered Retrain Fork",
        "",
        f"- status: `{manifest.get('status')}`",
        f"- source_external_root: `{manifest.get('source_external_root')}`",
        f"- source_exported_hdf5: `{manifest.get('source_exported_hdf5')}`",
        f"- filtered_hdf5: `{manifest.get('filtered_hdf5')}`",
        f"- filter_stage: `{manifest.get('filter_stage')}`",
        f"- filter_metric: `{manifest.get('filter_metric')}`",
        f"- filter_max_threshold: `{manifest.get('filter_max_threshold')}`",
        f"- dropped_generated_count: `{manifest.get('dropped_generated_count')}`",
        f"- dropped_generated_indices: `{manifest.get('dropped_generated_indices')}`",
        f"- train_config: `{manifest.get('train_config')}`",
        f"- train_run_dir: `{manifest.get('train_run_dir')}`",
        f"- checkpoint_epoch: `{manifest.get('checkpoint_path')}`",
        f"- eval_output_dir: `{manifest.get('eval_output_dir')}`",
        "",
        "## Eval Summary",
        "",
    ]
    eval_summary = manifest.get("eval_summary") or {}
    if eval_summary:
        lines.append(f"- completed_seeds: `{eval_summary.get('completed_seeds')}`")
        lines.append(f"- mean_success_rate: `{eval_summary.get('mean_success_rate')}`")
        lines.append(f"- mean_avg_return: `{eval_summary.get('mean_avg_return')}`")
        lines.append(f"- mean_avg_horizon: `{eval_summary.get('mean_avg_horizon')}`")
    else:
        lines.append("- eval_summary: `not_run`")
    if manifest.get("error"):
        lines.extend(["", "## Error", "", f"`{manifest['error']}`"])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Fork an existing twostage experiment by filtering generated episodes "
            "using solve metrics, then retrain and optionally run multiseed eval."
        )
    )
    parser.add_argument("--source-external-root", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--filter-stage", default="motion2")
    parser.add_argument("--filter-metric", default="rel_final_err_norm")
    parser.add_argument("--max-threshold", type=float, default=0.1)
    parser.add_argument("--train-base-config", default=None)
    parser.add_argument("--train-epochs", type=int, default=60)
    parser.add_argument("--save-every-n-epochs", type=int, default=30)
    parser.add_argument("--rollout-every-n-epochs", type=int, default=30)
    parser.add_argument("--rollout-n", type=int, default=10)
    parser.add_argument("--rollout-horizon", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-checkpoint-epoch", type=int, default=60)
    parser.add_argument("--eval-rollouts", type=int, default=20)
    parser.add_argument("--eval-horizon", type=int, default=1200)
    parser.add_argument("--eval-parallel-jobs", type=int, default=3)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--custom-reset-source-episode", type=int, default=0)
    parser.add_argument("--custom-reset-object-translation", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--custom-reset-target-translation", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--gpu-log-interval-sec", type=int, default=30)
    parser.add_argument("--gpu-warn-temp-c", type=float, default=82.0)
    parser.add_argument("--gpu-stop-temp-c", type=float, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--repo-root", default=str(repo_root))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    conda_exe = conda_exe_path()
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    source_root = Path(args.source_external_root).expanduser().resolve()
    source_manifest_path = source_root / "manifest.json"
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"source manifest not found: {source_manifest_path}")
    source_manifest = load_json(source_manifest_path)

    source_exported_hdf5 = Path(str(source_manifest["exported_hdf5"])).expanduser().resolve()
    source_solve_json = Path(str(source_manifest["solve_json"])).expanduser().resolve()
    source_selected_demo_hdf5 = Path(str(source_manifest["selected_demo_hdf5"])).expanduser().resolve()
    train_base_config = (
        Path(args.train_base_config).expanduser().resolve()
        if args.train_base_config
        else Path(str(source_manifest.get("train_base_config_snapshot") or source_manifest["train_config"]))
        .expanduser()
        .resolve()
    )
    if not source_exported_hdf5.exists():
        raise FileNotFoundError(f"source exported hdf5 not found: {source_exported_hdf5}")
    if not source_solve_json.exists():
        raise FileNotFoundError(f"source solve json not found: {source_solve_json}")
    if not source_selected_demo_hdf5.exists():
        raise FileNotFoundError(f"source selected demo hdf5 not found: {source_selected_demo_hdf5}")
    if not train_base_config.exists():
        raise FileNotFoundError(f"train base config not found: {train_base_config}")

    source_demo_count = parse_selected_episode_count(source_manifest.get("selected_episodes"))
    if source_demo_count <= 0:
        raise ValueError("could not infer source demo count from source manifest selected_episodes")

    filter_tag = (
        f"{args.filter_stage}_{args.filter_metric}_le{threshold_tag(args.max_threshold)}"
    )
    default_run_tag = (
        f"{source_root.name}_filtered_{filter_tag}_train{args.train_epochs}_{timestamp}"
    )
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else Path("/media/willzhang/KINGSTON/zlxtemp") / (args.run_tag or default_run_tag)
    )
    paths = init_paths(output_root)
    pipeline_log = paths["pipeline_log"]
    gpu_log = paths["gpu_log"]

    filter_report_path = paths["analysis"] / f"filter_{filter_tag}.json"
    filtered_hdf5 = paths["robomimic"] / f"{source_exported_hdf5.stem}_{filter_tag}.hdf5"
    train_config = paths["robomimic"] / "train_config.json"

    ensure_dir(paths["inputs"])
    shutil.copy2(source_manifest_path, paths["inputs"] / "source_manifest.json")
    shutil.copy2(source_solve_json, paths["inputs"] / source_solve_json.name)
    shutil.copy2(train_base_config, paths["inputs"] / train_base_config.name)

    manifest = {
        "status": "running",
        "source_external_root": str(source_root),
        "source_manifest": str(source_manifest_path),
        "source_exported_hdf5": str(source_exported_hdf5),
        "source_solve_json": str(source_solve_json),
        "source_selected_demo_hdf5": str(source_selected_demo_hdf5),
        "source_demo_count": int(source_demo_count),
        "filtered_hdf5": str(filtered_hdf5),
        "filter_report": str(filter_report_path),
        "filter_stage": args.filter_stage,
        "filter_metric": args.filter_metric,
        "filter_max_threshold": float(args.max_threshold),
        "train_base_config": str(train_base_config),
        "train_config": str(train_config),
        "train_epochs": int(args.train_epochs),
        "save_every_n_epochs": int(args.save_every_n_epochs),
        "rollout_every_n_epochs": int(args.rollout_every_n_epochs),
        "rollout_n": int(args.rollout_n),
        "rollout_horizon": int(args.rollout_horizon),
        "batch_size": int(args.batch_size),
        "eval_checkpoint_epoch": int(args.eval_checkpoint_epoch),
        "eval_rollouts": int(args.eval_rollouts),
        "eval_horizon": int(args.eval_horizon),
        "eval_seeds": [int(seed) for seed in args.eval_seeds],
        "custom_reset_source_episode": int(args.custom_reset_source_episode),
        "custom_reset_object_translation": [float(x) for x in args.custom_reset_object_translation],
        "custom_reset_target_translation": [float(x) for x in args.custom_reset_target_translation],
        "started_at": now_str(),
        "updated_at": now_str(),
        "error": None,
    }
    write_json(paths["manifest"], manifest)
    write_report(paths["report"], manifest)

    log_line(pipeline_log, f"FORK ROOT: {output_root}")
    log_line(pipeline_log, f"SOURCE ROOT: {source_root}")
    log_gpu_health(gpu_log=gpu_log, pipeline_log=pipeline_log, context="START", warn_temp_c=args.gpu_warn_temp_c)

    try:
        if filtered_hdf5.exists() and filter_report_path.exists():
            filter_report = load_json(filter_report_path)
            log_line(pipeline_log, f"REUSE FILTERED DATASET: {filtered_hdf5}")
        else:
            solve_payload = load_json(source_solve_json)
            solve_episodes = solve_payload.get("episodes", [])
            if not solve_episodes:
                raise ValueError(f"solve json has no episodes: {source_solve_json}")

            dropped_generated_indices = []
            dropped_rows = []
            for ep in solve_episodes:
                generated_idx = int(ep["generated_episode_idx"])
                stage_payload = ep.get(args.filter_stage)
                if not isinstance(stage_payload, dict):
                    raise KeyError(f"episode {generated_idx} missing stage '{args.filter_stage}'")
                if args.filter_metric not in stage_payload:
                    raise KeyError(
                        f"episode {generated_idx} stage '{args.filter_stage}' missing metric '{args.filter_metric}'"
                    )
                metric_value = float(stage_payload[args.filter_metric])
                if not math.isfinite(metric_value):
                    raise ValueError(
                        f"episode {generated_idx} stage '{args.filter_stage}' metric '{args.filter_metric}' "
                        f"is not finite: {metric_value}"
                    )
                if metric_value > float(args.max_threshold):
                    dropped_generated_indices.append(generated_idx)
                    dropped_rows.append(
                        {
                            "generated_episode_idx": generated_idx,
                            "source_episode_idx": int(ep["source_episode_idx"]),
                            "translation": ep.get("translation"),
                            "metric_value": metric_value,
                        }
                    )

            with h5py.File(source_exported_hdf5, "r") as src_f:
                if "data" not in src_f:
                    raise KeyError(f"{source_exported_hdf5} missing /data group")
                src_data = src_f["data"]
                src_keys = sorted_demo_keys(src_data)
                generated_demo_count = len(src_keys) - source_demo_count
                if generated_demo_count != len(solve_episodes):
                    raise ValueError(
                        "generated demo count does not match solve episode count: "
                        f"hdf5_generated={generated_demo_count} solve_episodes={len(solve_episodes)}"
                    )
                selected_src_keys = []
                old_to_new = {}
                total_samples = 0
                for out_idx, src_key in enumerate(src_keys):
                    src_numeric_idx = numeric_suffix(src_key)[0]
                    if out_idx >= source_demo_count:
                        generated_idx = out_idx - source_demo_count
                        if generated_idx in dropped_generated_indices:
                            continue
                    selected_src_keys.append(src_key)
                    old_to_new[src_key] = f"demo_{len(old_to_new)}"
                    total_samples += int(src_data[src_key].attrs.get("num_samples", 0))

            ensure_dir(filtered_hdf5.parent)
            if filtered_hdf5.exists():
                filtered_hdf5.unlink()
            with h5py.File(source_exported_hdf5, "r") as src_f, h5py.File(filtered_hdf5, "w") as dst_f:
                for key, value in src_f.attrs.items():
                    dst_f.attrs[key] = value
                src_data = src_f["data"]
                dst_data = dst_f.create_group("data")
                for key, value in src_data.attrs.items():
                    dst_data.attrs[key] = value
                dst_data.attrs["total"] = int(total_samples)
                dst_data.attrs["source_hdf5"] = str(source_exported_hdf5)
                dst_data.attrs["source_external_root"] = str(source_root)
                dst_data.attrs["filtered_from_solve_json"] = str(source_solve_json)
                dst_data.attrs["filtered_stage"] = args.filter_stage
                dst_data.attrs["filtered_metric"] = args.filter_metric
                dst_data.attrs["filtered_max_threshold"] = float(args.max_threshold)
                dst_data.attrs["selected_demo_keys"] = json.dumps(selected_src_keys)
                dst_data.attrs["source_to_filtered_demo_key_map"] = json.dumps(old_to_new)
                dst_data.attrs["dropped_generated_indices"] = json.dumps(dropped_generated_indices)
                for src_key in selected_src_keys:
                    src_f.copy(src_data[src_key], dst_data, name=old_to_new[src_key])

            filter_report = {
                "source_external_root": str(source_root),
                "source_exported_hdf5": str(source_exported_hdf5),
                "filtered_hdf5": str(filtered_hdf5),
                "source_demo_count": int(source_demo_count),
                "source_total_demo_count": int(source_demo_count + len(solve_episodes)),
                "generated_demo_count": int(len(solve_episodes)),
                "kept_generated_count": int(len(solve_episodes) - len(dropped_generated_indices)),
                "dropped_generated_count": int(len(dropped_generated_indices)),
                "filter_stage": args.filter_stage,
                "filter_metric": args.filter_metric,
                "max_threshold": float(args.max_threshold),
                "dropped_generated_indices": dropped_generated_indices,
                "dropped_rows": dropped_rows,
                "selected_demo_keys": selected_src_keys,
            }
            write_json(filter_report_path, filter_report)
            log_line(
                pipeline_log,
                f"FILTER DONE: kept_generated={filter_report['kept_generated_count']} "
                f"dropped_generated={filter_report['dropped_generated_count']}",
            )

        dropped_count = int(filter_report["dropped_generated_count"])
        dropped_indices = list(filter_report["dropped_generated_indices"])
        filter_suffix = f"{filter_tag}_drop{dropped_count}"
        base_name = str(source_manifest.get("generated_name") or source_exported_hdf5.stem)
        train_run_name = (
            f"{base_name}_{filter_suffix}_replayobs_dp_{args.save_every_n_epochs}save"
            f"{args.rollout_every_n_epochs}rollout{args.rollout_n}_h{args.rollout_horizon}_"
            f"{args.train_epochs}epoch_{timestamp}"
        )
        eval_output_dir = paths["evals"] / (
            f"epoch{args.eval_checkpoint_epoch:03d}_custom_reset_sourceep{args.custom_reset_source_episode}"
            f"_r{args.eval_rollouts}_seeds{''.join(str(seed) for seed in args.eval_seeds)}_{timestamp}"
        )

        update_manifest(
            paths["manifest"],
            manifest,
            status="prepared",
            train_run_name=train_run_name,
            dropped_generated_count=dropped_count,
            dropped_generated_indices=dropped_indices,
            filter_report=str(filter_report_path),
            eval_output_dir=str(eval_output_dir),
        )
        write_report(paths["report"], manifest)

        build_train_config(
            base_config_path=train_base_config,
            output_path=train_config,
            output_dir=paths["train_output_dir"],
            num_epochs=args.train_epochs,
            save_every_n_epochs=args.save_every_n_epochs,
            rollout_every_n_epochs=args.rollout_every_n_epochs,
            rollout_n=args.rollout_n,
            rollout_horizon=args.rollout_horizon,
            batch_size=args.batch_size,
            experiment_name=train_run_name,
        )
        log_line(pipeline_log, f"TRAIN CONFIG READY: {train_config}")

        train_run_dir = None
        checkpoint_path = None
        if args.skip_train:
            update_manifest(paths["manifest"], manifest, status="prepared_skip_train")
        else:
            train_run_dir = launch_training_and_wait(
                conda_exe=conda_exe,
                repo_root=repo_root,
                pipeline_log=pipeline_log,
                gpu_log=gpu_log,
                gpu_log_interval_sec=args.gpu_log_interval_sec,
                gpu_warn_temp_c=args.gpu_warn_temp_c,
                gpu_stop_temp_c=args.gpu_stop_temp_c,
                train_config=train_config,
                dataset_path=filtered_hdf5,
                train_output_dir=paths["train_output_dir"],
                run_name=train_run_name,
                launcher_log=paths["logs"] / "train_launcher.log",
                watcher_title=f"Filtered Retrain: {train_run_name}",
            )
            checkpoint_path = train_run_dir / "models" / f"model_epoch_{args.eval_checkpoint_epoch}.pth"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"expected checkpoint not found: {checkpoint_path}")
            update_manifest(
                paths["manifest"],
                manifest,
                status="trained",
                train_run_dir=str(train_run_dir),
                checkpoint_path=str(checkpoint_path),
            )
            write_report(paths["report"], manifest)

        if args.skip_eval:
            update_manifest(paths["manifest"], manifest, status=manifest["status"] + "_skip_eval")
            write_report(paths["report"], manifest)
        else:
            if checkpoint_path is None:
                raise ValueError("cannot run eval when training was skipped and no checkpoint path was produced")
            run_step(
                "eval_checkpoint_multiseed_custom_reset",
                [
                    conda_exe,
                    "run",
                    "--no-capture-output",
                    "-n",
                    "robomimic",
                    "python",
                    str(repo_root / "scripts" / "eval_robomimic_checkpoint_multiseed.py"),
                    "--checkpoint",
                    str(checkpoint_path),
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
                    "robomimic",
                    "--custom-reset-source-demo",
                    str(source_selected_demo_hdf5),
                    "--custom-reset-source-episode",
                    str(args.custom_reset_source_episode),
                    "--custom-reset-object-translation",
                    *[str(x) for x in args.custom_reset_object_translation],
                    "--custom-reset-target-translation",
                    *[str(x) for x in args.custom_reset_target_translation],
                ],
                cwd=repo_root,
                pipeline_log=pipeline_log,
                step_log=paths["step_logs"] / "eval_checkpoint_multiseed_custom_reset.log",
                gpu_log=gpu_log,
                gpu_log_interval_sec=args.gpu_log_interval_sec,
                gpu_warn_temp_c=args.gpu_warn_temp_c,
                gpu_stop_temp_c=args.gpu_stop_temp_c,
                wait_for_gpu_before=True,
            )
            eval_summary_path = eval_output_dir / "reports" / "summary.json"
            if not eval_summary_path.exists():
                raise FileNotFoundError(f"eval summary missing: {eval_summary_path}")
            eval_summary = summarize_eval_results(eval_summary_path)
            update_manifest(
                paths["manifest"],
                manifest,
                status="completed",
                eval_output_dir=str(eval_output_dir),
                eval_summary=eval_summary,
            )
            write_report(paths["report"], manifest)

        log_gpu_health(gpu_log=gpu_log, pipeline_log=pipeline_log, context="END", warn_temp_c=args.gpu_warn_temp_c)
        return 0
    except Exception as exc:
        update_manifest(paths["manifest"], manifest, status="failed", error=str(exc))
        write_report(paths["report"], manifest)
        log_line(pipeline_log, f"FAILED: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
