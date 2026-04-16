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


def now_str() -> str:
    return time.strftime("%F %T")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run the NutAssemblyRound external twostage pipeline end to end: raw demo.hdf5 "
            "-> selected4 source bundle -> 81-grid generation -> relalign twostage solve "
            "-> replayobs export -> 120-epoch robomimic training -> epoch60 multiseed eval."
        )
    )
    parser.add_argument(
        "--external-root",
        default=None,
        help=(
            "Experiment root on the external disk. Defaults to "
            "/media/willzhang/KINGSTON/zlxtemp/<timestamped-experiment-dir>."
        ),
    )
    parser.add_argument(
        "--raw-demo-hdf5",
        default=str(
            repo_root / "data" / "raw" / "nutassemblyround_0" / "1776068238_0811434" / "demo.hdf5"
        ),
    )
    parser.add_argument(
        "--selected-episodes",
        default="0,1,2,3",
        help="Selected source demos from the raw HDF5. Default keeps only the first four successful demos.",
    )
    parser.add_argument(
        "--generate-config",
        default=str(
            repo_root
            / "repos"
            / "DemoGen"
            / "demo_generation"
            / "demo_generation"
            / "config"
            / "nutassemblyround_0_v1_replayh1_twostage_selected4.yaml"
        ),
    )
    parser.add_argument(
        "--train-base-config",
        default=str(
            repo_root
            / "configs"
            / "robomimic"
            / "diffusion_policy_nutassemblyround_relalign_replayobs_lowdim_external_120epoch.json"
        ),
    )
    parser.add_argument(
        "--source-name",
        default="nutassemblyround_0_v1_replayh1_twostage_source_selected4",
    )
    parser.add_argument(
        "--generated-name",
        default=None,
        help="Generated dataset stem. Defaults to nutassemblyround_0_v1_replayh1_twostage_selected4_grid<N>.",
    )
    parser.add_argument("--n-gen-per-source", type=int, default=81)
    parser.add_argument("--train-epochs", type=int, default=120)
    parser.add_argument("--save-every-n-epochs", type=int, default=30)
    parser.add_argument("--rollout-every-n-epochs", type=int, default=60)
    parser.add_argument("--rollout-n", type=int, default=10)
    parser.add_argument("--rollout-horizon", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--control-steps", type=int, default=1)
    parser.add_argument("--action-deviation-weight", type=float, default=1e-4)
    parser.add_argument("--motion1-relative-tail-steps", type=int, default=40)
    parser.add_argument("--motion1-relative-cost-weight", type=float, default=4.0)
    parser.add_argument("--motion2-relative-tail-steps", type=int, default=40)
    parser.add_argument("--motion2-relative-cost-weight", type=float, default=4.0)
    parser.add_argument("--eval-checkpoint-epoch", type=int, default=60)
    parser.add_argument("--eval-rollouts", type=int, default=20)
    parser.add_argument("--eval-horizon", type=int, default=800)
    parser.add_argument("--eval-parallel-jobs", type=int, default=3)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--gpu-log-interval-sec", type=int, default=30)
    parser.add_argument(
        "--gpu-warn-temp-c",
        type=float,
        default=82.0,
        help="Only warn when GPU temp crosses this threshold; does not stop the run.",
    )
    parser.add_argument(
        "--gpu-stop-temp-c",
        type=float,
        default=None,
        help="Optional hard stop threshold. Leave unset to avoid aborting an overnight run.",
    )
    parser.add_argument("--skip-source-pipeline", action="store_true")
    parser.add_argument("--skip-source-smoke", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-solve", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional experiment name suffix. Defaults to a timestamped nut twostage tag.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def log_line(path: Path, message: str) -> None:
    line = f"[{now_str()}] {message}"
    print(line, flush=True)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_remove(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


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
    if compute:
        compute_desc = ", ".join(
            f"pid={row['pid']} name={row['name']} mem={row['mem_mib']:.0f}MiB" for row in compute
        )
    else:
        compute_desc = "none"

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
            log_line(
                pipeline_log,
                f"GPU BUSY before {context}; waiting for compute processes to exit",
            )
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
    env: dict[str, str] | None = None,
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
            env=env,
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
                        f"{name} terminated because GPU temp {max_temp:.1f}C exceeded "
                        f"{gpu_stop_temp_c:.1f}C"
                    )
                next_gpu_log = time.time() + float(gpu_log_interval_sec)
            time.sleep(5.0)
        return_code = proc.wait()
        if return_code != 0:
            log_line(
                pipeline_log,
                f"STEP FAIL {name}: returncode={return_code} log={step_log}",
            )
            raise RuntimeError(f"{name} failed with returncode={return_code}. See {step_log}")
    log_line(pipeline_log, f"STEP DONE {name}: log={step_log}")


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


def find_train_run_dir(run_root: Path) -> Path | None:
    if not run_root.exists():
        return None
    children = sorted([p for p in run_root.iterdir() if p.is_dir()])
    if not children:
        return None
    return children[-1]


def launch_training_and_wait(
    *,
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
        "conda",
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
        env=os.environ.copy(),
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
                        "conda",
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
            raise RuntimeError(
                f"Training failed with returncode={return_code}. See launcher log: {launcher_log}"
            )
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


def write_report(report_path: Path, manifest: dict) -> None:
    lines = [
        "# NutAssemblyRound Twostage Replayobs Train120",
        "",
        f"- status: `{manifest.get('status')}`",
        f"- external_root: `{manifest.get('external_root')}`",
        f"- raw_demo_hdf5: `{manifest.get('raw_demo_hdf5')}`",
        f"- selected_demo_hdf5: `{manifest.get('selected_demo_hdf5')}`",
        f"- low_dim_hdf5: `{manifest.get('low_dim_hdf5')}`",
        f"- depth_hdf5: `{manifest.get('depth_hdf5')}`",
        f"- source_zarr: `{manifest.get('source_zarr')}`",
        f"- generated_zarr: `{manifest.get('generated_zarr')}`",
        f"- solved_zarr: `{manifest.get('solved_zarr')}`",
        f"- exported_hdf5: `{manifest.get('exported_hdf5')}`",
        f"- train_config: `{manifest.get('train_config')}`",
        f"- train_run_dir: `{manifest.get('train_run_dir')}`",
        f"- checkpoint_epoch60: `{manifest.get('checkpoint_epoch60')}`",
        f"- checkpoint_epoch120: `{manifest.get('checkpoint_epoch120')}`",
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


def snapshot_input_file(src: Path, dst_dir: Path) -> Path:
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def init_paths(external_root: Path) -> dict[str, Path]:
    paths = {
        "root": external_root,
        "data_root": external_root / "data",
        "source_root": external_root / "data" / "datasets" / "source",
        "generated_root": external_root / "data" / "datasets" / "generated",
        "sam_mask_root": external_root / "data" / "sam_mask",
        "raw_root": external_root / "raw",
        "analysis_root": external_root / "outputs" / "analysis",
        "generated_out_root": external_root / "outputs" / "generated",
        "robomimic_root": external_root / "outputs" / "robomimic",
        "train_output_dir": external_root / "outputs" / "robomimic" / "diffusion_policy_demogen",
        "hydra_root": external_root / "outputs" / "hydra",
        "config_snapshot_root": external_root / "inputs",
        "log_root": external_root / "logs",
        "step_log_root": external_root / "logs" / "steps",
        "pipeline_log": external_root / "logs" / "pipeline.log",
        "gpu_log": external_root / "logs" / "gpu_health.log",
        "manifest_path": external_root / "manifest.json",
        "report_path": external_root / "report.md",
    }
    for key, path in paths.items():
        if key.endswith("_log") or key.endswith("_path"):
            continue
        ensure_dir(path)
    return paths


def update_manifest(manifest_path: Path, manifest: dict, **updates) -> dict:
    manifest.update(updates)
    manifest["updated_at"] = now_str()
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    default_tag = f"nutassemblyround_twostage_grid{args.n_gen_per_source}_train{args.train_epochs}_{timestamp}"
    run_tag = args.run_tag or default_tag
    external_root = (
        Path(args.external_root).expanduser().resolve()
        if args.external_root
        else Path("/media/willzhang/KINGSTON/zlxtemp") / run_tag
    )
    paths = init_paths(external_root)
    pipeline_log = paths["pipeline_log"]
    gpu_log = paths["gpu_log"]

    raw_demo_hdf5 = Path(args.raw_demo_hdf5).expanduser().resolve()
    generate_config = Path(args.generate_config).expanduser().resolve()
    train_base_config = Path(args.train_base_config).expanduser().resolve()
    if not raw_demo_hdf5.exists():
        raise FileNotFoundError(f"raw demo not found: {raw_demo_hdf5}")
    if not generate_config.exists():
        raise FileNotFoundError(f"generate config not found: {generate_config}")
    if not train_base_config.exists():
        raise FileNotFoundError(f"train base config not found: {train_base_config}")

    raw_bundle_root = paths["raw_root"] / raw_demo_hdf5.parent.parent.name / raw_demo_hdf5.parent.name
    ensure_dir(raw_bundle_root)
    selected_demo_hdf5 = raw_bundle_root / "demo_selected4.hdf5"
    low_dim_hdf5 = raw_bundle_root / "low_dim_selected4.hdf5"
    depth_hdf5 = raw_bundle_root / "depth_selected4.hdf5"
    source_pipeline_manifest = raw_bundle_root / "nutassemblyround_source_pipeline_manifest.json"

    generated_name = args.generated_name or f"nutassemblyround_0_v1_replayh1_twostage_selected4_grid{args.n_gen_per_source}"
    generated_zarr = paths["generated_root"] / f"{generated_name}_test_{args.n_gen_per_source}.zarr"
    solved_stem = f"{generated_name}_relalign_twostage_all"
    solved_zarr = paths["generated_out_root"] / f"{solved_stem}.zarr"
    solve_json = paths["analysis_root"] / f"{solved_stem}.json"
    exported_hdf5 = paths["robomimic_root"] / f"{generated_name}_relalign_twostage_replayobs_lowdim.hdf5"
    train_config = paths["robomimic_root"] / "train120_config.json"
    train_run_name = (
        f"{generated_name}_relalign_replayobs_dp_{args.save_every_n_epochs}save"
        f"{args.rollout_every_n_epochs}rollout{args.rollout_n}_h{args.rollout_horizon}_"
        f"{args.train_epochs}epoch_{timestamp}"
    )

    snapshot_root = paths["config_snapshot_root"]
    snapped_generate_config = snapshot_input_file(generate_config, snapshot_root)
    snapped_train_base_config = snapshot_input_file(train_base_config, snapshot_root)
    args_snapshot = snapshot_root / "run_args.json"
    args_snapshot.write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "status": "running",
        "external_root": str(external_root),
        "raw_demo_hdf5": str(raw_demo_hdf5),
        "selected_episodes": args.selected_episodes,
        "selected_demo_hdf5": str(selected_demo_hdf5),
        "low_dim_hdf5": str(low_dim_hdf5),
        "depth_hdf5": str(depth_hdf5),
        "source_name": args.source_name,
        "source_zarr": str(paths["source_root"] / f"{args.source_name}.zarr"),
        "generated_name": generated_name,
        "generated_zarr": str(generated_zarr),
        "solved_zarr": str(solved_zarr),
        "solve_json": str(solve_json),
        "exported_hdf5": str(exported_hdf5),
        "train_config": str(train_config),
        "train_run_name": train_run_name,
        "train_base_config_snapshot": str(snapped_train_base_config),
        "generate_config_snapshot": str(snapped_generate_config),
        "args_snapshot": str(args_snapshot),
        "source_pipeline_manifest": str(source_pipeline_manifest),
        "eval_checkpoint_epoch": int(args.eval_checkpoint_epoch),
        "eval_rollouts": int(args.eval_rollouts),
        "eval_horizon": int(args.eval_horizon),
        "eval_seeds": [int(seed) for seed in args.eval_seeds],
        "started_at": now_str(),
        "updated_at": now_str(),
        "error": None,
    }
    write_json(paths["manifest_path"], manifest)
    write_report(paths["report_path"], manifest)

    log_line(pipeline_log, f"EXPERIMENT ROOT: {external_root}")
    log_line(pipeline_log, f"INPUT SNAPSHOTS: config={snapped_generate_config} train_base={snapped_train_base_config}")
    log_gpu_health(gpu_log=gpu_log, pipeline_log=pipeline_log, context="START", warn_temp_c=args.gpu_warn_temp_c)

    try:
        if args.skip_source_pipeline:
            if not selected_demo_hdf5.exists():
                raise FileNotFoundError(
                    f"--skip-source-pipeline was set but selected demo is missing: {selected_demo_hdf5}"
                )
            if not low_dim_hdf5.exists():
                raise FileNotFoundError(f"--skip-source-pipeline was set but low_dim is missing: {low_dim_hdf5}")
            if not depth_hdf5.exists():
                raise FileNotFoundError(f"--skip-source-pipeline was set but depth is missing: {depth_hdf5}")
            if not Path(manifest["source_zarr"]).exists():
                raise FileNotFoundError(
                    f"--skip-source-pipeline was set but source zarr is missing: {manifest['source_zarr']}"
                )
            log_line(pipeline_log, f"REUSE SOURCE PIPELINE OUTPUTS: {selected_demo_hdf5}")
        else:
            run_step(
                "prepare_source_from_raw_hdf5",
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "demogen",
                    "python",
                    str(repo_root / "scripts" / "run_twostage_raw_hdf5_pipeline.py"),
                    "--config",
                    str(generate_config),
                    "--raw-demo-hdf5",
                    str(raw_demo_hdf5),
                    "--selected-episodes",
                    args.selected_episodes,
                    "--data-root",
                    str(paths["data_root"]),
                    "--selected-demo-hdf5",
                    str(selected_demo_hdf5),
                    "--low-dim-hdf5",
                    str(low_dim_hdf5),
                    "--depth-hdf5",
                    str(depth_hdf5),
                    "--manifest-out",
                    str(source_pipeline_manifest),
                    "--overwrite",
                    *([] if not args.skip_source_smoke else ["--skip-smoke"]),
                ],
                cwd=repo_root,
                pipeline_log=pipeline_log,
                step_log=paths["step_log_root"] / "prepare_source_from_raw_hdf5.log",
                gpu_log=gpu_log,
                gpu_log_interval_sec=args.gpu_log_interval_sec,
                gpu_warn_temp_c=args.gpu_warn_temp_c,
                gpu_stop_temp_c=args.gpu_stop_temp_c,
            )

        if args.skip_generate:
            if not generated_zarr.exists():
                raise FileNotFoundError(f"--skip-generate was set but generated zarr is missing: {generated_zarr}")
            log_line(pipeline_log, f"REUSE GENERATED ZARR: {generated_zarr}")
        elif generated_zarr.exists():
            log_line(pipeline_log, f"GENERATED ZARR ALREADY EXISTS, REUSING: {generated_zarr}")
        else:
            run_step(
                "generate_grid81_dataset",
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "demogen",
                    "python",
                    "-W",
                    "ignore",
                    str(repo_root / "repos" / "DemoGen" / "demo_generation" / "gen_demo.py"),
                    f"--config-name={generate_config.name}",
                    f"data_root={paths['data_root']}",
                    f"source_name={args.source_name}",
                    f"source_demo_hdf5={selected_demo_hdf5}",
                    f"generated_name={generated_name}",
                    "generation.range_name=test",
                    "generation.mode=grid",
                    f"generation.n_gen_per_source={args.n_gen_per_source}",
                    "generation.render_video=False",
                    f"hydra.run.dir={paths['hydra_root'] / 'generate'}",
                ],
                cwd=generate_config.parents[2],
                pipeline_log=pipeline_log,
                step_log=paths["step_log_root"] / "generate_grid81_dataset.log",
                gpu_log=gpu_log,
                gpu_log_interval_sec=args.gpu_log_interval_sec,
                gpu_warn_temp_c=args.gpu_warn_temp_c,
                gpu_stop_temp_c=args.gpu_stop_temp_c,
            )

        if args.skip_solve:
            if not solved_zarr.exists() or not solve_json.exists():
                raise FileNotFoundError(
                    f"--skip-solve was set but solved outputs are missing: solved_zarr={solved_zarr} "
                    f"solve_json={solve_json}"
                )
            log_line(pipeline_log, f"REUSE SOLVED OUTPUTS: {solved_zarr}")
        elif solved_zarr.exists() and solve_json.exists():
            log_line(pipeline_log, f"SOLVED OUTPUTS ALREADY EXIST, REUSING: {solved_zarr}")
        else:
            safe_remove(solved_zarr)
            safe_remove(solve_json)
            run_step(
                "solve_relalign_twostage_all",
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "demogen",
                    "python",
                    str(repo_root / "scripts" / "export_stack_solved_from_template_zarr_relalign_twostage.py"),
                    "--config",
                    str(generate_config),
                    "--data-root",
                    str(paths["data_root"]),
                    "--source-demo",
                    str(selected_demo_hdf5),
                    "--template-zarr",
                    str(generated_zarr),
                    "--episodes",
                    "all",
                    "--control-steps",
                    str(args.control_steps),
                    "--action-deviation-weight",
                    str(args.action_deviation_weight),
                    "--motion1-relative-tail-steps",
                    str(args.motion1_relative_tail_steps),
                    "--motion1-relative-cost-weight",
                    str(args.motion1_relative_cost_weight),
                    "--motion2-relative-tail-steps",
                    str(args.motion2_relative_tail_steps),
                    "--motion2-relative-cost-weight",
                    str(args.motion2_relative_cost_weight),
                    "--output-zarr",
                    str(solved_zarr),
                    "--output-json",
                    str(solve_json),
                ],
                cwd=repo_root,
                pipeline_log=pipeline_log,
                step_log=paths["step_log_root"] / "solve_relalign_twostage_all.log",
                gpu_log=gpu_log,
                gpu_log_interval_sec=args.gpu_log_interval_sec,
                gpu_warn_temp_c=args.gpu_warn_temp_c,
                gpu_stop_temp_c=args.gpu_stop_temp_c,
            )

        if args.skip_export:
            if not exported_hdf5.exists():
                raise FileNotFoundError(f"--skip-export was set but exported hdf5 is missing: {exported_hdf5}")
            log_line(pipeline_log, f"REUSE EXPORTED HDF5: {exported_hdf5}")
        else:
            safe_remove(exported_hdf5)
            run_step(
                "export_replayobs_hdf5",
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "demogen",
                    "python",
                    str(
                        repo_root
                        / "repos"
                        / "DemoGen"
                        / "real_world"
                        / "export_demogen_zarr_to_robomimic_lowdim_replayobs_twophase.py"
                    ),
                    "--generated-zarr",
                    str(solved_zarr),
                    "--source-low-dim-hdf5",
                    str(low_dim_hdf5),
                    "--output-hdf5",
                    str(exported_hdf5),
                    "--include-source-demos",
                    "--overwrite",
                    "--control-steps",
                    str(args.control_steps),
                ],
                cwd=repo_root,
                pipeline_log=pipeline_log,
                step_log=paths["step_log_root"] / "export_replayobs_hdf5.log",
                gpu_log=gpu_log,
                gpu_log_interval_sec=args.gpu_log_interval_sec,
                gpu_warn_temp_c=args.gpu_warn_temp_c,
                gpu_stop_temp_c=args.gpu_stop_temp_c,
            )

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

        train_run_dir: Path | None = None
        if args.skip_train:
            log_line(pipeline_log, "SKIP TRAIN: requested by flag")
        else:
            train_run_dir = launch_training_and_wait(
                repo_root=repo_root,
                pipeline_log=pipeline_log,
                gpu_log=gpu_log,
                gpu_log_interval_sec=args.gpu_log_interval_sec,
                gpu_warn_temp_c=args.gpu_warn_temp_c,
                gpu_stop_temp_c=args.gpu_stop_temp_c,
                train_config=train_config,
                dataset_path=exported_hdf5,
                train_output_dir=paths["train_output_dir"],
                run_name=train_run_name,
                launcher_log=paths["log_root"] / "train_launcher.log",
                watcher_title="Nut Replayobs",
            )
            checkpoint_epoch60 = train_run_dir / "models" / f"model_epoch_{args.eval_checkpoint_epoch}.pth"
            checkpoint_epoch120 = train_run_dir / "models" / f"model_epoch_{args.train_epochs}.pth"
            if not checkpoint_epoch60.exists():
                raise FileNotFoundError(f"Expected eval checkpoint not found: {checkpoint_epoch60}")
            if not checkpoint_epoch120.exists():
                raise FileNotFoundError(f"Expected final checkpoint not found: {checkpoint_epoch120}")
            update_manifest(
                paths["manifest_path"],
                manifest,
                train_run_dir=str(train_run_dir),
                checkpoint_epoch60=str(checkpoint_epoch60),
                checkpoint_epoch120=str(checkpoint_epoch120),
            )
            write_report(paths["report_path"], manifest)

        if args.skip_eval:
            log_line(pipeline_log, "SKIP EVAL: requested by flag")
        else:
            if train_run_dir is None:
                run_root = paths["train_output_dir"] / train_run_name
                train_run_dir = find_train_run_dir(run_root)
                if train_run_dir is None:
                    raise FileNotFoundError(f"Could not infer train run dir under {run_root} for eval")
            checkpoint_epoch60 = train_run_dir / "models" / f"model_epoch_{args.eval_checkpoint_epoch}.pth"
            if not checkpoint_epoch60.exists():
                raise FileNotFoundError(f"Expected eval checkpoint not found: {checkpoint_epoch60}")
            eval_output_dir = (
                train_run_dir
                / f"eval_epoch{args.eval_checkpoint_epoch}_r{args.eval_rollouts}_"
                f"seeds{''.join(str(seed) for seed in args.eval_seeds)}_{timestamp}"
            )
            run_step(
                "eval_epoch60_multiseed",
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "robomimic",
                    "python",
                    str(repo_root / "scripts" / "eval_robomimic_checkpoint_multiseed.py"),
                    "--checkpoint",
                    str(checkpoint_epoch60),
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
                ],
                cwd=repo_root,
                pipeline_log=pipeline_log,
                step_log=paths["step_log_root"] / "eval_epoch60_multiseed.log",
                gpu_log=gpu_log,
                gpu_log_interval_sec=args.gpu_log_interval_sec,
                gpu_warn_temp_c=args.gpu_warn_temp_c,
                gpu_stop_temp_c=args.gpu_stop_temp_c,
                wait_for_gpu_before=True,
            )
            eval_summary = summarize_eval_results(eval_output_dir / "reports" / "summary.json")
            update_manifest(
                paths["manifest_path"],
                manifest,
                eval_output_dir=str(eval_output_dir),
                eval_summary=eval_summary,
            )
            write_report(paths["report_path"], manifest)

        update_manifest(paths["manifest_path"], manifest, status="completed", error=None)
        write_report(paths["report_path"], manifest)
        log_gpu_health(gpu_log=gpu_log, pipeline_log=pipeline_log, context="END", warn_temp_c=args.gpu_warn_temp_c)
        log_line(paths["pipeline_log"], f"PIPELINE DONE: report={paths['report_path']}")
        return 0
    except Exception as exc:
        update_manifest(paths["manifest_path"], manifest, status="failed", error=str(exc))
        write_report(paths["report_path"], manifest)
        log_line(pipeline_log, f"PIPELINE FAIL: {exc}")
        log_gpu_health(gpu_log=gpu_log, pipeline_log=pipeline_log, context="FAIL", warn_temp_c=args.gpu_warn_temp_c)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
