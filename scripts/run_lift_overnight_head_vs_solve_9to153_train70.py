#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path


def now_str() -> str:
    return time.strftime("%F %T")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run the overnight serial comparison for Lift 9->153: "
            "(1) old head-gen route -> replayobs export -> train@70 -> eval 20x3, "
            "(2) v37 solve route -> replayobs export -> train@70 -> eval 20x3. "
            "Logs all outputs to external storage and keeps periodic GPU health traces."
        )
    )
    parser.add_argument(
        "--external-root",
        default=None,
        help=(
            "Top-level experiment root on external storage. Defaults to "
            "/media/willzhang/KINGSTON/zlxtemp/<timestamped-dir>."
        ),
    )
    parser.add_argument(
        "--full-demo-hdf5",
        default=str(repo_root / "data" / "raw" / "lift_0" / "1774702988_8036063" / "demo.hdf5"),
    )
    parser.add_argument(
        "--full-low-dim-hdf5",
        default=str(repo_root / "data" / "raw" / "lift_0" / "1774702988_8036063" / "low_dim.hdf5"),
    )
    parser.add_argument(
        "--head-generated-zarr",
        default=(
            "/media/willzhang/KINGSTON/zlxtemp/repro_archive_20260409/"
            "datasets_generated/lift_0_v28_originalschedule_phasecopy_statedelta_halfcorr_v9_s220_test_16.zarr"
        ),
        help="Existing full 9->153 old-route generated zarr to reuse for the head-gen branch.",
    )
    parser.add_argument(
        "--head-source-zarr",
        default=str(
            repo_root
            / "repos"
            / "DemoGen"
            / "data"
            / "datasets"
            / "source"
            / "lift_0_v21_originalschedule_motion_v9_s220.zarr"
        ),
        help="Fallback old-route source zarr if --head-generated-zarr is missing and we need to regenerate.",
    )
    parser.add_argument(
        "--solve-source-zarr",
        default=str(
            repo_root
            / "repos"
            / "DemoGen"
            / "data"
            / "datasets"
            / "source"
            / "lift_0_v31_replayh1_light_schedule_source.zarr"
        ),
    )
    parser.add_argument(
        "--solve-config",
        default=str(
            repo_root
            / "repos"
            / "DemoGen"
            / "demo_generation"
            / "demo_generation"
            / "config"
            / "lift_0_v37_replayh1_light_schedule_phasecopy_replayconsistent_full9_diagfix_m20.yaml"
        ),
    )
    parser.add_argument(
        "--oldroute-config-name",
        default="lift_0_v28_originalschedule_phasecopy_statedelta_halfcorr_v9_s220.yaml",
    )
    parser.add_argument(
        "--train-base-config",
        default=str(
            repo_root
            / "configs"
            / "robomimic"
            / "diffusion_policy_lift_v37_selected4_d2467_relalign_all_diagfix_replayobs_lowdim_30save_30rollout10_h1000_external_300epoch.json"
        ),
    )
    parser.add_argument("--n-gen-per-source", type=int, default=16)
    parser.add_argument("--train-epochs", type=int, default=70)
    parser.add_argument("--save-every-n-epochs", type=int, default=10)
    parser.add_argument("--eval-rollouts", type=int, default=20)
    parser.add_argument("--eval-horizon", type=int, default=1000)
    parser.add_argument(
        "--eval-parallel-jobs",
        type=int,
        default=1,
        help="Default is 1 to keep eval fully serial on a single GPU overnight.",
    )
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--control-steps", type=int, default=1)
    parser.add_argument(
        "--solve-num-workers",
        type=int,
        default=4,
        help="CPU workers for relalign solve export. GPU work remains serial.",
    )
    parser.add_argument("--gpu-log-interval-sec", type=int, default=30)
    parser.add_argument("--gpu-max-temp-c", type=float, default=82.0)
    parser.add_argument(
        "--branches",
        nargs="+",
        default=["solve", "head_gen"],
        help="Branch order to run serially. Supported values: solve, head_gen",
    )
    parser.add_argument(
        "--force-head-regen",
        action="store_true",
        help="Ignore --head-generated-zarr and regenerate the head branch from source.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the full overnight queue after the first experiment failure.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Stop after training finishes and checkpoint@target epoch is saved.",
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


def rewrite_yaml_scalars(
    *,
    input_path: Path,
    output_path: Path,
    updates: dict[str, str],
) -> None:
    text = input_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    seen: set[str] = set()
    out_lines: list[str] = []
    pattern_map = {
        key: re.compile(rf"^(\s*){re.escape(key)}\s*:\s*.*$")
        for key in updates
    }
    for line in lines:
        replaced = False
        for key, pattern in pattern_map.items():
            match = pattern.match(line)
            if match is None:
                continue
            indent = match.group(1)
            out_lines.append(f"{indent}{key}: {updates[key]}")
            seen.add(key)
            replaced = True
            break
        if not replaced:
            out_lines.append(line)

    missing = sorted(set(updates) - seen)
    if missing:
        raise KeyError(f"Could not rewrite YAML keys {missing} in {input_path}")

    output_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    ensure_dir(link_path.parent)
    if link_path.is_symlink():
        if link_path.resolve() == target_path.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        safe_remove(link_path)
    link_path.symlink_to(target_path, target_is_directory=target_path.is_dir())


def prepare_source_bundle_link(
    *,
    source_zarr: Path,
    external_data_root: Path,
    pipeline_log: Path,
) -> tuple[Path, Path | None]:
    source_name = source_zarr.stem
    source_link = external_data_root / "datasets" / "source" / source_zarr.name
    ensure_symlink(source_link, source_zarr)

    sam_mask_target = source_zarr.parents[2] / "sam_mask" / source_name
    sam_mask_link = external_data_root / "sam_mask" / source_name
    if sam_mask_target.exists():
        ensure_symlink(sam_mask_link, sam_mask_target)
        log_line(
            pipeline_log,
            f"SOURCE BUNDLE READY: source={source_link} sam_mask={sam_mask_link}",
        )
        return source_link, sam_mask_link

    log_line(
        pipeline_log,
        f"SOURCE BUNDLE READY: source={source_link} sam_mask_missing={sam_mask_target}",
    )
    return source_link, None


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
    return max((row["temp_c"] for row in gpus), default=None)


def wait_for_gpu_compute_idle(
    *,
    pipeline_log: Path,
    gpu_log: Path,
    context: str,
    poll_interval_sec: float = 10.0,
    required_idle_samples: int = 2,
) -> None:
    idle = 0
    while idle < required_idle_samples:
        max_temp = log_gpu_health(gpu_log=gpu_log, pipeline_log=pipeline_log, context=f"WAIT {context}")
        compute = query_gpu_compute_processes()
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
    env: dict[str, str] | None = None,
    gpu_log: Path | None = None,
    gpu_log_interval_sec: int = 30,
    gpu_max_temp_c: float | None = None,
    wait_for_gpu_before: bool = False,
) -> None:
    ensure_dir(step_log.parent)
    if wait_for_gpu_before and gpu_log is not None:
        wait_for_gpu_compute_idle(
            pipeline_log=pipeline_log,
            gpu_log=gpu_log,
            context=name,
        )

    log_line(pipeline_log, f"STEP START {name}: {shlex.join(cmd)}")
    with step_log.open("w", encoding="utf-8") as f:
        f.write(f"$ {shlex.join(cmd)}\n\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        next_gpu_log = time.time()
        while proc.poll() is None:
            if gpu_log is not None and time.time() >= next_gpu_log:
                max_temp = log_gpu_health(
                    gpu_log=gpu_log,
                    pipeline_log=pipeline_log,
                    context=f"RUN {name}",
                )
                if gpu_max_temp_c is not None and max_temp is not None and max_temp >= gpu_max_temp_c:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise RuntimeError(
                        f"{name} terminated because GPU temp {max_temp:.1f}C exceeded {gpu_max_temp_c:.1f}C"
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


def preflight_generator(
    *,
    repo_root: Path,
    config_path: Path,
    data_root: Path,
    pipeline_log: Path,
    step_log: Path,
) -> None:
    code = (
        "import sys; "
        "from pathlib import Path; "
        "sys.path.insert(0, sys.argv[3]); "
        "from solve_lift_prefix_xyz_actions import load_cfg, instantiate_generator; "
        "cfg = load_cfg(Path(sys.argv[1]), Path(sys.argv[2])); "
        "print(f'source_demo_hdf5={cfg.source_demo_hdf5}'); "
        "g = instantiate_generator(cfg); "
        "episode_ends = getattr(g.replay_buffer, 'episode_ends', None); "
        "n = len(episode_ends) if episode_ends is not None else 'unknown'; "
        "print(f'generator_ok episodes={n}')"
    )
    run_step(
        "solve_generator_preflight",
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "demogen",
            "python",
            "-c",
            code,
            str(config_path),
            str(data_root),
            str(repo_root / "scripts"),
        ],
        cwd=repo_root,
        pipeline_log=pipeline_log,
        step_log=step_log,
    )


def load_train_config(path: Path) -> dict:
    return load_json(path)


def build_train_config(
    *,
    base_config_path: Path,
    output_path: Path,
    output_dir: Path,
    num_epochs: int,
    save_every_n_epochs: int,
    experiment_name: str,
) -> None:
    cfg = load_train_config(base_config_path)
    cfg["experiment"]["name"] = experiment_name
    cfg["experiment"]["save"]["every_n_epochs"] = int(save_every_n_epochs)
    cfg["experiment"]["render_video"] = False
    cfg["experiment"]["rollout"]["enabled"] = False
    cfg["experiment"]["rollout"]["rate"] = int(num_epochs)
    cfg["train"]["output_dir"] = str(output_dir)
    cfg["train"]["num_epochs"] = int(num_epochs)
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
    gpu_max_temp_c: float,
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

    env = os.environ.copy()
    proc = subprocess.Popen(
        train_cmd,
        cwd=str(repo_root),
        stdout=launcher_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
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
                    watcher_log = run_dir / "logs" / "progress_watcher.log"
                    watcher_cmd = [
                        "conda",
                        "run",
                        "--no-capture-output",
                        "-n",
                        "robomimic",
                        "python",
                        str(repo_root / "scripts" / "watch_robomimic_progress_notify.py"),
                        "--run-dir",
                        str(run_dir),
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
                    )
                    log_line(
                        pipeline_log,
                        f"TRAIN RUN DIR READY: run_dir={run_dir} watcher_log={watcher_log}",
                    )

            if time.time() >= next_gpu_log:
                max_temp = log_gpu_health(
                    gpu_log=gpu_log,
                    pipeline_log=pipeline_log,
                    context=f"TRAIN {run_name}",
                )
                if max_temp is not None and max_temp >= gpu_max_temp_c:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise RuntimeError(
                        f"Training {run_name} terminated because GPU temp "
                        f"{max_temp:.1f}C exceeded {gpu_max_temp_c:.1f}C"
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
            watcher_proc.terminate()
            try:
                watcher_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                watcher_proc.kill()
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


def write_experiment_report(
    *,
    report_path: Path,
    title: str,
    manifest: dict,
) -> None:
    lines = [
        f"# {title}",
        "",
        f"- status: `{manifest.get('status')}`",
        f"- experiment_root: `{manifest.get('experiment_root')}`",
    ]
    if manifest.get("generated_zarr"):
        lines.append(f"- generated_zarr: `{manifest['generated_zarr']}`")
    if manifest.get("exported_hdf5"):
        lines.append(f"- exported_hdf5: `{manifest['exported_hdf5']}`")
    if manifest.get("train_run_dir"):
        lines.append(f"- train_run_dir: `{manifest['train_run_dir']}`")
    if manifest.get("checkpoint_epoch70"):
        lines.append(f"- checkpoint_epoch70: `{manifest['checkpoint_epoch70']}`")
    if manifest.get("eval_output_dir"):
        lines.append(f"- eval_output_dir: `{manifest['eval_output_dir']}`")
    if manifest.get("error"):
        lines.extend(["", "## Error", "", f"`{manifest['error']}`"])

    success_payload = manifest.get("success_gate")
    consistency_payload = manifest.get("consistency_gate")
    eval_payload = manifest.get("eval_summary")
    if success_payload or consistency_payload or eval_payload:
        lines.extend(["", "## Summary", ""])
    if success_payload:
        lines.append(
            f"- generated_success: `{success_payload.get('success_count')}/{success_payload.get('n_checked')}` "
            f"(rate={success_payload.get('success_rate')})"
        )
    if consistency_payload:
        lines.append(
            f"- consistency_pass: `{consistency_payload.get('n_pass')}/{consistency_payload.get('n_checked')}` "
            f"(rate={consistency_payload.get('pass_rate')})"
        )
    if eval_payload:
        lines.append(
            f"- eval_mean_success_rate: `{eval_payload.get('mean_success_rate')}` "
            f"over `{eval_payload.get('completed_seeds')}` seeds"
        )
        lines.append(f"- eval_mean_avg_return: `{eval_payload.get('mean_avg_return')}`")
        lines.append(f"- eval_mean_avg_horizon: `{eval_payload.get('mean_avg_horizon')}`")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_top_level_report(report_path: Path, payload: dict) -> None:
    lines = [
        "# Lift Overnight Head-vs-Solve 9->153 Train70",
        "",
        f"- root: `{payload.get('external_root')}`",
        f"- started_at: `{payload.get('started_at')}`",
        f"- updated_at: `{payload.get('updated_at')}`",
        "",
        "## Branches",
        "",
    ]
    for key in ["head_gen", "solve"]:
        exp = payload.get("experiments", {}).get(key, {})
        lines.append(f"- {key}: status=`{exp.get('status')}` root=`{exp.get('experiment_root')}`")
        eval_summary = exp.get("eval_summary") or {}
        if eval_summary.get("mean_success_rate") is not None:
            lines.append(
                f"  mean_success_rate=`{eval_summary.get('mean_success_rate')}` "
                f"completed_seeds=`{eval_summary.get('completed_seeds')}`"
            )
        if exp.get("error"):
            lines.append(f"  error=`{exp.get('error')}`")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def init_experiment_root(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "data_root": root / "data",
        "source_root": root / "data" / "datasets" / "source",
        "generated_root": root / "data" / "datasets" / "generated",
        "sam_mask_root": root / "data" / "sam_mask",
        "analysis_root": root / "outputs" / "analysis",
        "generated_out_root": root / "outputs" / "generated",
        "robomimic_root": root / "outputs" / "robomimic",
        "train_output_dir": root / "outputs" / "robomimic" / "diffusion_policy_demogen",
        "hydra_root": root / "outputs" / "hydra",
        "log_root": root / "logs",
        "step_log_root": root / "logs" / "steps",
        "pipeline_log": root / "logs" / "pipeline.log",
        "gpu_log": root / "logs" / "gpu_health.log",
        "manifest_path": root / "manifest.json",
        "report_path": root / "report.md",
    }
    for key, path in paths.items():
        if key.endswith("_log") or key.endswith("_path"):
            continue
        ensure_dir(path)
    return paths


def run_head_gen_experiment(args: argparse.Namespace, repo_root: Path, exp_root: Path) -> dict:
    paths = init_experiment_root(exp_root)
    pipeline_log = paths["pipeline_log"]
    gpu_log = paths["gpu_log"]
    manifest = {
        "name": "head_gen",
        "title": "Lift Head-Gen 9->153 Replayobs Train70",
        "status": "running",
        "experiment_root": str(exp_root),
        "started_at": now_str(),
        "updated_at": now_str(),
    }
    write_json(paths["manifest_path"], manifest)
    log_line(pipeline_log, f"EXPERIMENT ROOT: {exp_root}")

    full_demo_hdf5 = Path(args.full_demo_hdf5).expanduser().resolve()
    full_low_dim_hdf5 = Path(args.full_low_dim_hdf5).expanduser().resolve()
    head_generated_zarr = Path(args.head_generated_zarr).expanduser().resolve()
    head_source_zarr = Path(args.head_source_zarr).expanduser().resolve()

    if args.force_head_regen or not head_generated_zarr.exists():
        prepare_source_bundle_link(
            source_zarr=head_source_zarr,
            external_data_root=paths["data_root"],
            pipeline_log=pipeline_log,
        )
        generated_name = "lift_0_v28_originalschedule_phasecopy_statedelta_halfcorr_v9_s220_full9_train70"
        head_generated_zarr = paths["generated_root"] / f"{generated_name}_test_{args.n_gen_per_source}.zarr"
        run_step(
            "generate_head_oldroute_full9",
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
                f"--config-name={args.oldroute_config_name}",
                f"data_root={paths['data_root']}",
                f"source_name={head_source_zarr.stem}",
                f"generated_name={generated_name}",
                "generation.range_name=test",
                "generation.mode=grid",
                f"generation.n_gen_per_source={args.n_gen_per_source}",
                "generation.render_video=False",
                f"hydra.run.dir={paths['hydra_root'] / 'generate'}",
            ],
            cwd=repo_root / "repos" / "DemoGen" / "demo_generation",
            pipeline_log=pipeline_log,
            step_log=paths["step_log_root"] / "generate_head_oldroute_full9.log",
        )
    else:
        log_line(
            pipeline_log,
            f"REUSE HEAD GENERATED ZARR: {head_generated_zarr}",
        )

    manifest["generated_zarr"] = str(head_generated_zarr)
    manifest["updated_at"] = now_str()
    write_json(paths["manifest_path"], manifest)

    success_json = paths["analysis_root"] / "generated_success.json"
    run_step(
        "generated_success_check",
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "demogen",
            "python",
            str(repo_root / "scripts" / "eval_generated_zarr_success_rate.py"),
            "--zarr",
            str(head_generated_zarr),
            "--source-demo",
            str(full_demo_hdf5),
            "--control-steps",
            str(args.control_steps),
            "--output-json",
            str(success_json),
        ],
        cwd=repo_root,
        pipeline_log=pipeline_log,
        step_log=paths["step_log_root"] / "generated_success_check.log",
    )
    manifest["success_gate"] = load_json(success_json)
    manifest["updated_at"] = now_str()
    write_json(paths["manifest_path"], manifest)

    consistency_json = paths["analysis_root"] / "generated_consistency.json"
    run_step(
        "generated_consistency_check",
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "demogen",
            "python",
            str(repo_root / "scripts" / "validate_generated_zarr_consistency.py"),
            "--zarr",
            str(head_generated_zarr),
            "--source-demo",
            str(full_demo_hdf5),
            "--source-low-dim",
            str(full_low_dim_hdf5),
            "--control-steps",
            str(args.control_steps),
            "--rmse-threshold",
            "0.015",
            "--final-threshold",
            "0.015",
            "--require-pass",
            "--output-json",
            str(consistency_json),
        ],
        cwd=repo_root,
        pipeline_log=pipeline_log,
        step_log=paths["step_log_root"] / "generated_consistency_check.log",
    )
    manifest["consistency_gate"] = load_json(consistency_json)
    manifest["updated_at"] = now_str()
    write_json(paths["manifest_path"], manifest)

    exported_hdf5 = paths["robomimic_root"] / "lift_0_v28_full9_oldroute_replayobs_lowdim.hdf5"
    run_step(
        "export_replayobs_hdf5",
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "demogen",
            "python",
            str(repo_root / "repos" / "DemoGen" / "real_world" / "export_demogen_zarr_to_robomimic_lowdim_replayobs.py"),
            "--generated-zarr",
            str(head_generated_zarr),
            "--source-low-dim-hdf5",
            str(full_low_dim_hdf5),
            "--output-hdf5",
            str(exported_hdf5),
            "--include-source-demos",
            "--overwrite",
            "--generated-obs-mode",
            "replay",
            "--control-steps",
            str(args.control_steps),
        ],
        cwd=repo_root,
        pipeline_log=pipeline_log,
        step_log=paths["step_log_root"] / "export_replayobs_hdf5.log",
    )
    manifest["exported_hdf5"] = str(exported_hdf5)
    manifest["updated_at"] = now_str()
    write_json(paths["manifest_path"], manifest)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    train_run_name = f"lift_0_v28_full9_oldroute_replayobs_train70_{timestamp}"
    train_config = paths["robomimic_root"] / "train70_config.json"
    build_train_config(
        base_config_path=Path(args.train_base_config).expanduser().resolve(),
        output_path=train_config,
        output_dir=paths["train_output_dir"],
        num_epochs=args.train_epochs,
        save_every_n_epochs=args.save_every_n_epochs,
        experiment_name=train_run_name,
    )
    manifest["train_config"] = str(train_config)
    manifest["train_run_name"] = train_run_name
    manifest["updated_at"] = now_str()
    write_json(paths["manifest_path"], manifest)

    train_run_dir = launch_training_and_wait(
        repo_root=repo_root,
        pipeline_log=pipeline_log,
        gpu_log=gpu_log,
        gpu_log_interval_sec=args.gpu_log_interval_sec,
        gpu_max_temp_c=args.gpu_max_temp_c,
        train_config=train_config,
        dataset_path=exported_hdf5,
        train_output_dir=paths["train_output_dir"],
        run_name=train_run_name,
        launcher_log=paths["log_root"] / "train_launcher.log",
        watcher_title="Lift HeadGen Full9",
    )
    manifest["train_run_dir"] = str(train_run_dir)
    manifest["updated_at"] = now_str()
    write_json(paths["manifest_path"], manifest)

    checkpoint_epoch70 = train_run_dir / "models" / f"model_epoch_{args.train_epochs}.pth"
    if not checkpoint_epoch70.exists():
        raise FileNotFoundError(f"Expected checkpoint not found: {checkpoint_epoch70}")
    manifest["checkpoint_epoch70"] = str(checkpoint_epoch70)
    manifest["updated_at"] = now_str()
    write_json(paths["manifest_path"], manifest)

    if args.skip_eval:
        manifest["status"] = "training_completed_skip_eval"
        manifest["updated_at"] = now_str()
        write_json(paths["manifest_path"], manifest)
        write_experiment_report(report_path=paths["report_path"], title=manifest["title"], manifest=manifest)
        log_line(pipeline_log, f"TRAINING DONE (skip eval): report={paths['report_path']}")
        return manifest

    eval_output_dir = train_run_dir / (
        f"eval_epoch{args.train_epochs}_r{args.eval_rollouts}_"
        f"seeds{''.join(str(s) for s in args.eval_seeds)}_{timestamp}"
    )
    run_step(
        "eval_epoch70_multiseed",
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "robomimic",
            "python",
            str(repo_root / "scripts" / "eval_robomimic_checkpoint_multiseed.py"),
            "--checkpoint",
            str(checkpoint_epoch70),
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
        step_log=paths["step_log_root"] / "eval_epoch70_multiseed.log",
        gpu_log=gpu_log,
        gpu_log_interval_sec=args.gpu_log_interval_sec,
        gpu_max_temp_c=args.gpu_max_temp_c,
        wait_for_gpu_before=True,
    )
    manifest["eval_output_dir"] = str(eval_output_dir)
    manifest["eval_summary"] = summarize_eval_results(eval_output_dir / "reports" / "summary.json")
    manifest["status"] = "completed"
    manifest["updated_at"] = now_str()
    write_json(paths["manifest_path"], manifest)
    write_experiment_report(report_path=paths["report_path"], title=manifest["title"], manifest=manifest)
    log_line(pipeline_log, f"PIPELINE DONE: report={paths['report_path']}")
    return manifest


def run_solve_experiment(args: argparse.Namespace, repo_root: Path, exp_root: Path) -> dict:
    paths = init_experiment_root(exp_root)
    pipeline_log = paths["pipeline_log"]
    gpu_log = paths["gpu_log"]
    manifest = {
        "name": "solve",
        "title": "Lift Solve 9->153 Replayobs Train70",
        "status": "running",
        "experiment_root": str(exp_root),
        "started_at": now_str(),
        "updated_at": now_str(),
    }
    write_json(paths["manifest_path"], manifest)
    log_line(pipeline_log, f"EXPERIMENT ROOT: {exp_root}")

    full_demo_hdf5 = Path(args.full_demo_hdf5).expanduser().resolve()
    full_low_dim_hdf5 = Path(args.full_low_dim_hdf5).expanduser().resolve()
    solve_source_zarr = Path(args.solve_source_zarr).expanduser().resolve()
    solve_config = Path(args.solve_config).expanduser().resolve()
    resolved_solve_config = paths["root"] / "resolved_solve_config.yaml"

    rewrite_yaml_scalars(
        input_path=solve_config,
        output_path=resolved_solve_config,
        updates={
            "source_demo_hdf5": json.dumps(str(full_demo_hdf5)),
        },
    )
    manifest["resolved_config"] = str(resolved_solve_config)
    manifest["updated_at"] = now_str()
    write_json(paths["manifest_path"], manifest)

    prepare_source_bundle_link(
        source_zarr=solve_source_zarr,
        external_data_root=paths["data_root"],
        pipeline_log=pipeline_log,
    )

    preflight_generator(
        repo_root=repo_root,
        config_path=resolved_solve_config,
        data_root=paths["data_root"],
        pipeline_log=pipeline_log,
        step_log=paths["step_log_root"] / "solve_generator_preflight.log",
    )

    template_name = "lift_0_v37_replayh1_light_schedule_phasecopy_replayconsistent_full9_train70"
    template_zarr = paths["generated_root"] / f"{template_name}_test_{args.n_gen_per_source}.zarr"
    if template_zarr.exists():
        log_line(pipeline_log, f"REUSE SOLVE TEMPLATE ZARR: {template_zarr}")
    else:
        run_step(
            "generate_solve_template_full9",
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
                f"--config-path={resolved_solve_config.parent}",
                f"--config-name={resolved_solve_config.name}",
                f"data_root={paths['data_root']}",
                f"source_name={solve_source_zarr.stem}",
                f"generated_name={template_name}",
                "generation.range_name=test",
                "generation.mode=grid",
                f"generation.n_gen_per_source={args.n_gen_per_source}",
                "generation.render_video=False",
                f"hydra.run.dir={paths['hydra_root'] / 'generate_template'}",
            ],
            cwd=repo_root / "repos" / "DemoGen" / "demo_generation",
            pipeline_log=pipeline_log,
            step_log=paths["step_log_root"] / "generate_solve_template_full9.log",
        )
    manifest["template_zarr"] = str(template_zarr)
    manifest["updated_at"] = now_str()
    write_json(paths["manifest_path"], manifest)

    solved_zarr = paths["generated_out_root"] / "lift_0_v37_full9_relalign_all_train70.zarr"
    solve_json = paths["analysis_root"] / "relalign_solve_summary.json"
    solve_env = os.environ.copy()
    solve_env["OMP_NUM_THREADS"] = "1"
    solve_env["MKL_NUM_THREADS"] = "1"
    solve_env["OPENBLAS_NUM_THREADS"] = "1"
    run_step(
        "relalign_solve_export",
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "demogen",
            "python",
            str(repo_root / "scripts" / "export_lift_solved_from_template_zarr_relalign.py"),
            "--config",
            str(resolved_solve_config),
            "--data-root",
            str(paths["data_root"]),
            "--source-demo",
            str(full_demo_hdf5),
            "--template-zarr",
            str(template_zarr),
            "--control-steps",
            str(args.control_steps),
            "--num-workers",
            str(args.solve_num_workers),
            "--output-zarr",
            str(solved_zarr),
            "--output-json",
            str(solve_json),
        ],
        cwd=repo_root,
        pipeline_log=pipeline_log,
        step_log=paths["step_log_root"] / "relalign_solve_export.log",
        env=solve_env,
    )
    manifest["generated_zarr"] = str(solved_zarr)
    manifest["solve_summary_json"] = str(solve_json)
    manifest["updated_at"] = now_str()
    write_json(paths["manifest_path"], manifest)

    success_json = paths["analysis_root"] / "generated_success.json"
    run_step(
        "generated_success_check",
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "demogen",
            "python",
            str(repo_root / "scripts" / "eval_generated_zarr_success_rate.py"),
            "--zarr",
            str(solved_zarr),
            "--source-demo",
            str(full_demo_hdf5),
            "--control-steps",
            str(args.control_steps),
            "--output-json",
            str(success_json),
        ],
        cwd=repo_root,
        pipeline_log=pipeline_log,
        step_log=paths["step_log_root"] / "generated_success_check.log",
    )
    manifest["success_gate"] = load_json(success_json)
    manifest["updated_at"] = now_str()
    write_json(paths["manifest_path"], manifest)

    consistency_json = paths["analysis_root"] / "generated_consistency.json"
    run_step(
        "generated_consistency_check",
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "demogen",
            "python",
            str(repo_root / "scripts" / "validate_generated_zarr_consistency.py"),
            "--zarr",
            str(solved_zarr),
            "--source-demo",
            str(full_demo_hdf5),
            "--source-low-dim",
            str(full_low_dim_hdf5),
            "--control-steps",
            str(args.control_steps),
            "--rmse-threshold",
            "0.015",
            "--final-threshold",
            "0.015",
            "--require-pass",
            "--output-json",
            str(consistency_json),
        ],
        cwd=repo_root,
        pipeline_log=pipeline_log,
        step_log=paths["step_log_root"] / "generated_consistency_check.log",
    )
    manifest["consistency_gate"] = load_json(consistency_json)
    manifest["updated_at"] = now_str()
    write_json(paths["manifest_path"], manifest)

    exported_hdf5 = paths["robomimic_root"] / "lift_0_v37_full9_relalign_replayobs_lowdim.hdf5"
    run_step(
        "export_replayobs_hdf5",
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "demogen",
            "python",
            str(repo_root / "repos" / "DemoGen" / "real_world" / "export_demogen_zarr_to_robomimic_lowdim_replayobs.py"),
            "--generated-zarr",
            str(solved_zarr),
            "--source-low-dim-hdf5",
            str(full_low_dim_hdf5),
            "--output-hdf5",
            str(exported_hdf5),
            "--include-source-demos",
            "--overwrite",
            "--generated-obs-mode",
            "replay",
            "--control-steps",
            str(args.control_steps),
        ],
        cwd=repo_root,
        pipeline_log=pipeline_log,
        step_log=paths["step_log_root"] / "export_replayobs_hdf5.log",
    )
    manifest["exported_hdf5"] = str(exported_hdf5)
    manifest["updated_at"] = now_str()
    write_json(paths["manifest_path"], manifest)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    train_run_name = f"lift_0_v37_full9_relalign_replayobs_train70_{timestamp}"
    train_config = paths["robomimic_root"] / "train70_config.json"
    build_train_config(
        base_config_path=Path(args.train_base_config).expanduser().resolve(),
        output_path=train_config,
        output_dir=paths["train_output_dir"],
        num_epochs=args.train_epochs,
        save_every_n_epochs=args.save_every_n_epochs,
        experiment_name=train_run_name,
    )
    manifest["train_config"] = str(train_config)
    manifest["train_run_name"] = train_run_name
    manifest["updated_at"] = now_str()
    write_json(paths["manifest_path"], manifest)

    train_run_dir = launch_training_and_wait(
        repo_root=repo_root,
        pipeline_log=pipeline_log,
        gpu_log=gpu_log,
        gpu_log_interval_sec=args.gpu_log_interval_sec,
        gpu_max_temp_c=args.gpu_max_temp_c,
        train_config=train_config,
        dataset_path=exported_hdf5,
        train_output_dir=paths["train_output_dir"],
        run_name=train_run_name,
        launcher_log=paths["log_root"] / "train_launcher.log",
        watcher_title="Lift Solve Full9",
    )
    manifest["train_run_dir"] = str(train_run_dir)
    manifest["updated_at"] = now_str()
    write_json(paths["manifest_path"], manifest)

    checkpoint_epoch70 = train_run_dir / "models" / f"model_epoch_{args.train_epochs}.pth"
    if not checkpoint_epoch70.exists():
        raise FileNotFoundError(f"Expected checkpoint not found: {checkpoint_epoch70}")
    manifest["checkpoint_epoch70"] = str(checkpoint_epoch70)
    manifest["updated_at"] = now_str()
    write_json(paths["manifest_path"], manifest)

    if args.skip_eval:
        manifest["status"] = "training_completed_skip_eval"
        manifest["updated_at"] = now_str()
        write_json(paths["manifest_path"], manifest)
        write_experiment_report(report_path=paths["report_path"], title=manifest["title"], manifest=manifest)
        log_line(pipeline_log, f"TRAINING DONE (skip eval): report={paths['report_path']}")
        return manifest

    eval_output_dir = train_run_dir / (
        f"eval_epoch{args.train_epochs}_r{args.eval_rollouts}_"
        f"seeds{''.join(str(s) for s in args.eval_seeds)}_{timestamp}"
    )
    run_step(
        "eval_epoch70_multiseed",
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "robomimic",
            "python",
            str(repo_root / "scripts" / "eval_robomimic_checkpoint_multiseed.py"),
            "--checkpoint",
            str(checkpoint_epoch70),
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
        step_log=paths["step_log_root"] / "eval_epoch70_multiseed.log",
        gpu_log=gpu_log,
        gpu_log_interval_sec=args.gpu_log_interval_sec,
        gpu_max_temp_c=args.gpu_max_temp_c,
        wait_for_gpu_before=True,
    )
    manifest["eval_output_dir"] = str(eval_output_dir)
    manifest["eval_summary"] = summarize_eval_results(eval_output_dir / "reports" / "summary.json")
    manifest["status"] = "completed"
    manifest["updated_at"] = now_str()
    write_json(paths["manifest_path"], manifest)
    write_experiment_report(report_path=paths["report_path"], title=manifest["title"], manifest=manifest)
    log_line(pipeline_log, f"PIPELINE DONE: report={paths['report_path']}")
    return manifest


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    default_root = Path("/media/willzhang/KINGSTON/zlxtemp") / f"lift_overnight_head_vs_solve_9to153_train70_{timestamp}"
    external_root = Path(args.external_root).expanduser().resolve() if args.external_root else default_root
    ensure_dir(external_root)

    top_manifest_path = external_root / "manifest.json"
    top_report_path = external_root / "report.md"
    top_pipeline_log = external_root / "pipeline.log"
    top_gpu_log = external_root / "gpu_health.log"

    payload = {
        "external_root": str(external_root),
        "started_at": now_str(),
        "updated_at": now_str(),
        "args": vars(args),
        "experiments": {},
    }
    write_json(top_manifest_path, payload)
    log_line(top_pipeline_log, f"TOP ROOT: {external_root}")
    log_gpu_health(gpu_log=top_gpu_log, pipeline_log=top_pipeline_log, context="START")

    runners = {
        "head_gen": (run_head_gen_experiment, external_root / "head_gen"),
        "solve": (run_solve_experiment, external_root / "solve"),
    }
    seen = set()
    plan = []
    for exp_name in args.branches:
        if exp_name not in runners:
            raise ValueError(f"Unsupported branch '{exp_name}'. Supported: {sorted(runners)}")
        if exp_name in seen:
            continue
        seen.add(exp_name)
        runner, exp_root = runners[exp_name]
        plan.append((exp_name, runner, exp_root))

    for exp_name, runner, exp_root in plan:
        try:
            payload["experiments"][exp_name] = runner(args, repo_root, exp_root)
        except Exception as exc:
            err = {
                "name": exp_name,
                "status": "failed",
                "experiment_root": str(exp_root),
                "error": str(exc),
                "updated_at": now_str(),
            }
            payload["experiments"][exp_name] = err
            log_line(top_pipeline_log, f"EXPERIMENT FAIL {exp_name}: {exc}")
            if args.stop_on_error:
                payload["updated_at"] = now_str()
                write_json(top_manifest_path, payload)
                write_top_level_report(top_report_path, payload)
                raise
        finally:
            payload["updated_at"] = now_str()
            write_json(top_manifest_path, payload)
            write_top_level_report(top_report_path, payload)

    log_gpu_health(gpu_log=top_gpu_log, pipeline_log=top_pipeline_log, context="END")
    log_line(top_pipeline_log, f"ALL DONE: report={top_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
