#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py


def now_str() -> str:
    return time.strftime("%F %T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the selected4 old-route lift pipeline end to end: subset the original "
            "bundle, regenerate with the README v28 route, run success / consistency checks, "
            "export replayobs low-dim HDF5, train on external storage to epoch 70, and then "
            "launch multiseed external evaluation for epoch 70."
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
        "--episodes",
        default="2,4,6,7",
        help="Comma-separated original demos to keep from the full lift source bundle.",
    )
    parser.add_argument("--n-gen-per-source", type=int, default=25)
    parser.add_argument("--skill1-frame", type=int, default=190)
    parser.add_argument("--z-step-size", type=float, default=0.015)
    parser.add_argument("--train-epochs", type=int, default=70)
    parser.add_argument("--save-every-n-epochs", type=int, default=10)
    parser.add_argument("--eval-rollouts", type=int, default=20)
    parser.add_argument("--eval-horizon", type=int, default=1000)
    parser.add_argument("--eval-parallel-jobs", type=int, default=3)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--control-steps", type=int, default=1)
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional experiment name suffix. Defaults to a timestamped selected4 old-route tag.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def log_status(log_path: Path, message: str) -> None:
    line = f"[{now_str()}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_step(
    name: str,
    cmd: list[str],
    *,
    cwd: Path,
    pipeline_log: Path,
    step_log: Path,
    env: dict[str, str] | None = None,
) -> None:
    ensure_dir(step_log.parent)
    log_status(pipeline_log, f"STEP START {name}: {shlex.join(cmd)}")
    with step_log.open("w", encoding="utf-8") as f:
        f.write(f"$ {shlex.join(cmd)}\n\n")
        try:
            subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            log_status(
                pipeline_log,
                f"STEP FAIL {name}: returncode={exc.returncode} log={step_log}",
            )
            raise
    log_status(pipeline_log, f"STEP DONE {name}: log={step_log}")


def sorted_demo_keys(group: h5py.Group) -> list[str]:
    return sorted(group.keys(), key=lambda name: int(name.split("_")[-1]))


def parse_episode_selection(raw: str, full_demo_hdf5: Path) -> tuple[list[str], list[int]]:
    with h5py.File(full_demo_hdf5, "r") as f:
        available = sorted_demo_keys(f["data"])

    selected_keys: list[str] = []
    selected_indices: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        key = token if token.startswith("demo_") else f"demo_{int(token)}"
        if key not in available:
            raise KeyError(f"Requested {key}, but available demos are: {available}")
        selected_keys.append(key)
        selected_indices.append(available.index(key))
    if not selected_keys:
        raise ValueError("No episodes selected")
    if len(set(selected_keys)) != len(selected_keys):
        raise ValueError(f"Duplicate episode selection: {selected_keys}")
    return selected_keys, selected_indices


def copy_selected_sam_mask(
    *,
    repo_root: Path,
    input_source_name: str,
    output_data_root: Path,
    output_source_name: str,
    selected_episode_indices: list[int],
) -> None:
    src_root = repo_root / "repos" / "DemoGen" / "data" / "sam_mask" / input_source_name
    dst_root = output_data_root / "sam_mask" / output_source_name
    if not src_root.exists():
        raise FileNotFoundError(f"sam_mask source not found: {src_root}")
    if dst_root.exists():
        shutil.rmtree(dst_root)
    ensure_dir(dst_root)
    for new_idx, old_idx in enumerate(selected_episode_indices):
        src_dir = src_root / str(old_idx)
        if not src_dir.exists():
            raise FileNotFoundError(f"sam_mask episode dir not found: {src_dir}")
        shutil.copytree(src_dir, dst_root / str(new_idx))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_train_config(
    *,
    base_config_path: Path,
    output_path: Path,
    output_dir: Path,
    num_epochs: int,
    save_every_n_epochs: int,
    experiment_name: str,
) -> None:
    cfg = load_json(base_config_path)
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
    train_config: Path,
    dataset_path: Path,
    train_output_dir: Path,
    run_name: str,
    launcher_log: Path,
) -> Path:
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
    log_status(pipeline_log, f"TRAIN START: {shlex.join(train_cmd)}")
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
                        "Lift OldRoute Selected4",
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
                    log_status(
                        pipeline_log,
                        f"TRAIN RUN DIR READY: run_dir={run_dir} watcher_log={watcher_log}",
                    )
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
        log_status(pipeline_log, f"TRAIN DONE: run_dir={run_dir}")
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


def write_markdown_summary(
    *,
    summary_path: Path,
    manifest: dict,
    success_payload: dict,
    consistency_payload: dict,
    eval_payload: dict,
) -> None:
    lines = [
        "# Lift Selected4 Old-Route Replayobs Train70 Report",
        "",
        f"- external_root: `{manifest['external_root']}`",
        f"- selected_episodes: `{manifest['selected_episode_keys']}`",
        f"- generated_zarr: `{manifest['generated_zarr']}`",
        f"- exported_hdf5: `{manifest['exported_hdf5']}`",
        f"- train_run_dir: `{manifest['train_run_dir']}`",
        f"- checkpoint_epoch70: `{manifest['checkpoint_epoch70']}`",
        f"- eval_output_dir: `{manifest['eval_output_dir']}`",
        "",
        "## Generated Zarr Gate",
        "",
        f"- success_count: `{success_payload.get('success_count')}` / `{success_payload.get('n_checked')}`",
        f"- success_rate: `{success_payload.get('success_rate')}`",
        f"- consistency_pass: `{consistency_payload.get('n_pass')}` / `{consistency_payload.get('n_checked')}`",
        f"- consistency_pass_rate: `{consistency_payload.get('pass_rate')}`",
        "",
        "## Epoch 70 External Eval",
        "",
        f"- completed_seeds: `{eval_payload.get('completed_seeds')}`",
        f"- mean_success_rate: `{eval_payload.get('mean_success_rate')}`",
        f"- mean_avg_return: `{eval_payload.get('mean_avg_return')}`",
        f"- mean_avg_horizon: `{eval_payload.get('mean_avg_horizon')}`",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_tag = args.run_tag or f"lift_selected4_oldroute_replayobs_train70_{timestamp}"
    default_root = Path("/media/willzhang/KINGSTON/zlxtemp") / run_tag
    external_root = Path(args.external_root).expanduser().resolve() if args.external_root else default_root

    raw_root = external_root / "raw"
    data_root = external_root / "data"
    source_root = data_root / "datasets" / "source"
    generated_root = data_root / "datasets" / "generated"
    sam_mask_root = data_root / "sam_mask"
    analysis_root = external_root / "outputs" / "analysis"
    robomimic_root = external_root / "outputs" / "robomimic"
    train_output_dir = robomimic_root / "diffusion_policy_demogen"
    log_root = external_root / "logs"
    step_log_root = log_root / "steps"
    pipeline_log = log_root / "pipeline.log"
    manifest_path = external_root / "manifest.json"
    report_path = external_root / "report.md"

    for path in [
        raw_root,
        source_root,
        generated_root,
        sam_mask_root,
        analysis_root,
        robomimic_root,
        step_log_root,
    ]:
        ensure_dir(path)

    full_demo_hdf5 = repo_root / "data" / "raw" / "lift_0" / "1774702988_8036063" / "demo.hdf5"
    full_low_dim_hdf5 = repo_root / "data" / "raw" / "lift_0" / "1774702988_8036063" / "low_dim.hdf5"
    full_source_zarr = (
        repo_root
        / "repos"
        / "DemoGen"
        / "data"
        / "datasets"
        / "source"
        / "lift_0_v9_execmotion_xzfullfir4sum.zarr"
    )
    base_train_config = (
        repo_root
        / "configs"
        / "robomimic"
        / "diffusion_policy_lift_v37_selected4_d2467_relalign_all_diagfix_replayobs_lowdim_30save_30rollout10_h1000_external_300epoch.json"
    )

    selected_keys, selected_indices = parse_episode_selection(args.episodes, full_demo_hdf5)

    subset_demo_hdf5 = raw_root / "demo_selected4_d2467.hdf5"
    subset_low_dim_hdf5 = raw_root / "low_dim_selected4_d2467.hdf5"
    subset_source_name = "lift_0_v9_execmotion_xzfullfir4sum_selected4_d2467"
    subset_source_zarr = source_root / f"{subset_source_name}.zarr"
    originalschedule_source_name = "lift_0_v21_originalschedule_motion_v9_s220_selected4_d2467"
    originalschedule_source_zarr = source_root / f"{originalschedule_source_name}.zarr"
    generated_name = "lift_0_v28_originalschedule_phasecopy_statedelta_halfcorr_v9_s220_selected4_d2467"
    generated_zarr = generated_root / f"{generated_name}_test_{args.n_gen_per_source}.zarr"
    exported_hdf5 = robomimic_root / "lift_0_v28_selected4_d2467_originalschedule_replayobs_lowdim.hdf5"
    train_config = robomimic_root / "train70_config.json"
    train_run_name = f"lift_0_v28_selected4_d2467_oldroute_replayobs_train70_{timestamp}"

    manifest = {
        "external_root": str(external_root),
        "selected_episode_keys": selected_keys,
        "selected_episode_indices": selected_indices,
        "subset_demo_hdf5": str(subset_demo_hdf5),
        "subset_low_dim_hdf5": str(subset_low_dim_hdf5),
        "subset_source_zarr": str(subset_source_zarr),
        "originalschedule_source_zarr": str(originalschedule_source_zarr),
        "generated_zarr": str(generated_zarr),
        "exported_hdf5": str(exported_hdf5),
        "train_config": str(train_config),
        "train_run_name": train_run_name,
        "updated_at": now_str(),
    }
    write_json(manifest_path, manifest)
    log_status(pipeline_log, f"EXPERIMENT ROOT: {external_root}")

    run_step(
        "subset_bundle",
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "demogen",
            "python",
            str(repo_root / "scripts" / "subset_demogen_source_bundle.py"),
            "--demo-hdf5",
            str(full_demo_hdf5),
            "--demo-output",
            str(subset_demo_hdf5),
            "--low-dim-hdf5",
            str(full_low_dim_hdf5),
            "--low-dim-output",
            str(subset_low_dim_hdf5),
            "--source-zarr",
            str(full_source_zarr),
            "--source-output-zarr",
            str(subset_source_zarr),
            "--episodes",
            args.episodes,
            "--overwrite",
        ],
        cwd=repo_root,
        pipeline_log=pipeline_log,
        step_log=step_log_root / "subset_bundle.log",
    )

    copy_selected_sam_mask(
        repo_root=repo_root,
        input_source_name="lift_0_v9_execmotion_xzfullfir4sum",
        output_data_root=data_root,
        output_source_name=subset_source_name,
        selected_episode_indices=selected_indices,
    )
    log_status(
        pipeline_log,
        f"SAM MASK READY: {data_root / 'sam_mask' / subset_source_name}",
    )

    run_step(
        "convert_originalschedule_source",
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "demogen",
            "python",
            str(repo_root / "repos" / "DemoGen" / "real_world" / "convert_source_zarr_original_schedule_motion.py"),
            "--input-zarr",
            str(subset_source_zarr),
            "--output-zarr",
            str(originalschedule_source_zarr),
            "--source-name",
            originalschedule_source_name,
            "--skill1-frame",
            str(args.skill1_frame),
            "--z-step-size",
            str(args.z_step_size),
            "--copy-sam-mask",
        ],
        cwd=repo_root,
        pipeline_log=pipeline_log,
        step_log=step_log_root / "convert_originalschedule_source.log",
    )

    run_step(
        "generate_oldroute_selected4",
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
            "--config-name=lift_0_v28_originalschedule_phasecopy_statedelta_halfcorr_v9_s220.yaml",
            f"data_root={data_root}",
            f"source_name={originalschedule_source_name}",
            f"generated_name={generated_name}",
            "generation.range_name=test",
            "generation.mode=grid",
            f"generation.n_gen_per_source={args.n_gen_per_source}",
            "generation.render_video=False",
            f"hydra.run.dir={external_root / 'outputs' / 'hydra' / 'generate'}",
        ],
        cwd=repo_root / "repos" / "DemoGen" / "demo_generation",
        pipeline_log=pipeline_log,
        step_log=step_log_root / "generate_oldroute_selected4.log",
    )

    success_json = analysis_root / "generated_success.json"
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
            str(generated_zarr),
            "--source-demo",
            str(subset_demo_hdf5),
            "--control-steps",
            str(args.control_steps),
            "--output-json",
            str(success_json),
        ],
        cwd=repo_root,
        pipeline_log=pipeline_log,
        step_log=step_log_root / "generated_success_check.log",
    )
    success_payload = load_json(success_json)
    log_status(
        pipeline_log,
        f"GENERATED SUCCESS: {success_payload.get('success_count')}/{success_payload.get('n_checked')} rate={success_payload.get('success_rate')}",
    )

    consistency_json = analysis_root / "generated_consistency.json"
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
            str(generated_zarr),
            "--source-demo",
            str(subset_demo_hdf5),
            "--source-low-dim",
            str(subset_low_dim_hdf5),
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
        step_log=step_log_root / "generated_consistency_check.log",
    )
    consistency_payload = load_json(consistency_json)
    log_status(
        pipeline_log,
        f"GENERATED CONSISTENCY: {consistency_payload.get('n_pass')}/{consistency_payload.get('n_checked')} pass_rate={consistency_payload.get('pass_rate')}",
    )

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
            str(generated_zarr),
            "--source-low-dim-hdf5",
            str(subset_low_dim_hdf5),
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
        step_log=step_log_root / "export_replayobs_hdf5.log",
    )

    build_train_config(
        base_config_path=base_train_config,
        output_path=train_config,
        output_dir=train_output_dir,
        num_epochs=args.train_epochs,
        save_every_n_epochs=args.save_every_n_epochs,
        experiment_name=train_run_name,
    )
    log_status(pipeline_log, f"TRAIN CONFIG READY: {train_config}")

    train_run_dir = launch_training_and_wait(
        repo_root=repo_root,
        pipeline_log=pipeline_log,
        train_config=train_config,
        dataset_path=exported_hdf5,
        train_output_dir=train_output_dir,
        run_name=train_run_name,
        launcher_log=log_root / "train_launcher.log",
    )
    manifest["train_run_dir"] = str(train_run_dir)
    manifest["updated_at"] = now_str()
    write_json(manifest_path, manifest)

    checkpoint_epoch70 = train_run_dir / "models" / f"model_epoch_{args.train_epochs}.pth"
    if not checkpoint_epoch70.exists():
        raise FileNotFoundError(f"Expected checkpoint not found: {checkpoint_epoch70}")
    manifest["checkpoint_epoch70"] = str(checkpoint_epoch70)
    manifest["updated_at"] = now_str()
    write_json(manifest_path, manifest)

    eval_output_dir = train_run_dir / f"eval_epoch{args.train_epochs}_r{args.eval_rollouts}_seeds{''.join(str(s) for s in args.eval_seeds)}_{timestamp}"
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
        step_log=step_log_root / "eval_epoch70_multiseed.log",
    )

    eval_payload = summarize_eval_results(eval_output_dir / "reports" / "summary.json")
    manifest["eval_output_dir"] = str(eval_output_dir)
    manifest["eval_summary"] = eval_payload
    manifest["updated_at"] = now_str()
    write_json(manifest_path, manifest)

    write_markdown_summary(
        summary_path=report_path,
        manifest=manifest,
        success_payload=success_payload,
        consistency_payload=consistency_payload,
        eval_payload=eval_payload,
    )
    log_status(pipeline_log, f"PIPELINE DONE: report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
