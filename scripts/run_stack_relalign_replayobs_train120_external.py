#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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
            "Run the Stack external pipeline end to end: generate 81-grid two-phase demos, "
            "solve with relalign twostage, export replayobs low-dim HDF5, and train a "
            "robomimic Diffusion Policy run to epoch 120 with checkpoints every 30 epochs."
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
        "--source-demo",
        default=str(repo_root / "data" / "raw" / "stack_cube_0" / "1775663680_9007828" / "demo.hdf5"),
    )
    parser.add_argument(
        "--source-low-dim-hdf5",
        default=str(repo_root / "data" / "raw" / "stack_cube_0" / "1775663680_9007828" / "low_dim.hdf5"),
    )
    parser.add_argument(
        "--source-zarr",
        default=str(
            repo_root
            / "repos"
            / "DemoGen"
            / "data"
            / "datasets"
            / "source"
            / "stack_cube_0_v1_replayh1_twophase_source.zarr"
        ),
    )
    parser.add_argument(
        "--source-name",
        default="stack_cube_0_v1_replayh1_twophase_source",
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
            / "stack_cube_0_v5_replayh1_twophase_noschedule.yaml"
        ),
    )
    parser.add_argument(
        "--train-base-config",
        default=str(
            repo_root
            / "configs"
            / "robomimic"
            / "diffusion_policy_stack_cube_relalign_replayobs_lowdim_30save_30rollout10_h800_external_120epoch.json"
        ),
    )
    parser.add_argument("--n-gen-per-source", type=int, default=81)
    parser.add_argument(
        "--generated-name",
        default=None,
        help="Generated dataset stem. Defaults to stack_cube_0_v10_replayh1_twophase_noschedule_grid<N>.",
    )
    parser.add_argument("--train-epochs", type=int, default=120)
    parser.add_argument("--save-every-n-epochs", type=int, default=30)
    parser.add_argument("--rollout-n", type=int, default=10)
    parser.add_argument("--rollout-horizon", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--control-steps", type=int, default=1)
    parser.add_argument("--action-deviation-weight", type=float, default=1e-4)
    parser.add_argument("--motion1-relative-tail-steps", type=int, default=40)
    parser.add_argument("--motion1-relative-cost-weight", type=float, default=4.0)
    parser.add_argument("--motion2-relative-tail-steps", type=int, default=40)
    parser.add_argument("--motion2-relative-cost-weight", type=float, default=4.0)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-solve", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional experiment name suffix. Defaults to a timestamped stack replayobs tag.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def log_status(log_path: Path, message: str) -> None:
    line = f"[{now_str()}] {message}"
    print(line, flush=True)
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
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


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    ensure_dir(link_path.parent)
    if link_path.is_symlink():
        if link_path.resolve() == target_path.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        safe_remove(link_path)
    link_path.symlink_to(target_path, target_is_directory=target_path.is_dir())


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


def build_train_config(
    *,
    base_config_path: Path,
    output_path: Path,
    output_dir: Path,
    num_epochs: int,
    save_every_n_epochs: int,
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
    cfg["experiment"]["rollout"]["rate"] = int(save_every_n_epochs)
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
                        "Stack Replayobs",
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


def write_markdown_summary(*, summary_path: Path, manifest: dict) -> None:
    lines = [
        "# Stack Relalign Replayobs Train120 Report",
        "",
        f"- external_root: `{manifest['external_root']}`",
        f"- source_demo: `{manifest['source_demo']}`",
        f"- source_low_dim_hdf5: `{manifest['source_low_dim_hdf5']}`",
        f"- source_zarr: `{manifest['source_zarr']}`",
        f"- generated_zarr: `{manifest['generated_zarr']}`",
        f"- solved_zarr: `{manifest['solved_zarr']}`",
        f"- solve_json: `{manifest['solve_json']}`",
        f"- exported_hdf5: `{manifest['exported_hdf5']}`",
        f"- train_config: `{manifest['train_config']}`",
    ]
    if manifest.get("train_run_dir") is not None:
        lines.extend(
            [
                f"- train_run_dir: `{manifest['train_run_dir']}`",
                f"- checkpoint_epoch_target: `{manifest.get('checkpoint_epoch_target')}`",
            ]
        )
    lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    default_tag = f"stack_relalign_replayobs_grid{args.n_gen_per_source}_train{args.train_epochs}_{timestamp}"
    run_tag = args.run_tag or default_tag
    external_root = Path(args.external_root).expanduser().resolve() if args.external_root else Path("/media/willzhang/KINGSTON/zlxtemp") / run_tag

    data_root = external_root / "data"
    source_root = data_root / "datasets" / "source"
    generated_root = data_root / "datasets" / "generated"
    sam_mask_root = data_root / "sam_mask"
    outputs_root = external_root / "outputs"
    generated_output_root = outputs_root / "generated"
    analysis_root = outputs_root / "analysis"
    robomimic_root = outputs_root / "robomimic"
    train_output_dir = robomimic_root / "diffusion_policy_demogen"
    log_root = external_root / "logs"
    step_log_root = log_root / "steps"
    pipeline_log = log_root / "pipeline.log"
    manifest_path = external_root / "manifest.json"
    report_path = external_root / "report.md"

    for path in [
        source_root,
        generated_root,
        sam_mask_root,
        generated_output_root,
        analysis_root,
        robomimic_root,
        step_log_root,
    ]:
        ensure_dir(path)

    source_demo = Path(args.source_demo).expanduser().resolve()
    source_low_dim = Path(args.source_low_dim_hdf5).expanduser().resolve()
    source_zarr = Path(args.source_zarr).expanduser().resolve()
    generate_config = Path(args.generate_config).expanduser().resolve()
    train_base_config = Path(args.train_base_config).expanduser().resolve()

    if not source_demo.exists():
        raise FileNotFoundError(f"source demo not found: {source_demo}")
    if not source_low_dim.exists():
        raise FileNotFoundError(f"source low_dim not found: {source_low_dim}")
    if not source_zarr.exists():
        raise FileNotFoundError(f"source zarr not found: {source_zarr}")
    if not generate_config.exists():
        raise FileNotFoundError(f"generate config not found: {generate_config}")
    if not train_base_config.exists():
        raise FileNotFoundError(f"train base config not found: {train_base_config}")

    source_link = source_root / source_zarr.name
    ensure_symlink(source_link, source_zarr)
    sam_mask_target = source_zarr.parents[2] / "sam_mask" / args.source_name
    if sam_mask_target.exists():
        ensure_symlink(sam_mask_root / args.source_name, sam_mask_target)

    generated_name = args.generated_name or f"stack_cube_0_v10_replayh1_twophase_noschedule_grid{args.n_gen_per_source}"
    generated_zarr = generated_root / f"{generated_name}_test_{args.n_gen_per_source}.zarr"
    solved_stem = f"{generated_name}_relalign_twostage_all"
    solved_zarr = generated_output_root / f"{solved_stem}.zarr"
    solve_json = analysis_root / f"{solved_stem}.json"
    exported_hdf5 = robomimic_root / f"{generated_name}_relalign_twostage_replayobs_lowdim.hdf5"
    train_config = robomimic_root / "train120_config.json"
    train_run_name = (
        f"{generated_name}_relalign_replayobs_dp_"
        f"{args.save_every_n_epochs}save{args.save_every_n_epochs}rollout{args.rollout_n}_"
        f"h{args.rollout_horizon}_{args.train_epochs}epoch_{timestamp}"
    )

    manifest = {
        "external_root": str(external_root),
        "source_demo": str(source_demo),
        "source_low_dim_hdf5": str(source_low_dim),
        "source_zarr": str(source_link),
        "source_name": args.source_name,
        "generated_name": generated_name,
        "generated_zarr": str(generated_zarr),
        "solved_zarr": str(solved_zarr),
        "solve_json": str(solve_json),
        "exported_hdf5": str(exported_hdf5),
        "train_config": str(train_config),
        "train_run_name": train_run_name,
        "updated_at": now_str(),
    }
    write_json(manifest_path, manifest)
    log_status(pipeline_log, f"EXPERIMENT ROOT: {external_root}")
    log_status(pipeline_log, f"SOURCE BUNDLE READY: source={source_link} sam_mask={sam_mask_target}")

    if args.skip_generate:
        if not generated_zarr.exists():
            raise FileNotFoundError(f"--skip-generate was set but generated zarr is missing: {generated_zarr}")
        log_status(pipeline_log, f"REUSE GENERATED ZARR: {generated_zarr}")
    elif generated_zarr.exists():
        log_status(pipeline_log, f"GENERATED ZARR ALREADY EXISTS, REUSING: {generated_zarr}")
    else:
        run_step(
            "generate_grid_dataset",
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
                f"data_root={data_root}",
                f"source_name={args.source_name}",
                f"generated_name={generated_name}",
                "generation.range_name=test",
                "generation.mode=grid",
                f"generation.n_gen_per_source={args.n_gen_per_source}",
                "generation.render_video=False",
                f"hydra.run.dir={external_root / 'outputs' / 'hydra' / 'generate'}",
            ],
            cwd=generate_config.parents[2],
            pipeline_log=pipeline_log,
            step_log=step_log_root / "generate_grid_dataset.log",
        )

    if args.skip_solve:
        if not solved_zarr.exists() or not solve_json.exists():
            raise FileNotFoundError(
                f"--skip-solve was set but solved outputs are missing: solved_zarr={solved_zarr} solve_json={solve_json}"
            )
        log_status(pipeline_log, f"REUSE SOLVED ZARR: {solved_zarr}")
    elif solved_zarr.exists() and solve_json.exists():
        log_status(pipeline_log, f"SOLVED OUTPUTS ALREADY EXIST, REUSING: {solved_zarr}")
    else:
        safe_remove(solved_zarr)
        safe_remove(solve_json)
        run_step(
            "solve_relalign_twostage",
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
                str(data_root),
                "--source-demo",
                str(source_demo),
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
            step_log=step_log_root / "solve_relalign_twostage.log",
        )

    if args.skip_export:
        if not exported_hdf5.exists():
            raise FileNotFoundError(f"--skip-export was set but exported hdf5 is missing: {exported_hdf5}")
        log_status(pipeline_log, f"REUSE EXPORTED HDF5: {exported_hdf5}")
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
                str(source_low_dim),
                "--output-hdf5",
                str(exported_hdf5),
                "--include-source-demos",
                "--overwrite",
                "--control-steps",
                str(args.control_steps),
            ],
            cwd=repo_root,
            pipeline_log=pipeline_log,
            step_log=step_log_root / "export_replayobs_hdf5.log",
        )

    build_train_config(
        base_config_path=train_base_config,
        output_path=train_config,
        output_dir=train_output_dir,
        num_epochs=args.train_epochs,
        save_every_n_epochs=args.save_every_n_epochs,
        rollout_n=args.rollout_n,
        rollout_horizon=args.rollout_horizon,
        batch_size=args.batch_size,
        experiment_name=train_run_name,
    )
    log_status(pipeline_log, f"TRAIN CONFIG READY: {train_config}")

    if not args.skip_train:
        train_run_dir = launch_training_and_wait(
            repo_root=repo_root,
            pipeline_log=pipeline_log,
            train_config=train_config,
            dataset_path=exported_hdf5,
            train_output_dir=train_output_dir,
            run_name=train_run_name,
            launcher_log=log_root / "train_launcher.log",
        )
        target_ckpt = train_run_dir / "models" / f"model_epoch_{args.train_epochs}.pth"
        if not target_ckpt.exists():
            raise FileNotFoundError(f"Expected checkpoint not found: {target_ckpt}")
        manifest["train_run_dir"] = str(train_run_dir)
        manifest["checkpoint_epoch_target"] = str(target_ckpt)
        manifest["updated_at"] = now_str()
        write_json(manifest_path, manifest)
    else:
        log_status(pipeline_log, "SKIP TRAIN: requested by flag")

    write_markdown_summary(summary_path=report_path, manifest=manifest)
    log_status(pipeline_log, f"PIPELINE DONE: report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
