#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


def now_str() -> str:
    return time.strftime("%F %T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a two-stage DemoGen source bundle directly from a raw robosuite demo.hdf5, "
            "including selected-episode fork, low_dim / depth exports, source zarr creation, "
            "schedule fork, and an optional end-to-end smoke test."
        )
    )
    parser.add_argument("--config", required=True, help="Path to the two-stage DemoGen config YAML")
    parser.add_argument("--raw-demo-hdf5", required=True, help="Path to the original raw demo.hdf5")
    parser.add_argument(
        "--selected-episodes",
        required=True,
        help="Comma-separated episode indices or demo keys to keep, for example '0,1,2,3'.",
    )
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "data"),
        help="Root used by DemoGen, typically the repo data directory.",
    )
    parser.add_argument(
        "--selected-demo-hdf5",
        default=None,
        help="Optional explicit path for the filtered demo.hdf5. Defaults to cfg.source_demo_hdf5 when present.",
    )
    parser.add_argument("--low-dim-hdf5", default=None, help="Optional output path for low_dim.hdf5")
    parser.add_argument("--depth-hdf5", default=None, help="Optional output path for depth.hdf5")
    parser.add_argument(
        "--base-source-name",
        default=None,
        help="Optional intermediate source zarr stem. Defaults to '<cfg.source_name>_preschedule'.",
    )
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--camera-height", type=int, default=84)
    parser.add_argument("--camera-width", type=int, default=84)
    parser.add_argument("--done-mode", type=int, default=2)
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--mask-radius", type=float, default=0.045)
    parser.add_argument("--mask-dilation-iters", type=int, default=2)
    parser.add_argument(
        "--replay-prefix-frames",
        type=int,
        default=None,
        help="Optional debug limit for replay_h1 calibration during source conversion.",
    )
    parser.add_argument(
        "--source-episode-limit",
        type=int,
        default=None,
        help="Optional debug limit on how many selected episodes are converted into the source zarr.",
    )
    parser.add_argument("--control-steps", type=int, default=None, help="Override cfg.source_control_steps")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument(
        "--smoke-generated-name",
        default=None,
        help="Optional generated dataset stem for the smoke run. Defaults to '<cfg.generated_name>_smoke'.",
    )
    parser.add_argument(
        "--smoke-solve-episodes",
        default="0",
        help="Generated episode indices to run through solve/exportobs smoke. Defaults to '0'.",
    )
    parser.add_argument("--smoke-motion1-solve-steps", type=int, default=5)
    parser.add_argument("--smoke-motion2-solve-steps", type=int, default=5)
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="Optional manifest JSON path. Defaults next to the selected demo.hdf5.",
    )
    return parser.parse_args()


def derive_peer_hdf5(selected_demo_hdf5: Path, prefix: str) -> Path:
    name = selected_demo_hdf5.name
    if name.startswith("demo"):
        return selected_demo_hdf5.with_name(prefix + name[len("demo"):])
    return selected_demo_hdf5.with_name(f"{prefix}_{name}")


def to_plain(value):
    return OmegaConf.to_container(value, resolve=True)


def safe_remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def ensure_removed(path: Path, overwrite: bool, label: str) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"{label} already exists: {path}")
    safe_remove(path)


def run_step(step_name: str, cmd: list[str], *, cwd: Path, env_update: dict[str, str] | None = None) -> None:
    print(f"[{now_str()}] STEP {step_name}", flush=True)
    print(f"[{now_str()}] CMD  {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    if env_update:
        env.update(env_update)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    raw_demo_hdf5 = Path(args.raw_demo_hdf5).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)

    selected_demo_hdf5 = (
        Path(args.selected_demo_hdf5).expanduser().resolve()
        if args.selected_demo_hdf5 is not None
        else (
            Path(str(cfg.source_demo_hdf5)).expanduser().resolve()
            if getattr(cfg, "source_demo_hdf5", None) is not None
            else raw_demo_hdf5.with_name("demo_selected.hdf5")
        )
    )
    if selected_demo_hdf5 == raw_demo_hdf5:
        raise ValueError("selected demo path must differ from the raw demo path")

    low_dim_hdf5 = (
        Path(args.low_dim_hdf5).expanduser().resolve()
        if args.low_dim_hdf5 is not None
        else derive_peer_hdf5(selected_demo_hdf5, "low_dim")
    )
    depth_hdf5 = (
        Path(args.depth_hdf5).expanduser().resolve()
        if args.depth_hdf5 is not None
        else derive_peer_hdf5(selected_demo_hdf5, "depth")
    )

    source_name = str(cfg.source_name)
    generated_name = str(cfg.generated_name)
    base_source_name = args.base_source_name or f"{source_name}_preschedule"
    base_source_zarr = data_root / "datasets" / "source" / f"{base_source_name}.zarr"
    scheduled_source_zarr = data_root / "datasets" / "source" / f"{source_name}.zarr"
    sam_mask_base_dir = data_root / "sam_mask" / base_source_name
    sam_mask_scheduled_dir = data_root / "sam_mask" / source_name

    smoke_generated_name = args.smoke_generated_name or f"{generated_name}_smoke"
    smoke_generated_zarr = data_root / "datasets" / "generated" / f"{smoke_generated_name}_src_1.zarr"
    smoke_success_json = selected_demo_hdf5.parent / f"{smoke_generated_name}_success.json"
    smoke_solved_zarr = data_root / "datasets" / "generated" / f"{smoke_generated_name}_solve_smoke.zarr"
    smoke_solve_json = selected_demo_hdf5.parent / f"{smoke_generated_name}_solve_smoke.json"
    smoke_export_hdf5 = (
        data_root / "datasets" / "generated" / f"{smoke_generated_name}_solve_smoke_replayobs_lowdim.hdf5"
    )

    manifest_out = (
        Path(args.manifest_out).expanduser().resolve()
        if args.manifest_out is not None
        else selected_demo_hdf5.parent / f"{config_path.stem}_twostage_manifest.json"
    )

    control_steps = int(args.control_steps if args.control_steps is not None else getattr(cfg, "source_control_steps", 1))
    mask_names = to_plain(cfg.mask_names)
    parsing_frames = to_plain(cfg.parsing_frames)

    outputs_to_prepare = [
        (selected_demo_hdf5, "selected demo"),
        (low_dim_hdf5, "low_dim export"),
        (depth_hdf5, "depth export"),
        (base_source_zarr, "base source zarr"),
        (scheduled_source_zarr, "scheduled source zarr"),
        (sam_mask_base_dir, "base sam_mask"),
        (sam_mask_scheduled_dir, "scheduled sam_mask"),
        (manifest_out, "manifest"),
    ]
    if not args.skip_smoke:
        outputs_to_prepare.extend(
            [
                (smoke_generated_zarr, "smoke generated zarr"),
                (smoke_success_json, "smoke success report"),
                (smoke_solved_zarr, "smoke solved zarr"),
                (smoke_solve_json, "smoke solve report"),
                (smoke_export_hdf5, "smoke replayobs export"),
            ]
        )
    for path, label in outputs_to_prepare:
        ensure_removed(path, args.overwrite, label)

    selected_demo_hdf5.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    (data_root / "datasets" / "source").mkdir(parents=True, exist_ok=True)
    (data_root / "datasets" / "generated").mkdir(parents=True, exist_ok=True)

    python = sys.executable
    robomimic_root = REPO_ROOT / "repos" / "robomimic"
    demogen_demo_root = REPO_ROOT / "repos" / "DemoGen" / "demo_generation"
    existing_pythonpath = os.environ.get("PYTHONPATH")
    robomimic_pythonpath = str(robomimic_root)
    if existing_pythonpath:
        robomimic_pythonpath = robomimic_pythonpath + os.pathsep + existing_pythonpath
    robomimic_env = {"PYTHONPATH": robomimic_pythonpath}

    run_step(
        "filter_selected_episodes",
        [
            python,
            str(REPO_ROOT / "scripts" / "filter_robomimic_hdf5_episodes.py"),
            "--input-hdf5",
            str(raw_demo_hdf5),
            "--output-hdf5",
            str(selected_demo_hdf5),
            "--episodes",
            args.selected_episodes,
        ],
        cwd=REPO_ROOT,
    )
    run_step(
        "convert_robosuite_metadata",
        [
            python,
            str(robomimic_root / "robomimic" / "scripts" / "conversion" / "convert_robosuite.py"),
            "--dataset",
            str(selected_demo_hdf5),
        ],
        cwd=robomimic_root,
        env_update=robomimic_env,
    )
    run_step(
        "extract_low_dim",
        [
            python,
            str(robomimic_root / "robomimic" / "scripts" / "dataset_states_to_obs.py"),
            "--dataset",
            str(selected_demo_hdf5),
            "--output_name",
            str(low_dim_hdf5),
            "--done_mode",
            str(args.done_mode),
        ],
        cwd=robomimic_root,
        env_update=robomimic_env,
    )
    run_step(
        "extract_depth",
        [
            python,
            str(robomimic_root / "robomimic" / "scripts" / "dataset_states_to_obs.py"),
            "--dataset",
            str(selected_demo_hdf5),
            "--output_name",
            str(depth_hdf5),
            "--done_mode",
            str(args.done_mode),
            "--camera_names",
            args.camera_name,
            "--camera_height",
            str(args.camera_height),
            "--camera_width",
            str(args.camera_width),
            "--depth",
        ],
        cwd=robomimic_root,
        env_update=robomimic_env,
    )

    source_cmd = [
        python,
        str(REPO_ROOT / "repos" / "DemoGen" / "real_world" / "convert_robomimic_hdf5_to_zarr_exec_replay_h1_light.py"),
        "--demo-hdf5",
        str(selected_demo_hdf5),
        "--low-dim-hdf5",
        str(low_dim_hdf5),
        "--depth-hdf5",
        str(depth_hdf5),
        "--output-zarr",
        str(base_source_zarr),
        "--source-name",
        base_source_name,
        "--camera-name",
        args.camera_name,
        "--n-points",
        str(args.n_points),
        "--mask-object-name",
        str(mask_names["object"]),
        "--mask-radius",
        str(args.mask_radius),
        "--mask-dilation-iters",
        str(args.mask_dilation_iters),
        "--control-steps",
        str(control_steps),
    ]
    if args.replay_prefix_frames is not None:
        source_cmd.extend(["--replay-prefix-frames", str(args.replay_prefix_frames)])
    if args.source_episode_limit is not None:
        source_cmd.extend(["--episode-limit", str(args.source_episode_limit)])
    run_step("convert_base_source_zarr", source_cmd, cwd=REPO_ROOT)

    mask_cmd = [
        python,
        str(REPO_ROOT / "scripts" / "generate_task_sam_masks_from_hdf5.py"),
        "--demo-hdf5",
        str(selected_demo_hdf5),
        "--depth-hdf5",
        str(depth_hdf5),
        "--source-name",
        base_source_name,
        "--output-data-root",
        str(data_root),
        "--camera-name",
        args.camera_name,
        "--episodes",
        "all",
        "--mask-object-name",
        str(mask_names["object"]),
        "--mask-radius",
        str(args.mask_radius),
        "--mask-dilation-iters",
        str(args.mask_dilation_iters),
    ]
    if mask_names.get("target", None) is not None:
        mask_cmd.extend(["--mask-target-name", str(mask_names["target"])])
    run_step("generate_task_masks", mask_cmd, cwd=REPO_ROOT)

    run_step(
        "convert_twostage_schedule_source",
        [
            python,
            str(REPO_ROOT / "repos" / "DemoGen" / "real_world" / "convert_source_zarr_twophase_schedule_replay_h1.py"),
            "--input-zarr",
            str(base_source_zarr),
            "--output-zarr",
            str(scheduled_source_zarr),
            "--source-name",
            source_name,
            "--skill1-frame",
            json.dumps(parsing_frames["skill-1"]),
            "--motion2-frame",
            json.dumps(parsing_frames["motion-2"]),
            "--skill2-frame",
            json.dumps(parsing_frames["skill-2"]),
            "--copy-sam-mask",
        ],
        cwd=REPO_ROOT,
    )

    smoke_result = None
    smoke_solve_result = None
    if not args.skip_smoke:
        hydra_dir = REPO_ROOT / "outputs" / "hydra" / f"{config_path.stem}_smoke"
        run_step(
            "smoke_generate",
            [
                python,
                "-W",
                "ignore",
                str(demogen_demo_root / "gen_demo.py"),
                f"--config-name={config_path.name}",
                f"data_root={data_root}",
                f"source_name={source_name}",
                f"generated_name={smoke_generated_name}",
                f"source_demo_hdf5={selected_demo_hdf5}",
                "generation.range_name=src",
                "generation.mode=random",
                "generation.n_gen_per_source=1",
                "generation.render_video=False",
                f"hydra.run.dir={hydra_dir}",
            ],
            cwd=demogen_demo_root,
        )
        run_step(
            "smoke_eval",
            [
                python,
                str(REPO_ROOT / "scripts" / "eval_generated_zarr_success_rate.py"),
                "--zarr",
                str(smoke_generated_zarr),
                "--source-demo",
                str(selected_demo_hdf5),
                "--control-steps",
                str(control_steps),
                "--output-json",
                str(smoke_success_json),
            ],
            cwd=REPO_ROOT,
        )
        smoke_result = json.loads(smoke_success_json.read_text(encoding="utf-8"))
        run_step(
            "smoke_solve",
            [
                python,
                str(REPO_ROOT / "scripts" / "export_stack_solved_from_template_zarr_relalign_twostage.py"),
                "--config",
                str(config_path),
                "--data-root",
                str(data_root),
                "--source-demo",
                str(selected_demo_hdf5),
                "--template-zarr",
                str(smoke_generated_zarr),
                "--episodes",
                str(args.smoke_solve_episodes),
                "--motion1-solve-steps",
                str(args.smoke_motion1_solve_steps),
                "--motion2-solve-steps",
                str(args.smoke_motion2_solve_steps),
                "--control-steps",
                str(control_steps),
                "--output-zarr",
                str(smoke_solved_zarr),
                "--output-json",
                str(smoke_solve_json),
            ],
            cwd=REPO_ROOT,
        )
        run_step(
            "smoke_exportobs",
            [
                python,
                str(
                    REPO_ROOT
                    / "repos"
                    / "DemoGen"
                    / "real_world"
                    / "export_demogen_zarr_to_robomimic_lowdim_replayobs_twophase.py"
                ),
                "--generated-zarr",
                str(smoke_solved_zarr),
                "--source-low-dim-hdf5",
                str(low_dim_hdf5),
                "--output-hdf5",
                str(smoke_export_hdf5),
                "--control-steps",
                str(control_steps),
                "--overwrite",
            ],
            cwd=REPO_ROOT,
        )
        smoke_solve_result = json.loads(smoke_solve_json.read_text(encoding="utf-8"))

    manifest = {
        "config": str(config_path),
        "raw_demo_hdf5": str(raw_demo_hdf5),
        "selected_episodes": args.selected_episodes,
        "selected_demo_hdf5": str(selected_demo_hdf5),
        "low_dim_hdf5": str(low_dim_hdf5),
        "depth_hdf5": str(depth_hdf5),
        "data_root": str(data_root),
        "source_name_preschedule": base_source_name,
        "source_name_scheduled": source_name,
        "base_source_zarr": str(base_source_zarr),
        "scheduled_source_zarr": str(scheduled_source_zarr),
        "mask_names": mask_names,
        "parsing_frames": parsing_frames,
        "control_steps": control_steps,
        "smoke_generated_zarr": None if args.skip_smoke else str(smoke_generated_zarr),
        "smoke_success": smoke_result,
        "smoke_solve_episodes": None if args.skip_smoke else str(args.smoke_solve_episodes),
        "smoke_motion1_solve_steps": None if args.skip_smoke else int(args.smoke_motion1_solve_steps),
        "smoke_motion2_solve_steps": None if args.skip_smoke else int(args.smoke_motion2_solve_steps),
        "smoke_solved_zarr": None if args.skip_smoke else str(smoke_solved_zarr),
        "smoke_solve": smoke_solve_result,
        "smoke_export_hdf5": None if args.skip_smoke else str(smoke_export_hdf5),
        "updated_at": now_str(),
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
