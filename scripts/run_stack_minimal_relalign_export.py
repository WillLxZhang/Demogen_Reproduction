#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "repos"
    / "DemoGen"
    / "demo_generation"
    / "demo_generation"
    / "config"
    / "stack_cube_0_v9_replayh1_twophase_noschedule_minimal.yaml"
)
DEFAULT_DATA_ROOT = REPO_ROOT / "repos" / "DemoGen" / "data"
DEFAULT_SOURCE_DEMO = REPO_ROOT / "data" / "raw" / "stack_cube_0" / "1775663680_9007828" / "demo.hdf5"
DEFAULT_SOURCE_LOW_DIM = (
    REPO_ROOT / "data" / "raw" / "stack_cube_0" / "1775663680_9007828" / "low_dim.hdf5"
)

if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the minimal Stack fork: generate the standard 16-grid template if needed, "
            "then keep only the four corner episodes for twostage rel-align solve and low-dim export."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--source-demo", default=str(DEFAULT_SOURCE_DEMO))
    parser.add_argument("--source-low-dim-hdf5", default=str(DEFAULT_SOURCE_LOW_DIM))
    parser.add_argument(
        "--selected-episodes",
        default="0,4,8,12",
        help="Comma-separated template episode ids to keep. Defaults to the four corner episodes.",
    )
    parser.add_argument(
        "--template-zarr",
        default=None,
        help="Optional existing template zarr. Defaults to the path implied by config.generated_name.",
    )
    parser.add_argument("--output-zarr", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-hdf5", default=None)
    parser.add_argument("--control-steps", type=int, default=1)
    parser.add_argument("--action-deviation-weight", type=float, default=1e-4)
    parser.add_argument("--motion1-relative-tail-steps", type=int, default=40)
    parser.add_argument("--motion1-relative-cost-weight", type=float, default=4.0)
    parser.add_argument("--motion2-relative-tail-steps", type=int, default=40)
    parser.add_argument("--motion2-relative-cost-weight", type=float, default=4.0)
    parser.add_argument(
        "--include-source-demos",
        dest="include_source_demos",
        action="store_true",
        help="Prepend the original source demos into the exported HDF5.",
    )
    parser.add_argument(
        "--no-include-source-demos",
        dest="include_source_demos",
        action="store_false",
        help="Export generated demos only, without prepending source demos.",
    )
    parser.set_defaults(include_source_demos=True)
    parser.add_argument(
        "--regenerate-template",
        action="store_true",
        help="Force regeneration even if the template zarr already exists.",
    )
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-solve", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the resolved plan and commands without executing them.",
    )
    return parser.parse_args()


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def ensure_writable_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path}")
        remove_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def build_template_zarr_path(cfg, data_root: Path) -> Path:
    return (
        data_root
        / "datasets"
        / "generated"
        / f"{cfg.generated_name}_{cfg.generation.range_name}_{cfg.generation.n_gen_per_source}.zarr"
    )


def build_default_outputs(cfg) -> tuple[Path, Path, Path]:
    stem = f"{cfg.generated_name}_relalign_twostage_objcorners4"
    output_zarr = REPO_ROOT / "outputs" / "generated" / f"{stem}.zarr"
    output_json = REPO_ROOT / "outputs" / "analysis" / f"{stem}.json"
    output_hdf5 = REPO_ROOT / "outputs" / "generated" / f"{stem}.hdf5"
    return output_zarr, output_json, output_hdf5


def load_runtime_cfg(config_path: Path, data_root: Path):
    from solve_lift_prefix_xyz_actions import load_cfg

    return load_cfg(config_path, data_root)


def instantiate_runtime_generator(cfg):
    from solve_lift_prefix_xyz_actions import instantiate_generator

    return instantiate_generator(cfg)


def run_subprocess(cmd: list[str], *, print_only: bool) -> None:
    print(json.dumps({"cmd": cmd}, ensure_ascii=False))
    if print_only:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    source_demo = Path(args.source_demo).expanduser().resolve()
    source_low_dim = Path(args.source_low_dim_hdf5).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not source_demo.exists():
        raise FileNotFoundError(f"Source demo not found: {source_demo}")
    if not source_low_dim.exists():
        raise FileNotFoundError(f"Source low_dim.hdf5 not found: {source_low_dim}")

    cfg = load_runtime_cfg(config_path, data_root)
    cfg.source_demo_hdf5 = str(source_demo)

    template_zarr = (
        Path(args.template_zarr).expanduser().resolve()
        if args.template_zarr is not None
        else build_template_zarr_path(cfg, data_root)
    )
    default_output_zarr, default_output_json, default_output_hdf5 = build_default_outputs(cfg)
    output_zarr = (
        Path(args.output_zarr).expanduser().resolve()
        if args.output_zarr is not None
        else default_output_zarr
    )
    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json is not None
        else default_output_json
    )
    output_hdf5 = (
        Path(args.output_hdf5).expanduser().resolve()
        if args.output_hdf5 is not None
        else default_output_hdf5
    )

    if not args.skip_solve and not args.print_only:
        ensure_writable_output(output_zarr, args.overwrite)
        ensure_writable_output(output_json, args.overwrite)
    elif args.skip_solve and not output_zarr.exists() and not args.print_only:
        raise FileNotFoundError(f"--skip-solve was set but solved zarr does not exist: {output_zarr}")

    if not args.skip_export and not args.print_only:
        ensure_writable_output(output_hdf5, args.overwrite)

    plan = {
        "config": str(config_path),
        "source_demo": str(source_demo),
        "source_low_dim_hdf5": str(source_low_dim),
        "template_zarr": str(template_zarr),
        "selected_episodes": args.selected_episodes,
        "output_zarr": str(output_zarr),
        "output_json": str(output_json),
        "output_hdf5": str(output_hdf5),
        "include_source_demos": bool(args.include_source_demos),
        "skip_generate": bool(args.skip_generate),
        "skip_solve": bool(args.skip_solve),
        "skip_export": bool(args.skip_export),
        "print_only": bool(args.print_only),
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))

    if not args.skip_generate:
        if args.regenerate_template or not template_zarr.exists():
            if args.regenerate_template and not args.print_only:
                ensure_writable_output(template_zarr, True)
            print(
                json.dumps(
                    {
                        "stage": "generate",
                        "message": "Building the standard 16-grid template before selecting the four corner episodes.",
                    },
                    ensure_ascii=False,
                )
            )
            if not args.print_only:
                generator = instantiate_runtime_generator(cfg)
                generator.generate_demo()
            else:
                print(
                    json.dumps(
                        {
                            "stage": "generate",
                            "message": "print-only mode: skipped generator.generate_demo()",
                        },
                        ensure_ascii=False,
                    )
                )
        else:
            print(
                json.dumps(
                    {
                        "stage": "generate",
                        "message": "Template zarr already exists, reusing it.",
                    },
                    ensure_ascii=False,
                )
            )

    if not template_zarr.exists() and not args.print_only:
        raise FileNotFoundError(f"Template zarr not found after generate stage: {template_zarr}")

    if not args.skip_solve:
        solve_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "export_stack_solved_from_template_zarr_relalign_twostage.py"),
            "--config",
            str(config_path),
            "--data-root",
            str(data_root),
            "--source-demo",
            str(source_demo),
            "--template-zarr",
            str(template_zarr),
            "--episodes",
            args.selected_episodes,
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
            str(output_zarr),
            "--output-json",
            str(output_json),
        ]
        run_subprocess(solve_cmd, print_only=args.print_only)

    if not args.skip_export:
        export_cmd = [
            sys.executable,
            str(REPO_ROOT / "repos" / "DemoGen" / "real_world" / "export_demogen_zarr_to_robomimic_lowdim_twophase.py"),
            "--generated-zarr",
            str(output_zarr),
            "--source-low-dim-hdf5",
            str(source_low_dim),
            "--output-hdf5",
            str(output_hdf5),
            "--overwrite",
        ]
        if args.include_source_demos:
            export_cmd.append("--include-source-demos")
        run_subprocess(export_cmd, print_only=args.print_only)

    summary = {
        "status": "planned" if args.print_only else "completed",
        "template_zarr": str(template_zarr),
        "selected_episodes": args.selected_episodes,
        "solved_zarr": str(output_zarr),
        "solve_report": str(output_json),
        "exported_hdf5": str(output_hdf5),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
