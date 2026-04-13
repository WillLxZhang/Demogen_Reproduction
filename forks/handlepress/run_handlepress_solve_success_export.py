#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run HandlePress solve automation: smoke solve, full solve, success gate, "
            "and conditional replayobs export."
        )
    )
    parser.add_argument(
        "--config",
        default=str(
            REPO_ROOT
            / "repos"
            / "DemoGen"
            / "demo_generation"
            / "demo_generation"
            / "config"
            / "handlepress_0_v37_replayh1_light_schedule_phasecopy_replayconsistent.yaml"
        ),
    )
    parser.add_argument(
        "--template-zarr",
        default=str(
            REPO_ROOT
            / "repos"
            / "DemoGen"
            / "data"
            / "datasets"
            / "generated"
            / "handlepress_0_v37_replayh1_light_schedule_phasecopy_replayconsistent_test_25.zarr"
        ),
    )
    parser.add_argument(
        "--source-demo",
        default=str(
            REPO_ROOT
            / "data"
            / "raw"
            / "handlepress_0"
            / "1776042489_1873188"
            / "demo.hdf5"
        ),
    )
    parser.add_argument(
        "--source-low-dim",
        default=str(
            REPO_ROOT
            / "data"
            / "raw"
            / "handlepress_0"
            / "1776042489_1873188"
            / "low_dim.hdf5"
        ),
    )
    parser.add_argument("--control-steps", type=int, default=1)
    parser.add_argument("--action-deviation-weight", type=float, default=1e-4)
    parser.add_argument("--relative-tail-steps", type=int, default=40)
    parser.add_argument("--relative-cost-weight", type=float, default=4.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=0.5,
        help="Only export if solved dataset success_rate is at least this value.",
    )
    parser.add_argument("--include-source-demos", action="store_true")
    parser.add_argument("--export-overwrite", action="store_true")
    return parser.parse_args()


def run_and_log(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write("$ " + " ".join(cmd) + "\n\n")
        log_f.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
            log_f.flush()
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)


def main() -> None:
    args = parse_args()
    python_bin = sys.executable

    config = Path(args.config).expanduser().resolve()
    template_zarr = Path(args.template_zarr).expanduser().resolve()
    source_demo = Path(args.source_demo).expanduser().resolve()
    source_low_dim = Path(args.source_low_dim).expanduser().resolve()

    outputs_generated = REPO_ROOT / "outputs" / "generated"
    outputs_analysis = REPO_ROOT / "outputs" / "analysis"
    outputs_logs = REPO_ROOT / "outputs" / "logs"
    outputs_robomimic = REPO_ROOT / "outputs" / "robomimic"

    stem = template_zarr.stem
    smoke_zarr = outputs_generated / f"{stem}_relalign_smoke_ep0.zarr"
    smoke_json = outputs_analysis / f"{stem}_relalign_smoke_ep0.json"
    full_zarr = outputs_generated / f"{stem}_relalign_all.zarr"
    full_json = outputs_analysis / f"{stem}_relalign_all.json"
    success_json = outputs_analysis / f"{stem}_relalign_all_success.json"
    export_hdf5 = outputs_robomimic / f"{stem}_relalign_all_replayobs_lowdim.hdf5"

    solve_script = REPO_ROOT / "scripts" / "export_handlepress_solved_from_template_zarr_relalign.py"
    success_script = REPO_ROOT / "scripts" / "eval_handlepress_generated_zarr_success_rate.py"
    export_script = (
        REPO_ROOT
        / "repos"
        / "DemoGen"
        / "real_world"
        / "export_demogen_handlepress_zarr_to_robomimic_lowdim_replayobs.py"
    )

    smoke_cmd = [
        python_bin,
        str(solve_script),
        "--config",
        str(config),
        "--template-zarr",
        str(template_zarr),
        "--source-demo",
        str(source_demo),
        "--episodes",
        "0",
        "--control-steps",
        str(args.control_steps),
        "--action-deviation-weight",
        str(args.action_deviation_weight),
        "--relative-tail-steps",
        str(args.relative_tail_steps),
        "--relative-cost-weight",
        str(args.relative_cost_weight),
        "--num-workers",
        "1",
        "--output-zarr",
        str(smoke_zarr),
        "--output-json",
        str(smoke_json),
    ]
    run_and_log(smoke_cmd, outputs_logs / f"{stem}_smoke_solve.log")

    smoke_payload = json.loads(smoke_json.read_text(encoding="utf-8"))
    if int(smoke_payload["aggregate"]["n_episodes"]) != 1:
        raise RuntimeError("Smoke solve did not produce exactly one episode")

    full_cmd = [
        python_bin,
        str(solve_script),
        "--config",
        str(config),
        "--template-zarr",
        str(template_zarr),
        "--source-demo",
        str(source_demo),
        "--episodes",
        "all",
        "--control-steps",
        str(args.control_steps),
        "--action-deviation-weight",
        str(args.action_deviation_weight),
        "--relative-tail-steps",
        str(args.relative_tail_steps),
        "--relative-cost-weight",
        str(args.relative_cost_weight),
        "--num-workers",
        str(args.num_workers),
        "--output-zarr",
        str(full_zarr),
        "--output-json",
        str(full_json),
    ]
    run_and_log(full_cmd, outputs_logs / f"{stem}_full_solve.log")

    success_cmd = [
        python_bin,
        str(success_script),
        "--zarr",
        str(full_zarr),
        "--source-demo",
        str(source_demo),
        "--control-steps",
        str(args.control_steps),
        "--output-json",
        str(success_json),
    ]
    run_and_log(success_cmd, outputs_logs / f"{stem}_success_eval.log")

    success_payload = json.loads(success_json.read_text(encoding="utf-8"))
    success_rate = float(success_payload["success_rate"])
    print(f"\nSuccess rate: {success_rate:.6f} (threshold {args.min_success_rate:.6f})")

    if success_rate < float(args.min_success_rate):
        print("Success gate did not pass. Skipping export.")
        return

    export_cmd = [
        python_bin,
        str(export_script),
        "--generated-zarr",
        str(full_zarr),
        "--source-low-dim-hdf5",
        str(source_low_dim),
        "--output-hdf5",
        str(export_hdf5),
        "--control-steps",
        str(args.control_steps),
    ]
    if args.include_source_demos:
        export_cmd.append("--include-source-demos")
    if args.export_overwrite:
        export_cmd.append("--overwrite")
    run_and_log(export_cmd, outputs_logs / f"{stem}_export.log")


if __name__ == "__main__":
    main()
