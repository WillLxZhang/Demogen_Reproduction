#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EPOCH_RE = re.compile(r"model_epoch_(\d+)\.pth$")


@dataclass
class JobResult:
    checkpoint: str
    epoch: int
    seed: int
    status: str
    return_code: int | None = None
    success_rate: float | None = None
    num_success: float | None = None
    avg_return: float | None = None
    avg_horizon: float | None = None
    log_path: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    error: str | None = None


def now_str() -> str:
    return time.strftime("%F %T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple robomimic checkpoints across multiple seeds with a shared job queue."
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Checkpoint paths, e.g. model_epoch_60.pth model_epoch_150.pth model_epoch_270.pth",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to store logs and reports")
    parser.add_argument("--n-rollouts", type=int, default=20, help="Episodes per seed")
    parser.add_argument("--horizon", type=int, default=1000, help="Rollout horizon")
    parser.add_argument("--parallel-jobs", type=int, default=3, help="Concurrent jobs")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Seeds to evaluate, e.g. --seeds 1 2 3",
    )
    parser.add_argument("--conda-env", default="robomimic", help="Conda env name")
    parser.add_argument("--nice", type=int, default=0, help="Nice level for evaluation subprocesses")
    return parser.parse_args()


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "logs": root / "logs",
        "reports": root / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def checkpoint_epoch(path: Path) -> int:
    match = EPOCH_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse epoch from checkpoint name: {path}")
    return int(match.group(1))


def extract_avg_rollout_stats(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    marker = "Average Rollout Stats"
    idx = text.rfind(marker)
    if idx < 0:
        raise ValueError(f"Could not find '{marker}' in {log_path}")
    start = text.find("{", idx)
    if start < 0:
        raise ValueError(f"Could not find JSON block after '{marker}' in {log_path}")
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(text[start:])
    return payload


def build_command(
    checkpoint: Path,
    seed: int,
    n_rollouts: int,
    horizon: int,
    conda_env: str,
    nice_level: int,
) -> list[str]:
    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        "repos/robomimic/robomimic/scripts/run_trained_agent.py",
        "--agent",
        str(checkpoint),
        "--n_rollouts",
        str(n_rollouts),
        "--horizon",
        str(horizon),
        "--seed",
        str(seed),
    ]
    if nice_level != 0:
        return ["nice", "-n", str(nice_level), *cmd]
    return cmd


def save_reports(
    checkpoints: list[Path],
    n_rollouts: int,
    horizon: int,
    seeds: list[int],
    results: dict[tuple[int, int], JobResult],
    reports_dir: Path,
) -> None:
    payload = {
        "checkpoints": [str(p) for p in checkpoints],
        "n_rollouts": n_rollouts,
        "horizon": horizon,
        "seeds": seeds,
        "results": {
            f"{epoch}:{seed}": asdict(result)
            for (epoch, seed), result in sorted(results.items())
        },
    }
    (reports_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (reports_dir / "per_seed.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "seed",
                "status",
                "success_rate",
                "num_success",
                "avg_return",
                "avg_horizon",
                "return_code",
                "checkpoint",
                "log_path",
                "started_at",
                "updated_at",
                "error",
            ]
        )
        for (_, _), result in sorted(results.items()):
            writer.writerow(
                [
                    result.epoch,
                    result.seed,
                    result.status,
                    result.success_rate,
                    result.num_success,
                    result.avg_return,
                    result.avg_horizon,
                    result.return_code,
                    result.checkpoint,
                    result.log_path,
                    result.started_at,
                    result.updated_at,
                    result.error,
                ]
            )

    per_checkpoint_rows: list[dict[str, Any]] = []
    lines = [
        "checkpoint evaluations",
        f"n_rollouts: {n_rollouts}",
        f"horizon: {horizon}",
        f"seeds: {seeds}",
        "",
        "epoch | status      | completed_seeds | mean_success_rate | mean_num_success | mean_return | mean_horizon",
    ]
    for checkpoint in checkpoints:
        epoch = checkpoint_epoch(checkpoint)
        ckpt_results = [r for (e, _), r in results.items() if e == epoch]
        completed = [r for r in ckpt_results if r.status == "completed" and r.success_rate is not None]
        status = "completed" if len(completed) == len(seeds) else "partial"
        mean_sr = None
        mean_ns = None
        mean_ret = None
        mean_hor = None
        if completed:
            mean_sr = sum(r.success_rate for r in completed if r.success_rate is not None) / len(completed)
            mean_ns = sum(r.num_success for r in completed if r.num_success is not None) / len(completed)
            mean_ret = sum(r.avg_return for r in completed if r.avg_return is not None) / len(completed)
            mean_hor = sum(r.avg_horizon for r in completed if r.avg_horizon is not None) / len(completed)
        per_checkpoint_rows.append(
            {
                "epoch": epoch,
                "checkpoint": str(checkpoint),
                "status": status,
                "completed_seeds": len(completed),
                "mean_success_rate": mean_sr,
                "mean_num_success": mean_ns,
                "mean_return": mean_ret,
                "mean_horizon": mean_hor,
            }
        )
        sr_str = "-" if mean_sr is None else f"{mean_sr:.4f}"
        ns_str = "-" if mean_ns is None else f"{mean_ns:.2f}"
        ret_str = "-" if mean_ret is None else f"{mean_ret:.4f}"
        hor_str = "-" if mean_hor is None else f"{mean_hor:.2f}"
        lines.append(
            f"{epoch:>5} | {status:<11} | {len(completed):>15} | {sr_str:>17} | {ns_str:>16} | {ret_str:>11} | {hor_str:>12}"
        )

    with (reports_dir / "per_checkpoint.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "checkpoint",
                "status",
                "completed_seeds",
                "mean_success_rate",
                "mean_num_success",
                "mean_return",
                "mean_horizon",
            ],
        )
        writer.writeheader()
        for row in per_checkpoint_rows:
            writer.writerow(row)

    (reports_dir / "leaderboard.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    checkpoints = [Path(p).expanduser().resolve() for p in args.checkpoints]
    for checkpoint in checkpoints:
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    checkpoints = sorted(checkpoints, key=checkpoint_epoch)

    output_paths = ensure_dirs(Path(args.output_dir).expanduser().resolve())
    results: dict[tuple[int, int], JobResult] = {}
    running: dict[tuple[int, int], tuple[subprocess.Popen[str], Any, Path]] = {}
    pending: list[tuple[Path, int]] = []
    for checkpoint in checkpoints:
        for seed in args.seeds:
            pending.append((checkpoint, seed))

    save_reports(checkpoints, args.n_rollouts, args.horizon, args.seeds, results, output_paths["reports"])
    print(f"[{now_str()}] output -> {output_paths['root']}", flush=True)
    print(f"[{now_str()}] checkpoints -> {[str(p) for p in checkpoints]}", flush=True)
    print(f"[{now_str()}] seeds -> {args.seeds}", flush=True)
    print(f"[{now_str()}] parallel_jobs -> {args.parallel_jobs}", flush=True)

    while pending or running:
        while pending and len(running) < max(1, int(args.parallel_jobs)):
            checkpoint, seed = pending.pop(0)
            epoch = checkpoint_epoch(checkpoint)
            log_path = output_paths["logs"] / f"epoch_{epoch}_seed_{seed}_r{args.n_rollouts}_h{args.horizon}.txt"
            log_file = log_path.open("w", encoding="utf-8")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"
            proc = subprocess.Popen(
                build_command(
                    checkpoint=checkpoint,
                    seed=seed,
                    n_rollouts=args.n_rollouts,
                    horizon=args.horizon,
                    conda_env=args.conda_env,
                    nice_level=args.nice,
                ),
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[(epoch, seed)] = (proc, log_file, log_path)
            results[(epoch, seed)] = JobResult(
                checkpoint=str(checkpoint),
                epoch=epoch,
                seed=seed,
                status="running",
                log_path=str(log_path),
                started_at=now_str(),
                updated_at=now_str(),
            )
            save_reports(checkpoints, args.n_rollouts, args.horizon, args.seeds, results, output_paths["reports"])
            print(f"[{now_str()}] launched epoch {epoch} seed {seed}", flush=True)

        finished: list[tuple[int, int]] = []
        for key, (proc, log_file, log_path) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log_file.close()
            result = results[key]
            result.return_code = rc
            result.updated_at = now_str()
            if rc == 0:
                try:
                    stats = extract_avg_rollout_stats(log_path)
                    result.status = "completed"
                    result.success_rate = float(stats.get("Success_Rate")) if "Success_Rate" in stats else None
                    result.num_success = float(stats.get("Num_Success")) if "Num_Success" in stats else None
                    result.avg_return = float(stats.get("Return")) if "Return" in stats else None
                    result.avg_horizon = float(stats.get("Horizon")) if "Horizon" in stats else None
                except Exception as exc:
                    result.status = "parse_failed"
                    result.error = str(exc)
            else:
                result.status = "failed"
                result.error = f"returncode={rc}"
            save_reports(checkpoints, args.n_rollouts, args.horizon, args.seeds, results, output_paths["reports"])
            print(
                f"[{now_str()}] epoch {result.epoch} seed {result.seed} done: "
                f"status={result.status}, success_rate={result.success_rate}, num_success={result.num_success}",
                flush=True,
            )
            finished.append(key)

        for key in finished:
            running.pop(key, None)

        if running:
            time.sleep(2.0)

    print(f"[{now_str()}] all jobs finished", flush=True)
    save_reports(checkpoints, args.n_rollouts, args.horizon, args.seeds, results, output_paths["reports"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
