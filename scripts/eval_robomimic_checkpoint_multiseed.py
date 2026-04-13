#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class SeedResult:
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
        description="Evaluate one robomimic checkpoint on multiple seeds and summarize results."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--output-dir", required=True, help="Directory to store logs and reports")
    parser.add_argument("--n-rollouts", type=int, default=20, help="Episodes per seed")
    parser.add_argument("--horizon", type=int, default=1000, help="Rollout horizon")
    parser.add_argument("--parallel-jobs", type=int, default=3, help="Concurrent seed jobs")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Seeds to evaluate, e.g. --seeds 1 2 3",
    )
    parser.add_argument("--conda-env", default="robomimic", help="Conda env name")
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


def save_reports(
    checkpoint: Path,
    n_rollouts: int,
    horizon: int,
    results: dict[int, SeedResult],
    reports_dir: Path,
) -> None:
    payload = {
        "checkpoint": str(checkpoint),
        "n_rollouts": n_rollouts,
        "horizon": horizon,
        "results": {str(seed): asdict(result) for seed, result in sorted(results.items())},
    }

    (reports_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (reports_dir / "leaderboard.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed",
                "status",
                "success_rate",
                "num_success",
                "avg_return",
                "avg_horizon",
                "return_code",
                "log_path",
                "started_at",
                "updated_at",
                "error",
            ]
        )
        for seed, result in sorted(results.items()):
            writer.writerow(
                [
                    seed,
                    result.status,
                    result.success_rate,
                    result.num_success,
                    result.avg_return,
                    result.avg_horizon,
                    result.return_code,
                    result.log_path,
                    result.started_at,
                    result.updated_at,
                    result.error,
                ]
            )

    lines = [
        f"checkpoint: {checkpoint}",
        f"n_rollouts: {n_rollouts}",
        f"horizon: {horizon}",
        "",
        "seed | status    | success_rate | num_success | avg_return | avg_horizon",
    ]
    completed = [r for r in results.values() if r.status == "completed" and r.success_rate is not None]
    for seed, result in sorted(results.items()):
        sr = "-" if result.success_rate is None else f"{result.success_rate:.4f}"
        ns = "-" if result.num_success is None else f"{result.num_success:.0f}"
        ret = "-" if result.avg_return is None else f"{result.avg_return:.4f}"
        hor = "-" if result.avg_horizon is None else f"{result.avg_horizon:.2f}"
        lines.append(f"{seed:>4} | {result.status:<9} | {sr:>12} | {ns:>11} | {ret:>10} | {hor:>11}")
    if completed:
        mean_sr = sum(r.success_rate for r in completed if r.success_rate is not None) / len(completed)
        mean_ret = sum(r.avg_return for r in completed if r.avg_return is not None) / len(completed)
        mean_hor = sum(r.avg_horizon for r in completed if r.avg_horizon is not None) / len(completed)
        mean_num_success = sum(r.num_success for r in completed if r.num_success is not None) / len(completed)
        lines.extend(
            [
                "",
                f"completed_seeds: {len(completed)}",
                f"mean_success_rate: {mean_sr:.4f}",
                f"mean_num_success: {mean_num_success:.2f}",
                f"mean_return: {mean_ret:.4f}",
                f"mean_horizon: {mean_hor:.2f}",
            ]
        )

    (reports_dir / "leaderboard.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_command(checkpoint: Path, seed: int, n_rollouts: int, horizon: int, conda_env: str) -> list[str]:
    return [
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


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    output_paths = ensure_dirs(Path(args.output_dir).expanduser().resolve())
    results: dict[int, SeedResult] = {}
    running: dict[int, tuple[subprocess.Popen[str], Any, Path]] = {}
    pending = list(args.seeds)

    save_reports(checkpoint, args.n_rollouts, args.horizon, results, output_paths["reports"])
    print(f"[{now_str()}] checkpoint -> {checkpoint}", flush=True)
    print(f"[{now_str()}] output -> {output_paths['root']}", flush=True)
    print(f"[{now_str()}] seeds -> {args.seeds}", flush=True)
    print(f"[{now_str()}] parallel_jobs -> {args.parallel_jobs}", flush=True)

    while pending or running:
        while pending and len(running) < max(1, int(args.parallel_jobs)):
            seed = pending.pop(0)
            log_path = output_paths["logs"] / f"seed_{seed}_r{args.n_rollouts}_h{args.horizon}.txt"
            log_file = log_path.open("w", encoding="utf-8")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"
            proc = subprocess.Popen(
                build_command(checkpoint, seed, args.n_rollouts, args.horizon, args.conda_env),
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[seed] = (proc, log_file, log_path)
            results[seed] = SeedResult(
                seed=seed,
                status="running",
                log_path=str(log_path),
                started_at=now_str(),
                updated_at=now_str(),
            )
            save_reports(checkpoint, args.n_rollouts, args.horizon, results, output_paths["reports"])
            print(f"[{now_str()}] launched seed {seed}", flush=True)

        finished: list[int] = []
        for seed, (proc, log_file, log_path) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log_file.close()
            result = results[seed]
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
            save_reports(checkpoint, args.n_rollouts, args.horizon, results, output_paths["reports"])
            print(
                f"[{now_str()}] seed {seed} done: status={result.status}, "
                f"success_rate={result.success_rate}, num_success={result.num_success}",
                flush=True,
            )
            finished.append(seed)

        for seed in finished:
            running.pop(seed, None)

        if running:
            time.sleep(2.0)

    print(f"[{now_str()}] all seeds finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
