#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CHECKPOINT_RE = re.compile(r"model_epoch_(\d+)\.pth$")


@dataclass
class EvalResult:
    epoch: int
    checkpoint: str
    status: str
    return_code: int | None = None
    success_rate: float | None = None
    num_success: float | None = None
    avg_return: float | None = None
    avg_horizon: float | None = None
    log_path: str | None = None
    dataset_path: str | None = None
    updated_at: str | None = None
    error: str | None = None


@dataclass
class State:
    run_dir: str
    n_rollouts: int
    horizon: int
    seed: int
    results: dict[str, EvalResult] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously evaluate robomimic checkpoints in a selected epoch range, "
            "and write a persistent leaderboard to disk."
        )
    )
    parser.add_argument("--run-dir", required=True, help="Timestamped robomimic run directory")
    parser.add_argument("--n-rollouts", type=int, default=100, help="Rollouts per checkpoint")
    parser.add_argument("--horizon", type=int, default=1000, help="Rollout horizon override")
    parser.add_argument("--seed", type=int, default=1, help="Seed passed to run_trained_agent.py")
    parser.add_argument("--poll-seconds", type=float, default=20.0, help="Polling interval")
    parser.add_argument(
        "--eval-subdir",
        default=None,
        help="Optional relative subdirectory under run-dir for outputs. Defaults to auto_eval_r{n}_h{h}_seed{s}.",
    )
    parser.add_argument(
        "--min-epoch",
        type=int,
        default=None,
        help="Minimum checkpoint epoch to evaluate, inclusive.",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=None,
        help="Maximum checkpoint epoch to evaluate, inclusive.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force evaluation onto CPU.",
    )
    parser.add_argument(
        "--nice",
        type=int,
        default=10,
        help="Unix nice level for evaluation subprocesses.",
    )
    parser.add_argument(
        "--save-rollout-hdf5",
        action="store_true",
        help="Persist rollout datasets from run_trained_agent.py for later inspection.",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send desktop notifications when an evaluation finishes.",
    )
    parser.add_argument(
        "--exit-when-done",
        action="store_true",
        help="Exit after all target checkpoints have been evaluated.",
    )
    return parser.parse_args()


def now_str() -> str:
    return time.strftime("%F %T")


def send_notification(title: str, body: str) -> None:
    try:
        subprocess.run(
            ["notify-send", "-u", "normal", title, body],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass


def checkpoint_epoch(path: Path) -> int | None:
    match = CHECKPOINT_RE.search(path.name)
    return int(match.group(1)) if match else None


def list_checkpoints(models_dir: Path) -> list[tuple[int, Path]]:
    items: list[tuple[int, Path]] = []
    if not models_dir.exists():
        return items
    for path in sorted(models_dir.glob("model_epoch_*.pth")):
        epoch = checkpoint_epoch(path)
        if epoch is not None:
            items.append((epoch, path))
    items.sort(key=lambda x: x[0])
    return items


def training_finished(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "finished run successfully!" in text


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "logs": root / "logs",
        "rollouts": root / "rollouts",
        "reports": root / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def state_to_jsonable(state: State) -> dict[str, Any]:
    return {
        "run_dir": state.run_dir,
        "n_rollouts": state.n_rollouts,
        "horizon": state.horizon,
        "seed": state.seed,
        "results": {
            key: {
                "epoch": value.epoch,
                "checkpoint": value.checkpoint,
                "status": value.status,
                "return_code": value.return_code,
                "success_rate": value.success_rate,
                "num_success": value.num_success,
                "avg_return": value.avg_return,
                "avg_horizon": value.avg_horizon,
                "log_path": value.log_path,
                "dataset_path": value.dataset_path,
                "updated_at": value.updated_at,
                "error": value.error,
            }
            for key, value in state.results.items()
        },
    }


def load_state(path: Path, run_dir: Path, n_rollouts: int, horizon: int, seed: int) -> State:
    if not path.exists():
        return State(run_dir=str(run_dir), n_rollouts=n_rollouts, horizon=horizon, seed=seed)
    data = json.loads(path.read_text(encoding="utf-8"))
    state = State(
        run_dir=data.get("run_dir", str(run_dir)),
        n_rollouts=int(data.get("n_rollouts", n_rollouts)),
        horizon=int(data.get("horizon", horizon)),
        seed=int(data.get("seed", seed)),
    )
    for key, value in data.get("results", {}).items():
        state.results[key] = EvalResult(
            epoch=int(value["epoch"]),
            checkpoint=str(value["checkpoint"]),
            status=str(value["status"]),
            return_code=value.get("return_code"),
            success_rate=value.get("success_rate"),
            num_success=value.get("num_success"),
            avg_return=value.get("avg_return"),
            avg_horizon=value.get("avg_horizon"),
            log_path=value.get("log_path"),
            dataset_path=value.get("dataset_path"),
            updated_at=value.get("updated_at"),
            error=value.get("error"),
        )
    return state


def save_state(state: State, reports_dir: Path) -> None:
    state_path = reports_dir / "state.json"
    state_path.write_text(json.dumps(state_to_jsonable(state), indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary(state: State, reports_dir: Path) -> None:
    rows = sorted(state.results.values(), key=lambda r: r.epoch)

    summary_json = reports_dir / "summary.json"
    summary_json.write_text(json.dumps(state_to_jsonable(state), indent=2, ensure_ascii=False), encoding="utf-8")

    summary_csv = reports_dir / "leaderboard.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "status",
                "success_rate",
                "num_success",
                "avg_return",
                "avg_horizon",
                "return_code",
                "checkpoint",
                "log_path",
                "dataset_path",
                "updated_at",
                "error",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.epoch,
                    row.status,
                    row.success_rate,
                    row.num_success,
                    row.avg_return,
                    row.avg_horizon,
                    row.return_code,
                    row.checkpoint,
                    row.log_path,
                    row.dataset_path,
                    row.updated_at,
                    row.error,
                ]
            )

    summary_txt = reports_dir / "leaderboard.txt"
    lines = [
        f"run_dir: {state.run_dir}",
        f"n_rollouts: {state.n_rollouts}",
        f"horizon: {state.horizon}",
        f"seed: {state.seed}",
        "",
        "epoch | status | success_rate | num_success | avg_return | avg_horizon",
    ]
    for row in rows:
        sr = "-" if row.success_rate is None else f"{row.success_rate:.4f}"
        ns = "-" if row.num_success is None else f"{row.num_success:.0f}"
        ret = "-" if row.avg_return is None else f"{row.avg_return:.4f}"
        hor = "-" if row.avg_horizon is None else f"{row.avg_horizon:.2f}"
        lines.append(f"{row.epoch:>5} | {row.status:<9} | {sr:>12} | {ns:>11} | {ret:>10} | {hor:>11}")
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def build_eval_command(
    *,
    checkpoint_path: Path,
    dataset_path: Path | None,
    n_rollouts: int,
    horizon: int,
    seed: int,
    cpu_only: bool,
    nice_level: int,
) -> tuple[list[str], dict[str, str]]:
    eval_cmd = [
        "conda",
        "run",
        "-n",
        "robomimic",
        "python",
        "repos/robomimic/robomimic/scripts/run_trained_agent.py",
        "--agent",
        str(checkpoint_path),
        "--n_rollouts",
        str(n_rollouts),
        "--horizon",
        str(horizon),
        "--seed",
        str(seed),
    ]
    if dataset_path is not None:
        eval_cmd.extend(["--dataset_path", str(dataset_path)])

    shell_parts = []
    if nice_level != 0:
        shell_parts.extend(["nice", "-n", str(nice_level)])
    shell_parts.extend(eval_cmd)

    env = os.environ.copy()
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    return shell_parts, env


def evaluate_checkpoint(
    *,
    epoch: int,
    checkpoint_path: Path,
    output_paths: dict[str, Path],
    state: State,
    args: argparse.Namespace,
) -> EvalResult:
    log_path = output_paths["logs"] / f"model_epoch_{epoch}_r{args.n_rollouts}_h{args.horizon}_seed{args.seed}.txt"
    dataset_path = None
    if args.save_rollout_hdf5:
        dataset_path = output_paths["rollouts"] / f"model_epoch_{epoch}_r{args.n_rollouts}_h{args.horizon}_seed{args.seed}.hdf5"

    print(f"[{now_str()}] evaluating epoch {epoch} -> {checkpoint_path}", flush=True)
    cmd, env = build_eval_command(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        n_rollouts=args.n_rollouts,
        horizon=args.horizon,
        seed=args.seed,
        cpu_only=args.cpu_only,
        nice_level=args.nice,
    )

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    result = EvalResult(
        epoch=epoch,
        checkpoint=str(checkpoint_path),
        status="failed" if proc.returncode != 0 else "completed",
        return_code=proc.returncode,
        log_path=str(log_path),
        dataset_path=str(dataset_path) if dataset_path is not None else None,
        updated_at=now_str(),
    )

    if proc.returncode == 0:
        try:
            stats = extract_avg_rollout_stats(log_path)
            result.success_rate = float(stats.get("Success_Rate")) if "Success_Rate" in stats else None
            result.num_success = float(stats.get("Num_Success")) if "Num_Success" in stats else None
            result.avg_return = float(stats.get("Return")) if "Return" in stats else None
            result.avg_horizon = float(stats.get("Horizon")) if "Horizon" in stats else None
        except Exception as exc:
            result.status = "parse_failed"
            result.error = str(exc)
    else:
        result.error = f"returncode={proc.returncode}"

    state.results[str(epoch)] = result
    save_state(state, output_paths["reports"])
    write_summary(state, output_paths["reports"])

    if args.notify:
        if result.status == "completed":
            body = (
                f"epoch {epoch}: success_rate={result.success_rate:.4f}, "
                f"num_success={result.num_success:.0f}/{args.n_rollouts}"
            )
        else:
            body = f"epoch {epoch}: status={result.status}"
        send_notification("Auto Eval", body)

    print(
        f"[{now_str()}] epoch {epoch} done: status={result.status}, "
        f"success_rate={result.success_rate}, num_success={result.num_success}",
        flush=True,
    )
    return result


def epoch_in_range(epoch: int, min_epoch: int | None, max_epoch: int | None) -> bool:
    if min_epoch is not None and epoch < min_epoch:
        return False
    if max_epoch is not None and epoch > max_epoch:
        return False
    return True


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run-dir not found: {run_dir}")

    models_dir = run_dir / "models"
    train_log_path = run_dir / "logs" / "log.txt"
    eval_subdir = args.eval_subdir or f"auto_eval_r{args.n_rollouts}_h{args.horizon}_seed{args.seed}"
    output_paths = ensure_dirs(run_dir / eval_subdir)
    state = load_state(
        output_paths["reports"] / "state.json",
        run_dir=run_dir,
        n_rollouts=args.n_rollouts,
        horizon=args.horizon,
        seed=args.seed,
    )
    save_state(state, output_paths["reports"])
    write_summary(state, output_paths["reports"])

    stop = False

    def handle_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    print(f"[{now_str()}] watching checkpoints in {models_dir}", flush=True)
    print(f"[{now_str()}] outputs -> {output_paths['root']}", flush=True)
    print(f"[{now_str()}] epoch range -> min={args.min_epoch}, max={args.max_epoch}", flush=True)

    while not stop:
        checkpoints = [
            (epoch, path)
            for epoch, path in list_checkpoints(models_dir)
            if epoch_in_range(epoch, args.min_epoch, args.max_epoch)
        ]
        pending = []
        for epoch, path in checkpoints:
            current = state.results.get(str(epoch))
            if current is None or current.status == "running":
                pending.append((epoch, path))

        if pending:
            epoch, ckpt = pending[0]
            state.results[str(epoch)] = EvalResult(
                epoch=epoch,
                checkpoint=str(ckpt),
                status="running",
                updated_at=now_str(),
            )
            save_state(state, output_paths["reports"])
            write_summary(state, output_paths["reports"])
            evaluate_checkpoint(
                epoch=epoch,
                checkpoint_path=ckpt,
                output_paths=output_paths,
                state=state,
                args=args,
            )
            continue

        if args.exit_when_done:
            if training_finished(train_log_path):
                print(f"[{now_str()}] no pending checkpoints in target range and training is finished, exiting", flush=True)
                break

            if args.max_epoch is not None:
                seen_epochs = {epoch for epoch, _ in checkpoints}
                if seen_epochs and max(seen_epochs) >= args.max_epoch:
                    print(f"[{now_str()}] no pending checkpoints in target range and max epoch is available, exiting", flush=True)
                    break

        time.sleep(max(1.0, float(args.poll_seconds)))

    save_state(state, output_paths["reports"])
    write_summary(state, output_paths["reports"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
