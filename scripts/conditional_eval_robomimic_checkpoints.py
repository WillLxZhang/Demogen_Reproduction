#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageState:
    status: str = "pending"
    output_dir: str | None = None
    log_path: str | None = None
    checkpoint: str | None = None
    return_code: int | None = None
    mean_success_rate: float | None = None
    completed_seeds: int | None = None
    started_at: str | None = None
    updated_at: str | None = None
    error: str | None = None


@dataclass
class EpochState:
    epoch: int
    short: StageState = field(default_factory=StageState)
    full: StageState = field(default_factory=StageState)
    decision: str = "pending"
    decision_reason: str | None = None


@dataclass
class GuardState:
    external_root: str
    manifest_path: str
    train_run_name: str
    run_dir: str | None = None
    checkpoints: dict[str, EpochState] = field(default_factory=dict)
    updated_at: str | None = None


def now_str() -> str:
    return time.strftime("%F %T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for selected robomimic checkpoints to appear, run a short eval first, "
            "and only launch a longer multiseed eval when the short eval passes a threshold."
        )
    )
    parser.add_argument("--external-root", required=True, help="Timestamped external experiment root")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest path. Defaults to <external-root>/manifest.json",
    )
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[60, 120])
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--short-rollouts", type=int, default=10)
    parser.add_argument("--short-seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--short-threshold", type=float, default=0.5)
    parser.add_argument("--full-rollouts", type=int, default=20)
    parser.add_argument("--full-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--horizon", type=int, default=800)
    parser.add_argument("--short-parallel-jobs", type=int, default=1)
    parser.add_argument("--full-parallel-jobs", type=int, default=1)
    parser.add_argument("--conda-env", default="robomimic")
    parser.add_argument(
        "--eval-script",
        default=None,
        help="Path to eval_robomimic_checkpoint_multiseed.py. Defaults to repo script.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def log_message(log_path: Path, message: str) -> None:
    line = f"[{now_str()}] {message}"
    print(line, flush=True)
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def epoch_state_to_dict(state: EpochState) -> dict[str, Any]:
    return {
        "epoch": state.epoch,
        "short": vars(state.short),
        "full": vars(state.full),
        "decision": state.decision,
        "decision_reason": state.decision_reason,
    }


def save_state(path: Path, state: GuardState) -> None:
    payload = {
        "external_root": state.external_root,
        "manifest_path": state.manifest_path,
        "train_run_name": state.train_run_name,
        "run_dir": state.run_dir,
        "checkpoints": {key: epoch_state_to_dict(value) for key, value in sorted(state.checkpoints.items(), key=lambda item: int(item[0]))},
        "updated_at": state.updated_at,
    }
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_train_run_dir(run_root: Path) -> Path | None:
    if not run_root.exists():
        return None
    children = sorted([p for p in run_root.iterdir() if p.is_dir()])
    if not children:
        return None
    return children[-1]


def training_finished(run_dir: Path) -> bool:
    log_path = run_dir / "logs" / "log.txt"
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "finished run successfully!" in text


def stage_dir_name(epoch: int, stage: str, n_rollouts: int, seeds: list[int]) -> str:
    seed_part = "seed" + "".join(str(seed) for seed in seeds) if len(seeds) == 1 else "seeds" + "".join(str(seed) for seed in seeds)
    return f"epoch{epoch:03d}_{stage}_r{n_rollouts}_{seed_part}"


def is_terminal_status(status: str) -> bool:
    return status in {"completed", "failed", "parse_failed", "skipped", "blocked"}


def extract_mean_success(summary_path: Path) -> tuple[float, int]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    completed: list[float] = []
    for result in payload.get("results", {}).values():
        if result.get("status") == "completed" and result.get("success_rate") is not None:
            completed.append(float(result["success_rate"]))
    if not completed:
        raise ValueError(f"no completed seed results found in {summary_path}")
    return sum(completed) / len(completed), len(completed)


def run_eval_stage(
    *,
    repo_root: Path,
    eval_script: Path,
    checkpoint: Path,
    output_dir: Path,
    n_rollouts: int,
    horizon: int,
    seeds: list[int],
    parallel_jobs: int,
    conda_env: str,
    stage_state: StageState,
    stage_name: str,
    log_path: Path,
    guard_state_path: Path,
    guard_state: GuardState,
) -> StageState:
    stage_state.status = "running"
    stage_state.output_dir = str(output_dir)
    stage_state.log_path = str(output_dir / "stage.log")
    stage_state.checkpoint = str(checkpoint)
    stage_state.started_at = stage_state.started_at or now_str()
    stage_state.updated_at = now_str()
    save_state(guard_state_path, guard_state)

    ensure_dir(output_dir)
    cmd = [
        sys.executable,
        str(eval_script),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--n-rollouts",
        str(n_rollouts),
        "--horizon",
        str(horizon),
        "--parallel-jobs",
        str(parallel_jobs),
        "--conda-env",
        conda_env,
        "--seeds",
        *[str(seed) for seed in seeds],
    ]
    log_message(log_path, f"STAGE START {stage_name}: {' '.join(cmd)}")
    with (output_dir / "stage.log").open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    stage_state.return_code = proc.returncode
    stage_state.updated_at = now_str()
    if proc.returncode != 0:
        stage_state.status = "failed"
        stage_state.error = f"returncode={proc.returncode}"
        save_state(guard_state_path, guard_state)
        log_message(log_path, f"STAGE FAIL {stage_name}: returncode={proc.returncode} output_dir={output_dir}")
        return stage_state

    summary_path = output_dir / "reports" / "summary.json"
    try:
        mean_success_rate, completed_seeds = extract_mean_success(summary_path)
    except Exception as exc:
        stage_state.status = "parse_failed"
        stage_state.error = str(exc)
        save_state(guard_state_path, guard_state)
        log_message(log_path, f"STAGE PARSE FAIL {stage_name}: error={exc}")
        return stage_state

    stage_state.status = "completed"
    stage_state.mean_success_rate = mean_success_rate
    stage_state.completed_seeds = completed_seeds
    stage_state.error = None
    save_state(guard_state_path, guard_state)
    log_message(
        log_path,
        f"STAGE DONE {stage_name}: mean_success_rate={mean_success_rate:.4f} completed_seeds={completed_seeds}",
    )
    return stage_state


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    external_root = Path(args.external_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else external_root / "manifest.json"
    eval_script = Path(args.eval_script).expanduser().resolve() if args.eval_script else repo_root / "scripts" / "eval_robomimic_checkpoint_multiseed.py"

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not eval_script.exists():
        raise FileNotFoundError(f"eval script not found: {eval_script}")

    manifest = load_manifest(manifest_path)
    train_run_name = str(manifest["train_run_name"])
    run_root = external_root / "outputs" / "robomimic" / "diffusion_policy_demogen" / train_run_name
    guard_root = external_root / "outputs" / "robomimic" / "checkpoint_eval_guard"
    state_path = guard_root / "state.json"
    log_path = external_root / "logs" / "checkpoint_eval_guard.log"

    state = GuardState(
        external_root=str(external_root),
        manifest_path=str(manifest_path),
        train_run_name=train_run_name,
        checkpoints={str(epoch): EpochState(epoch=epoch) for epoch in sorted(set(args.checkpoints))},
        updated_at=now_str(),
    )

    stop = False

    def handle_stop(signum, frame):
        nonlocal stop
        stop = True
        log_message(log_path, f"RECEIVED SIGNAL {signum}, stopping after current step")

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    ensure_dir(guard_root)
    save_state(state_path, state)
    log_message(
        log_path,
        (
            f"GUARD START external_root={external_root} train_run_name={train_run_name} "
            f"checkpoints={sorted(set(args.checkpoints))} short_threshold={args.short_threshold:.3f}"
        ),
    )

    run_dir: Path | None = None
    training_done_logged = False

    while not stop:
        if run_dir is None:
            run_dir = find_train_run_dir(run_root)
            if run_dir is not None:
                state.run_dir = str(run_dir)
                state.updated_at = now_str()
                save_state(state_path, state)
                log_message(log_path, f"RUN DIR READY: {run_dir}")
            else:
                time.sleep(max(1.0, float(args.poll_seconds)))
                continue

        all_done = True
        for epoch in sorted(set(args.checkpoints)):
            epoch_key = str(epoch)
            epoch_state = state.checkpoints[epoch_key]
            checkpoint_path = run_dir / "models" / f"model_epoch_{epoch}.pth"

            if is_terminal_status(epoch_state.full.status):
                continue
            if epoch_state.decision in {"skipped", "blocked", "full_failed"}:
                continue
            all_done = False

            if not checkpoint_path.exists():
                continue

            if epoch_state.short.status == "pending":
                short_output_dir = guard_root / "evals" / stage_dir_name(
                    epoch,
                    "short",
                    args.short_rollouts,
                    list(args.short_seeds),
                )
                run_eval_stage(
                    repo_root=repo_root,
                    eval_script=eval_script,
                    checkpoint=checkpoint_path,
                    output_dir=short_output_dir,
                    n_rollouts=args.short_rollouts,
                    horizon=args.horizon,
                    seeds=list(args.short_seeds),
                    parallel_jobs=args.short_parallel_jobs,
                    conda_env=args.conda_env,
                    stage_state=epoch_state.short,
                    stage_name=f"epoch{epoch}_short",
                    log_path=log_path,
                    guard_state_path=state_path,
                    guard_state=state,
                )
                state.updated_at = now_str()
                save_state(state_path, state)

            if epoch_state.short.status in {"failed", "parse_failed"} and epoch_state.decision == "pending":
                epoch_state.decision = "blocked"
                epoch_state.decision_reason = f"short eval {epoch_state.short.status}: {epoch_state.short.error}"
                epoch_state.full.status = "blocked"
                epoch_state.full.updated_at = now_str()
                log_message(log_path, f"EPOCH {epoch} BLOCKED: {epoch_state.decision_reason}")
                state.updated_at = now_str()
                save_state(state_path, state)

            if epoch_state.short.status == "completed" and epoch_state.decision == "pending":
                if epoch_state.short.mean_success_rate is not None and epoch_state.short.mean_success_rate >= args.short_threshold:
                    epoch_state.decision = "promote"
                    epoch_state.decision_reason = (
                        f"short mean success {epoch_state.short.mean_success_rate:.4f} >= threshold {args.short_threshold:.4f}"
                    )
                    log_message(log_path, f"EPOCH {epoch} PROMOTED: {epoch_state.decision_reason}")
                else:
                    epoch_state.decision = "skipped"
                    if epoch_state.short.mean_success_rate is None:
                        epoch_state.decision_reason = "short eval completed without a parsed success rate"
                    else:
                        epoch_state.decision_reason = (
                            f"short mean success {epoch_state.short.mean_success_rate:.4f} < threshold {args.short_threshold:.4f}"
                        )
                    epoch_state.full.status = "skipped"
                    epoch_state.full.updated_at = now_str()
                    log_message(log_path, f"EPOCH {epoch} FULL SKIPPED: {epoch_state.decision_reason}")
                state.updated_at = now_str()
                save_state(state_path, state)

            if epoch_state.decision == "promote" and epoch_state.full.status == "pending":
                full_output_dir = guard_root / "evals" / stage_dir_name(
                    epoch,
                    "full",
                    args.full_rollouts,
                    list(args.full_seeds),
                )
                run_eval_stage(
                    repo_root=repo_root,
                    eval_script=eval_script,
                    checkpoint=checkpoint_path,
                    output_dir=full_output_dir,
                    n_rollouts=args.full_rollouts,
                    horizon=args.horizon,
                    seeds=list(args.full_seeds),
                    parallel_jobs=args.full_parallel_jobs,
                    conda_env=args.conda_env,
                    stage_state=epoch_state.full,
                    stage_name=f"epoch{epoch}_full",
                    log_path=log_path,
                    guard_state_path=state_path,
                    guard_state=state,
                )
                state.updated_at = now_str()
                save_state(state_path, state)

                if epoch_state.full.status in {"failed", "parse_failed"}:
                    epoch_state.decision = "full_failed"
                    epoch_state.decision_reason = f"full eval {epoch_state.full.status}: {epoch_state.full.error}"
                    log_message(log_path, f"EPOCH {epoch} FULL FAILED: {epoch_state.decision_reason}")
                    state.updated_at = now_str()
                    save_state(state_path, state)

        if all_done:
            log_message(log_path, "ALL TARGET CHECKPOINTS REACHED TERMINAL STATE, exiting guard")
            break

        if training_finished(run_dir):
            if not training_done_logged:
                training_done_logged = True
                missing = [
                    str(epoch)
                    for epoch in sorted(set(args.checkpoints))
                    if state.checkpoints[str(epoch)].short.status == "pending"
                ]
                if missing:
                    for epoch_str in missing:
                        epoch_state = state.checkpoints[epoch_str]
                        epoch_state.decision = "blocked"
                        epoch_state.decision_reason = "training finished before checkpoint appeared"
                        epoch_state.full.status = "blocked"
                        epoch_state.full.updated_at = now_str()
                    state.updated_at = now_str()
                    save_state(state_path, state)
                    log_message(
                        log_path,
                        f"TRAINING FINISHED BEFORE SOME TARGET CHECKPOINTS APPEARED: missing={','.join(missing)}",
                    )
                else:
                    log_message(log_path, "TRAINING FINISHED, waiting only for in-flight eval state to settle")
        else:
            training_done_logged = False

        time.sleep(max(1.0, float(args.poll_seconds)))

    state.updated_at = now_str()
    save_state(state_path, state)
    log_message(log_path, "GUARD EXIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
