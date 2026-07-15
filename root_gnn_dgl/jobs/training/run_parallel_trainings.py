#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import socket

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGETS = (
    REPO_ROOT / "configs/higgs_production_updated_leptons",
    REPO_ROOT / "configs/triboson",
)
FIXED_GPU_TOKENS = ("0", "1", "2", "3")
DEFAULT_TRAINING_ARGS = ("--preshuffle", "--nocompile", "--lazy")
FORBIDDEN_TRAINING_ARGS = {"--config", "--multigpu", "--multinode"}


@dataclass(frozen=True)
class ExecutionSlot:
    gpu_token: str


@dataclass
class TrainingJob:
    config_path: Path
    training_directory: str
    pretraining_path: Path | None


@dataclass
class ActiveProcess:
    job: TrainingJob
    slot: ExecutionSlot
    log_path: Path
    log_handle: object
    process: subprocess.Popen


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Queue local single-GPU trainings on the current node only. This launcher "
            "is intentionally hard-coded to use GPUs 0,1,2,3 with no srun or multi-node logic."
        ),
        epilog=(
            "Unknown arguments are forwarded to scripts/training_script.py. "
            "Example: python jobs/training/run_parallel_trainings.py --split 1/2 --restart --seed 7"
        ),
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help=(
            "Config files or directories. Defaults to "
            "configs/higgs_production_updated_leptons and configs/triboson."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        help=(
            "Run only one deterministic slice of the config queue, formatted as K/N. "
            "Example: --split 1/2 for the first half and --split 2/2 for the second."
        ),
    )
    parser.add_argument(
        "--logs-root",
        type=str,
        default="jobs/slurm/parallel_training_logs",
        help="Directory where launcher stdout/stderr logs should be written.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop launching new trainings after the first failure.",
    )
    parser.add_argument(
        "--ignore-missing-pretraining",
        action="store_true",
        help="Launch even if a finetuning config points to a missing pretraining checkpoint.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run preflight checks and print the launch plan without starting trainings.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=10.0,
        help="Seconds between checks for finished trainings.",
    )

    # Accepted only so older commands do not fail; these are ignored in the
    # single-node hard-coded launcher.
    parser.add_argument("--gpus", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--nodelist", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--local-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--srun-binary", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--scontrol-binary", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--max-parallel", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)

    args, forwarded_args = parser.parse_known_args()
    return args, forwarded_args


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def resolve_target(raw_target: str) -> Path:
    raw_path = Path(raw_target).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()

    repo_candidate = (REPO_ROOT / raw_path).resolve()
    if repo_candidate.exists():
        return repo_candidate

    return (Path.cwd() / raw_path).resolve()


def discover_configs(raw_targets: list[str]) -> list[Path]:
    targets = [resolve_target(raw_target) for raw_target in raw_targets] if raw_targets else list(DEFAULT_TARGETS)
    configs: list[Path] = []

    for target in targets:
        if target.is_file():
            if target.suffix != ".yaml":
                raise FileNotFoundError(f"Expected a .yaml config file, got {target}")
            configs.append(target)
            continue

        if not target.is_dir():
            raise FileNotFoundError(f"Config target does not exist: {target}")

        configs.extend(sorted(path.resolve() for path in target.glob("*.yaml")))

    unique_configs = []
    seen = set()
    for config in configs:
        if config not in seen:
            unique_configs.append(config)
            seen.add(config)
    return unique_configs


def resolve_checkpoint_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def load_job(config_path: Path) -> TrainingJob:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config is not a YAML mapping: {config_path}")

    training_directory = config.get("Training_Directory")
    if not training_directory:
        raise ValueError(f"Config is missing Training_Directory: {config_path}")

    pretraining_path = None
    model_args = config.get("Model", {}).get("args", {})
    if isinstance(model_args, dict) and model_args.get("pretraining_path"):
        pretraining_path = resolve_checkpoint_path(model_args["pretraining_path"])

    return TrainingJob(
        config_path=config_path.resolve(),
        training_directory=str(training_directory),
        pretraining_path=pretraining_path,
    )


def parse_split_spec(raw_split: str | None) -> tuple[int, int] | None:
    if raw_split is None:
        return None

    match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", raw_split)
    if not match:
        raise ValueError(
            f"Invalid --split value {raw_split!r}. Use the format K/N, for example 1/2."
        )

    split_index = int(match.group(1))
    split_count = int(match.group(2))

    if split_count <= 0:
        raise ValueError("--split requires N > 0.")
    if split_index <= 0:
        raise ValueError("--split requires K >= 1.")
    if split_index > split_count:
        raise ValueError("--split requires K <= N.")

    return split_index, split_count


def select_jobs_for_split(
    jobs: list[TrainingJob],
    split_spec: tuple[int, int] | None,
) -> tuple[list[TrainingJob], str | None]:
    if split_spec is None:
        return jobs, None

    split_index, split_count = split_spec
    selected_jobs = [
        job for index, job in enumerate(jobs)
        if index % split_count == (split_index - 1)
    ]
    return selected_jobs, f"{split_index}/{split_count}"


def validate_forwarded_args(forwarded_args: list[str]) -> None:
    normalized_args = [arg.split("=", 1)[0] for arg in forwarded_args]
    forbidden = sorted(arg for arg in normalized_args if arg in FORBIDDEN_TRAINING_ARGS)
    if forbidden:
        joined = ", ".join(forbidden)
        raise ValueError(
            f"These training arguments are controlled by the launcher and cannot be forwarded: {joined}"
        )


def warn_about_ignored_args(args: argparse.Namespace) -> None:
    ignored = []

    if args.gpus:
        ignored.append("--gpus")
    if args.nodelist:
        ignored.append("--nodelist")
    if args.local_only:
        ignored.append("--local-only")
    if args.srun_binary:
        ignored.append("--srun-binary")
    if args.scontrol_binary:
        ignored.append("--scontrol-binary")
    if args.max_parallel is not None:
        ignored.append("--max-parallel")
    if args.dry_run:
        ignored.append("--dry-run")

    if ignored:
        print(
            "Ignoring launcher options not used by the single-node version: "
            + ", ".join(ignored),
            flush=True,
        )


def format_slot(slot: ExecutionSlot, hostname: str) -> str:
    return f"{hostname}:gpu{slot.gpu_token}"


def sanitize_log_name(config_path: Path) -> str:
    try:
        relative = config_path.resolve().relative_to(REPO_ROOT)
        name = "__".join(relative.parts)
    except ValueError:
        name = config_path.name
    return f"{name}.log"


def terminate_active_processes(active_processes: list[ActiveProcess]) -> None:
    for active in active_processes:
        if active.process.poll() is None:
            active.process.terminate()

    deadline = time.time() + 10
    while time.time() < deadline:
        if all(active.process.poll() is not None for active in active_processes):
            break
        time.sleep(0.5)

    for active in active_processes:
        if active.process.poll() is None:
            active.process.kill()

    for active in active_processes:
        if not active.log_handle.closed:
            active.log_handle.close()


def build_training_command(job: TrainingJob, forwarded_args: list[str]) -> list[str]:
    return [
        sys.executable,
        "scripts/training_script.py",
        "--config",
        str(job.config_path),
        *DEFAULT_TRAINING_ARGS,
        *forwarded_args,
    ]


def launch_job(
    job: TrainingJob,
    slot: ExecutionSlot,
    forwarded_args: list[str],
    log_dir: Path,
    hostname: str,
) -> ActiveProcess:
    log_path = log_dir / sanitize_log_name(job.config_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")

    command = build_training_command(job, forwarded_args)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = slot.gpu_token

    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )

    print(
        f"[launch] slot={format_slot(slot, hostname)} config={repo_relative(job.config_path)} "
        f"log={repo_relative(log_path)}",
        flush=True,
    )

    return ActiveProcess(
        job=job,
        slot=slot,
        log_path=log_path,
        log_handle=log_handle,
        process=process,
    )


def main() -> int:
    args, forwarded_args = parse_args()
    validate_forwarded_args(forwarded_args)
    warn_about_ignored_args(args)
    split_spec = parse_split_spec(args.split)

    config_paths = discover_configs(args.targets)
    if not config_paths:
        raise RuntimeError("No YAML configs were found for the requested targets.")

    all_jobs = [load_job(config_path) for config_path in config_paths]
    jobs, split_label = select_jobs_for_split(all_jobs, split_spec)
    if not jobs:
        raise RuntimeError("No configs were selected for this split.")

    duplicated_training_dirs: dict[str, Path] = {}
    duplicate_messages = []
    missing_pretraining = []
    for job in jobs:
        if job.training_directory in duplicated_training_dirs:
            duplicate_messages.append(
                f"{repo_relative(job.config_path)} conflicts with "
                f"{repo_relative(duplicated_training_dirs[job.training_directory])} "
                f"on Training_Directory={job.training_directory}"
            )
        else:
            duplicated_training_dirs[job.training_directory] = job.config_path

        if job.pretraining_path is not None and not job.pretraining_path.exists():
            missing_pretraining.append(
                f"{repo_relative(job.config_path)} -> missing {job.pretraining_path}"
            )

    if duplicate_messages:
        raise RuntimeError(
            "Refusing to launch configs with overlapping Training_Directory values:\n"
            + "\n".join(duplicate_messages)
        )

    if missing_pretraining and not args.ignore_missing_pretraining:
        raise RuntimeError(
            "Missing pretraining checkpoints were found:\n"
            + "\n".join(missing_pretraining)
            + "\nRe-run with --ignore-missing-pretraining to launch anyway."
        )

    if missing_pretraining:
        print("Pretraining checkpoint warnings:", flush=True)
        for message in missing_pretraining:
            print(f"  {message}", flush=True)

    hostname = socket.gethostname().split(".", 1)[0]
    execution_slots = [ExecutionSlot(gpu_token=gpu_token) for gpu_token in FIXED_GPU_TOKENS]
    max_parallel = len(execution_slots)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = (REPO_ROOT / args.logs_root / timestamp).resolve()

    print(f"Repo root: {REPO_ROOT}", flush=True)
    print(f"Configs discovered: {len(all_jobs)}", flush=True)
    if split_label:
        print(f"Configs selected for split {split_label}: {len(jobs)}", flush=True)
    else:
        print(f"Configs selected: {len(jobs)}", flush=True)
    print(f"Node selected: {hostname}", flush=True)
    print(f"GPUs selected: {', '.join(FIXED_GPU_TOKENS)}", flush=True)
    print(f"Max parallel trainings: {max_parallel}", flush=True)
    print(f"Training args: {' '.join(DEFAULT_TRAINING_ARGS + tuple(forwarded_args))}", flush=True)
    print(f"Launcher logs: {repo_relative(log_dir)}", flush=True)
    print("Launch mode: local queued trainings only", flush=True)

    if args.test or args.dry_run:
        print("Running launcher self-test; no trainings will be started.", flush=True)
        for index, job in enumerate(jobs, start=1):
            slot = execution_slots[(index - 1) % len(execution_slots)]
            command = build_training_command(job, forwarded_args)
            print(
                f"[dry-run {index:02d}] slot={format_slot(slot, hostname)} "
                f"dir={job.training_directory} cmd={shlex.join(command)}",
                flush=True,
            )
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)

    pending_jobs = deque(jobs)
    free_slots = deque(execution_slots)
    active_processes: list[ActiveProcess] = []
    failed_jobs: list[tuple[TrainingJob, int, Path]] = []
    completed_jobs = 0

    try:
        while pending_jobs or active_processes:
            while pending_jobs and free_slots:
                job = pending_jobs.popleft()
                slot = free_slots.popleft()
                active_processes.append(launch_job(job, slot, forwarded_args, log_dir, hostname))

            if not active_processes:
                break

            time.sleep(args.poll_seconds)

            still_active: list[ActiveProcess] = []
            for active in active_processes:
                return_code = active.process.poll()
                if return_code is None:
                    still_active.append(active)
                    continue

                active.log_handle.close()
                free_slots.append(active.slot)

                if return_code == 0:
                    completed_jobs += 1
                    print(
                        f"[done] slot={format_slot(active.slot, hostname)} "
                        f"config={repo_relative(active.job.config_path)} "
                        f"completed={completed_jobs}/{len(jobs)}",
                        flush=True,
                    )
                else:
                    failed_jobs.append((active.job, return_code, active.log_path))
                    print(
                        f"[fail] slot={format_slot(active.slot, hostname)} "
                        f"config={repo_relative(active.job.config_path)} "
                        f"exit={return_code} log={repo_relative(active.log_path)}",
                        flush=True,
                    )
                    if args.stop_on_failure:
                        pending_jobs.clear()

            active_processes = still_active

            if failed_jobs and args.stop_on_failure:
                terminate_active_processes(active_processes)
                active_processes = []
                break

    except KeyboardInterrupt:
        print("\nInterrupted. Terminating active trainings...", flush=True)
        terminate_active_processes(active_processes)
        return 130

    print(f"Completed {completed_jobs} training(s); failed {len(failed_jobs)}.", flush=True)
    if failed_jobs:
        for job, return_code, log_path in failed_jobs:
            print(
                f"  {repo_relative(job.config_path)} exit={return_code} "
                f"log={repo_relative(log_path)}",
                flush=True,
            )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
