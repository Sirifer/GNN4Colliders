#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


REPO_ROOT = Path(__file__).resolve().parents[2]
GPUS_PER_NODE = 4


@dataclass(frozen=True)
class InferenceTask:
    target: Path
    destination: Path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()

    repo_path = (REPO_ROOT / path).resolve()
    if repo_path.exists() or str(raw_path).startswith(("configs/", "jobs/", "scripts/")):
        return repo_path

    return (Path.cwd() / path).resolve()


def sanitize_branch_name(name: str) -> str:
    cleaned = re.sub(r"\W+", "_", name).strip("_")
    if not cleaned:
        cleaned = "gnn"
    if cleaned[0].isdigit():
        cleaned = f"gnn_{cleaned}"
    return f"{cleaned}_score"


def discover_model_configs(raw_configs: list[str], raw_config_dirs: list[str]) -> list[Path]:
    configs = []
    for raw_config in raw_configs:
        config = resolve_path(raw_config)
        if not config.is_file():
            raise FileNotFoundError(f"Config file does not exist: {config}")
        configs.append(config)

    for raw_dir in raw_config_dirs:
        config_dir = resolve_path(raw_dir)
        if not config_dir.is_dir():
            raise NotADirectoryError(f"Config directory does not exist: {config_dir}")
        configs.extend(sorted(path.resolve() for path in config_dir.glob("*.yaml")))

    unique_configs = []
    seen = set()
    for config in configs:
        if config not in seen:
            unique_configs.append(config)
            seen.add(config)
    if not unique_configs:
        raise ValueError("No model configs were provided.")
    return unique_configs


def default_branch_names(config_paths: list[Path]) -> list[str]:
    branch_names = []
    for config_path in config_paths:
        config = load_config(config_path)
        training_name = config.get("Training_Name") or config_path.stem
        branch_names.append(sanitize_branch_name(str(training_name)))
    return branch_names


def discover_targets_from_config(sample_config_path: Path) -> list[Path]:
    config = load_config(sample_config_path)
    targets = []

    for dataset_name, dataset_config in config.get("Datasets", {}).items():
        args = dataset_config.get("args", {})
        raw_dir = args.get("raw_dir")
        file_names = args.get("file_names")
        if not raw_dir or not file_names:
            raise ValueError(f"Dataset {dataset_name} is missing raw_dir or file_names.")

        patterns = file_names if isinstance(file_names, list) else [file_names]
        for pattern in patterns:
            matches = sorted(Path(path).resolve() for path in glob.glob(os.path.join(raw_dir, pattern)))
            if not matches:
                raise FileNotFoundError(f"No files matched {os.path.join(raw_dir, pattern)}")
            targets.extend(matches)

    unique_targets = []
    seen = set()
    for target in targets:
        if target not in seen:
            unique_targets.append(target)
            seen.add(target)
    return unique_targets


def discover_targets(raw_targets: list[str], sample_config: str | None) -> list[Path]:
    targets = []
    if sample_config:
        targets.extend(discover_targets_from_config(resolve_path(sample_config)))

    for raw_target in raw_targets:
        matches = sorted(Path(path).resolve() for path in glob.glob(str(resolve_path(raw_target))))
        if matches:
            targets.extend(matches)
        else:
            target = resolve_path(raw_target)
            if not target.exists():
                raise FileNotFoundError(f"Target does not exist: {target}")
            targets.append(target)

    unique_targets = []
    seen = set()
    for target in targets:
        if target not in seen:
            unique_targets.append(target)
            seen.add(target)
    if not unique_targets:
        raise ValueError("No inference targets were provided.")
    return unique_targets


def build_tasks(targets: list[Path], output_dir: Path) -> list[InferenceTask]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [InferenceTask(target=target, destination=(output_dir / target.name).resolve()) for target in targets]


def build_command(
    task: InferenceTask,
    config_paths: list[Path],
    branch_names: list[str],
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/inference.py",
        "--target",
        str(task.target),
        "--destination",
        str(task.destination),
        "--config",
        *[str(path) for path in config_paths],
        "--branch_name",
        *branch_names,
        "--chunks",
        str(args.chunks),
        "--chunkno",
        str(args.chunkno),
    ]

    if args.write:
        command.append("--write")
    if args.clobber:
        command.append("--clobber")
    if args.ckpt is not None:
        command.extend(["--ckpt", str(args.ckpt)])
    if args.var:
        command.extend(["--var", args.var])
    if args.mode:
        command.extend(["--mode", args.mode])
    if args.tree:
        command.extend(["--tree", args.tree])
    return command


def run_task(
    task: InferenceTask,
    config_paths: list[Path],
    branch_names: list[str],
    args: argparse.Namespace,
    rank: int,
) -> int:
    gpu_id = rank % GPUS_PER_NODE
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    command = build_command(task, config_paths, branch_names, args)

    print(
        f"[rank {rank}] target={task.target} destination={task.destination} gpu={gpu_id}",
        flush=True,
    )
    print(f"[rank {rank}] command={' '.join(command)}", flush=True)
    if args.test:
        return 0
    return subprocess.run(command, cwd=REPO_ROOT, env=env).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scripts/inference.py over many ROOT files with many model configs."
    )
    parser.add_argument("--sample-config", help="Config whose dataset file_names define inference targets.")
    parser.add_argument("--target", nargs="*", default=[], help="Target ROOT files or glob patterns.")
    parser.add_argument("--config", nargs="*", default=[], help="Model config files to score.")
    parser.add_argument("--config-dir", nargs="*", default=[], help="Directories containing model YAML configs.")
    parser.add_argument("--branch-name", nargs="*", default=[], help="Score branch names. Defaults to Training_Name_score.")
    parser.add_argument("--output-dir", required=True, help="Directory for output ROOT/NPZ files.")
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--chunkno", type=int, default=0)
    parser.add_argument("--write", action="store_true", help="Write ROOT files with score branches.")
    parser.add_argument("--clobber", action="store_true")
    parser.add_argument("--ckpt", type=int, default=None, help="Checkpoint epoch. Omit to let scripts/inference.py choose best epoch.")
    parser.add_argument("--var", default="Test_AUC")
    parser.add_argument("--mode", default="max")
    parser.add_argument("--tree", default="")
    parser.add_argument("--test", action="store_true", help="Print planned commands without running inference.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_paths = discover_model_configs(args.config, args.config_dir)
    branch_names = args.branch_name or default_branch_names(config_paths)
    if len(branch_names) != len(config_paths):
        raise ValueError("--branch-name count must match the number of model configs.")

    targets = discover_targets(args.target, args.sample_config)
    tasks = build_tasks(targets, resolve_path(args.output_dir))

    print(f"Repo root: {REPO_ROOT}", flush=True)
    print(f"Targets: {len(tasks)}", flush=True)
    print(f"Model configs: {len(config_paths)}", flush=True)
    print("Branches:", " ".join(branch_names), flush=True)

    if MPI is None:
        failures = 0
        for task in tasks:
            failures += int(run_task(task, config_paths, branch_names, args, rank=0) != 0)
        return 1 if failures else 0

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    failures = 0
    for index, task in enumerate(tasks):
        if index % size != rank:
            continue
        failures += int(run_task(task, config_paths, branch_names, args, rank=rank) != 0)

    total_failures = comm.allreduce(failures, op=MPI.SUM)
    if rank == 0:
        print(f"Completed {len(tasks) - total_failures} task(s); failed {total_failures}.", flush=True)
    return 1 if total_failures else 0


if __name__ == "__main__":
    sys.exit(main())
