#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class JobRecord:
    lane: str
    attempt: int
    job_name: str
    script: str
    dependency: str | None
    env: dict[str, str]
    command: list[str]
    job_id: str


def repo_root_default() -> Path:
    return Path(__file__).resolve().parents[1]


def manifest_path_default(logs_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return logs_dir / "submissions" / f"pipeline_{timestamp}.json"


def export_arg(env: dict[str, str]) -> str:
    items = ["ALL"] + [f"{key}={value}" for key, value in sorted(env.items())]
    return ",".join(items)


def build_sbatch_command(
    *,
    script: Path,
    repo_root: Path,
    job_name: str,
    dependency: str | None,
    partition: str,
    account: str,
    qos: str,
    gres: str,
    mem: str,
    time_limit: str,
    env: dict[str, str],
) -> list[str]:
    command = [
        "sbatch",
        "--chdir",
        str(repo_root),
        "--job-name",
        job_name,
        "--partition",
        partition,
        "--account",
        account,
        "--qos",
        qos,
        "--gres",
        gres,
        "--mem",
        mem,
        "--time",
        time_limit,
        "--export",
        export_arg(env),
    ]
    if dependency:
        command.extend(["--dependency", dependency])
    command.append(str(script))
    return command


def parse_job_id(output: str) -> str:
    parts = output.strip().split()
    if not parts:
        raise ValueError(f"Unrecognised sbatch output: {output!r}")
    return parts[-1]


def submit_job(command: list[str], *, dry_run: bool, sequence_number: int) -> str:
    if dry_run:
        return f"dryrun-{sequence_number:02d}"
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return parse_job_id(completed.stdout)


def add_if_set(env: dict[str, str], key: str, value) -> None:
    if value is not None:
        env[key] = str(value)


def train_lane_env(
    *,
    repo_root: Path,
    env_name: str,
    output_root: str,
    preset: str,
    run_name: str,
    initial_checkpoint: str | None = None,
    num_iterations: int | None = None,
    selfplay_games: int | None = None,
    num_simulations: int | None = None,
    max_plies: int | None = None,
    eval_games: int | None = None,
) -> dict[str, str]:
    env = {
        "REPO_ROOT": str(repo_root),
        "ENV_NAME": env_name,
        "OUTPUT_ROOT": output_root,
        "PRESET": preset,
        "RUN_NAME": run_name,
        "RESUME": "1",
    }
    add_if_set(env, "INITIAL_CHECKPOINT", initial_checkpoint)
    add_if_set(env, "NUM_ITERATIONS", num_iterations)
    add_if_set(env, "SELFPLAY_GAMES", selfplay_games)
    add_if_set(env, "NUM_SIMULATIONS", num_simulations)
    add_if_set(env, "MAX_PLIES", max_plies)
    add_if_set(env, "EVAL_GAMES", eval_games)
    return env


def transfer_lane_env(
    *,
    repo_root: Path,
    env_name: str,
    output_root: str,
    pretrained_checkpoint: str,
    iterations_finetune: int | None = None,
    selfplay_games: int | None = None,
    num_simulations: int | None = None,
    max_plies_5x5: int | None = None,
    max_plies_8x8: int | None = None,
    eval_games: int | None = None,
) -> dict[str, str]:
    env = {
        "REPO_ROOT": str(repo_root),
        "ENV_NAME": env_name,
        "OUTPUT_ROOT": output_root,
        "PRETRAINED_CHECKPOINT": pretrained_checkpoint,
        "RESUME": "1",
    }
    add_if_set(env, "ITERATIONS_FINETUNE", iterations_finetune)
    add_if_set(env, "SELFPLAY_GAMES", selfplay_games)
    add_if_set(env, "NUM_SIMULATIONS", num_simulations)
    add_if_set(env, "MAX_PLIES_5X5", max_plies_5x5)
    add_if_set(env, "MAX_PLIES_8X8", max_plies_8x8)
    add_if_set(env, "EVAL_GAMES", eval_games)
    return env


def postprocess_env(
    *,
    repo_root: Path,
    env_name: str,
    output_root: str,
    output_dir: str,
    n_games: int | None,
    n_sim: int | None,
    max_plies: int | None,
) -> dict[str, str]:
    env = {
        "REPO_ROOT": str(repo_root),
        "ENV_NAME": env_name,
        "GNN_SCRATCH_DIR": f"{output_root}/gnn_8x8_scratch",
        "CNN_SCRATCH_DIR": f"{output_root}/cnn_8x8_scratch",
        "PRETRAIN_DIR": f"{output_root}/gnn_5x5_pretrain",
        "FINETUNE_DIR": f"{output_root}/gnn_8x8_finetune",
        "TRANSFER_SUMMARY": f"{output_root}/gnn_transfer_summary.json",
        "TRANSFER_ZERO_SHOT": f"{output_root}/gnn_transfer_zero_shot.json",
        "HEAD_TO_HEAD_OUTPUT": f"{output_root}/head_to_head_gnn_vs_cnn.json",
        "SUMMARY_OUTPUT": f"{output_root}/postprocess_summary.json",
        "OUTPUT_DIR": output_dir,
    }
    add_if_set(env, "N_GAMES", n_games)
    add_if_set(env, "N_SIM", n_sim)
    add_if_set(env, "MAX_PLIES", max_plies)
    return env


def submit_attempt_chain(
    *,
    lane: str,
    script: Path,
    repo_root: Path,
    partition: str,
    account: str,
    qos: str,
    gres: str,
    mem: str,
    time_limit: str,
    env: dict[str, str],
    max_attempts: int,
    initial_dependency: str | None,
    dry_run: bool,
    sequence_start: int,
) -> tuple[list[JobRecord], str, int]:
    records: list[JobRecord] = []
    previous_job_id = ""
    sequence_number = sequence_start
    for attempt in range(1, max_attempts + 1):
        dependency = initial_dependency if attempt == 1 else f"afterany:{previous_job_id}"
        job_name = f"{lane}-a{attempt}"
        command = build_sbatch_command(
            script=script,
            repo_root=repo_root,
            job_name=job_name,
            dependency=dependency,
            partition=partition,
            account=account,
            qos=qos,
            gres=gres,
            mem=mem,
            time_limit=time_limit,
            env=env,
        )
        job_id = submit_job(command, dry_run=dry_run, sequence_number=sequence_number)
        records.append(
            JobRecord(
                lane=lane,
                attempt=attempt,
                job_name=job_name,
                script=str(script),
                dependency=dependency,
                env=env,
                command=command,
                job_id=job_id,
            )
        )
        previous_job_id = job_id
        sequence_number += 1
    return records, previous_job_id, sequence_number


def format_commands(records: list[JobRecord]) -> list[str]:
    return [" ".join(shlex.quote(part) for part in record.command) for record in records]


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit the full McGill Slurm experiment DAG from mimi.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--partition", default="all")
    parser.add_argument("--account", default="winter2026-comp579")
    parser.add_argument("--qos", default="comp579-1gpu-12h")
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--mem", default="16G")
    parser.add_argument("--time-train", default="12:00:00")
    parser.add_argument("--time-postprocess", default="02:00:00")
    parser.add_argument("--env-name", default="comp579-breakthrough")
    parser.add_argument("--jax-extra", default="jax[cuda12]")
    parser.add_argument("--repo-root", type=Path, default=repo_root_default())
    parser.add_argument("--output-root", default="artifacts/experiments")
    parser.add_argument("--figures-dir", default="report/figures")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/slurm"))
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--selfplay-games", type=int, default=None)
    parser.add_argument("--num-simulations", type=int, default=None)
    parser.add_argument("--max-plies-5x5", type=int, default=None)
    parser.add_argument("--max-plies-8x8", type=int, default=None)
    parser.add_argument("--eval-games", type=int, default=None)
    parser.add_argument("--head-to-head-games", type=int, default=None)
    parser.add_argument("--head-to-head-simulations", type=int, default=None)
    parser.add_argument("--head-to-head-max-plies", type=int, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    logs_dir = (repo_root / args.logs_dir).resolve() if not args.logs_dir.is_absolute() else args.logs_dir
    manifest_path = args.manifest_path or manifest_path_default(logs_dir)
    if not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    sequence_number = 1
    records: list[JobRecord] = []

    train_script = repo_root / "slurm" / "train_experiment.sbatch"
    transfer_script = repo_root / "slurm" / "transfer_stage.sbatch"
    postprocess_script = repo_root / "slurm" / "postprocess.sbatch"

    gnn_env = train_lane_env(
        repo_root=repo_root,
        env_name=args.env_name,
        output_root=args.output_root,
        preset="gnn_8x8_scratch",
        run_name="gnn_8x8_scratch",
        num_iterations=args.num_iterations,
        selfplay_games=args.selfplay_games,
        num_simulations=args.num_simulations,
        max_plies=args.max_plies_8x8,
        eval_games=args.eval_games,
    )
    lane_records, gnn_terminal, sequence_number = submit_attempt_chain(
        lane="gnn-8x8-scratch",
        script=train_script,
        repo_root=repo_root,
        partition=args.partition,
        account=args.account,
        qos=args.qos,
        gres=args.gres,
        mem=args.mem,
        time_limit=args.time_train,
        env=gnn_env,
        max_attempts=args.max_attempts,
        initial_dependency=None,
        dry_run=args.dry_run,
        sequence_start=sequence_number,
    )
    records.extend(lane_records)

    cnn_env = train_lane_env(
        repo_root=repo_root,
        env_name=args.env_name,
        output_root=args.output_root,
        preset="cnn_8x8_scratch",
        run_name="cnn_8x8_scratch",
        num_iterations=args.num_iterations,
        selfplay_games=args.selfplay_games,
        num_simulations=args.num_simulations,
        max_plies=args.max_plies_8x8,
        eval_games=args.eval_games,
    )
    lane_records, cnn_terminal, sequence_number = submit_attempt_chain(
        lane="cnn-8x8-scratch",
        script=train_script,
        repo_root=repo_root,
        partition=args.partition,
        account=args.account,
        qos=args.qos,
        gres=args.gres,
        mem=args.mem,
        time_limit=args.time_train,
        env=cnn_env,
        max_attempts=args.max_attempts,
        initial_dependency=None,
        dry_run=args.dry_run,
        sequence_start=sequence_number,
    )
    records.extend(lane_records)

    pretrain_env = train_lane_env(
        repo_root=repo_root,
        env_name=args.env_name,
        output_root=args.output_root,
        preset="gnn_5x5_pretrain",
        run_name="gnn_5x5_pretrain",
        num_iterations=args.num_iterations,
        selfplay_games=args.selfplay_games,
        num_simulations=args.num_simulations,
        max_plies=args.max_plies_5x5,
        eval_games=args.eval_games,
    )
    lane_records, pretrain_terminal, sequence_number = submit_attempt_chain(
        lane="gnn-5x5-pretrain",
        script=train_script,
        repo_root=repo_root,
        partition=args.partition,
        account=args.account,
        qos=args.qos,
        gres=args.gres,
        mem=args.mem,
        time_limit=args.time_train,
        env=pretrain_env,
        max_attempts=args.max_attempts,
        initial_dependency=None,
        dry_run=args.dry_run,
        sequence_start=sequence_number,
    )
    records.extend(lane_records)

    transfer_env = transfer_lane_env(
        repo_root=repo_root,
        env_name=args.env_name,
        output_root=args.output_root,
        pretrained_checkpoint=f"{args.output_root}/gnn_5x5_pretrain/checkpoints/final.pkl",
        iterations_finetune=args.num_iterations,
        selfplay_games=args.selfplay_games,
        num_simulations=args.num_simulations,
        max_plies_5x5=args.max_plies_5x5,
        max_plies_8x8=args.max_plies_8x8,
        eval_games=args.eval_games,
    )
    lane_records, transfer_terminal, sequence_number = submit_attempt_chain(
        lane="gnn-transfer",
        script=transfer_script,
        repo_root=repo_root,
        partition=args.partition,
        account=args.account,
        qos=args.qos,
        gres=args.gres,
        mem=args.mem,
        time_limit=args.time_train,
        env=transfer_env,
        max_attempts=args.max_attempts,
        initial_dependency=f"afterok:{pretrain_terminal}",
        dry_run=args.dry_run,
        sequence_start=sequence_number,
    )
    records.extend(lane_records)

    post_env = postprocess_env(
        repo_root=repo_root,
        env_name=args.env_name,
        output_root=args.output_root,
        output_dir=args.figures_dir,
        n_games=args.head_to_head_games,
        n_sim=args.head_to_head_simulations,
        max_plies=args.head_to_head_max_plies,
    )
    post_dependency = f"afterok:{gnn_terminal}:{cnn_terminal}:{transfer_terminal}"
    lane_records, postprocess_terminal, sequence_number = submit_attempt_chain(
        lane="postprocess",
        script=postprocess_script,
        repo_root=repo_root,
        partition=args.partition,
        account=args.account,
        qos=args.qos,
        gres=args.gres,
        mem=args.mem,
        time_limit=args.time_postprocess,
        env=post_env,
        max_attempts=1,
        initial_dependency=post_dependency,
        dry_run=args.dry_run,
        sequence_start=sequence_number,
    )
    records.extend(lane_records)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": args.dry_run,
        "repo_root": str(repo_root),
        "logs_dir": str(logs_dir),
        "setup": {
            "script": "scripts/setup_mimi_env.sh",
            "env_name": args.env_name,
            "jax_extra": args.jax_extra,
        },
        "slurm": {
            "partition": args.partition,
            "account": args.account,
            "qos": args.qos,
            "gres": args.gres,
            "mem": args.mem,
            "time_train": args.time_train,
            "time_postprocess": args.time_postprocess,
            "max_attempts": args.max_attempts,
        },
        "jobs": [asdict(record) for record in records],
        "commands": format_commands(records),
        "lane_terminals": {
            "gnn_8x8_scratch": gnn_terminal,
            "cnn_8x8_scratch": cnn_terminal,
            "gnn_5x5_pretrain": pretrain_terminal,
            "gnn_transfer": transfer_terminal,
            "postprocess": postprocess_terminal,
        },
        "expected_outputs": [
            f"{args.output_root}/gnn_8x8_scratch/checkpoints/final.pkl",
            f"{args.output_root}/cnn_8x8_scratch/checkpoints/final.pkl",
            f"{args.output_root}/gnn_5x5_pretrain/checkpoints/final.pkl",
            f"{args.output_root}/gnn_8x8_finetune/checkpoints/final.pkl",
            f"{args.output_root}/gnn_transfer_summary.json",
            f"{args.output_root}/gnn_transfer_zero_shot.json",
            f"{args.output_root}/head_to_head_gnn_vs_cnn.json",
            f"{args.output_root}/postprocess_summary.json",
            f"{args.figures_dir}/gnn_8x8_scratch_curve.png",
            f"{args.figures_dir}/transfer_curve.png",
            f"{args.figures_dir}/encoding_visualisation.png",
        ],
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps({"manifest_path": str(manifest_path), "postprocess_job": postprocess_terminal}, indent=2))


if __name__ == "__main__":
    main()
