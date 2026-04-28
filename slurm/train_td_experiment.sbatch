#!/bin/bash
#SBATCH --partition=all
# Set SLURM_ACCOUNT and SLURM_QOS in your environment or pass --account / --qos to sbatch.
# Example for McGill COMP-579: --account=winter2026-comp579 --qos=comp579-1gpu-12h
#SBATCH --account=winter2026-comp579
#SBATCH --qos=comp579-1gpu-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --account=winter2026-comp579
#SBATCH --qos=comp579-1gpu-12h
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?set REPO_ROOT to the cloned repository path on mimi}"
ENV_NAME="${ENV_NAME:-comp579-breakthrough}"
PROJECT_SHARED_ROOT="${PROJECT_SHARED_ROOT:-/mnt/teaching/slurm/$USER}"
ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/comp579-breakthrough}"
PRESET="${PRESET:?set PRESET to an experiment preset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/experiments}"
LOGS_DIR="${LOGS_DIR:-$REPO_ROOT/logs/slurm}"
RUN_NAME="${RUN_NAME:-$PRESET}"
RESUME="${RESUME:-1}"

mkdir -p "$LOGS_DIR"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/bootstrap_mimi_shell.sh"
bootstrap_mimi_shell
activate_conda_target

cmd=(python scripts/train_td_experiment.py "$PRESET" --output-root "$OUTPUT_ROOT" --run-name "$RUN_NAME")

if [[ "$RESUME" == "1" ]]; then
  cmd+=(--resume)
fi
if [[ -n "${INITIAL_CHECKPOINT:-}" ]]; then
  cmd+=(--initial-checkpoint "$INITIAL_CHECKPOINT")
fi
if [[ -n "${NUM_ITERATIONS:-}" ]]; then
  cmd+=(--num-iterations "$NUM_ITERATIONS")
fi
if [[ -n "${EPISODES_PER_ITERATION:-}" ]]; then
  cmd+=(--episodes-per-iteration "$EPISODES_PER_ITERATION")
fi
if [[ -n "${DISCOUNT_FACTOR:-}" ]]; then
  cmd+=(--discount-factor "$DISCOUNT_FACTOR")
fi
if [[ -n "${MAX_PLIES:-}" ]]; then
  cmd+=(--max-plies "$MAX_PLIES")
fi
if [[ -n "${EVAL_GAMES:-}" ]]; then
  cmd+=(--eval-games "$EVAL_GAMES")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
