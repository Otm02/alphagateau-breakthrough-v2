#!/bin/bash
#SBATCH --partition=all
#SBATCH --account=winter2026-comp579
#SBATCH --qos=comp579-1gpu-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?set REPO_ROOT to the cloned repository path on mimi}"
ENV_NAME="${ENV_NAME:-comp579-breakthrough}"
PRESET="${PRESET:?set PRESET to an experiment preset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/experiments}"
RUN_NAME="${RUN_NAME:-$PRESET}"
RESUME="${RESUME:-1}"

mkdir -p "$REPO_ROOT/logs/slurm"
cd "$REPO_ROOT"

module load anaconda
module load cuda/11.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cmd=(python scripts/train_experiment.py "$PRESET" --output-root "$OUTPUT_ROOT" --run-name "$RUN_NAME")

if [[ "$RESUME" == "1" ]]; then
  cmd+=(--resume)
fi
if [[ -n "${INITIAL_CHECKPOINT:-}" ]]; then
  cmd+=(--initial-checkpoint "$INITIAL_CHECKPOINT")
fi
if [[ -n "${NUM_ITERATIONS:-}" ]]; then
  cmd+=(--num-iterations "$NUM_ITERATIONS")
fi
if [[ -n "${SELFPLAY_GAMES:-}" ]]; then
  cmd+=(--selfplay-games "$SELFPLAY_GAMES")
fi
if [[ -n "${NUM_SIMULATIONS:-}" ]]; then
  cmd+=(--num-simulations "$NUM_SIMULATIONS")
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
