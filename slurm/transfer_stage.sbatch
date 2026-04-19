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
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/experiments}"
RESUME="${RESUME:-1}"

mkdir -p "$REPO_ROOT/logs/slurm"
cd "$REPO_ROOT"

module load anaconda
module load cuda/11.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cmd=(python scripts/run_transfer_pipeline.py --output-root "$OUTPUT_ROOT")

if [[ "$RESUME" == "1" ]]; then
  cmd+=(--resume)
fi
if [[ -n "${PRETRAINED_CHECKPOINT:-}" ]]; then
  cmd+=(--pretrained-checkpoint "$PRETRAINED_CHECKPOINT")
fi
if [[ -n "${ITERATIONS_PRETRAIN:-}" ]]; then
  cmd+=(--iterations-pretrain "$ITERATIONS_PRETRAIN")
fi
if [[ -n "${ITERATIONS_FINETUNE:-}" ]]; then
  cmd+=(--iterations-finetune "$ITERATIONS_FINETUNE")
fi
if [[ -n "${SELFPLAY_GAMES:-}" ]]; then
  cmd+=(--selfplay-games "$SELFPLAY_GAMES")
fi
if [[ -n "${NUM_SIMULATIONS:-}" ]]; then
  cmd+=(--num-simulations "$NUM_SIMULATIONS")
fi
if [[ -n "${MAX_PLIES_5X5:-}" ]]; then
  cmd+=(--max-plies-5x5 "$MAX_PLIES_5X5")
fi
if [[ -n "${MAX_PLIES_8X8:-}" ]]; then
  cmd+=(--max-plies-8x8 "$MAX_PLIES_8X8")
fi
if [[ -n "${EVAL_GAMES:-}" ]]; then
  cmd+=(--eval-games "$EVAL_GAMES")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
