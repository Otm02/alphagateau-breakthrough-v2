#!/bin/bash
#SBATCH --partition=all
# Set SLURM_ACCOUNT and SLURM_QOS in your environment or pass --account / --qos to sbatch.
# Example for McGill COMP-579: --account=winter2026-comp579 --qos=comp579-1gpu-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# extended_transfer_stage_p1.sbatch
set -euo pipefail

REPO_ROOT="${REPO_ROOT:?set REPO_ROOT to the cloned repository path on mimi}"
ENV_NAME="${ENV_NAME:-comp579-breakthrough}"
ENV_PREFIX="${ENV_PREFIX:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/experiments}"
LOGS_DIR="${LOGS_DIR:-$REPO_ROOT/logs/slurm}"
RESUME="${RESUME:-1}"

mkdir -p "$LOGS_DIR"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/bootstrap_mimi_shell.sh"
bootstrap_mimi_shell
activate_conda_target

cmd=(python scripts/run_extended_transfer_pipeline_p1.py --output-root "$OUTPUT_ROOT")

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
if [[ -n "${MAX_PLIES_6X6:-}" ]]; then
  cmd+=(--max-plies-6x6 "$MAX_PLIES_6X6")
fi
if [[ -n "${EVAL_GAMES:-}" ]]; then
  cmd+=(--eval-games "$EVAL_GAMES")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
