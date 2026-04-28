#!/bin/bash
#SBATCH --partition=all
# Set SLURM_ACCOUNT and SLURM_QOS in your environment or pass --account / --qos to sbatch.
# Example for McGill COMP-579: --account=winter2026-comp579 --qos=comp579-1gpu-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?set REPO_ROOT to the cloned repository path on mimi}"
ENV_NAME="${ENV_NAME:-comp579-breakthrough}"
ENV_PREFIX="${ENV_PREFIX:-}"
GNN_SCRATCH_DIR="${GNN_SCRATCH_DIR:?set GNN_SCRATCH_DIR}"
CNN_SCRATCH_DIR="${CNN_SCRATCH_DIR:?set CNN_SCRATCH_DIR}"
PRETRAIN_DIR="${PRETRAIN_DIR:?set PRETRAIN_DIR}"
FINETUNE_DIR="${FINETUNE_DIR:?set FINETUNE_DIR}"
HEAD_TO_HEAD_OUTPUT="${HEAD_TO_HEAD_OUTPUT:-artifacts/experiments/head_to_head_gnn_vs_cnn.json}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-artifacts/experiments/postprocess_summary.json}"
OUTPUT_DIR="${OUTPUT_DIR:-report/figures}"
LOGS_DIR="${LOGS_DIR:-$REPO_ROOT/logs/slurm}"

mkdir -p "$LOGS_DIR"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/bootstrap_mimi_shell.sh"
bootstrap_mimi_shell
activate_conda_target

cmd=(
  python scripts/postprocess_experiments.py
  --gnn-scratch-dir "$GNN_SCRATCH_DIR"
  --cnn-scratch-dir "$CNN_SCRATCH_DIR"
  --pretrain-dir "$PRETRAIN_DIR"
  --finetune-dir "$FINETUNE_DIR"
  --head-to-head-output "$HEAD_TO_HEAD_OUTPUT"
  --summary-output "$SUMMARY_OUTPUT"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "${TRANSFER_SUMMARY:-}" ]]; then
  cmd+=(--transfer-summary "$TRANSFER_SUMMARY")
fi
if [[ -n "${TRANSFER_ZERO_SHOT:-}" ]]; then
  cmd+=(--transfer-zero-shot "$TRANSFER_ZERO_SHOT")
fi
if [[ -n "${N_GAMES:-}" ]]; then
  cmd+=(--n-games "$N_GAMES")
fi
if [[ -n "${N_SIM:-}" ]]; then
  cmd+=(--n-sim "$N_SIM")
fi
if [[ -n "${MAX_PLIES:-}" ]]; then
  cmd+=(--max-plies "$MAX_PLIES")
fi
if [[ -n "${SEED:-}" ]]; then
  cmd+=(--seed "$SEED")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
