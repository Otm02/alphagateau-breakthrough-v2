#!/bin/bash
#SBATCH --partition=all
# Set SLURM_ACCOUNT and SLURM_QOS in your environment or pass --account / --qos to sbatch.
# Example for McGill COMP-579: --account=winter2026-comp579 --qos=comp579-1gpu-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# extended_transfer_pipeline.sbatch
# Runs Part 1 (5x5 pretrain -> 6x6 finetune) and then submits Part 2 (8x8) as a dependent job.
# Usage:
#   REPO_ROOT=/path/to/repo sbatch scripts/extended_transfer_pipeline.sbatch

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?set REPO_ROOT to the cloned repository path on mimi}"
ENV_NAME="${ENV_NAME:-comp579-breakthrough}"
ENV_PREFIX="${ENV_PREFIX:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/experiments}"
LOGS_DIR="${LOGS_DIR:-$REPO_ROOT/logs/slurm}"
RESUME="${RESUME:-1}"
PART1_SUMMARY="${OUTPUT_ROOT}/gnn_transfer_summary_part1.json"

mkdir -p "$LOGS_DIR"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/bootstrap_mimi_shell.sh"
bootstrap_mimi_shell
activate_conda_target

# --- Part 1 ---

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

printf 'Running Part 1:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

# --- Read 6x6 checkpoint from Part 1 summary ---

FINETUNED_6X6_CHECKPOINT=$(python -c "
import json, sys
try:
    d = json.load(open('${PART1_SUMMARY}'))
    print(d['finetune_6x6']['final_checkpoint'])
except Exception as e:
    print(f'ERROR reading ${PART1_SUMMARY}: {e}', file=sys.stderr)
    sys.exit(1)
")

echo "Part 1 complete. 6x6 checkpoint: $FINETUNED_6X6_CHECKPOINT"

# --- Submit Part 2 as a new job (no GPU time wasted waiting) ---

p2_cmd=(
  sbatch
  --partition=all
  --gres=gpu:1
  --mem=16G
  --time=12:00:00
  --output="${LOGS_DIR}/%x_%j.out"
  --error="${LOGS_DIR}/%x_%j.err"
)
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  p2_cmd+=(--account="$SLURM_ACCOUNT")
fi
if [[ -n "${SLURM_QOS:-}" ]]; then
  p2_cmd+=(--qos="$SLURM_QOS")
fi

# Pass through all relevant env vars to the p2 job
p2_env="REPO_ROOT=${REPO_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},RESUME=${RESUME},FINETUNED_6X6_CHECKPOINT=${FINETUNED_6X6_CHECKPOINT}"
if [[ -n "${ITERATIONS_FINETUNE:-}" ]]; then
  p2_env+=",ITERATIONS_FINETUNE=${ITERATIONS_FINETUNE}"
fi
if [[ -n "${SELFPLAY_GAMES:-}" ]]; then
  p2_env+=",SELFPLAY_GAMES=${SELFPLAY_GAMES}"
fi
if [[ -n "${NUM_SIMULATIONS:-}" ]]; then
  p2_env+=",NUM_SIMULATIONS=${NUM_SIMULATIONS}"
fi
if [[ -n "${MAX_PLIES_8X8:-}" ]]; then
  p2_env+=",MAX_PLIES_8X8=${MAX_PLIES_8X8}"
fi
if [[ -n "${EVAL_GAMES:-}" ]]; then
  p2_env+=",EVAL_GAMES=${EVAL_GAMES}"
fi

p2_cmd+=(--export="$p2_env")
p2_cmd+=("$REPO_ROOT/scripts/extended_transfer_stage_p2.sbatch")

printf 'Submitting Part 2:'
printf ' %q' "${p2_cmd[@]}"
printf '\n'
"${p2_cmd[@]}"
