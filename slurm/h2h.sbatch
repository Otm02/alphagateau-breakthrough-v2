#!/bin/bash
#SBATCH --partition=all
# Set SLURM_ACCOUNT and SLURM_QOS in your environment or pass --account / --qos to sbatch.
# Example for McGill COMP-579: --account=winter2026-comp579 --qos=comp579-1gpu-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --account=winter2026-comp579
#SBATCH --qos=comp579-1gpu-12h
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

  export REPO=/home/2025/asaffa4/alphagateau-breakthrough-v2
  export OUT=/home/2025/asaffa4/alphagateau-breakthrough-v2/artifacts/experiments
  export LOGS=/home/2025/asaffa4/alphagateau-breakthrough-v2/logs/slurm
  export ENV_PREFIX=/home/2025/asaffa4/envs/comp579-breakthrough/
  mkdir -p "$OUT" "$LOGS"
  cd "$REPO"


# shellcheck disable=SC1091
source "$REPO/scripts/bootstrap_mimi_shell.sh"
bootstrap_mimi_shell
activate_conda_target
 sbatch \
    --account winter2026-comp579 \
    --qos comp579-1gpu-12h \
    --gres=gpu:1 \
    --mem=32G \
    --time=02:00:00 \
    --output="$LOGS/h2h_5x5_to_8x8_vs_cnn_%j.out" \
    --error="$LOGS/h2h_5x5_to_8x8_vs_cnn_%j.err" \
    --wrap="python scripts/evaluate_models.py '$OUT/gnn_8x8_finetune_cosine/checkpoints/final.pkl' '$OUT/
cnn_8x8_scratch/checkpoints/final.pkl' --output '$OUT/head_to_head_gnn_5x5_to_8x8_vs_cnn.json'"