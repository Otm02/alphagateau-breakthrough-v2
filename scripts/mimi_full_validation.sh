#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_SHARED_ROOT="${PROJECT_SHARED_ROOT:-/mnt/teaching/slurm/$USER}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_SHARED_ROOT/artifacts/full_wrapper_validation}"
export FIGURES_DIR="${FIGURES_DIR:-$PROJECT_SHARED_ROOT/report/full_wrapper_validation_figures}"
export NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
export SELFPLAY_GAMES="${SELFPLAY_GAMES:-2}"
export NUM_SIMULATIONS="${NUM_SIMULATIONS:-2}"
export MAX_PLIES_5X5="${MAX_PLIES_5X5:-8}"
export MAX_PLIES_8X8="${MAX_PLIES_8X8:-8}"
export EVAL_GAMES="${EVAL_GAMES:-2}"
export HEAD_TO_HEAD_GAMES="${HEAD_TO_HEAD_GAMES:-2}"
export HEAD_TO_HEAD_SIMULATIONS="${HEAD_TO_HEAD_SIMULATIONS:-2}"
export HEAD_TO_HEAD_MAX_PLIES="${HEAD_TO_HEAD_MAX_PLIES:-8}"
export MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
export ALLOW_TINY_FULL_RUN="${ALLOW_TINY_FULL_RUN:-1}"

exec bash "$REPO_ROOT/scripts/mimi_project.sh" full "$@"
