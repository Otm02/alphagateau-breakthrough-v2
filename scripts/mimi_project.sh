#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJECT_SHARED_ROOT="${PROJECT_SHARED_ROOT:-/mnt/teaching/slurm/$USER}"
ENV_PREFIX="${ENV_PREFIX:-$PROJECT_SHARED_ROOT/envs/comp579-breakthrough}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$PROJECT_SHARED_ROOT/conda-pkgs}"
LOGS_DIR="${LOGS_DIR:-$PROJECT_SHARED_ROOT/logs/slurm}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_SHARED_ROOT/artifacts/experiments}"
FIGURES_DIR="${FIGURES_DIR:-$PROJECT_SHARED_ROOT/report/figures}"
SMOKE_OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT:-$PROJECT_SHARED_ROOT/artifacts/smoke_dag}"
SMOKE_FIGURES_DIR="${SMOKE_FIGURES_DIR:-$PROJECT_SHARED_ROOT/report/smoke_figures}"
DRY_RUN_MANIFEST="${DRY_RUN_MANIFEST:-$LOGS_DIR/submissions/pipeline_dry_run.json}"
SLURM_MODULE="${SLURM_MODULE:-slurm/slurm-24.05.4}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-winter2026-comp579}"
SLURM_QOS="${SLURM_QOS:-comp579-1gpu-12h}"
PARTITION="${PARTITION:-all}"
GRES="${GRES:-gpu:1}"
SETUP_TIME="${SETUP_TIME:-00:45:00}"
SETUP_MEM="${SETUP_MEM:-8G}"
CHECK_TIME="${CHECK_TIME:-00:05:00}"
CHECK_MEM="${CHECK_MEM:-2G}"
SMOKE_POLL_SECONDS="${SMOKE_POLL_SECONDS:-15}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
TRAIN_TIME_LIMIT="${TRAIN_TIME_LIMIT:-12:00:00}"
TRAIN_MEM_LIMIT="${TRAIN_MEM_LIMIT:-32G}"
POSTPROCESS_TIME_LIMIT="${POSTPROCESS_TIME_LIMIT:-02:00:00}"
POSTPROCESS_MEM_LIMIT="${POSTPROCESS_MEM_LIMIT:-16G}"
RESUME="${RESUME:-1}"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/bootstrap_mimi_shell.sh"

usage() {
  cat <<EOF
Usage: bash scripts/mimi_project.sh <command>

Commands:
  paths       Show the shared paths used by this project
  bootstrap   Create or update the shared Slurm-visible conda env
  runtime     Verify the shared env and CUDA runtime on a GPU node
  dry-run     Generate a dry-run DAG manifest on shared storage
  smoke       Submit the tiny smoke DAG with shared defaults
  full        Submit the full DAG with shared defaults
  collect     Mirror useful shared outputs back into this repo
  queue       Show your Slurm queue

Environment overrides:
  PROJECT_SHARED_ROOT  default: /mnt/teaching/slurm/\$USER
  ENV_PREFIX           default: \$PROJECT_SHARED_ROOT/envs/comp579-breakthrough
  CONDA_PKGS_DIRS      default: \$PROJECT_SHARED_ROOT/conda-pkgs
  LOGS_DIR             default: \$PROJECT_SHARED_ROOT/logs/slurm
  OUTPUT_ROOT          default: \$PROJECT_SHARED_ROOT/artifacts/experiments
  FIGURES_DIR          default: \$PROJECT_SHARED_ROOT/report/figures
  MAX_ATTEMPTS         default: 3
  TRAIN_TIME_LIMIT     default: 12:00:00
  TRAIN_MEM_LIMIT      default: 32G
  POSTPROCESS_TIME_LIMIT default: 02:00:00
  POSTPROCESS_MEM_LIMIT  default: 16G
EOF
}

load_slurm_support() {
  if [[ -n "${SBATCH_BIN:-}" && -x "${SBATCH_BIN:-}" ]]; then
    return 0
  fi

  ensure_module_command || {
    echo "ERROR: environment modules are unavailable in this shell." >&2
    exit 1
  }

  load_first_available_module slurm \
    "$SLURM_MODULE" \
    slurm/slurm-24.05.4 \
    slurm/slurm-24.05.3 \
    slurm >/dev/null || {
    echo "ERROR: unable to load a Slurm module." >&2
    exit 1
  }

  if [[ -z "${SBATCH_BIN:-}" ]]; then
    SBATCH_BIN="$(bash -lc 'command -v sbatch' 2>/dev/null || true)"
    if [[ -z "$SBATCH_BIN" && -x /usr/local/pkgs/slurm/24.05.4/root/bin/sbatch ]]; then
      SBATCH_BIN="/usr/local/pkgs/slurm/24.05.4/root/bin/sbatch"
    fi
    export SBATCH_BIN
  fi
}

ensure_shared_dirs() {
  load_slurm_support
  srun \
    --partition="$PARTITION" \
    --gres="$GRES" \
    --time="$CHECK_TIME" \
    --mem="$CHECK_MEM" \
    -N1 -n1 \
    env \
      PROJECT_SHARED_ROOT="$PROJECT_SHARED_ROOT" \
      CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS" \
      LOGS_DIR="$LOGS_DIR" \
      OUTPUT_ROOT="$OUTPUT_ROOT" \
      FIGURES_DIR="$FIGURES_DIR" \
      SMOKE_OUTPUT_ROOT="$SMOKE_OUTPUT_ROOT" \
      SMOKE_FIGURES_DIR="$SMOKE_FIGURES_DIR" \
      bash -lc '
        mkdir -p \
          "$PROJECT_SHARED_ROOT" \
          "$PROJECT_SHARED_ROOT/envs" \
          "$CONDA_PKGS_DIRS" \
          "$LOGS_DIR/submissions" \
          "$OUTPUT_ROOT" \
          "$FIGURES_DIR" \
          "$SMOKE_OUTPUT_ROOT" \
          "$SMOKE_FIGURES_DIR"
      '
}

run_on_gpu() {
  local time_limit="$1"
  local mem_limit="$2"
  local command="$3"

  load_slurm_support
  srun \
    --partition="$PARTITION" \
    --gres="$GRES" \
    --time="$time_limit" \
    --mem="$mem_limit" \
    -N1 -n1 \
    env \
      REPO_ROOT="$REPO_ROOT" \
      PROJECT_SHARED_ROOT="$PROJECT_SHARED_ROOT" \
      ENV_PREFIX="$ENV_PREFIX" \
      CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS" \
      LOGS_DIR="$LOGS_DIR" \
      OUTPUT_ROOT="$OUTPUT_ROOT" \
      FIGURES_DIR="$FIGURES_DIR" \
      SMOKE_OUTPUT_ROOT="$SMOKE_OUTPUT_ROOT" \
      SMOKE_FIGURES_DIR="$SMOKE_FIGURES_DIR" \
      DRY_RUN_MANIFEST="$DRY_RUN_MANIFEST" \
      SBATCH_BIN="${SBATCH_BIN:-}" \
      bash -lc "$command"
}

collect_outputs_to_repo() {
  run_on_gpu "$CHECK_TIME" "$CHECK_MEM" '
    set -euo pipefail

    sync_dir() {
      local src="$1"
      local dst="$2"

      [[ -d "$src" ]] || return 0
      if [[ "$src" == "$dst" ]]; then
        return 0
      fi

      mkdir -p "$dst"
      if command -v rsync >/dev/null 2>&1; then
        rsync -au "$src"/ "$dst"/
      else
        cp -au "$src"/. "$dst"/
      fi
      printf "Synced newer files %s -> %s\n" "$src" "$dst"
    }

    sync_dir "$OUTPUT_ROOT" "$REPO_ROOT/artifacts/$(basename "$OUTPUT_ROOT")"
    sync_dir "$FIGURES_DIR" "$REPO_ROOT/report/$(basename "$FIGURES_DIR")"
    sync_dir "$SMOKE_OUTPUT_ROOT" "$REPO_ROOT/artifacts/$(basename "$SMOKE_OUTPUT_ROOT")"
    sync_dir "$SMOKE_FIGURES_DIR" "$REPO_ROOT/report/$(basename "$SMOKE_FIGURES_DIR")"
    sync_dir "$LOGS_DIR" "$REPO_ROOT/logs/slurm"

    shopt -s nullglob
    src=""
    for src in "$PROJECT_SHARED_ROOT"/artifacts/full_wrapper_*; do
      sync_dir "$src" "$REPO_ROOT/artifacts/$(basename "$src")"
    done
    for src in "$PROJECT_SHARED_ROOT"/report/full_wrapper_*; do
      sync_dir "$src" "$REPO_ROOT/report/$(basename "$src")"
    done
  '
}

print_paths() {
  cat <<EOF
REPO_ROOT=$REPO_ROOT
PROJECT_SHARED_ROOT=$PROJECT_SHARED_ROOT
ENV_PREFIX=$ENV_PREFIX
CONDA_PKGS_DIRS=$CONDA_PKGS_DIRS
LOGS_DIR=$LOGS_DIR
OUTPUT_ROOT=$OUTPUT_ROOT
FIGURES_DIR=$FIGURES_DIR
SMOKE_OUTPUT_ROOT=$SMOKE_OUTPUT_ROOT
SMOKE_FIGURES_DIR=$SMOKE_FIGURES_DIR
DRY_RUN_MANIFEST=$DRY_RUN_MANIFEST
PARTITION=$PARTITION
GRES=$GRES
EOF
}

submit_pipeline_on_gpu() {
  local extra_args=""
  if (($# > 0)); then
    printf -v extra_args ' %q' "$@"
  fi
  run_on_gpu "$CHECK_TIME" "$CHECK_MEM" "
    cd \"\$REPO_ROOT\"
    \"\$ENV_PREFIX/bin/python\" scripts/submit_mimi_pipeline.py \
      --env-prefix \"\$ENV_PREFIX\" \
      --logs-dir \"\$LOGS_DIR\" \
      --output-root \"\$OUTPUT_ROOT\" \
      --figures-dir \"\$FIGURES_DIR\"$extra_args
  "
}

cmd_bootstrap() {
  ensure_shared_dirs
  run_on_gpu "$SETUP_TIME" "$SETUP_MEM" '
    cd "$REPO_ROOT"
    export ENV_PREFIX CONDA_PKGS_DIRS CREATE_ENV_IF_MISSING=1
    bash scripts/setup_mimi_env.sh
  '
}

cmd_runtime() {
  ensure_shared_dirs
  run_on_gpu "$CHECK_TIME" "$CHECK_MEM" '
    cd "$REPO_ROOT"
    export ENV_PREFIX CONDA_PKGS_DIRS SKIP_DEP_INSTALL=1
    bash scripts/setup_mimi_env.sh
  '
}

cmd_dry_run() {
  ensure_shared_dirs
  submit_pipeline_on_gpu --dry-run --manifest-path "$DRY_RUN_MANIFEST"
}

cmd_collect() {
  load_slurm_support
  collect_outputs_to_repo
}

slurm_bin_dir() {
  dirname "${SBATCH_BIN:?SBATCH_BIN is unset}"
}

submit_sbatch_job() {
  local dependency="${1:-}"
  shift

  local command=("$SBATCH_BIN" "$@")
  if [[ -n "$dependency" ]]; then
    command+=(--dependency "$dependency")
  fi

  local output
  if ! output="$("${command[@]}" 2>&1)"; then
    echo "ERROR: sbatch failed: $output" >&2
    return 1
  fi
  awk 'END { print $NF }' <<<"$output"
}

build_export_arg() {
  local -a entries=("ALL" "$@")
  local joined=""
  local entry
  for entry in "${entries[@]}"; do
    if [[ -n "$joined" ]]; then
      joined+=","
    fi
    joined+="$entry"
  done
  printf '%s\n' "$joined"
}

append_export_if_set() {
  local -n export_items_ref="$1"
  local key="$2"
  local value="${3:-}"

  if [[ -n "$value" ]]; then
    export_items_ref+=("$key=$value")
  fi
}

append_env_override() {
  local -n export_items_ref="$1"
  local source_key="$2"
  local target_key="${3:-$source_key}"
  local value="${!source_key:-}"

  if [[ -n "$value" ]]; then
    export_items_ref+=("$target_key=$value")
  fi
}

append_suspicious_override() {
  local -n suspicious_ref="$1"
  local key="$2"
  local min_value="$3"
  local value="${!key:-}"

  [[ -n "$value" ]] || return 0
  [[ "$value" =~ ^[0-9]+$ ]] || return 0
  if (( value < min_value )); then
    suspicious_ref+=("$key=$value (expected >= $min_value for a full run)")
  fi
}

validate_full_run_overrides() {
  local -a suspicious=()

  append_suspicious_override suspicious NUM_ITERATIONS 5
  append_suspicious_override suspicious SELFPLAY_GAMES 8
  append_suspicious_override suspicious NUM_SIMULATIONS 8
  append_suspicious_override suspicious EVAL_GAMES 4
  append_suspicious_override suspicious MAX_PLIES_5X5 32
  append_suspicious_override suspicious MAX_PLIES_8X8 64
  append_suspicious_override suspicious HEAD_TO_HEAD_GAMES 4
  append_suspicious_override suspicious HEAD_TO_HEAD_SIMULATIONS 4
  append_suspicious_override suspicious HEAD_TO_HEAD_MAX_PLIES 64

  if ((${#suspicious[@]} == 0)) || [[ "${ALLOW_TINY_FULL_RUN:-0}" == "1" ]]; then
    return 0
  fi

  {
    echo "ERROR: refusing to launch scripts/mimi_project.sh full with tiny overrides."
    echo "These values look like smoke/validation settings and would make the staged full DAG inconclusive:"
    local item
    for item in "${suspicious[@]}"; do
      echo "  - $item"
    done
    cat <<'EOF'

Use `bash scripts/mimi_full_validation.sh` for a tiny end-to-end check.
If you intentionally want a tiny staged full run, rerun with `ALLOW_TINY_FULL_RUN=1`.
Otherwise unset the small override variables and submit `bash scripts/mimi_full_dag.sh` again.
EOF
  } >&2
  exit 1
}

is_terminal_slurm_state() {
  local state="$1"
  case "$state" in
    BOOT_FAIL*|CANCELLED*|COMPLETED*|DEADLINE*|FAILED*|NODE_FAIL*|OUT_OF_MEMORY*|PREEMPTED*|REVOKED*|SPECIAL_EXIT*|TIMEOUT*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

slurm_terminal_state() {
  local job_id="$1"
  local squeue_bin sacct_bin queue_state acct_state
  squeue_bin="$(slurm_bin_dir)/squeue"
  sacct_bin="$(slurm_bin_dir)/sacct"

  queue_state="$("$squeue_bin" -h -j "$job_id" -o "%T" 2>/dev/null || true)"
  if [[ -n "$queue_state" ]]; then
    return 1
  fi

  acct_state="$("$sacct_bin" -n -X -j "$job_id" -o State 2>/dev/null | awk 'NF { print $1; exit }')"
  if [[ -z "$acct_state" ]]; then
    return 1
  fi

  if ! is_terminal_slurm_state "$acct_state"; then
    return 1
  fi

  printf '%s\n' "$acct_state"
}

wait_for_slurm_job() {
  local job_id="$1"
  local state=""

  echo "Waiting for job $job_id..."
  until state="$(slurm_terminal_state "$job_id")"; do
    sleep "$SMOKE_POLL_SECONDS"
  done

  echo "Job $job_id finished with state $state"
  [[ "$state" == COMPLETED* ]]
}

smoke_submit_train() {
  local job_name="$1"
  local preset="$2"
  local run_name="$3"
  local max_plies="$4"

  submit_sbatch_job "" \
    --chdir "$REPO_ROOT" \
    --job-name "$job_name" \
    --partition "$PARTITION" \
    --account "$SLURM_ACCOUNT" \
    --qos "$SLURM_QOS" \
    --gres gpu:1 \
    --mem 16G \
    --time 12:00:00 \
    --output "$LOGS_DIR/%x_%j.out" \
    --error "$LOGS_DIR/%x_%j.err" \
    --export "ALL,ENV_PREFIX=$ENV_PREFIX,EVAL_GAMES=2,LOGS_DIR=$LOGS_DIR,MAX_PLIES=$max_plies,NUM_ITERATIONS=1,NUM_SIMULATIONS=2,OUTPUT_ROOT=$SMOKE_OUTPUT_ROOT,PRESET=$preset,REPO_ROOT=$REPO_ROOT,RESUME=1,RUN_NAME=$run_name,SELFPLAY_GAMES=2" \
    "$REPO_ROOT/slurm/train_experiment.sbatch"
}

full_submit_train() {
  local job_name="$1"
  local preset="$2"
  local run_name="$3"
  local max_plies="${4:-}"
  local -a export_items=(
    "ENV_PREFIX=$ENV_PREFIX"
    "LOGS_DIR=$LOGS_DIR"
    "OUTPUT_ROOT=$OUTPUT_ROOT"
    "PRESET=$preset"
    "REPO_ROOT=$REPO_ROOT"
    "RESUME=$RESUME"
    "RUN_NAME=$run_name"
  )

  append_env_override export_items NUM_ITERATIONS
  append_env_override export_items SELFPLAY_GAMES
  append_env_override export_items NUM_SIMULATIONS
  append_env_override export_items EVAL_GAMES
  append_export_if_set export_items MAX_PLIES "$max_plies"

  submit_sbatch_job "" \
    --chdir "$REPO_ROOT" \
    --job-name "$job_name" \
    --partition "$PARTITION" \
    --account "$SLURM_ACCOUNT" \
    --qos "$SLURM_QOS" \
    --gres gpu:1 \
    --mem "$TRAIN_MEM_LIMIT" \
    --time "$TRAIN_TIME_LIMIT" \
    --output "$LOGS_DIR/%x_%j.out" \
    --error "$LOGS_DIR/%x_%j.err" \
    --export "$(build_export_arg "${export_items[@]}")" \
    "$REPO_ROOT/slurm/train_experiment.sbatch"
}

full_submit_transfer() {
  local job_name="$1"
  local -a export_items=(
    "ENV_PREFIX=$ENV_PREFIX"
    "LOGS_DIR=$LOGS_DIR"
    "OUTPUT_ROOT=$OUTPUT_ROOT"
    "PRETRAINED_CHECKPOINT=$OUTPUT_ROOT/gnn_5x5_pretrain/checkpoints/final.pkl"
    "REPO_ROOT=$REPO_ROOT"
    "RESUME=$RESUME"
  )

  append_export_if_set export_items ITERATIONS_FINETUNE "${NUM_ITERATIONS:-}"
  append_env_override export_items SELFPLAY_GAMES
  append_env_override export_items NUM_SIMULATIONS
  append_env_override export_items EVAL_GAMES
  append_export_if_set export_items MAX_PLIES_5X5 "${MAX_PLIES_5X5:-}"
  append_export_if_set export_items MAX_PLIES_8X8 "${MAX_PLIES_8X8:-}"

  submit_sbatch_job "" \
    --chdir "$REPO_ROOT" \
    --job-name "$job_name" \
    --partition "$PARTITION" \
    --account "$SLURM_ACCOUNT" \
    --qos "$SLURM_QOS" \
    --gres gpu:1 \
    --mem "$TRAIN_MEM_LIMIT" \
    --time "$TRAIN_TIME_LIMIT" \
    --output "$LOGS_DIR/%x_%j.out" \
    --error "$LOGS_DIR/%x_%j.err" \
    --export "$(build_export_arg "${export_items[@]}")" \
    "$REPO_ROOT/slurm/transfer_stage.sbatch"
}

full_submit_postprocess() {
  local job_name="$1"
  local -a export_items=(
    "CNN_SCRATCH_DIR=$OUTPUT_ROOT/cnn_8x8_scratch"
    "ENV_PREFIX=$ENV_PREFIX"
    "FINETUNE_DIR=$OUTPUT_ROOT/gnn_8x8_finetune"
    "GNN_SCRATCH_DIR=$OUTPUT_ROOT/gnn_8x8_scratch"
    "HEAD_TO_HEAD_OUTPUT=$OUTPUT_ROOT/head_to_head_gnn_vs_cnn.json"
    "LOGS_DIR=$LOGS_DIR"
    "OUTPUT_DIR=$FIGURES_DIR"
    "PRETRAIN_DIR=$OUTPUT_ROOT/gnn_5x5_pretrain"
    "REPO_ROOT=$REPO_ROOT"
    "SUMMARY_OUTPUT=$OUTPUT_ROOT/postprocess_summary.json"
    "TRANSFER_SUMMARY=$OUTPUT_ROOT/gnn_transfer_summary.json"
    "TRANSFER_ZERO_SHOT=$OUTPUT_ROOT/gnn_transfer_zero_shot.json"
  )

  append_export_if_set export_items N_GAMES "${HEAD_TO_HEAD_GAMES:-}"
  append_export_if_set export_items N_SIM "${HEAD_TO_HEAD_SIMULATIONS:-}"
  append_export_if_set export_items MAX_PLIES "${HEAD_TO_HEAD_MAX_PLIES:-}"

  submit_sbatch_job "" \
    --chdir "$REPO_ROOT" \
    --job-name "$job_name" \
    --partition "$PARTITION" \
    --account "$SLURM_ACCOUNT" \
    --qos "$SLURM_QOS" \
    --gres gpu:1 \
    --mem "$POSTPROCESS_MEM_LIMIT" \
    --time "$POSTPROCESS_TIME_LIMIT" \
    --output "$LOGS_DIR/%x_%j.out" \
    --error "$LOGS_DIR/%x_%j.err" \
    --export "$(build_export_arg "${export_items[@]}")" \
    "$REPO_ROOT/slurm/postprocess.sbatch"
}

run_retrying_job() {
  local -n result_ref="$1"
  local lane="$2"
  local max_attempts="$3"
  local submit_fn="$4"
  shift 4

  local attempt job_name job_id
  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    job_name="${lane}-a${attempt}"
    job_id="$("$submit_fn" "$job_name" "$@")"
    echo "Submitted $lane attempt $attempt as job $job_id"
    if wait_for_slurm_job "$job_id"; then
      result_ref="$job_id"
      return 0
    fi
    if (( attempt == max_attempts )); then
      echo "ERROR: $lane failed after $max_attempts attempts." >&2
      return 1
    fi
    echo "$lane attempt $attempt failed; retrying."
  done
}

run_full_scratch_phase() {
  local -n gnn_result_ref="$1"
  local -n cnn_result_ref="$2"
  local gnn_attempt=1
  local cnn_attempt=1
  local gnn_active_job_id cnn_active_job_id gnn_state cnn_state

  gnn_active_job_id="$(full_submit_train "gnn-8x8-scratch-a1" "gnn_8x8_scratch" "gnn_8x8_scratch" "${MAX_PLIES_8X8:-}")"
  cnn_active_job_id="$(full_submit_train "cnn-8x8-scratch-a1" "cnn_8x8_scratch" "cnn_8x8_scratch" "${MAX_PLIES_8X8:-}")"
  echo "Submitted full scratch jobs: gnn=$gnn_active_job_id cnn=$cnn_active_job_id"

  while [[ -z "$gnn_result_ref" || -z "$cnn_result_ref" ]]; do
    if [[ -z "$gnn_result_ref" ]]; then
      gnn_state="$(slurm_terminal_state "$gnn_active_job_id" || true)"
      if [[ -n "$gnn_state" ]]; then
        echo "Job $gnn_active_job_id finished with state $gnn_state"
        if [[ "$gnn_state" == COMPLETED* ]]; then
          gnn_result_ref="$gnn_active_job_id"
        elif (( gnn_attempt < MAX_ATTEMPTS )); then
          ((gnn_attempt += 1))
          gnn_active_job_id="$(full_submit_train "gnn-8x8-scratch-a${gnn_attempt}" "gnn_8x8_scratch" "gnn_8x8_scratch" "${MAX_PLIES_8X8:-}")"
          echo "Resubmitted gnn-8x8-scratch attempt $gnn_attempt as job $gnn_active_job_id"
        else
          echo "ERROR: gnn-8x8-scratch failed after $MAX_ATTEMPTS attempts." >&2
          return 1
        fi
      fi
    fi

    if [[ -z "$cnn_result_ref" ]]; then
      cnn_state="$(slurm_terminal_state "$cnn_active_job_id" || true)"
      if [[ -n "$cnn_state" ]]; then
        echo "Job $cnn_active_job_id finished with state $cnn_state"
        if [[ "$cnn_state" == COMPLETED* ]]; then
          cnn_result_ref="$cnn_active_job_id"
        elif (( cnn_attempt < MAX_ATTEMPTS )); then
          ((cnn_attempt += 1))
          cnn_active_job_id="$(full_submit_train "cnn-8x8-scratch-a${cnn_attempt}" "cnn_8x8_scratch" "cnn_8x8_scratch" "${MAX_PLIES_8X8:-}")"
          echo "Resubmitted cnn-8x8-scratch attempt $cnn_attempt as job $cnn_active_job_id"
        else
          echo "ERROR: cnn-8x8-scratch failed after $MAX_ATTEMPTS attempts." >&2
          return 1
        fi
      fi
    fi

    if [[ -z "$gnn_result_ref" || -z "$cnn_result_ref" ]]; then
      sleep "$SMOKE_POLL_SECONDS"
    fi
  done
}

smoke_submit_transfer() {
  local pretrain_job_id="$1"

  submit_sbatch_job "afterok:$pretrain_job_id" \
    --chdir "$REPO_ROOT" \
    --job-name "gnn-transfer-a1" \
    --partition "$PARTITION" \
    --account "$SLURM_ACCOUNT" \
    --qos "$SLURM_QOS" \
    --gres gpu:1 \
    --mem 16G \
    --time 12:00:00 \
    --output "$LOGS_DIR/%x_%j.out" \
    --error "$LOGS_DIR/%x_%j.err" \
    --export "ALL,ENV_PREFIX=$ENV_PREFIX,EVAL_GAMES=2,ITERATIONS_FINETUNE=1,LOGS_DIR=$LOGS_DIR,MAX_PLIES_5X5=8,MAX_PLIES_8X8=8,NUM_SIMULATIONS=2,OUTPUT_ROOT=$SMOKE_OUTPUT_ROOT,PRETRAINED_CHECKPOINT=$SMOKE_OUTPUT_ROOT/gnn_5x5_pretrain/checkpoints/final.pkl,REPO_ROOT=$REPO_ROOT,RESUME=1,SELFPLAY_GAMES=2" \
    "$REPO_ROOT/slurm/transfer_stage.sbatch"
}

smoke_submit_postprocess() {
  local gnn_job_id="$1"
  local cnn_job_id="$2"
  local transfer_job_id="$3"

  submit_sbatch_job "afterok:$gnn_job_id:$cnn_job_id:$transfer_job_id" \
    --chdir "$REPO_ROOT" \
    --job-name "postprocess-a1" \
    --partition "$PARTITION" \
    --account "$SLURM_ACCOUNT" \
    --qos "$SLURM_QOS" \
    --gres gpu:1 \
    --mem 16G \
    --time 02:00:00 \
    --output "$LOGS_DIR/%x_%j.out" \
    --error "$LOGS_DIR/%x_%j.err" \
    --export "ALL,CNN_SCRATCH_DIR=$SMOKE_OUTPUT_ROOT/cnn_8x8_scratch,ENV_PREFIX=$ENV_PREFIX,FINETUNE_DIR=$SMOKE_OUTPUT_ROOT/gnn_8x8_finetune,GNN_SCRATCH_DIR=$SMOKE_OUTPUT_ROOT/gnn_8x8_scratch,HEAD_TO_HEAD_OUTPUT=$SMOKE_OUTPUT_ROOT/head_to_head_gnn_vs_cnn.json,LOGS_DIR=$LOGS_DIR,MAX_PLIES=8,N_GAMES=2,N_SIM=2,OUTPUT_DIR=$SMOKE_FIGURES_DIR,PRETRAIN_DIR=$SMOKE_OUTPUT_ROOT/gnn_5x5_pretrain,REPO_ROOT=$REPO_ROOT,SUMMARY_OUTPUT=$SMOKE_OUTPUT_ROOT/postprocess_summary.json,TRANSFER_SUMMARY=$SMOKE_OUTPUT_ROOT/gnn_transfer_summary.json,TRANSFER_ZERO_SHOT=$SMOKE_OUTPUT_ROOT/gnn_transfer_zero_shot.json" \
    "$REPO_ROOT/slurm/postprocess.sbatch"
}

cmd_smoke() {
  ensure_shared_dirs
  load_slurm_support

  run_on_gpu "$CHECK_TIME" "$CHECK_MEM" '
    rm -rf "$SMOKE_OUTPUT_ROOT" "$SMOKE_FIGURES_DIR"
    mkdir -p "$SMOKE_OUTPUT_ROOT" "$SMOKE_FIGURES_DIR"
  '

  local gnn_job_id cnn_job_id pretrain_job_id transfer_job_id postprocess_job_id
  gnn_job_id="$(smoke_submit_train "gnn-8x8-scratch-a1" "gnn_8x8_scratch" "gnn_8x8_scratch" 8)"
  cnn_job_id="$(smoke_submit_train "cnn-8x8-scratch-a1" "cnn_8x8_scratch" "cnn_8x8_scratch" 8)"
  echo "Submitted smoke training jobs: gnn=$gnn_job_id cnn=$cnn_job_id"

  wait_for_slurm_job "$gnn_job_id" || exit 1
  wait_for_slurm_job "$cnn_job_id" || exit 1

  pretrain_job_id="$(smoke_submit_train "gnn-5x5-pretrain-a1" "gnn_5x5_pretrain" "gnn_5x5_pretrain" 8)"
  echo "Submitted smoke pretrain job: pretrain=$pretrain_job_id"
  wait_for_slurm_job "$pretrain_job_id" || exit 1

  transfer_job_id="$(smoke_submit_transfer "$pretrain_job_id")"
  echo "Submitted smoke transfer job: transfer=$transfer_job_id"
  wait_for_slurm_job "$transfer_job_id" || exit 1

  postprocess_job_id="$(smoke_submit_postprocess "$gnn_job_id" "$cnn_job_id" "$transfer_job_id")"
  echo "Submitted smoke postprocess job: postprocess=$postprocess_job_id"
  wait_for_slurm_job "$postprocess_job_id" || exit 1

  echo "Smoke DAG completed successfully."
  echo "Syncing useful outputs back into the repo..."
  if ! cmd_collect; then
    echo "warning: unable to sync outputs automatically; rerun bash scripts/mimi_collect_outputs.sh later." >&2
  fi
  cmd_queue
}

cmd_full() {
  validate_full_run_overrides
  ensure_shared_dirs
  load_slurm_support

  local gnn_job_id=""
  local cnn_job_id=""
  local pretrain_job_id=""
  local transfer_job_id=""
  local postprocess_job_id=""

  run_full_scratch_phase gnn_job_id cnn_job_id
  run_retrying_job pretrain_job_id "gnn-5x5-pretrain" "$MAX_ATTEMPTS" full_submit_train "gnn_5x5_pretrain" "gnn_5x5_pretrain" "${MAX_PLIES_5X5:-}"
  run_retrying_job transfer_job_id "gnn-transfer" "$MAX_ATTEMPTS" full_submit_transfer
  run_retrying_job postprocess_job_id "postprocess" 1 full_submit_postprocess

  echo "Full staged DAG completed successfully."
  echo "Final job ids: gnn=$gnn_job_id cnn=$cnn_job_id pretrain=$pretrain_job_id transfer=$transfer_job_id postprocess=$postprocess_job_id"
  echo "Syncing useful outputs back into the repo..."
  if ! cmd_collect; then
    echo "warning: unable to sync outputs automatically; rerun bash scripts/mimi_collect_outputs.sh later." >&2
  fi
  cmd_queue
}

cmd_queue() {
  load_slurm_support
  squeue -u "$USER"
}

main() {
  local command="${1:-}"
  case "$command" in
    paths)
      print_paths
      ;;
    bootstrap)
      cmd_bootstrap
      ;;
    runtime)
      cmd_runtime
      ;;
    dry-run)
      cmd_dry_run
      ;;
    smoke)
      cmd_smoke
      ;;
    full)
      cmd_full
      ;;
    collect)
      cmd_collect
      ;;
    queue)
      cmd_queue
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
