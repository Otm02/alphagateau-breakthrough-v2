#!/bin/bash

# Shared bootstrap for Mimi login shells and Slurm jobs.
# It prefers an existing conda on PATH, but can fall back to common module names.

ensure_module_command() {
  if type module >/dev/null 2>&1; then
    return 0
  fi

  local init_script
  for init_script in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
    if [[ -f "$init_script" ]]; then
      # shellcheck disable=SC1090
      source "$init_script"
      if type module >/dev/null 2>&1; then
        return 0
      fi
    fi
  done

  return 1
}

module_exists() {
  local candidate="${1:-}"
  [[ -n "$candidate" ]] || return 1
  local available
  available="$(module -t avail "$candidate" 2>&1 || true)"
  grep -qxF "$candidate" <<<"$available" || grep -qxF "${candidate}(default)" <<<"$available"
}

load_first_available_module() {
  local kind="$1"
  shift

  local candidate
  for candidate in "$@"; do
    [[ -n "$candidate" ]] || continue
    if module_exists "$candidate"; then
      module load "$candidate"
      printf 'Loaded %s module: %s\n' "$kind" "$candidate" >&2
      return 0
    fi
  done

  return 1
}

bootstrap_mimi_shell() {
  if ! type conda >/dev/null 2>&1 && ensure_module_command; then
    load_first_available_module conda \
      "${MIMI_CONDA_MODULE:-}" \
      anaconda \
      anaconda/latest \
      miniconda3 \
      miniconda \
      conda || true
  fi

  if ! type conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available. Put conda on PATH or set MIMI_CONDA_MODULE to a valid module name." >&2
    return 1
  fi

  if [[ "${MIMI_SKIP_CUDA_MODULE:-0}" != "1" ]] && ensure_module_command; then
    load_first_available_module cuda \
      "${MIMI_CUDA_MODULE:-}" \
      cuda/cuda-12.6 \
      cuda/cuda-11.8 \
      cuda/12.6 \
      cuda/11.8 \
      cuda || echo "warning: no CUDA module was loaded; continuing with the current shell environment." >&2
  fi

  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
}

resolve_conda_target() {
  if [[ -n "${ENV_PREFIX:-}" ]]; then
    printf '%s\n' "$ENV_PREFIX"
    return 0
  fi

  if [[ -n "${CONDA_PREFIX:-}" && "${CONDA_DEFAULT_ENV:-}" != "base" ]]; then
    printf '%s\n' "$CONDA_PREFIX"
    return 0
  fi

  if [[ -n "${ENV_NAME:-}" ]]; then
    printf '%s\n' "$ENV_NAME"
    return 0
  fi

  return 1
}

prepend_to_ld_library_path() {
  local libdir="${1:-}"
  [[ -n "$libdir" && -d "$libdir" ]] || return 0

  case ":${LD_LIBRARY_PATH:-}:" in
    *":$libdir:"*) ;;
    *)
      export LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      ;;
  esac
}

add_python_cuda_libs_to_ld_library_path() {
  [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX:-}" ]] || return 0

  local libdir
  shopt -s nullglob
  for libdir in "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/*/lib; do
    prepend_to_ld_library_path "$libdir"
  done
  shopt -u nullglob
}

activate_conda_target() {
  local target
  target="$(resolve_conda_target)" || {
    echo "ERROR: no conda environment target was provided. Set ENV_PREFIX or ENV_NAME." >&2
    return 1
  }

  conda activate "$target"
  add_python_cuda_libs_to_ld_library_path
}
