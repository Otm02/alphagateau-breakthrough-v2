#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-comp579-breakthrough}"
ENV_PREFIX="${ENV_PREFIX:-}"
JAX_EXTRA="${JAX_EXTRA:-jax[cuda12]}"
CREATE_ENV_IF_MISSING="${CREATE_ENV_IF_MISSING:-0}"
SKIP_DEP_INSTALL="${SKIP_DEP_INSTALL:-0}"

cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/bootstrap_mimi_shell.sh"
bootstrap_mimi_shell

if [[ -n "$ENV_PREFIX" ]]; then
  if [[ ! -d "$ENV_PREFIX" ]]; then
    if [[ "$CREATE_ENV_IF_MISSING" != "1" ]]; then
      echo "ERROR: shared conda prefix does not exist: $ENV_PREFIX" >&2
      echo "Set CREATE_ENV_IF_MISSING=1 to create it in place." >&2
      exit 1
    fi
    mkdir -p "$(dirname "$ENV_PREFIX")"
    conda create -y -p "$ENV_PREFIX" python=3.11 pip
  fi
elif [[ -n "${CONDA_PREFIX:-}" && "${CONDA_DEFAULT_ENV:-}" != "base" ]]; then
  ENV_PREFIX="$CONDA_PREFIX"
elif ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  if [[ "$CREATE_ENV_IF_MISSING" != "1" ]]; then
    cat >&2 <<EOF
ERROR: conda env '$ENV_NAME' does not exist.
Use an existing shared env with:
  ENV_PREFIX=/path/to/shared/env bash scripts/setup_mimi_env.sh
Or opt into a personal env with:
  CREATE_ENV_IF_MISSING=1 ENV_NAME=$ENV_NAME bash scripts/setup_mimi_env.sh
EOF
    exit 1
  fi
  conda create -y -n "$ENV_NAME" python=3.11 pip
fi

activate_conda_target

if [[ "$SKIP_DEP_INSTALL" == "1" ]]; then
  python scripts/show_runtime_info.py
  exit 0
fi

python -m pip install --upgrade pip
python -m pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt jax-cuda13-plugin jax-cuda13-pjrt || true
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-cache-dir -e .
python -m pip install --no-cache-dir --upgrade "$JAX_EXTRA"
python scripts/show_runtime_info.py
