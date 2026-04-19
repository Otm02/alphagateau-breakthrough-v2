#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-comp579-breakthrough}"
JAX_EXTRA="${JAX_EXTRA:-jax[cuda12]}"

cd "$REPO_ROOT"

module load anaconda
module load cuda/11.8

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.11 pip
fi

conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt jax-cuda13-plugin jax-cuda13-pjrt || true
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install --upgrade "$JAX_EXTRA"
python scripts/show_runtime_info.py
