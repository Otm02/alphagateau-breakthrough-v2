# Run Guide

Three supported paths, in order of complexity:

| Path | When to use |
|---|---|
| **[A] Standalone GPU / notebook](#a-standalone-gpu--notebook)** | Any Linux GPU, Google Colab, Kaggle, Lambda Labs — no Slurm needed |
| **[B] McGill Slurm (personal env)](#b-mcgill-slurm-personal-env)** | `mimi.cs.mcgill.ca` with your own conda env |
| **[C] McGill Slurm (shared env)](#c-mcgill-slurm-shared-env--mimi-wrappers)** | `mimi` with the pre-built team env via the `mimi_project.sh` wrappers |

For Path C only (the shared-env wrappers validated for this repo), see also [RUN_GUIDE_CUSTOM.md](RUN_GUIDE_CUSTOM.md).

---

## A. Standalone GPU / Notebook

Use this path on any Linux machine with a GPU: your own workstation, Google Colab, Kaggle, Lambda
Labs, or any cloud instance. No Slurm access required.

See [notebooks/demo.ipynb](notebooks/demo.ipynb) for a self-contained notebook that runs this
path end to end.

### Step 1 — Clone and install

```bash
git clone <repo-url>
cd alphagateau-breakthrough-v2
pip install -e .
```

CPU-safe by default. For GPU acceleration install exactly one JAX CUDA extra **after** the base
install — do not mix two CUDA variants in the same environment:

```bash
# CUDA 12.x (most common on recent machines and cloud instances):
pip install --upgrade "jax[cuda12]"

# CUDA 13.x:
pip install --upgrade "jax[cuda13]"
```

### Step 2 — Verify GPU is visible

```bash
python scripts/show_runtime_info.py
```

Expected on GPU: `jax devices: [CudaDevice(id=0)]`

### Step 3 — Run the smoke suite (1-iteration sanity check)

```bash
python scripts/run_smoke_suite.py
```

This runs one iteration of every preset on CPU/GPU and writes figures to `report/figures/`.
Takes ~2 minutes on GPU, ~10 minutes on CPU.

### Step 4 — Run the full experiments

Each command supports `--resume` so you can restart safely if interrupted.

```bash
# GNN and CNN on 8x8 from scratch (run in parallel if you have two GPUs or two sessions):
python scripts/train_experiment.py gnn_8x8_scratch --resume
python scripts/train_experiment.py cnn_8x8_scratch --resume

# 5x5 pretraining, then transfer to 8x8:
python scripts/train_experiment.py gnn_5x5_pretrain --resume
python scripts/run_transfer_pipeline.py \
  --pretrained-checkpoint artifacts/experiments/gnn_5x5_pretrain/checkpoints/final.pkl \
  --resume
```

Outputs land under `artifacts/experiments/<run_name>/`.


## B. McGill Slurm (personal env)

Use this path when you have access to `mimi.cs.mcgill.ca` and want a personal conda environment.

### Step 1 — SSH to mimi and clone

```bash
git clone <repo-url>
cd alphagateau-breakthrough-v2
```

### Step 2 — Bootstrap your conda environment once

```bash
bash scripts/setup_mimi_env.sh
```

Creates a `comp579-breakthrough` conda env under `~/.conda/envs/` with JAX CUDA 12.
Override the JAX CUDA version if your allocation uses CUDA 13:

```bash
JAX_EXTRA='jax[cuda13]' bash scripts/setup_mimi_env.sh
```


### Step 3 — Submit (with your Slurm account and QoS)

```bash
python3 scripts/submit_mimi_pipeline.py \
  --account <your-slurm-account> \
  --qos <your-slurm-qos>
```

McGill COMP-579 example:

```bash
python3 scripts/submit_mimi_pipeline.py \
  --account winter2026-comp579 \
  --qos comp579-1gpu-12h
```

If your cluster does not require an account or QoS, omit those flags.

### Step 4 — Monitor

```bash
squeue -u "$USER"
```

Each training lane is submitted as a retry chain. If a job times out, the next attempt resumes
from `latest_resume.pkl`. Do not restart manually from scratch.


---
## C. McGill Slurm (shared env + mimi wrappers)

Use this path when using the pre-built team conda environment on shared storage.
The `mimi_project.sh` wrapper handles environment setup, Slurm module loading, and staged DAG
submission within the course job cap.

```bash
# One-time shared env bootstrap (only the first team member needs to do this):
bash scripts/mimi_bootstrap_shared_env.sh

# Verify GPU and env on a compute node:
bash scripts/mimi_runtime_check.sh

# Preview the DAG:
bash scripts/mimi_dry_run.sh

# Submit the full staged DAG:
bash scripts/mimi_full_dag.sh

# Mirror results back to the repo after the DAG finishes:
bash scripts/mimi_collect_outputs.sh
```

Override the Slurm account and QoS if needed (defaults to COMP-579 values):

```bash
SLURM_ACCOUNT=<your-account> SLURM_QOS=<your-qos> bash scripts/mimi_full_dag.sh
```

See [RUN_GUIDE_CUSTOM.md](RUN_GUIDE_CUSTOM.md) for the full shared-path layout and all available
environment overrides.

---

## Expected outputs

After any complete run:

| Path | Contents |
|---|---|
| `artifacts/experiments/<run>/checkpoints/` | Model checkpoints every 5 iterations + `final.pkl` |
| `artifacts/experiments/<run>/metrics.csv` | Per-iteration training loss |
| `artifacts/experiments/<run>/evaluation.csv` | Greedy-evaluator win rate every 5 iterations |
| `artifacts/experiments/<run>/summary.json` | Final metrics snapshot |
| `artifacts/experiments/head_to_head_gnn_vs_cnn.json` | GNN vs CNN head-to-head result |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `runtime-info` shows CPU on Linux | Install exactly one JAX CUDA extra; run `pip install "jax[cuda12]"` |
| Two `jax-cuda*-plugin` packages installed | Uninstall all JAX, reinstall one: `pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt jax-cuda13-plugin jax-cuda13-pjrt && pip install "jax[cuda12]"` |
| Slurm job times out | Do not restart manually. Let the retry chain pick it up from `latest_resume.pkl` |
| `sbatch: error: Invalid account` | Pass `--account <your-account>` or set `SLURM_ACCOUNT=<account>` |
