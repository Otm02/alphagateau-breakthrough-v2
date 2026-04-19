# Run Guide

This repository now has two supported operating modes:

- **Local validation**: Pixi on Windows PowerShell or Linux for tests, smoke runs, figure regeneration from existing artifacts, and paper compilation.
- **Canonical full training**: the McGill CS department GPUs on `mimi.cs.mcgill.ca` through Slurm.

## Important GPU Note

Treat native Windows PowerShell as **CPU-only** for JAX. The official JAX installation guide documents NVIDIA GPU wheels as Linux-only, with Windows GPU support limited to WSL2 experimentation.

That means:

- use local Pixi for quick validation and report work
- use `mimi` + Slurm for the real course experiments

## 1. Local Validation

Use this path when you want to verify the repo, run the smoke suite, or compile the paper locally.

### Step 1: Open the repo

```powershell
cd "C:\Users\alger\OneDrive\Bureau\Projects\COMP 579\alphagateau-breakthrough-v2"
```

### Step 2: Create the Pixi environment

```powershell
pixi install
```

### Step 3: Check the runtime

```powershell
pixi run runtime-info
```

Expected on native Windows:

- `jax.devices()` will usually show `CpuDevice(...)`
- that is fine for tests, smoke runs, and report generation

### Step 4: Run the validation suite

```powershell
pixi run test
pixi run smoke
```

This writes:

- `artifacts/smoke_suite/...`
- `report/figures/gnn_8x8_scratch_curve.png`
- `report/figures/transfer_curve.png`
- `report/figures/encoding_visualisation.png`

### Step 5: Compile the paper PDF

```powershell
pixi run paper
```

This writes:

- `report/paper.pdf`

## 2. Canonical Full Training On McGill Slurm

This is the default path for the real experiments.

The Slurm workers and submitter are aligned with the TA’s Ed post and follow these defaults:

- `partition=all`
- `account=winter2026-comp579`
- `qos=comp579-1gpu-12h`
- `gres=gpu:1`
- `mem=16G`
- training jobs use `--signal=B:USR1@120` so the training loop can checkpoint resume-state before timeout

### Step 1: SSH to `mimi`

Use your CS account and SSH key from a terminal that already has access configured.

### Step 2: Clone the repo on `mimi`

Use a Linux path on the server, ideally without spaces in it.

```bash
git clone <your-repo-url>
cd alphagateau-breakthrough-v2
```

### Step 3: Bootstrap the environment once

```bash
bash scripts/setup_mimi_env.sh
```

Defaults:

- conda env name: `comp579-breakthrough`
- GPU JAX extra: `jax[cuda12]`

If `scripts/show_runtime_info.py` indicates the cluster supports CUDA 13 cleanly, you can override the default:

```bash
JAX_EXTRA='jax[cuda13]' bash scripts/setup_mimi_env.sh
```

### Step 4: Inspect the Slurm DAG before submitting

```bash
python3 scripts/submit_mimi_pipeline.py --dry-run
```

This writes a manifest under `logs/slurm/submissions/` showing:

- every `sbatch` command
- job names
- dependency edges
- expected output files

### Step 5: Run a tiny cluster smoke DAG

Use this once before the full experiment schedule:

```bash
python3 scripts/submit_mimi_pipeline.py \
  --max-attempts 1 \
  --num-iterations 1 \
  --selfplay-games 2 \
  --num-simulations 2 \
  --max-plies-5x5 8 \
  --max-plies-8x8 8 \
  --eval-games 2 \
  --head-to-head-games 2 \
  --head-to-head-simulations 2 \
  --head-to-head-max-plies 8
```

Then monitor it:

```bash
squeue -u "$USER"
```

### Step 6: Submit the full course DAG

```bash
python3 scripts/submit_mimi_pipeline.py
```

The submitter launches:

- `gnn_8x8_scratch`
- `cnn_8x8_scratch`
- `gnn_5x5_pretrain`
- `gnn_transfer` zero-shot eval + `gnn_8x8_finetune`
- final postprocessing for head-to-head evaluation and report figures

Each training lane is submitted as an `afterany` retry chain. If an earlier attempt completes successfully, later attempts exit cleanly because the run is already complete. If an attempt times out, the next job resumes from `latest_resume.pkl`.

### Step 7: Inspect logs and outputs

Logs live under:

- `logs/slurm/*.out`
- `logs/slurm/*.err`
- `logs/slurm/submissions/*.json`

Per-run experiment outputs live under:

- `artifacts/experiments/<run_name>/checkpoints/`
- `artifacts/experiments/<run_name>/metrics.csv`
- `artifacts/experiments/<run_name>/evaluation.csv`
- `artifacts/experiments/<run_name>/latest_resume.pkl`
- `artifacts/experiments/<run_name>/status.json`
- `artifacts/experiments/<run_name>/summary.json`

### Step 8: Compile the paper after the DAG finishes

The Slurm DAG stops at figures and evaluation summaries. Build the PDF separately:

```bash
tectonic report/paper.tex
```

## 3. Optional Local Linux GPU Fallback

If you cannot use `mimi`, you can still run the long jobs in WSL2/Linux, but that is now a fallback path rather than the default.

Use the same experiment CLIs directly:

```bash
python3 scripts/train_experiment.py gnn_8x8_scratch --resume
python3 scripts/train_experiment.py gnn_5x5_pretrain --resume
python3 scripts/train_experiment.py cnn_8x8_scratch --resume
python3 scripts/run_transfer_pipeline.py \
  --pretrained-checkpoint artifacts/experiments/gnn_5x5_pretrain/checkpoints/final.pkl \
  --resume
python3 scripts/postprocess_experiments.py \
  --gnn-scratch-dir artifacts/experiments/gnn_8x8_scratch \
  --cnn-scratch-dir artifacts/experiments/cnn_8x8_scratch \
  --pretrain-dir artifacts/experiments/gnn_5x5_pretrain \
  --finetune-dir artifacts/experiments/gnn_8x8_finetune \
  --transfer-summary artifacts/experiments/gnn_transfer_summary.json \
  --transfer-zero-shot artifacts/experiments/gnn_transfer_zero_shot.json
```

If JAX only sees CPU, reinstall exactly one GPU extra, not both:

```bash
python3 -m pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt jax-cuda13-plugin jax-cuda13-pjrt
python3 -m pip install --upgrade "jax[cuda13]"
# or: python3 -m pip install --upgrade "jax[cuda12]"
```

## 4. Recommended End-To-End Order

Use this order:

1. Local `pixi run test`
2. Local `pixi run smoke`
3. On `mimi`: `bash scripts/setup_mimi_env.sh`
4. On `mimi`: `python3 scripts/submit_mimi_pipeline.py --dry-run`
5. On `mimi`: one tiny smoke submission
6. On `mimi`: `python3 scripts/submit_mimi_pipeline.py`
7. After the DAG finishes: `tectonic report/paper.tex`

## 5. Expected Final Outputs

After a full Slurm run, the important files are:

- `artifacts/experiments/gnn_8x8_scratch/...`
- `artifacts/experiments/gnn_5x5_pretrain/...`
- `artifacts/experiments/gnn_8x8_finetune/...`
- `artifacts/experiments/cnn_8x8_scratch/...`
- `artifacts/experiments/gnn_transfer_summary.json`
- `artifacts/experiments/gnn_transfer_zero_shot.json`
- `artifacts/experiments/head_to_head_gnn_vs_cnn.json`
- `artifacts/experiments/postprocess_summary.json`
- `report/figures/gnn_8x8_scratch_curve.png`
- `report/figures/transfer_curve.png`
- `report/figures/encoding_visualisation.png`
- `report/paper.pdf`

## 6. If Something Looks Wrong

- If `runtime-info` shows CPU only in native Windows PowerShell: that is expected.
- If `runtime-info` on `mimi` shows CPU only: rerun `bash scripts/setup_mimi_env.sh` and verify the installed JAX extra.
- If `runtime-info` shows more than one `jax-cuda*-plugin` package: clean the environment and reinstall only one CUDA extra.
- If a Slurm training job times out: do not restart manually from scratch. Let the next `afterany` retry continue from `latest_resume.pkl`.
- If you only want a fast local sanity check: use `pixi run smoke`.
