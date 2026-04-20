# McGill Mimi — Shared-Env Guide

This guide is the shortest path for this repository on the McGill COMP-579 Slurm cluster.
Any team member with a `mimi.cs.mcgill.ca` account can follow it independently — shared
storage is keyed by `$USER`, so each person gets their own isolated output paths.

It assumes:

- login node shell on `mimi`
- shared Slurm-visible storage under `/mnt/teaching/slurm/$USER`
- shared conda env at `/mnt/teaching/slurm/$USER/envs/comp579-breakthrough`

## One-time Bootstrap

Create or update the shared GPU environment:

```bash
bash scripts/mimi_bootstrap_shared_env.sh
```

Check that JAX sees the GPU:

```bash
bash scripts/mimi_runtime_check.sh
```

Expected signal:

- `jax devices: [CudaDevice(id=0)]`

## Everyday Commands

Show the exact shared paths baked into the wrapper:

```bash
bash scripts/mimi_project.sh paths
```

Generate a dry-run Slurm manifest:

```bash
bash scripts/mimi_dry_run.sh
```

Submit the tiny smoke DAG:

```bash
bash scripts/mimi_smoke_dag.sh
```

This smoke wrapper stages jobs in phases so it stays under the course Slurm submit cap.

Run a tiny end-to-end validation of the staged full wrapper:

```bash
bash scripts/mimi_full_validation.sh
```

Check the queue:

```bash
bash scripts/mimi_queue.sh
```

Mirror the useful shared outputs back into this repo, refreshing local copies when the shared version is newer:

```bash
bash scripts/mimi_collect_outputs.sh
```

Submit the full course DAG:

```bash
bash scripts/mimi_full_dag.sh
```

The full wrapper now stages the real DAG to stay under the Slurm submit cap. We validated that staged flow end-to-end with the tiny wrapper above.
It also refuses obviously smoke-sized overrides unless you explicitly opt in with `ALLOW_TINY_FULL_RUN=1`.

## Shared Paths

These scripts default to:

- shared root: `/mnt/teaching/slurm/$USER`
- env: `/mnt/teaching/slurm/$USER/envs/comp579-breakthrough`
- conda package cache: `/mnt/teaching/slurm/$USER/conda-pkgs`
- Slurm logs: `/mnt/teaching/slurm/$USER/logs/slurm`
- experiment outputs: `/mnt/teaching/slurm/$USER/artifacts/experiments`
- figure outputs: `/mnt/teaching/slurm/$USER/report/figures`
- smoke outputs: `/mnt/teaching/slurm/$USER/artifacts/smoke_dag`
- smoke figures: `/mnt/teaching/slurm/$USER/report/smoke_figures`
- full-wrapper validation outputs: `/mnt/teaching/slurm/$USER/artifacts/full_wrapper_validation`
- full-wrapper validation figures: `/mnt/teaching/slurm/$USER/report/full_wrapper_validation_figures`
- dry-run manifest: `/mnt/teaching/slurm/$USER/logs/slurm/submissions/pipeline_dry_run.json`

## What The Wrappers Do

- load a Slurm module if needed
- run setup and runtime checks on a GPU node through `srun`
- submit DAGs using the shared env interpreter
- pass the shared env prefix and shared output paths into the existing project submitter
- run the smoke DAG through a staged launcher that respects the Slurm job cap
- run the full DAG through a staged launcher with retries, so it never over-submits jobs under this account

## Optional Overrides

If you ever need to move the shared root or change defaults, set environment variables before calling the wrappers:

```bash
PROJECT_SHARED_ROOT=/some/other/shared/path bash scripts/mimi_project.sh paths
```

Useful overrides:

- `SLURM_ACCOUNT` — Slurm account (default: `winter2026-comp579`)
- `SLURM_QOS` — Slurm QoS (default: `comp579-1gpu-12h`)
- `PROJECT_SHARED_ROOT`
- `ENV_PREFIX`
- `CONDA_PKGS_DIRS`
- `LOGS_DIR`
- `OUTPUT_ROOT`
- `FIGURES_DIR`
- `SMOKE_OUTPUT_ROOT`
- `SMOKE_FIGURES_DIR`
- `MAX_ATTEMPTS`
- `NUM_ITERATIONS`
- `SELFPLAY_GAMES`
- `NUM_SIMULATIONS`
- `MAX_PLIES_5X5`
- `MAX_PLIES_8X8`
- `EVAL_GAMES`
- `HEAD_TO_HEAD_GAMES`
- `HEAD_TO_HEAD_SIMULATIONS`
- `HEAD_TO_HEAD_MAX_PLIES`
- `ALLOW_TINY_FULL_RUN`
- `PARTITION`
- `GRES`

## Shared Path Visibility

On this setup, `/mnt/teaching/slurm/...` is easiest to inspect from a Slurm allocation. The wrappers already handle that when they need to touch shared paths. If you want to inspect those directories manually, use `srun` or one of the wrapper scripts rather than assuming the login node can see them directly.
