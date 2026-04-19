# AlphaGateau Breakthrough

This repository adapts the AlphaGateau idea from the original chess codebase to **Breakthrough**, with a board-size-invariant graph model, a simpler AlphaZero-style CNN baseline, self-play training, MCTS evaluation, Elo aggregation, automated figure generation, a short paper draft, and a 6-slide presentation package.

`AlphaGateau-master/` is kept as an upstream reference only. The submission-ready code lives under `src/alphagateau_breakthrough/`.

## Repository Layout

- `src/alphagateau_breakthrough/`: Breakthrough environment, graph encoder, models, MCTS, training, evaluation, Elo, and plotting.
- `scripts/`: CLI entrypoints for training, transfer, postprocessing, smoke validation, and McGill `mimi` environment/Slurm submission.
- `slurm/`: committed `sbatch` workers for single-experiment training, transfer, and postprocessing.
- `tests/`: unit tests and a tiny end-to-end smoke test.
- `report/`: project paper draft and figure output directory.
- `presentation/`: 6-slide presentation script and speaker notes.

## Setup

The code is written for Python `3.10+`.

For the **full course experiments**, the canonical path is now the McGill CS department GPU allocation on `mimi.cs.mcgill.ca` via Slurm.

Use local Pixi or a local virtualenv for:

- tests
- smoke runs
- figure generation from existing artifacts
- report compilation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The default dependency specs are **CPU-safe**. If you want GPU acceleration on Linux/WSL2, follow [RUN_GUIDE.md](RUN_GUIDE.md) and install exactly one official JAX CUDA extra inside the environment, such as `jax[cuda13]` or `jax[cuda12]`. Do not mix multiple CUDA extras in the same environment.

If you prefer Pixi, the repo now includes a root [pixi.toml](pixi.toml). The shortest setup path is:

```bash
pixi install
pixi run runtime-info
pixi run test
```

For the full operator guide, including the canonical **McGill Slurm** workflow and local validation commands, see [RUN_GUIDE.md](RUN_GUIDE.md).

## Core Commands

Run the local smoke suite:

```bash
python3 scripts/run_smoke_suite.py
```

Prepare the `mimi` environment once on the login node:

```bash
bash scripts/setup_mimi_env.sh
```

Inspect the Slurm DAG without submitting anything:

```bash
python3 scripts/submit_mimi_pipeline.py --dry-run
```

Submit a tiny cluster smoke DAG before the full run:

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

Submit the full course DAG on `mimi`:

```bash
python3 scripts/submit_mimi_pipeline.py
```

Train one preset directly, with resume support:

```bash
python3 scripts/train_experiment.py gnn_8x8_scratch --resume
```

Run the transfer curriculum from an existing 5x5 checkpoint:

```bash
python3 scripts/run_transfer_pipeline.py \
  --pretrained-checkpoint artifacts/experiments/gnn_5x5_pretrain/checkpoints/final.pkl \
  --resume
```

Postprocess completed experiment directories into the headline comparison and report figures:

```bash
python3 scripts/postprocess_experiments.py \
  --gnn-scratch-dir artifacts/experiments/gnn_8x8_scratch \
  --cnn-scratch-dir artifacts/experiments/cnn_8x8_scratch \
  --pretrain-dir artifacts/experiments/gnn_5x5_pretrain \
  --finetune-dir artifacts/experiments/gnn_8x8_finetune \
  --transfer-summary artifacts/experiments/gnn_transfer_summary.json \
  --transfer-zero-shot artifacts/experiments/gnn_transfer_zero_shot.json
```

Run the tests:

```bash
pytest -q -s tests
```

## Experiment Design

The repository ships the four course-facing presets from the plan:

- `gnn_8x8_scratch`
- `gnn_5x5_pretrain`
- `gnn_8x8_finetune`
- `cnn_8x8_scratch`

The default preset values are the course-feasible schedules from the project plan:

- hidden size `64`
- `4` residual blocks/layers
- `32` MCTS simulations
- replay window `50k`
- batch size `32`
- checkpoint/eval every `5` iterations
- `128` self-play games and `96` max plies on `5x5`
- `64` self-play games and `192` max plies on `8x8`

## Implementation Notes

- The Breakthrough action space is `from_square * 3 + move_type`, with move types `{forward, diag_left, diag_right}` in the **current player's canonical perspective**.
- The graph model uses fixed node and edge feature schemas so the same GNN checkpoint can be trained on `5x5`, evaluated zero-shot on `8x8`, and then fine-tuned on `8x8` without any parameter shape changes.
- The CNN baseline is restricted to `8x8`, matching the project plan’s “simpler AlphaZero-style baseline” replacement for the originally proposed TD baseline.
- Training outputs are written to `artifacts/experiments/<run_name>/` with checkpoints, metrics, evaluation summaries, representative move logs, `latest_resume.pkl`, `status.json`, and `summary.json`.
- The Slurm workers target the TA-recommended department configuration: `partition=all`, `account=winter2026-comp579`, `qos=comp579-1gpu-12h`, `gres=gpu:1`, `mem=16G`, with logs under `logs/slurm/`.
- Resume state includes model parameters, optimizer state, replay buffer contents, RNG state, and accumulated metric rows so long runs can survive the 12-hour QoS boundary.

## Submission Assets

- Paper draft: [report/paper.md](report/paper.md)
- Paper LaTeX source: [report/paper.tex](report/paper.tex)
- Presentation slides: [presentation/slides.md](presentation/slides.md)
- Speaker notes: [presentation/speaker_notes.md](presentation/speaker_notes.md)

The report and slide deck are written around the exact experiment and figure pipeline implemented in this repo. Long runs are configured in the presets; the smoke suite exists to validate the pipeline quickly on CPU-only environments.
