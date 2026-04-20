# AlphaGateau for Breakthrough

This repository adapts the AlphaGateau idea from chess to the abstract strategy game
**Breakthrough**. It contains a board-size-invariant GNN, an AlphaZero-style CNN baseline,
a full self-play training pipeline, cross-board transfer experiments, automated figure
generation, and the project paper and presentation.

## Quick start

**No Slurm? Any Linux GPU or notebook works:**

```bash
git clone <repo-url>
cd alphagateau-breakthrough-v2
pip install -e ".[cuda12]"        # or cuda13, or just . for CPU
python scripts/show_runtime_info.py
python scripts/run_smoke_suite.py  # 1-iter sanity check, writes figures
```

**On McGill mimi with the shared team environment:**

```bash
bash scripts/mimi_collect_outputs.sh   # pull existing results
bash scripts/mimi_full_dag.sh          # submit the full pipeline
```

See [RUN_GUIDE.md](RUN_GUIDE.md) for all three supported paths (standalone GPU, personal Slurm
env, shared team Slurm env).

## Repository layout

```
src/alphagateau_breakthrough/   Core library (env, graph, models, training, eval, plotting)
scripts/                        Training CLIs + Slurm submission + mimi wrappers
slurm/                          sbatch job templates
tests/                          Unit tests and end-to-end smoke tests
report/                         Paper (paper.tex / paper.md / paper.pdf) and figures
presentation/                   Slides and speaker notes
notebooks/                      Standalone GPU demo notebook
```

## Experiments

Four presets, all designed to run on a single GPU:

| Preset | Board | Model | Iterations |
|---|---|---|---|
| `gnn_8x8_scratch` | 8×8 | GNN | 40 |
| `cnn_8x8_scratch` | 8×8 | CNN | 40 |
| `gnn_5x5_pretrain` | 5×5 | GNN | 40 |
| `gnn_8x8_finetune` | 8×8 | GNN (from 5×5 ckpt) | 30 |

Shared hyperparameters: hidden size 64, 4 residual layers, 32 MCTS simulations, 50k replay
window, batch 32, LR 1e-3.

Train a single preset (supports `--resume` for interrupted runs):

```bash
python scripts/train_experiment.py gnn_8x8_scratch --resume
```

Run the 5×5 → 8×8 transfer curriculum:

```bash
python scripts/run_transfer_pipeline.py \
  --pretrained-checkpoint artifacts/experiments/gnn_5x5_pretrain/checkpoints/final.pkl \
  --resume
```

Postprocess and regenerate report figures:

```bash
python scripts/postprocess_experiments.py \
  --gnn-scratch-dir  artifacts/experiments/gnn_8x8_scratch \
  --cnn-scratch-dir  artifacts/experiments/cnn_8x8_scratch \
  --pretrain-dir     artifacts/experiments/gnn_5x5_pretrain \
  --finetune-dir     artifacts/experiments/gnn_8x8_finetune \
  --transfer-summary artifacts/experiments/gnn_transfer_summary.json \
  --transfer-zero-shot artifacts/experiments/gnn_transfer_zero_shot.json
```

## Tests

```bash
pytest -q -s tests
```

## Pixi (optional)

If you have [Pixi](https://pixi.sh) installed, a single command sets up the full environment
and runs the tests:

```bash
pixi run test
pixi run smoke
pixi run paper     # compile report/paper.pdf with tectonic
```

Available tasks: `runtime-info`, `test`, `smoke`, `train-gnn-8`, `train-gnn-5`, `train-cnn-8`,
`transfer`, `paper`.

## Key design notes

- **Action encoding**: `from_square × 3 + move_type` where `move_type ∈ {forward, diag_left,
  diag_right}` in the current player's canonical perspective.
- **Board-size invariance**: the GNN uses the same node and edge feature schema on 5×5 and 8×8,
  enabling direct cross-board transfer without any parameter shape changes.
- **Resume support**: training writes `latest_resume.pkl` at every checkpoint, capturing model
  params, optimizer state, replay buffer, RNG state, and accumulated metrics. Any run can be
  resumed after an interruption or cluster timeout.
- **Slurm**: the sbatch templates do not hardcode account or QoS. Pass `--account` / `--qos` to
  `submit_mimi_pipeline.py`, or set `SLURM_ACCOUNT` / `SLURM_QOS` when using the mimi wrappers.

## Submission assets

| File | Description |
|---|---|
| [report/paper.pdf](report/paper.pdf) | Compiled project paper |
| [report/paper.tex](report/paper.tex) | LaTeX source |
| [report/paper.md](report/paper.md) | Markdown draft |
| [presentation/slides.md](presentation/slides.md) | 7-slide presentation script |
| [presentation/speaker_notes.md](presentation/speaker_notes.md) | Speaker notes |
