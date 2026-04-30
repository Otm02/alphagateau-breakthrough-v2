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
```

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
