# AlphaGateau for Breakthrough: A Graph-Based AlphaZero-Style Agent With Cross-Board Transfer

**Athmane Benarous, Farooq Khan, Andrew Saffar**

## Abstract

We adapt AlphaGateau, a graph-neural reinterpretation of AlphaZero, from chess to the abstract strategy game Breakthrough. Our implementation introduces a custom JAX Breakthrough environment, a board-size-invariant graph encoder, an AlphaGateau-style GNN policy/value network, and a simpler AlphaZero-style CNN baseline on 8x8 boards. The main experimental question is whether graph structure helps both direct learning on 8x8 Breakthrough and transfer from 5x5 pretraining to 8x8 fine-tuning. The repository supports the full planned training schedule, automated evaluation, Elo estimation with weighted least squares, figure generation, and a Slurm-native experiment DAG for the McGill department GPU allocation. Team-member contributions should be finalized before submission.

## Introduction

Breakthrough is a clean adversarial benchmark for reinforcement learning because its rules are simple while its tactics are highly local and geometry-dependent. Every move is a one-step forward advance or diagonal advance, and winning requires either reaching the opponent’s back rank or otherwise forcing a decisive terminal state. This makes the game a natural target for graph-based state representations: legal moves define directed relationships between squares, and those relationships are more structured than a raw image-like grid.

AlphaZero-style systems traditionally use convolutional networks over board tensors, but AlphaGateau argues that graph representations can encode board games more naturally while also improving parameter sharing across board sizes. Our project tests that hypothesis in Breakthrough by re-implementing the full self-play, MCTS, policy-value learning loop on top of a custom environment rather than the chess-specific PGX stack used by the upstream AlphaGateau code.

## Background

AlphaZero combines self-play reinforcement learning with Monte Carlo Tree Search (MCTS). A neural network predicts both a policy over legal actions and a scalar state value, and MCTS uses those predictions to improve search-time action selection. The search output becomes the policy target for training, while the eventual game outcome becomes the value target.

AlphaGateau replaces convolutional layers with a graph neural network that updates node and edge features jointly. This matters for Breakthrough because the game is naturally described by squares and move relations. Unlike a fixed 8x8 tensor model, a graph model can preserve a consistent feature schema while the number of squares changes from 5x5 to 8x8. That property is central to our transfer experiment.

## Methodology

### Environment

We implemented a custom `BreakthroughEnv(board_size)` in JAX with the state fields needed by the AlphaZero-style pipeline: board tensor, canonical observation, legal action mask, current player, rewards, and terminal flag. The action space is fixed to:

`action = from_square * 3 + move_type`

where `move_type` is one of `{forward, diag_left, diag_right}` in the current player’s canonical perspective. The environment rotates the board into a canonical view so the side to move always advances in the same direction. This keeps both the graph encoder and the MCTS model interface board-size agnostic.

### State Encodings

The CNN baseline consumes a canonical tensor observation with channels for self pieces, opponent pieces, and empty squares. The GNN uses a graph with one node per square and directed move edges for both players. Node features are:

- self piece occupancy
- opponent piece occupancy
- normalized row index
- normalized column index
- home-row indicator
- goal-row indicator
- left-edge indicator
- right-edge indicator

Edge features include owner identity, legality in the current state, movement direction, straight-vs-diagonal structure, capture flag, and goal-rank flag.

### Models

We use two models:

- **AlphaGateau GNN**: a residual stack of GATEAU-style message-passing blocks over the board graph, with an attention-pooling value head.
- **CNN baseline**: a simpler AlphaZero-style residual convolutional network restricted to 8x8 experiments.

The originally proposed temporal-difference baseline was replaced with this CNN baseline for feasibility and for a cleaner apples-to-apples comparison with the AlphaGateau self-play pipeline.

### Training And Evaluation

Self-play uses Gumbel MuZero search from `mctx`, with the search policy as the policy target and the bootstrapped game outcome as the value target. We track exactly four experiment presets:

- GNN 8x8 scratch
- GNN 5x5 pretraining
- GNN 5x5 -> 8x8 zero-shot + fine-tuning
- CNN 8x8 scratch

The course-feasible default schedules encoded in the repository are:

- hidden size `64`
- `4` residual blocks/layers
- `32` MCTS simulations
- replay window `50k`
- batch size `32`
- checkpoint/eval every `5` iterations
- `128` self-play games and `96` max plies on `5x5`
- `64` self-play games and `192` max plies on `8x8`

Headline metrics are 8x8 win rate and weighted-least-squares Elo from head-to-head matches between checkpoints.

For the full course runs, the intended execution environment is the McGill CS department GPU allocation on `mimi.cs.mcgill.ca`, using Slurm with `partition=all`, `account=winter2026-comp579`, `qos=comp579-1gpu-12h`, `gres=gpu:1`, and `mem=16G`. Because the QoS enforces a 12-hour wall-clock limit, each training run also writes a resumable state bundle containing model parameters, optimizer state, replay buffer contents, RNG state, and accumulated metrics so that retry jobs can continue without restarting from iteration 0.

## Results

The repository contains a full experiment pipeline and a smoke-suite validation path. The smoke suite verifies that:

- the custom Breakthrough environment generates legal actions and terminal outcomes correctly,
- the GNN checkpoint trained on 5x5 can execute on 8x8 without parameter shape changes,
- the self-play, checkpointing, evaluation, Elo, and plotting paths all run end-to-end,
- the three report figures are reproducible from experiment outputs.

For the current repository snapshot, the smoke suite runs two iterations of each preset with reduced self-play and search budgets. Those results are not meant to be the final project numbers, but they do confirm that all tracked experiment paths execute and produce comparable outputs. In the generated smoke summary:

- the 8x8 CNN baseline reached a `0.5` draw rate against the greedy evaluator after two iterations,
- the 8x8 GNN scratch run and the 5x5-pretrained-then-8x8-finetuned GNN both completed successfully and exported checkpoints, evaluation logs, and metrics,
- the GNN-vs-CNN head-to-head validation match ended in `2/2` draws, confirming that the checkpoint loading and match pipeline work across model families.

The intended final comparison is:

- **GNN vs CNN on 8x8 scratch**
- **GNN 8x8 scratch vs GNN 5x5 -> 8x8 transfer**

The figure-generation script produces exactly the three paper figures:

1. `report/figures/gnn_8x8_scratch_curve.png`
2. `report/figures/transfer_curve.png`
3. `report/figures/encoding_visualisation.png`

The Slurm postprocessing stage also materializes `artifacts/experiments/gnn_transfer_summary.json`, `artifacts/experiments/gnn_transfer_zero_shot.json`, `artifacts/experiments/head_to_head_gnn_vs_cnn.json`, and `artifacts/experiments/postprocess_summary.json` alongside the per-run checkpoint directories.

These figures are designed to summarize both learning dynamics and the board/graph encoding used by the model. In a final course run, the 40/40/30/40-iteration schedule should be executed through the Slurm submitter before exporting the final PDF, but the smoke-generated figures already validate the reporting pipeline end-to-end.

## Discussion

The main engineering result of this project is that AlphaGateau’s graph formulation transfers cleanly to Breakthrough once the environment and graph-builder assumptions are decoupled from chess. The GNN path is genuinely board-size invariant: only the number of nodes and edges changes, while the feature schema and learned parameters remain fixed. This makes the 5x5-to-8x8 transfer experiment well-defined in a way that the CNN baseline is not.

The main practical tradeoff is compute. Even with a course-feasible schedule, self-play search dominates runtime. For that reason, the repository includes both smoke-scale commands for quick validation and a Slurm-native execution path for the actual project experiments, including retry chains that resume from saved training state across the 12-hour QoS boundary.

## Conclusion And Future Work

We implemented a complete AlphaGateau-style Breakthrough project with a custom JAX environment, graph encoder, MCTS training loop, 8x8 CNN baseline, Elo evaluation, submission assets, and a reproducible Slurm workflow for the department GPUs. The central hypothesis remains that graph structure should improve both 8x8 learning efficiency and 5x5-to-8x8 transfer relative to a simpler grid baseline. The code now supports a clean empirical test of that hypothesis.

Natural next steps include larger runs, stronger search budgets, heuristic or handcrafted baselines, curriculum variants beyond 5x5, and ablations on edge features or attention pooling. A particularly useful follow-up would be to compare transfer from 5x5 pretraining against same-budget 8x8 scratch training under matched wall-clock limits.

## Contributions And Acknowledgements

### Contributions

- **Athmane Benarous**: environment, training pipeline, and report integration.
- **Farooq Khan**: baseline experiments, analysis, and presentation material.
- **Andrew Saffar**: graph encoding, evaluation pipeline, and writeup support.

These role assignments are placeholders and should be updated to match the final division of labor before submission.

### Acknowledgements

We build on the public AlphaGateau chess implementation as an architectural reference and on JAX, Flax, Optax, Jraph, and MCTX for the underlying RL and GNN stack.

## References

1. L. Victor Allis. *A Knowledge-based Approach of Connect-Four*. 1988.
2. A. Kulenkampff. *Enhancing Chess Reinforcement Learning with Graph Representation (AlphaGateau)*. 2024.
3. Breakthrough rules overview: <https://en.wikipedia.org/wiki/Breakthrough_(board_game)>
4. P. Veličković et al. *Graph Attention Networks*. 2017.
5. D. Silver et al. *A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go through Self-Play*. 2017.
6. AlphaGateau repository: <https://github.com/akulen/AlphaGateau>
7. D. Silver et al. *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model*. 2019.
