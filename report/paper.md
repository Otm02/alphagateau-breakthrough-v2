# AlphaGateau for Breakthrough: Graph-Based Self-Play and Cross-Board Transfer

**Athmane Benarous, Farooq Khan, Andrew Saffar**

## Abstract

We adapt AlphaGateau from chess to the abstract strategy game Breakthrough using a custom JAX
environment, a board-size-invariant graph encoder, and an AlphaZero-style self-play training loop.
We compare a graph neural network (GNN) against a convolutional baseline (CNN) on 8×8 Breakthrough
over 40 self-play iterations, and also study knowledge transfer from 5×5 pretraining to 8×8
fine-tuning. The CNN converges cleanly to 100% greedy-evaluation win rate by iteration 40 while the
GNN plateaus at 42% with high variance throughout training. Transfer from 5×5 pretraining yields a
strong early benefit — 67% win rate at iteration 5 versus 17% from scratch — but does not close the
long-run gap with the CNN baseline. In a direct 12-game match, the CNN wins 10 of 12 games (83%).
*Roles*: Athmane Benarous led the training infrastructure and cluster orchestration; Farooq Khan led
experiments and analysis; Andrew Saffar led graph encoding and evaluation tooling.

## Introduction

Breakthrough is a useful reinforcement-learning benchmark because its rules are simple but its
tactics are strongly shaped by local geometry: pieces only move forward, captures are diagonal, and
promotion to the opponent's back rank ends the game. That structure makes the game a natural
candidate for graph-based state representations, where squares are nodes and directional moves are
edges, rather than for purely image-like encodings.

Our project addresses two questions:

- Does an AlphaGateau-style GNN outperform a simpler CNN baseline on 8×8 Breakthrough under a
  course-feasible self-play budget?
- Does the GNN's board-size-invariant representation make 5×5 → 8×8 transfer meaningful and
  useful?

To answer both, we built a complete pipeline: JAX environment, graph construction, MCTS self-play,
checkpointing with resume support, evaluation, figure generation, and Slurm orchestration for the
McGill course cluster. We report a **negative result** on the first question and a **qualified
positive result** on the second.

## Background

AlphaZero-style agents combine self-play reinforcement learning with Monte Carlo Tree Search (MCTS).
A neural network predicts a policy over actions and a scalar value estimate; MCTS then refines
action selection during play, and the search distribution becomes a policy target for training. We
use Gumbel MCTS from the `mctx` library, which selects actions by differentiably perturbing policy
logits with Gumbel noise.

AlphaGateau replaces the convolutional encoder in AlphaZero with a graph neural network using graph
attention (GAT) with residual message-passing layers, aiming to better represent the relational
structure of board games.

Breakthrough is especially attractive for this idea because the same move rules apply on 5×5 and
8×8 boards. A fixed-size CNN does not naturally support cross-board transfer. A graph encoder can
keep the same feature schema and architecture while only changing the number of nodes and edges.

## Methodology

### Environment and action space

We implemented `BreakthroughEnv(board_size)` in JAX. Actions use the fixed encoding:

```
action = from_square * 3 + move_type
```

where `move_type` ∈ {`forward`, `diag_left`, `diag_right`} in the canonical player-to-move frame.
We canonicalize the board so the player to move always advances upward, keeping both the tensor and
graph interfaces board-size agnostic.

### Representations

The CNN baseline uses a three-channel tensor (self pieces, opponent pieces, empty squares) fed into
a residual convolutional network. The GNN constructs a graph with one node per square and directed
move edges for both players. Node features encode occupancy, normalized coordinates, and
edge-of-board indicators; edge features encode move direction, owner, legality, capture structure,
and goal-rank information.

### Models and training

Both models use hidden size 64 and four residual layers. The GNN adds attention pooling over node
embeddings to produce a global representation.

Training hyperparameters:

| Setting | 8×8 scratch | 5×5 pretrain | 8×8 fine-tune |
|---|---|---|---|
| Iterations | 40 | 40 | 30 |
| Self-play games/iter | 64 | 128 | 64 |
| MCTS simulations | 32 | 32 | 32 |
| Replay window | 50 000 | 50 000 | 50 000 |
| Batch size | 32 | 32 | 32 |
| Learning rate | 1e-3 | 1e-3 | 1e-3 |
| Move cap (plies) | 192 | 96 | 192 |

### Evaluation

Every five iterations we play 12 games against a greedy evaluator and record the win rate. After
training we run a 12-game head-to-head match between the final GNN and CNN scratch checkpoints.
We also record zero-shot 8×8 performance of the 5×5 pretrained model before any fine-tuning.

## Results

All four experiments ran to completion.

| Run | Iters | Peak win rate | Final win rate |
| --- | ---: | --- | --- |
| GNN 8×8 scratch | 40 | 0.67 @ iter 20 | 0.42 @ iter 40 |
| CNN 8×8 scratch | 40 | 1.00 @ iter 40 | 1.00 @ iter 40 |
| GNN 5×5 pretrain | 40 | 1.00 @ iter 20 | 1.00 @ iter 40 |
| GNN 8×8 fine-tune | 30 | 0.75 @ iter 15 | 0.67 @ iter 30 |

### Scratch comparison

The CNN and GNN both start near zero win rate but diverge sharply from iteration 15 onward. The CNN
climbs monotonically, reaching 0.58 at iteration 15, 0.92 at iteration 25, and 1.00 at iteration
40. The GNN peaks at 0.67 at iteration 20, then oscillates between 0.25 and 0.58 for the rest of
training, ending at 0.42. The policy loss confirms this: the CNN's loss descends steadily to 1.97
by iteration 40, while the GNN's plateaus at ≈2.05 from iteration 36 onward.

Head-to-head: the CNN wins 10/12 games (83%), the GNN wins 2/12 (17%).

### Transfer learning

The 5×5 pretrained GNN converges cleanly, reaching 1.00 win rate by iteration 20. Applied
zero-shot to 8×8, it achieves only 17% (2/12), showing that the 5×5 policy does not transfer
directly.

Fine-tuning from the 5×5 checkpoint gives a **substantial early advantage**: 67% win rate at
iteration 5 versus 17% for the scratch GNN — a 4× improvement. By iteration 30, the fine-tuned
model holds 67% versus 58% for the scratch GNN, a persistent but narrower lead. Neither GNN
variant approaches the CNN's 75% at iteration 30.

## Discussion

### Systems outcome

The project delivers a complete, reproducible AlphaZero-style Breakthrough stack with graph and
convolutional models, resumable self-play, automated figure generation, and Slurm-native cluster
wrappers. The graph encoder is genuinely board-size invariant and the transfer experiment runs
end-to-end.

### Why the CNN outperforms the GNN

On a regular grid, convolutions share weights across all positions and efficiently capture the
local, directional patterns that dominate Breakthrough. The GNN introduces additional parameters
through attention pooling and edge-feature MLPs, making optimisation harder under a fixed budget.
The GNN policy-loss plateau suggests convergence to a suboptimal local minimum. A larger budget,
decaying learning rate, or simplified GNN architecture might close this gap.

### Transfer benefit and limits

Transfer from 5×5 pretraining produces a real, 4× early-training advantage. This confirms that the
GNN's board-size-invariant representation supports meaningful cross-board knowledge transfer.
However, the benefit is largest early in fine-tuning and shrinks over iterations, suggesting the
pretrained features provide a good initialisation rather than a compounding structural advantage.

### Evaluation noise

Each win-rate data point covers 12 games, giving a standard deviation of ≈0.14 for p ≈ 0.5. The
oscillations in the GNN curves are consistent with this noise level. Future runs should increase
eval games to 50–100 per checkpoint.

## Conclusion and Future Work

We successfully adapted AlphaGateau to Breakthrough, built a complete reproducible pipeline, and
ran all four planned experiments to completion. The main findings are: (1) the CNN outperforms the
GNN from scratch under the current 40-iteration budget; (2) transfer from 5×5 pretraining
meaningfully accelerates early 8×8 learning but does not overcome the GNN's convergence
disadvantage versus CNN.

Promising future directions: longer fine-tuning budget (60–80 iterations); GNN learning-rate
schedule tuning; ablations on attention pooling and edge features; increasing eval games to reduce
win-rate noise; and repeating with the updated 256-ply cap.

## Contributions and Acknowledgements

**Contributions**

- Athmane Benarous: JAX environment, training pipeline, Slurm orchestration, report integration
- Farooq Khan: CNN baseline experiments, result analysis, presentation assets
- Andrew Saffar: graph encoding, evaluation tooling, write-up support

**Acknowledgements**

We thank the AlphaGateau project for the architectural reference and build on JAX, Flax, Optax,
Jraph, MCTX, and the McGill `mimi` GPU infrastructure.

## References

1. D. Silver et al. *A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go
   through Self-Play*. Science, 2018.
2. I. Antonoglou, T. Hubert, A. Guez et al. *Policy Improvement by Planning with Gumbel*. ICLR,
   2022.
3. A. Kulenkampff. *Enhancing Chess Reinforcement Learning with Graph Representation
   (AlphaGateau)*. 2024.
4. P. Veličković et al. *Graph Attention Networks*. ICLR, 2018.
5. Breakthrough (board game): <https://en.wikipedia.org/wiki/Breakthrough_(board_game)>
6. AlphaGateau repository: <https://github.com/akulen/AlphaGateau>
