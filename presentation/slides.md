# Slide 1 — Title

**AlphaGateau for Breakthrough: Graph-Based Self-Play With Board-Size Transfer**

- Team members
- Course and project framing
- One-sentence goal: adapt AlphaGateau from chess to Breakthrough and test whether graph structure helps 8x8 learning and 5x5->8x8 transfer

# Slide 2 — Why Breakthrough?

- Simple rules, tactical play, and clean win condition
- Strong fit for graph reasoning because legal moves define directed square-to-square relations
- Good transfer benchmark because 5x5 and 8x8 share rules but not board size

# Slide 3 — What We Built

- Custom JAX Breakthrough environment
- Canonical action space: `from_square * 3 + {forward, diag_left, diag_right}`
- AlphaGateau-style GNN over board graphs
- 8x8 CNN AlphaZero-style baseline
- Self-play + MCTS + checkpointing + Elo pipeline

# Slide 4 — Graph Encoding

- One node per square
- Node features: occupancy, coordinates, board-edge flags
- Directed move edges for both players
- Edge features: owner, legal flag, direction, straight/diagonal, capture, goal-rank
- Show `report/figures/encoding_visualisation.png`

# Slide 5 — Experiments

- GNN 8x8 scratch
- GNN 5x5 pretraining
- GNN zero-shot 5x5->8x8 + fine-tuning
- CNN 8x8 scratch
- Headline comparisons: GNN vs CNN on 8x8, and scratch vs transfer for the GNN

# Slide 6 — Takeaways

- The repo now supports the complete Breakthrough project end-to-end
- The GNN path is board-size invariant, so 5x5 checkpoints transfer directly to 8x8
- Smoke-suite validation confirms environment, training, evaluation, Elo, and plotting
- Next step for final submission: run the full preset schedules and export the final PDF/report figures
