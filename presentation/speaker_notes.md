# Speaker Notes

## Slide 1

This project takes AlphaGateau, which was originally proposed for chess, and adapts it to Breakthrough. Our goal was not just to port the model, but to build a complete submission pipeline: environment, training, evaluation, figures, report, and presentation assets.

## Slide 2

Breakthrough is a strong RL project game because the rules are simple but the tactics are non-trivial. It is also well suited to graph representations, since every square is a node and every legal move is naturally an edge. That makes it a good test bed for AlphaGateau’s core claim.

## Slide 3

We replaced the chess-specific PGX assumptions from the upstream code with a local JAX environment for Breakthrough. On top of that, we implemented two learning systems: the AlphaGateau-style graph model and a simpler AlphaZero-style CNN baseline on 8x8. Both use self-play with MCTS and the same general training loop.

## Slide 4

The key methodological idea is the graph encoding. Nodes correspond to squares, and edges correspond to possible directional moves for both players. Because the feature schema is fixed, the same GNN parameters can process 5x5 and 8x8 boards. That is what makes the transfer experiment meaningful.

## Slide 5

Our experiments are organized around four tracked presets. The first compares GNN and CNN learning directly on 8x8 Breakthrough. The second asks whether pretraining on 5x5 gives the GNN a better starting point when it is moved to 8x8. The repository automates training, head-to-head evaluation, Elo estimation, and figure generation for these comparisons.

## Slide 6

The main outcome of the implementation is that the complete Breakthrough project now exists as an executable repo, not just as a proposal. The smoke suite validates the full stack on small runs, and the full course-feasible schedules are encoded as defaults for the final experiment runs.
