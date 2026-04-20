# Speaker Notes

## Slide 1 — Title

This project adapts AlphaGateau — originally proposed for chess — to the board game Breakthrough.
Our goal was to build a full project submission pipeline and use it to answer two concrete
questions: does a GNN outperform a CNN baseline on 8×8, and does the GNN's board-size-invariant
representation make 5×5 → 8×8 transfer useful?

## Slide 2 — Why This Project?

Breakthrough has simple rules but its tactics are strongly geometry-dependent: all pieces move
forward, captures are diagonal, and the first player to reach the opponent's back rank wins. These
properties mean the game is naturally relational — move legality depends on local piece
relationships — which is exactly what graph representations are designed for. The fact that 5×5 and
8×8 share the same rules makes transfer a meaningful experiment, unlike in most other games.

## Slide 3 — What We Built

We built everything from scratch: a JAX Breakthrough environment that supports arbitrary board
sizes, an action encoding that works identically for any size, a graph encoder that maps squares to
nodes and moves to directed edges, and a CNN baseline for comparison. On top of that, we
implemented a full AlphaZero-style training loop with Gumbel MCTS, a replay buffer, and a Slurm
DAG that runs the whole pipeline on the McGill course cluster with automatic retry and resume.

## Slide 4 — Experimental Setup

We ran four experiments to completion. The 5×5 pretraining used 128 games per iteration because
the smaller board means shorter games and lower variance. All other settings were held fixed across
experiments to keep the comparison clean. Evaluation every 5 iterations with 12 games against a
greedy opponent gives a reasonable signal, though with some noise — we discuss that on the last
slide.

## Slide 5 — Main Result: Scratch Comparison

This is our headline result. The CNN converges cleanly: win rate climbs from near zero at iter 5 to
100% at iter 40. The GNN peaks at 67% at iter 20, then oscillates and ends at 42%. The policy loss
curves tell the same story — the CNN's loss decreases steadily to 1.97, while the GNN's plateaus
at about 2.05 from iter 36 onward, indicating it converged to a suboptimal policy. The head-to-head
match between the two final checkpoints confirms this: CNN wins 10 of 12 games.

This is a negative result for our first hypothesis, and it's a meaningful one. It's not that the
GNN failed to learn — it did learn and reached 67% at one point — it's that it couldn't sustain
that level and the CNN pulled far ahead.

## Slide 6 — Transfer Result

Our second hypothesis gets a qualified positive answer. The 5×5 pretrained GNN converges to 100%
win rate on the small board, but applied zero-shot to 8×8, it achieves only 17%. The policy clearly
doesn't transfer directly.

Fine-tuning, however, shows a strong early benefit. At iteration 5, the fine-tuned model is already
at 67% — four times higher than the scratch GNN at the same point. This confirms the board-size-
invariant representation is doing something useful: the pretrained features provide a much better
starting point.

The benefit persists but shrinks. At iter 30, the fine-tuned model is at 67% and the scratch GNN
is at 58% — still better, but the gap has closed. And neither reaches the CNN's 75% at that point.

## Slide 7 — Takeaways

The key tension is this: the GNN has richer inductive biases for relational structure, but
Breakthrough on 8×8 is well-served by the CNN's spatial weight sharing and simpler optimisation
landscape. Our most important next experiment is a longer fine-tuning run — if we give the fine-
tuned GNN 60 or 80 iterations, can it match or beat the CNN? That would tell us whether the
bottleneck is the initialisation or the architecture. Evaluation noise from 12-game win rates is
real and worth fixing — moving to 50 games per checkpoint would reduce the visible oscillations
significantly.
