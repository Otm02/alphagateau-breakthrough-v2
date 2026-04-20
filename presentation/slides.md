# Slide 1 — Title

**AlphaGateau for Breakthrough: Graph-Based Self-Play and Cross-Board Transfer**

Athmane Benarous · Farooq Khan · Andrew Saffar — COMP-579, McGill University

Goal: adapt AlphaGateau from chess to Breakthrough, compare a GNN vs. a CNN baseline from scratch,
and test whether the board-size-invariant GNN supports useful 5×5 → 8×8 transfer.

---

# Slide 2 — Why This Project?

- Breakthrough is simple to describe but tactically geometry-dependent
- Legal moves define directed square-to-square relations — a natural fit for graphs
- 5×5 and 8×8 share the same rules, so transfer is a meaningful experiment for a
  board-size-invariant model
- Two clear hypotheses to test under a fixed compute budget

---

# Slide 3 — What We Built

- Custom JAX Breakthrough environment with canonical observation
- Action space: `from_square * 3 + {forward, diag_left, diag_right}`
- AlphaGateau-style GNN: one node per square, rich node and edge features, attention pooling
- AlphaZero-style CNN baseline: three-channel residual network on 8×8
- Self-play + Gumbel MCTS (mctx) + replay buffer + Slurm DAG on McGill cluster

---

# Slide 4 — Experimental Setup

Four experiments, all run to completion:

| Experiment | Iters | Self-play games/iter |
|---|---|---|
| GNN 8×8 scratch | 40 | 64 |
| CNN 8×8 scratch | 40 | 64 |
| GNN 5×5 pretrain | 40 | 128 |
| GNN 8×8 fine-tune | 30 | 64 |

Shared settings: hidden size 64, 4 residual layers, 32 MCTS simulations, replay window 50 000,
LR 1e-3. Evaluation: 12 games vs. greedy every 5 iterations.

---

# Slide 5 — Main Result: Scratch Comparison

**CNN clearly outperforms GNN from scratch on 8×8.**

| | CNN 8×8 scratch | GNN 8×8 scratch |
|---|---|---|
| Peak win rate | **1.00** @ iter 40 | 0.67 @ iter 20 |
| Final win rate | **1.00** | 0.42 |
| Policy loss @ iter 40 | **1.97** | 2.05 (plateaued) |

- CNN climbs monotonically from iter 15 onward; GNN oscillates and plateaus
- Head-to-head: **CNN wins 10/12** (83%)
- GNN policy loss flatlines from iter 36 → converged to suboptimal policy

---

# Slide 6 — Transfer Result: 5×5 → 8×8

**Transfer provides a real early benefit but doesn't close the long-run gap.**

| | @ iter 5 | @ iter 30 |
|---|---|---|
| GNN scratch | 0.17 | 0.58 |
| GNN fine-tune (from 5×5) | **0.67** | **0.67** |
| CNN scratch | 0.08 | 0.75 |

- Zero-shot 8×8 (before fine-tuning): **0.17** — policy doesn't transfer directly
- Fine-tuning starts 4× higher than scratch at iter 5 ✓
- Persistent but shrinking advantage over scratch GNN
- Neither GNN variant closes the gap with CNN by iter 30

---

# Slide 7 — Takeaways and Future Work

**Empirical findings:**
- CNN > GNN from scratch under current 40-iter budget (meaningful negative result)
- GNN transfer from 5×5 works early — board-size-invariant representation is useful
- Transfer benefit doesn't compound: fine-tuned GNN still lags CNN

**Why CNN wins:** weight sharing over a regular grid suits Breakthrough's local geometry;
GNN has extra parameters (attention, edge MLPs) that are harder to optimise in 40 iterations.

**Next steps:**
- Longer fine-tuning budget (60–80 iters) — can the fine-tuned GNN match CNN?
- GNN learning-rate schedule tuning
- Ablations: disable attention pooling or edge features
- More eval games (50–100) to reduce win-rate noise
