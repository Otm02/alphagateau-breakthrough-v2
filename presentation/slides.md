# Learning the game Breakthrough

Athmane Benarous · Farooq Khan · Andrew Saffar — COMP-579, McGill University

More details presented in our report.

---

# Goal
- Breakthrough is a 2 player game played on a chess-like board
  - It is simple to describe but tactically geometry-dependent

- Compare a GNN vs. a CNN baseline from scratch
- Test whether GNN benefits from transfer learning between board sizes

---

# Motivation

- AlphaGateau implemented GNN architecture for chess and was successful with low compute ressources
  - Can we achieve similar results on Breakthrough?
- Different board sizes share the same rules → transfer learning could be beneficial

---

# Experimental Setup

- CNN & GNN from scratch on 5x5 and 8x8 boards
  - Also implemented cosine decay learning rate schedule for GNN
- Transfer learning on GNN with fixed and decaying LRs: 
  - 5x5 → 8x8 
  - 5x5 → 6x6 → 8x8 
--
## Evaluation

- Models evaluated against a greedy version of themselves
- Head-to-head games:
  - GNN vs CNN
  - GNN (cosine) vs CNN

---

# Scratch Comparison

**CNN outperforms GNN from scratch on 8×8.**

| | CNN 8×8 scratch | GNN 8×8 scratch | GNN w/ cosine decay 8×8 scratch |
|---|---|---|---|
| Peak win rate | **1.00** @ iter 40 | 0.67 @ iter 20 | 0.58 @ iter 20 (still improving) |
| Final win rate | **1.00** | 0.42 | 0.58 |
| Policy loss @ iter 40 | 1.97 | 2.05 (plateaued) | **1.79** |

--
- Head-to-head: **CNN wins 10/12** (83%) against fixed LR GNN
  - GNN policy loss flatlines from iter 36 → converged to suboptimal policy
- Adding a cosine decay LR scheduler avoids the plateau and achieves best policy loss
  - Needs more iterations to converge

---

# Transfer Results:

**Transfer learning with GNN and cosine decay outperforms CNN.**

| | @ iter 5 | @ iter 30 |
|---|---|---|
| GNN scratch | 0.17 | 0.58 |
| GNN fine-tune (from 5×5) | 0.67 | 0.67 |
| GNN w/ cosine decay scratch | 0.25 | 0.25 |
| GNN w/ cosine decay fine-tune (from 5×5) | 0.75 | 0.83 |
| GNN w/ cosine decay fine-tune (from 5×5 → 6x6) | **0.92** | **TBD** |
| CNN scratch | 0.08 | 0.75 |
--
- Zero-shot 8×8 (before fine-tuning, for both fixed and decaying LR): **0.17** — policy doesn't transfer directly
- Fine-tuning starts 3-4× higher than scratch at iter 5
- Persistent but shrinking advantage over scratch GNN
- Fine-tuned GNN with cosine decay surpasses CNN by iter 30
- **5x5 → 6x6 → 8x8 pipeline promises to achieve 100% win rate**

---

# Takeaways

- CNN > GNN head-to-head under current 40-iter budget (meaningful negative result)
- We believe GNN with transfer learning and cosine decay will beat CNN head-to-head
- Confirms GNN is a good architecture for low compute budget
