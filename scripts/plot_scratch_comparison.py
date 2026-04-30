#!/usr/bin/env python3
"""
Plot greedy win rate vs iterations and policy loss vs iterations
for CNN scratch, GNN scratch, and GNN scratch with cosine decay.

Usage:
    python plot_scratch_comparison.py \
        --gnn-scratch-dir   artifacts/experiments/gnn_8x8_scratch \
        --gnn-cosine-dir    artifacts/experiments/gnn_8x8_scratch_cosine \
        --cnn-scratch-dir   artifacts/experiments/cnn_8x8_scratch \
        --output            report/figures/gnn_8x8_scratch_curve.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _read_metrics_jsonl(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / "metrics.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_win_rate(rows: list[dict]) -> tuple[list[int], list[float]]:
    iters, rates = [], []
    for row in rows:
        if row.get("greedy_win_rate") not in (None, ""):
            iters.append(int(row["iteration"]))
            rates.append(float(row["greedy_win_rate"]))
    return iters, rates


def _extract_policy_loss(rows: list[dict]) -> tuple[list[int], list[float]]:
    iters, losses = [], []
    for row in rows:
        if row.get("policy_loss") not in (None, ""):
            iters.append(int(row["iteration"]))
            losses.append(float(row["policy_loss"]))
    return iters, losses


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(
    gnn_scratch_dir: str | Path,
    gnn_cosine_dir: str | Path,
    cnn_scratch_dir: str | Path,
    output_path: str | Path,
) -> None:
    # Load data
    gnn_rows    = _read_metrics_jsonl(gnn_scratch_dir)
    cosine_rows = _read_metrics_jsonl(gnn_cosine_dir)
    cnn_rows    = _read_metrics_jsonl(cnn_scratch_dir)

    gnn_wr_x,    gnn_wr_y    = _extract_win_rate(gnn_rows)
    cosine_wr_x, cosine_wr_y = _extract_win_rate(cosine_rows)
    cnn_wr_x,    cnn_wr_y    = _extract_win_rate(cnn_rows)

    gnn_pl_x,    gnn_pl_y    = _extract_policy_loss(gnn_rows)
    cosine_pl_x, cosine_pl_y = _extract_policy_loss(cosine_rows)
    cnn_pl_x,    cnn_pl_y    = _extract_policy_loss(cnn_rows)

    # Colors matching the example image style
    COLOR_GNN    = "#1f77b4"   # blue
    COLOR_COSINE = "#2ca02c"   # green
    COLOR_CNN    = "#ff7f0e"   # orange

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- Left panel: Win rate ---
    ax = axes[0]
    ax.plot(gnn_wr_x,    gnn_wr_y,    marker="o", color=COLOR_GNN,    linewidth=1.8, markersize=5, label="GNN scratch")
    ax.plot(cosine_wr_x, cosine_wr_y, marker="D", color=COLOR_COSINE, linewidth=1.8, markersize=5, label="GNN cosine decay")
    ax.plot(cnn_wr_x,    cnn_wr_y,    marker="s", color=COLOR_CNN,    linewidth=1.8, markersize=5, label="CNN scratch")
    ax.set_title("8×8 Scratch Evaluation", fontsize=12)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Win rate vs greedy", fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # --- Right panel: Policy loss ---
    ax = axes[1]
    ax.plot(gnn_pl_x,    gnn_pl_y,    color=COLOR_GNN,    linewidth=1.8, label="GNN policy loss")
    ax.plot(cosine_pl_x, cosine_pl_y, color=COLOR_COSINE, linewidth=1.8, label="GNN cosine policy loss")
    ax.plot(cnn_pl_x,    cnn_pl_y,    color=COLOR_CNN,    linewidth=1.8, label="CNN policy loss")
    ax.set_title("8×8 Scratch Training", fontsize=12)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Policy loss", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout(pad=1.5)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 8x8 scratch comparison figure.")
    parser.add_argument("--gnn-scratch-dir",  required=True,
                        help="Path to gnn_8x8_scratch experiment directory")
    parser.add_argument("--gnn-cosine-dir",   required=True,
                        help="Path to gnn_8x8_scratch_cosine experiment directory")
    parser.add_argument("--cnn-scratch-dir",  required=True,
                        help="Path to cnn_8x8_scratch experiment directory")
    parser.add_argument("--output", default="report/figures/gnn_8x8_scratch_curve.png",
                        help="Output image path")
    args = parser.parse_args()

    plot(
        gnn_scratch_dir=args.gnn_scratch_dir,
        gnn_cosine_dir=args.gnn_cosine_dir,
        cnn_scratch_dir=args.cnn_scratch_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
