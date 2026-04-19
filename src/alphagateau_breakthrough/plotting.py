from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .env import BreakthroughEnv
from .graph import state_to_graph
from .utils import ensure_dir


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def plot_learning_curve(run_dir: str | Path, output_path: str | Path, title: str) -> None:
    rows = _read_csv(Path(run_dir) / "metrics.csv")
    if not rows:
        raise ValueError(f"No metrics found in {run_dir}.")
    iterations = [int(row["iteration"]) for row in rows]
    greedy_win_rate = [float(row.get("greedy_win_rate", 0.0) or 0.0) for row in rows]
    policy_loss = [float(row["policy_loss"]) for row in rows]
    value_loss = [float(row["value_loss"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(iterations, greedy_win_rate, marker="o", color="#1f77b4")
    axes[0].set_title(f"{title}\nGreedy Evaluation")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Win rate")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.3)
    axes[1].plot(iterations, policy_loss, label="Policy loss", color="#d62728")
    axes[1].plot(iterations, value_loss, label="Value loss", color="#2ca02c")
    axes[1].set_title(f"{title}\nTraining Loss")
    axes[1].set_xlabel("Iteration")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_transfer_curve(
    pretrain_dir: str | Path,
    finetune_dir: str | Path,
    output_path: str | Path,
) -> None:
    pretrain_rows = _read_csv(Path(pretrain_dir) / "metrics.csv")
    finetune_rows = _read_csv(Path(finetune_dir) / "metrics.csv")
    pre_x = [int(row["iteration"]) for row in pretrain_rows]
    pre_y = [float(row.get("greedy_win_rate", 0.0) or 0.0) for row in pretrain_rows]
    fin_x = [int(row["iteration"]) for row in finetune_rows]
    fin_y = [float(row.get("greedy_win_rate", 0.0) or 0.0) for row in finetune_rows]
    zero_shot = pre_y[-1] if pre_y else 0.0
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(pre_x, pre_y, marker="o", label="5x5 pretraining", color="#ff7f0e")
    if fin_x:
        ax.scatter([0], [zero_shot], color="#111111", label="Zero-shot 8x8")
        ax.plot(fin_x, fin_y, marker="o", label="8x8 fine-tuning", color="#1f77b4")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Win rate vs greedy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Transfer From 5x5 Pretraining To 8x8 Fine-Tuning")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_encoding_visualisation(output_path: str | Path, board_size: int = 8) -> None:
    env = BreakthroughEnv(board_size)
    state = env.init(np.array([0, 0], dtype=np.uint32))
    graph = state_to_graph(state._board, state.legal_action_mask)
    board = np.array(state._board)
    fig, ax = plt.subplots(figsize=(5, 5))
    for row in range(board_size):
        for col in range(board_size):
            color = "#f0d9b5" if (row + col) % 2 == 0 else "#b58863"
            ax.add_patch(plt.Rectangle((col, row), 1, 1, color=color))
            piece = board[row, col]
            if piece == 1:
                ax.text(col + 0.5, row + 0.5, "W", ha="center", va="center", fontsize=14, color="#111111")
            elif piece == -1:
                ax.text(col + 0.5, row + 0.5, "B", ha="center", va="center", fontsize=14, color="#ffffff")
    senders = np.array(graph.senders[: board_size * board_size * 3])
    receivers = np.array(graph.receivers[: board_size * board_size * 3])
    legal = np.array(graph.edges[: board_size * board_size * 3, 2]) > 0.5
    for sender, receiver, is_legal in zip(senders, receivers, legal):
        if not is_legal:
            continue
        from_row, from_col = divmod(int(sender), board_size)
        to_row, to_col = divmod(int(receiver), board_size)
        ax.annotate(
            "",
            xy=(to_col + 0.5, to_row + 0.5),
            xytext=(from_col + 0.5, from_row + 0.5),
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#1f77b4", "alpha": 0.5},
        )
    ax.set_xlim(0, board_size)
    ax.set_ylim(0, board_size)
    ax.set_xticks(range(board_size + 1))
    ax.set_yticks(range(board_size + 1))
    ax.set_aspect("equal")
    ax.set_title("Canonical Breakthrough Encoding and Legal Edges")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_submission_figures(
    *,
    gnn_scratch_dir: str | Path,
    pretrain_dir: str | Path,
    finetune_dir: str | Path,
    output_dir: str | Path = "report/figures",
) -> None:
    output_dir = ensure_dir(output_dir)
    plot_learning_curve(
        gnn_scratch_dir,
        output_dir / "gnn_8x8_scratch_curve.png",
        title="AlphaGateau 8x8 Scratch",
    )
    plot_transfer_curve(
        pretrain_dir,
        finetune_dir,
        output_dir / "transfer_curve.png",
    )
    plot_encoding_visualisation(output_dir / "encoding_visualisation.png", board_size=8)
