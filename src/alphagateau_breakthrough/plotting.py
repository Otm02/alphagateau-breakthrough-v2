from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _read_json(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_metrics(run_dir: str | Path) -> list[dict[str, str]]:
    return _read_csv(Path(run_dir) / "metrics.csv")


def _read_greedy_eval_points(run_dir: str | Path) -> tuple[list[int], list[float]]:
    eval_path = Path(run_dir) / "evaluation.csv"
    if eval_path.is_file():
        rows = _read_csv(eval_path)
        if rows:
            return (
                [int(row["iteration"]) for row in rows],
                [float(row["win_rate"]) for row in rows],
            )

    rows = _read_metrics(run_dir)
    filtered = [
        row
        for row in rows
        if row.get("greedy_win_rate") not in (None, "")
    ]
    return (
        [int(row["iteration"]) for row in filtered],
        [float(row["greedy_win_rate"]) for row in filtered],
    )


def _read_zero_shot_win_rate(
    *,
    transfer_zero_shot_path: str | Path | None = None,
    transfer_summary_path: str | Path | None = None,
) -> float | None:
    if transfer_zero_shot_path is not None:
        payload = _read_json(transfer_zero_shot_path)
        if payload is not None and "win_rate" in payload:
            return float(payload["win_rate"])

    if transfer_summary_path is not None:
        payload = _read_json(transfer_summary_path)
        if payload is not None:
            zero_shot = payload.get("zero_shot_8x8")
            if zero_shot is not None and "win_rate" in zero_shot:
                return float(zero_shot["win_rate"])

    return None


def _format_iteration_axis(ax, x_values: list[int]) -> None:
    if len(x_values) == 1:
        value = x_values[0]
        ax.set_xlim(value - 0.5, value + 0.5)
        ax.set_xticks([value])


def plot_scratch_comparison(
    gnn_run_dir: str | Path,
    gnn_cosine_run_dir: str | Path,
    cnn_run_dir: str | Path,
    output_path: str | Path,
) -> None:
    gnn_eval_x, gnn_eval_y = _read_greedy_eval_points(gnn_run_dir)
    gnn_cosine_eval_x, gnn_cosine_eval_y = _read_greedy_eval_points(gnn_cosine_run_dir)
    cnn_eval_x, cnn_eval_y = _read_greedy_eval_points(cnn_run_dir)
    gnn_metrics = _read_metrics(gnn_run_dir)
    gnn_cosine_metrics = _read_metrics(gnn_cosine_run_dir)
    cnn_metrics = _read_metrics(cnn_run_dir)

    gnn_loss_x = [int(row["iteration"]) for row in gnn_metrics]
    gnn_policy_loss = [float(row["policy_loss"]) for row in gnn_metrics]
    gnn_cosine_loss_x = [int(row["iteration"]) for row in gnn_cosine_metrics]
    gnn_cosine_policy_loss = [float(row["policy_loss"]) for row in gnn_cosine_metrics]
    cnn_loss_x = [int(row["iteration"]) for row in cnn_metrics]
    cnn_policy_loss = [float(row["policy_loss"]) for row in cnn_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(gnn_eval_x, gnn_eval_y, marker="o", color="#1f77b4", label="GNN scratch")
    axes[0].plot(gnn_cosine_eval_x, gnn_cosine_eval_y, marker="d", color="#42b41f", label="GNN cosine decay scratch")
    axes[0].plot(cnn_eval_x, cnn_eval_y, marker="s", color="#ff7f0e", label="CNN scratch")
    axes[0].set_title("8x8 Scratch Evaluation")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Win rate vs greedy")
    axes[0].set_ylim(0.0, 1.0)
    _format_iteration_axis(axes[0], sorted(set(gnn_eval_x + gnn_cosine_eval_x + cnn_eval_x)))
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(gnn_loss_x, gnn_policy_loss, color="#1f77b4", label="GNN policy loss")
    axes[1].plot(gnn_cosine_loss_x, gnn_cosine_policy_loss, color="#42b41f", label="GNN cosine decay policy loss")
    axes[1].plot(cnn_loss_x, cnn_policy_loss, color="#ff7f0e", label="CNN policy loss")
    axes[1].set_title("8x8 Scratch Training")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Policy loss")
    _format_iteration_axis(axes[1], sorted(set(gnn_loss_x + gnn_cosine_loss_x + cnn_loss_x)))
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curve(run_dir: str | Path, output_path: str | Path, title: str) -> None:
    rows = _read_metrics(run_dir)
    if not rows:
        raise ValueError(f"No metrics found in {run_dir}.")
    eval_x, eval_y = _read_greedy_eval_points(run_dir)
    loss_x = [int(row["iteration"]) for row in rows]
    policy_loss = [float(row["policy_loss"]) for row in rows]
    value_loss = [float(row["value_loss"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(eval_x, eval_y, marker="o", color="#1f77b4")
    axes[0].set_title(f"{title}\nGreedy Evaluation")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Win rate")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.3)
    axes[1].plot(loss_x, policy_loss, label="Policy loss", color="#d62728")
    axes[1].plot(loss_x, value_loss, label="Value loss", color="#2ca02c")
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
    *,
    transfer_zero_shot_path: str | Path | None = None,
    transfer_summary_path: str | Path | None = None,
    gnn_scratch_dir: str | Path | None = None,
) -> None:
    pre_x, pre_y = _read_greedy_eval_points(pretrain_dir)
    fin_x, fin_y = _read_greedy_eval_points(finetune_dir)
    zero_shot = _read_zero_shot_win_rate(
        transfer_zero_shot_path=transfer_zero_shot_path,
        transfer_summary_path=transfer_summary_path,
    )

    scratch_x: list[int] = []
    scratch_y: list[float] = []
    if gnn_scratch_dir is not None:
        scratch_x, scratch_y = _read_greedy_eval_points(gnn_scratch_dir)
        if fin_x:
            max_iter = max(fin_x)
            scratch_x, scratch_y = zip(
                *[(x, y) for x, y in zip(scratch_x, scratch_y) if x <= max_iter],
                strict=False,
            ) if any(x <= max_iter for x in scratch_x) else ([], [])
            scratch_x = list(scratch_x)
            scratch_y = list(scratch_y)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(pre_x, pre_y, marker="o", color="#ff7f0e")
    axes[0].set_title("5x5 Pretraining")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Win rate vs greedy")
    axes[0].set_ylim(0.0, 1.0)
    _format_iteration_axis(axes[0], pre_x)
    axes[0].grid(alpha=0.3)

    if zero_shot is not None:
        axes[1].scatter([0], [zero_shot], color="#111111", s=60, zorder=5, label="Zero-shot 8x8")
    if fin_x:
        axes[1].plot(fin_x, fin_y, marker="o", color="#1f77b4", label="GNN fine-tune")
    if scratch_x:
        axes[1].plot(scratch_x, scratch_y, marker="s", color="#2ca02c", linestyle="--", label="GNN scratch")
    axes[1].set_title("Transfer To 8x8")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Win rate vs greedy")
    axes[1].set_ylim(0.0, 1.0)
    _format_iteration_axis(axes[1], ([0] if zero_shot is not None else []) + fin_x)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_encoding_visualisation(output_path: str | Path, board_size: int = 8) -> None:
    try:
        from .env import BreakthroughEnv
        from .graph import state_to_graph
    except Exception:
        if Path(output_path).is_file():
            return
        raise

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


def plot_breakthrough_rules(output_path: str | Path, board_size: int = 8) -> None:
    board = np.zeros((board_size, board_size), dtype=int)
    board[:2, :] = 1
    board[-2:, :] = -1

    example = board.copy()
    example[2, 3] = 1
    example[3, 4] = -1
    example[2, 0] = 1

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))

    def draw_board(ax, position: np.ndarray, *, show_moves: bool) -> None:
        for row in range(board_size):
            for col in range(board_size):
                color = "#f3e7c9" if (row + col) % 2 == 0 else "#b08a5a"
                ax.add_patch(plt.Rectangle((col, row), 1, 1, facecolor=color, edgecolor="#4f3d27", linewidth=0.8))
                piece = position[row, col]
                if piece == 1:
                    ax.scatter(col + 0.5, row + 0.5, s=260, facecolor="#fffaf0", edgecolor="#2b2118", linewidth=1.0, zorder=3)
                elif piece == -1:
                    ax.scatter(col + 0.5, row + 0.5, s=260, facecolor="#1e1e1e", edgecolor="#f5f5f5", linewidth=0.8, zorder=3)

        if show_moves:
            arrow = {"arrowstyle": "->", "lw": 1.8, "color": "#111111", "shrinkA": 8, "shrinkB": 8}
            ax.annotate("", xy=(2.5, 3.5), xytext=(3.5, 2.5), arrowprops=arrow, zorder=4)
            ax.annotate("", xy=(3.5, 3.5), xytext=(3.5, 2.5), arrowprops=arrow, zorder=4)
            ax.annotate("", xy=(4.5, 3.5), xytext=(3.5, 2.5), arrowprops=arrow, zorder=4)
            ax.annotate("", xy=(1.5, 3.5), xytext=(0.5, 2.5), arrowprops=arrow, zorder=4)

        files = [chr(ord("a") + idx) for idx in range(board_size)]
        ranks = [str(board_size - idx) for idx in range(board_size)]
        ax.set_xticks(np.arange(board_size) + 0.5, files)
        ax.set_yticks(np.arange(board_size) + 0.5, ranks)
        ax.set_xlim(0, board_size)
        ax.set_ylim(0, board_size)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.tick_params(length=0, labelsize=10)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("#4f3d27")

    draw_board(axes[0], board, show_moves=False)
    axes[0].set_title("Starting Position", fontsize=12)
    draw_board(axes[1], example, show_moves=True)
    axes[1].set_title("Legal White Moves", fontsize=12)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_submission_figures(
    *,
    gnn_scratch_dir: str | Path,
    gnn_cosine_scratch_dir: str | Path,
    cnn_scratch_dir: str | Path | None = None,
    pretrain_dir: str | Path,
    finetune_dir: str | Path,
    transfer_zero_shot_path: str | Path | None = None,
    transfer_summary_path: str | Path | None = None,
    output_dir: str | Path = "report/figures",
) -> None:
    output_dir = _ensure_dir(output_dir)
    if cnn_scratch_dir is None:
        plot_learning_curve(
            gnn_scratch_dir,
            output_dir / "gnn_8x8_scratch_curve.png",
            title="AlphaGateau 8x8 Scratch",
        )
    else:
        plot_scratch_comparison(
            gnn_scratch_dir,
            gnn_cosine_scratch_dir,
            cnn_scratch_dir,
            output_dir / "gnn_8x8_scratch_curve.png",
        )
    plot_transfer_curve(
        pretrain_dir,
        finetune_dir,
        output_dir / "transfer_curve.png",
        transfer_zero_shot_path=transfer_zero_shot_path,
        transfer_summary_path=transfer_summary_path,
        gnn_scratch_dir=gnn_scratch_dir,
    )
    plot_breakthrough_rules(output_dir / "breakthrough_rules.png", board_size=8)
    plot_encoding_visualisation(output_dir / "encoding_visualisation.png", board_size=8)
