#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = REPO_ROOT / "artifacts" / "experiments"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "report" / "figures"

PIPELINES = {
    "direct_transfer": [
        ("5x5", "gnn_5x5_pretrain_cosine"),
        ("8x8", "gnn_8x8_finetune_cosine"),
    ],
    "progressive_transfer": [
        ("5x5", "gnn_5x5_pretrain_cosine_cat_forg"),
        ("6x6", "gnn_6x6_finetune_cosine_cat_forg"),
        ("8x8", "gnn_8x8_finetune_cosine_cat_forg"),
    ],
}

STAGE_COLORS = {
    "5x5": "#f6efe3",
    "6x6": "#e7f2ff",
    "8x8": "#edf7ed",
}


def load_metrics(exp_folder: Path) -> pd.DataFrame:
    metrics_path = exp_folder / "metrics.jsonl"
    records = []
    with metrics_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    required_cols = ["iteration", "policy_loss", "greedy_win_rate"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {metrics_path}")
    return df.sort_values("iteration").reset_index(drop=True)


def build_pipeline_dataframe(base_dir: Path, stages: list[tuple[str, str]]):
    dfs = []
    transition_points = []
    stage_ranges = []
    offset = 0

    for idx, (board_size, folder_name) in enumerate(stages):
        exp_path = base_dir / folder_name
        df = load_metrics(exp_path).copy()
        df["local_iteration"] = df["iteration"]
        df["global_iteration"] = df["iteration"] + offset
        df["board_size"] = board_size
        df["experiment"] = folder_name
        dfs.append(df)

        stage_length = int(df["iteration"].max())
        stage_start = offset
        stage_end = offset + stage_length
        stage_ranges.append((board_size, stage_start, stage_end))
        offset = stage_end
        if idx < len(stages) - 1:
            transition_points.append(offset)

    full_df = pd.concat(dfs, ignore_index=True)
    return full_df, transition_points, stage_ranges


def add_stage_markers(ax, stage_ranges: list[tuple[str, int, int]], transition_points: list[int]) -> None:
    for board_size, start, end in stage_ranges:
        ax.axvspan(start, end, color=STAGE_COLORS.get(board_size, "#f2f2f2"), alpha=0.35)
        midpoint = start + (end - start) / 2
        ax.text(
            midpoint,
            1.02,
            board_size,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    for x in transition_points:
        ax.axvline(x=x, linestyle="--", linewidth=1.5, alpha=0.7, color="#4c78a8")
        ax.text(
            x,
            1.0,
            f"transition @ {x}",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="right",
            va="top",
            fontsize=8,
            color="#4c78a8",
        )


def save_pipeline_plot(
    *,
    df: pd.DataFrame,
    transition_points: list[int],
    stage_ranges: list[tuple[str, int, int]],
    metric_col: str,
    ylabel: str,
    title: str,
    filename: str,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    clean_df = df.dropna(subset=[metric_col])
    ax.plot(
        clean_df["global_iteration"],
        clean_df[metric_col],
        linewidth=2.2,
        marker="o",
        markersize=4,
        color="#1f77b4",
    )
    add_stage_markers(ax, stage_ranges, transition_points)
    ax.set_xlabel("Global Training Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if metric_col == "greedy_win_rate":
        ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.99,
        0.02,
        "Each checkpoint win rate uses 12 games.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def save_eight_by_eight_comparison(
    *,
    direct_df: pd.DataFrame,
    progressive_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    direct_8x8 = direct_df[direct_df["board_size"] == "8x8"].dropna(subset=["greedy_win_rate"])
    progressive_8x8 = progressive_df[progressive_df["board_size"] == "8x8"].dropna(subset=["greedy_win_rate"])

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(
        direct_8x8["local_iteration"],
        direct_8x8["greedy_win_rate"],
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="Direct 5x5 -> 8x8",
    )
    ax.plot(
        progressive_8x8["local_iteration"],
        progressive_8x8["greedy_win_rate"],
        marker="s",
        linewidth=2,
        color="#d62728",
        label="Progressive 5x5 -> 6x6 -> 8x8",
    )
    ax.set_title("8x8 Fine-Tuning Comparison")
    ax.set_xlabel("8x8 Local Iteration")
    ax.set_ylabel("Greedy Win Rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.text(
        0.99,
        0.02,
        "Final win rates: direct 0.83, progressive 0.58",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    output_path = output_dir / "gnn_progressive_transfer_8x8_comparison.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot stagewise transfer figures for the Breakthrough report."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    direct_df, direct_transitions, direct_stage_ranges = build_pipeline_dataframe(
        args.base_dir,
        PIPELINES["direct_transfer"],
    )
    progressive_df, progressive_transitions, progressive_stage_ranges = build_pipeline_dataframe(
        args.base_dir,
        PIPELINES["progressive_transfer"],
    )

    save_pipeline_plot(
        df=direct_df,
        transition_points=direct_transitions,
        stage_ranges=direct_stage_ranges,
        metric_col="greedy_win_rate",
        ylabel="Greedy Win Rate",
        title="Direct Transfer Pipeline (5x5 -> 8x8): Win Rate Evolution",
        filename="gnn_5x5_to_8x8_winrate.png",
        output_dir=output_dir,
    )
    save_pipeline_plot(
        df=direct_df,
        transition_points=direct_transitions,
        stage_ranges=direct_stage_ranges,
        metric_col="policy_loss",
        ylabel="Policy Loss",
        title="Direct Transfer Pipeline (5x5 -> 8x8): Policy Loss Evolution",
        filename="gnn_5x5_to_8x8_policy_loss.png",
        output_dir=output_dir,
    )
    save_pipeline_plot(
        df=progressive_df,
        transition_points=progressive_transitions,
        stage_ranges=progressive_stage_ranges,
        metric_col="greedy_win_rate",
        ylabel="Greedy Win Rate",
        title="Progressive Transfer Pipeline (5x5 -> 6x6 -> 8x8): Win Rate Evolution",
        filename="gnn_5x5_6x6_8x8_progressive_winrate.png",
        output_dir=output_dir,
    )
    save_pipeline_plot(
        df=progressive_df,
        transition_points=progressive_transitions,
        stage_ranges=progressive_stage_ranges,
        metric_col="policy_loss",
        ylabel="Policy Loss",
        title="Progressive Transfer Pipeline (5x5 -> 6x6 -> 8x8): Policy Loss Evolution",
        filename="gnn_5x5_6x6_8x8_progressive_policy_loss.png",
        output_dir=output_dir,
    )
    save_eight_by_eight_comparison(
        direct_df=direct_df,
        progressive_df=progressive_df,
        output_dir=output_dir,
    )
    print("All stagewise transfer figures generated successfully.")


if __name__ == "__main__":
    main()
