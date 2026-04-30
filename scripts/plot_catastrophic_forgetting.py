import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# =========================
# CONFIG
# =========================
BASE_DIR = Path(r"C:\MCGILL_CODE\RL\ALPHAGATEAU-BREAKTHROUGH-V2\ARTIFACTS\EXPERIMENTS")
OUTPUT_DIR = Path(r"C:\mcgill_code\rl\ALPHAGATEAU-BREAKTHROUGH-V2\report\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Pipelines
PIPELINES = {
    "5x5_to_8x8": [
        ("5x5", "gnn_5x5_pretrain_cosine"),
        ("8x8", "gnn_8x8_finetune_cosine"),
    ],
    "5x5_to_6x6_to_8x8_cat_forg": [
        ("5x5", "gnn_5x5_pretrain_cosine_cat_forg"),
        ("6x6", "gnn_6x6_finetune_cosine_cat_forg"),
        ("8x8", "gnn_8x8_finetune_cosine_cat_forg"),
    ],
}


# =========================
# HELPERS
# =========================
def load_metrics(exp_folder: Path) -> pd.DataFrame:
    """
    Load metrics.jsonl from one experiment folder.
    """
    metrics_path = exp_folder / "metrics.jsonl"

    records = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    df = pd.DataFrame(records)

    required_cols = ["iteration", "policy_loss", "greedy_win_rate"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {metrics_path}")

    return df.sort_values("iteration").reset_index(drop=True)


def build_pipeline_dataframe(stages):
    """
    Concatenate multiple training stages into one continuous timeline.
    Returns:
        full_df
        transition_points (list of x positions where board size changes)
        stage_labels (list of labels)
    """
    dfs = []
    transition_points = []
    stage_labels = []

    offset = 0

    for idx, (board_size, folder_name) in enumerate(stages):
        exp_path = BASE_DIR / folder_name
        df = load_metrics(exp_path).copy()

        # Preserve local iteration
        df["local_iteration"] = df["iteration"]

        # Shift to global timeline
        df["global_iteration"] = df["iteration"] + offset
        df["board_size"] = board_size
        df["experiment"] = folder_name

        dfs.append(df)
        stage_labels.append(board_size)

        # Add vertical transition line after this stage (except last)
        stage_length = df["iteration"].max()
        offset += stage_length

        if idx < len(stages) - 1:
            transition_points.append(offset)

    full_df = pd.concat(dfs, ignore_index=True)

    return full_df, transition_points, stage_labels


def add_transition_lines(ax, transition_points):
    """
    Draw vertical dashed lines where board sizes change.
    """
    for x in transition_points:
        ax.axvline(
            x=x,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )


def save_plot(df, transition_points, metric_col, ylabel, title, filename):
    """
    Save one plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(
        df["global_iteration"],
        df[metric_col],
        linewidth=2,
        label=metric_col,
    )

    add_transition_lines(plt.gca(), transition_points)

    plt.xlabel("Training Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


# =========================
# MAIN
# =========================
def main():
    # -------- Pipeline 1: 5x5 -> 8x8 --------
    pipeline1_df, pipeline1_transitions, _ = build_pipeline_dataframe(
        PIPELINES["5x5_to_8x8"]
    )

    save_plot(
        df=pipeline1_df.dropna(),
        transition_points=pipeline1_transitions,
        metric_col="greedy_win_rate",
        ylabel="Greedy Win Rate",
        title="GNN Transfer Pipeline (5x5 -> 8x8): Win Rate Evolution",
        filename="gnn_5x5_to_8x8_winrate.png",
    )

    save_plot(
        df=pipeline1_df,
        transition_points=pipeline1_transitions,
        metric_col="policy_loss",
        ylabel="Policy Loss",
        title="GNN Transfer Pipeline (5x5 -> 8x8): Policy Loss Evolution",
        filename="gnn_5x5_to_8x8_policy_loss.png",
    )

    # -------- Pipeline 2: 5x5 -> 6x6 -> 8x8 --------
    pipeline2_df, pipeline2_transitions, _ = build_pipeline_dataframe(
        PIPELINES["5x5_to_6x6_to_8x8_cat_forg"]
    )
    
    save_plot(
        df=pipeline2_df.dropna(),
        transition_points=pipeline2_transitions,
        metric_col="greedy_win_rate",
        ylabel="Greedy Win Rate",
        title="GNN Progressive Transfer Pipeline (5x5 -> 6x6 -> 8x8): Win Rate Evolution",
        filename="gnn_5x5_6x6_8x8_cat_forg_winrate.png",
    )

    save_plot(
        df=pipeline2_df,
        transition_points=pipeline2_transitions,
        metric_col="policy_loss",
        ylabel="Policy Loss",
        title="GNN Progressive Transfer Pipeline (5x5 -> 6x6 -> 8x8): Policy Loss Evolution",
        filename="gnn_5x5_6x6_8x8_cat_forg_policy_loss.png",
    )

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
