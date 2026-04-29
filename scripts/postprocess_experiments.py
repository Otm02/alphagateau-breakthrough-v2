#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from alphagateau_breakthrough.evaluation import evaluate_checkpoint_pair
from alphagateau_breakthrough.plotting import generate_submission_figures
from alphagateau_breakthrough.utils import write_json


def read_json(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def final_checkpoint(run_dir: str | Path) -> Path:
    return Path(run_dir) / "checkpoints" / "final.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run final checkpoint comparison and regenerate report figures from completed experiment directories."
    )
    parser.add_argument("--gnn-scratch-dir", required=True)
    parser.add_argument("--gnn-cosine-scratch-dir", required=True)
    parser.add_argument("--cnn-scratch-dir", required=True)
    parser.add_argument("--pretrain-dir", required=True)
    parser.add_argument("--finetune-dir", required=True)
    parser.add_argument("--transfer-summary", default=None)
    parser.add_argument("--transfer-zero-shot", default=None)
    parser.add_argument("--gnn-vs-cnn-output", default="artifacts/experiments/head_to_head_gnn_vs_cnn.json")
    parser.add_argument("--gnn-cosine-vs-cnn-output", default="artifacts/experiments/head_to_head_gnn_cosine_vs_cnn.json")
    parser.add_argument("--summary-output", default="artifacts/experiments/postprocess_summary.json")
    parser.add_argument("--output-dir", default="report/figures")
    parser.add_argument("--n-games", type=int, default=12)
    parser.add_argument("--n-sim", type=int, default=32)
    parser.add_argument("--max-plies", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    gnn_vs_cnn = evaluate_checkpoint_pair(
        final_checkpoint(args.gnn_scratch_dir),
        final_checkpoint(args.cnn_scratch_dir),
        n_games=args.n_games,
        n_sim=args.n_sim,
        max_plies=args.max_plies,
        seed=args.seed,
        output_path=args.gnn_vs_cnn_output,
    )
    gnn_cosine_vs_cnn = evaluate_checkpoint_pair(
        final_checkpoint(args.gnn_cosine_scratch_dir),
        final_checkpoint(args.cnn_scratch_dir),
        n_games=args.n_games,
        n_sim=args.n_sim,
        max_plies=args.max_plies,
        seed=args.seed,
        output_path=args.gnn_cosine_vs_cnn_output,
    )
    generate_submission_figures(
        gnn_scratch_dir=args.gnn_scratch_dir,
        gnn_cosine_scratch_dir=args.gnn_cosine_scratch_dir,
        cnn_scratch_dir=args.cnn_scratch_dir,
        pretrain_dir=args.pretrain_dir,
        finetune_dir=args.finetune_dir,
        transfer_zero_shot_path=args.transfer_zero_shot,
        transfer_summary_path=args.transfer_summary,
        output_dir=args.output_dir,
    )
    summary = {
        "gnn_scratch": read_json(Path(args.gnn_scratch_dir) / "summary.json"),
        "gnn_cosine_scratch": read_json(Path(args.gnn_cosine_scratch_dir) / "summary.json"),
        "cnn_scratch": read_json(Path(args.cnn_scratch_dir) / "summary.json"),
        "pretrain": read_json(Path(args.pretrain_dir) / "summary.json"),
        "finetune": read_json(Path(args.finetune_dir) / "summary.json"),
        "transfer_summary": read_json(args.transfer_summary) if args.transfer_summary else None,
        "transfer_zero_shot": read_json(args.transfer_zero_shot) if args.transfer_zero_shot else None,
        "gnn_vs_cnn": gnn_vs_cnn,
        "gnn_cosine_vs_cnn": gnn_cosine_vs_cnn,
        "figures": {
            "gnn_8x8_scratch_curve": str(Path(args.output_dir) / "gnn_8x8_scratch_curve.png"),
            "transfer_curve": str(Path(args.output_dir) / "transfer_curve.png"),
            "encoding_visualisation": str(Path(args.output_dir) / "encoding_visualisation.png"),
        },
    }
    write_json(args.summary_output, summary)
    print(summary)


if __name__ == "__main__":
    main()
