#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")

from alphagateau_breakthrough.evaluation import evaluate_checkpoint_pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate two checkpoints head-to-head on 8x8 Breakthrough.")
    parser.add_argument("checkpoint_a")
    parser.add_argument("checkpoint_b")
    parser.add_argument("--n-games", type=int, default=12)
    parser.add_argument("--n-sim", type=int, default=32)
    parser.add_argument("--max-plies", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    summary = evaluate_checkpoint_pair(
        args.checkpoint_a,
        args.checkpoint_b,
        n_games=args.n_games,
        n_sim=args.n_sim,
        max_plies=args.max_plies,
        seed=args.seed,
        output_path=args.output,
    )
    print(summary)


if __name__ == "__main__":
    main()
