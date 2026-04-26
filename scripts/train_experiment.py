#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")

from alphagateau_breakthrough.configs import EXPERIMENT_PRESETS
from alphagateau_breakthrough.training import build_config_from_preset, train_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Breakthrough experiment preset.")
    parser.add_argument("preset", choices=sorted(EXPERIMENT_PRESETS.keys()))
    parser.add_argument("--output-root", default="artifacts/experiments")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--initial-checkpoint", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--selfplay-games", type=int, default=None)
    parser.add_argument("--num-simulations", type=int, default=None)
    parser.add_argument("--max-plies", type=int, default=None)
    parser.add_argument("--eval-games", type=int, default=None)
    parser.add_argument("--lr-schedule", default=None, choices=["constant", "cosine", "step"])
    parser.add_argument("--lr-decay-factor", type=float, default=None)
    parser.add_argument("--lr-warmup-steps", type=int, default=None)
    args = parser.parse_args()

    config = build_config_from_preset(
        EXPERIMENT_PRESETS[args.preset],
        initial_checkpoint=args.initial_checkpoint,
        num_iterations=args.num_iterations,
        selfplay_games=args.selfplay_games,
        num_simulations=args.num_simulations,
        max_plies=args.max_plies,
        eval_games=args.eval_games,
        lr_schedule=args.lr_schedule,
        lr_decay_factor=args.lr_decay_factor,
        lr_warmup_steps=args.lr_warmup_steps,
    )
    summary = train_experiment(
        config,
        output_root=args.output_root,
        run_name=args.run_name,
        resume=args.resume,
    )
    print(summary)


if __name__ == "__main__":
    main()
