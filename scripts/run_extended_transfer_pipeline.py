#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")

from alphagateau_breakthrough.configs import EXPERIMENT_PRESETS
from alphagateau_breakthrough.evaluation import evaluate_against_greedy
from alphagateau_breakthrough.models import build_model_manager, load_checkpoint
from alphagateau_breakthrough.training import build_config_from_preset, train_experiment
from alphagateau_breakthrough.utils import write_json
from alphagateau_breakthrough.env import BreakthroughEnv


def build_checkpoint_summary(checkpoint_path: str | Path) -> dict:
    payload = load_checkpoint(checkpoint_path)
    config = payload["config"]
    return {
        "experiment_name": config["experiment_name"],
        "run_name": config["experiment_name"],
        "board_size": config["board_size"],
        "model_type": config["model_type"],
        "final_checkpoint": str(checkpoint_path),
        "final_iteration": int(payload["iteration"]),
        "target_iterations": int(payload["iteration"]),
        "status": "completed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 5x5 pretrain -> 6x6 zero-shot -> 6x6 fine-tune pipeline -> 8x8 zero-shot -> 8x8 fine-tune pipeline.")
    parser.add_argument("--output-root", default="artifacts/experiments")
    parser.add_argument("--pretrained-checkpoint", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--iterations-pretrain", type=int, default=None)
    parser.add_argument("--iterations-finetune", type=int, default=None)
    parser.add_argument("--selfplay-games", type=int, default=None)
    parser.add_argument("--num-simulations", type=int, default=None)
    parser.add_argument("--max-plies-5x5", type=int, default=None)
    parser.add_argument("--max-plies-6x6", type=int, default=None)
    parser.add_argument("--max-plies-8x8", type=int, default=None)
    parser.add_argument("--eval-games", type=int, default=12)
    args = parser.parse_args()

    if args.pretrained_checkpoint:
        pretrain_summary = build_checkpoint_summary(args.pretrained_checkpoint)
        pretrain_payload = load_checkpoint(args.pretrained_checkpoint)
    else:
        pretrain_config = build_config_from_preset(
            EXPERIMENT_PRESETS["gnn_5x5_pretrain_cosine"],
            num_iterations=args.iterations_pretrain,
            selfplay_games=args.selfplay_games,
            num_simulations=args.num_simulations,
            max_plies=args.max_plies_5x5,
            eval_games=args.eval_games,
        )
        pretrain_summary = train_experiment(
            pretrain_config,
            output_root=args.output_root,
            run_name="gnn_5x5_pretrain_cosine",
            resume=args.resume,
        )
        pretrain_payload = load_checkpoint(pretrain_summary["final_checkpoint"])

    env_6x6 = BreakthroughEnv(6)
    zero_shot_model = build_model_manager(
        model_id="gnn_transfer_zero_shot_6x6",
        model_type="gnn",
        board_size=6,
        inner_size=pretrain_payload["config"]["hidden_size"],
        n_res_layers=pretrain_payload["config"]["n_res_layers"],
        attention_pooling=pretrain_payload["config"]["attention_pooling"],
        mix_edge_node=pretrain_payload["config"]["mix_edge_node"],
        add_features=pretrain_payload["config"]["add_features"],
        self_edges=pretrain_payload["config"]["self_edges"],
        simple_update=pretrain_payload["config"]["simple_update"],
        sync_updates=pretrain_payload["config"]["sync_updates"],
    )
    zero_shot_6x6_summary = evaluate_against_greedy(
        env=env_6x6,
        model=zero_shot_model,
        params={"params": pretrain_payload["params"], "batch_stats": pretrain_payload["batch_stats"]},
        n_games=args.eval_games,
        n_sim=args.num_simulations or 32,
        max_plies=args.max_plies_6x6 or 256,
        seed=pretrain_payload["iteration"],
        log_path=Path(args.output_root) / "gnn_transfer_zero_shot_moves_6x6.txt",
    )
    write_json(Path(args.output_root) / "gnn_transfer_zero_shot_6x6.json", zero_shot_6x6_summary)

    finetune_6x6_config = build_config_from_preset(
        EXPERIMENT_PRESETS["gnn_6x6_finetune_cosine"],
        initial_checkpoint=pretrain_summary["final_checkpoint"],
        num_iterations=args.iterations_finetune,
        selfplay_games=args.selfplay_games,
        num_simulations=args.num_simulations,
        max_plies=args.max_plies_6x6,
        eval_games=args.eval_games,
    )
    finetune_6x6_summary = train_experiment(
        finetune_6x6_config,
        output_root=args.output_root,
        run_name="gnn_6x6_finetune_cosine",
        resume=args.resume,
    )

    env_8x8 = BreakthroughEnv(8)
    zero_shot_model = build_model_manager(
        model_id="gnn_transfer_zero_shot",
        model_type="gnn",
        board_size=8,
        inner_size=pretrain_payload["config"]["hidden_size"],
        n_res_layers=pretrain_payload["config"]["n_res_layers"],
        attention_pooling=pretrain_payload["config"]["attention_pooling"],
        mix_edge_node=pretrain_payload["config"]["mix_edge_node"],
        add_features=pretrain_payload["config"]["add_features"],
        self_edges=pretrain_payload["config"]["self_edges"],
        simple_update=pretrain_payload["config"]["simple_update"],
        sync_updates=pretrain_payload["config"]["sync_updates"],
    )
    zero_shot_8x8_summary = evaluate_against_greedy(
        env=env_8x8,
        model=zero_shot_model,
        params={"params": pretrain_payload["params"], "batch_stats": pretrain_payload["batch_stats"]},
        n_games=args.eval_games,
        n_sim=args.num_simulations or 32,
        max_plies=args.max_plies_8x8 or 256,
        seed=pretrain_payload["iteration"],
        log_path=Path(args.output_root) / "gnn_transfer_zero_shot_moves.txt",
    )
    write_json(Path(args.output_root) / "gnn_transfer_zero_shot.json", zero_shot_8x8_summary)

    finetune_8x8_config = build_config_from_preset(
        EXPERIMENT_PRESETS["gnn_8x8_finetune_cosine"],
        initial_checkpoint=pretrain_summary["final_checkpoint"],
        num_iterations=args.iterations_finetune,
        selfplay_games=args.selfplay_games,
        num_simulations=args.num_simulations,
        max_plies=args.max_plies_8x8,
        eval_games=args.eval_games,
    )
    finetune_8x8_summary = train_experiment(
        finetune_8x8_config,
        output_root=args.output_root,
        run_name="gnn_8x8_finetune_cosine",
        resume=args.resume,
    )
    summary = {
        "pretrain": pretrain_summary,
        "zero_shot_6x6": zero_shot_6x6_summary,
        "finetune_6x6": finetune_6x6_summary,
        "zero_shot_8x8": zero_shot_8x8_summary,
        "finetune": finetune_8x8_summary
    }
    write_json(Path(args.output_root) / "gnn_transfer_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
