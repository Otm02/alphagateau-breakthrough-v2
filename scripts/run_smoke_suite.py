#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")

from alphagateau_breakthrough.configs import EXPERIMENT_PRESETS
from alphagateau_breakthrough.evaluation import evaluate_checkpoint_pair
from alphagateau_breakthrough.plotting import generate_submission_figures
from alphagateau_breakthrough.training import build_config_from_preset, train_experiment
from alphagateau_breakthrough.utils import write_json


def main() -> None:
    output_root = Path("artifacts/smoke_suite")
    summaries = {}
    for preset_name in ["gnn_8x8_scratch", "gnn_5x5_pretrain", "cnn_8x8_scratch"]:
        preset = build_config_from_preset(
            EXPERIMENT_PRESETS[preset_name],
            num_iterations=2,
            selfplay_games=4 if "8x8" in preset_name else 6,
            num_simulations=4,
            max_plies=16 if "8x8" in preset_name else 10,
            eval_games=2,
        )
        summaries[preset_name] = train_experiment(preset, output_root=output_root, run_name=preset_name)

    finetune_config = build_config_from_preset(
        EXPERIMENT_PRESETS["gnn_8x8_finetune"],
        initial_checkpoint=summaries["gnn_5x5_pretrain"]["final_checkpoint"],
        num_iterations=2,
        selfplay_games=4,
        num_simulations=4,
        max_plies=16,
        eval_games=2,
    )
    summaries["gnn_8x8_finetune"] = train_experiment(
        finetune_config,
        output_root=output_root,
        run_name="gnn_8x8_finetune",
    )
    summaries["head_to_head"] = evaluate_checkpoint_pair(
        summaries["gnn_8x8_scratch"]["final_checkpoint"],
        summaries["cnn_8x8_scratch"]["final_checkpoint"],
        n_games=2,
        n_sim=4,
        max_plies=16,
        seed=0,
        output_path=output_root / "head_to_head.json",
    )
    generate_submission_figures(
        gnn_scratch_dir=output_root / "gnn_8x8_scratch",
        pretrain_dir=output_root / "gnn_5x5_pretrain",
        finetune_dir=output_root / "gnn_8x8_finetune",
        output_dir="report/figures",
    )
    write_json(output_root / "smoke_summary.json", summaries)
    print(summaries)


if __name__ == "__main__":
    main()
