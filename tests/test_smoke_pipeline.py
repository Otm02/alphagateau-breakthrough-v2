from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")

from alphagateau_breakthrough.configs import EXPERIMENT_PRESETS
from alphagateau_breakthrough.evaluation import evaluate_checkpoint_pair
from alphagateau_breakthrough.training import build_config_from_preset, train_experiment


def test_smoke_training_and_evaluation(tmp_path: Path) -> None:
    gnn_config = build_config_from_preset(
        EXPERIMENT_PRESETS["gnn_5x5_pretrain"],
        num_iterations=1,
        selfplay_games=2,
        num_simulations=2,
        max_plies=8,
        eval_games=2,
    )
    cnn_config = build_config_from_preset(
        EXPERIMENT_PRESETS["cnn_8x8_scratch"],
        num_iterations=1,
        selfplay_games=2,
        num_simulations=2,
        max_plies=8,
        eval_games=2,
    )
    gnn_summary = train_experiment(gnn_config, output_root=tmp_path, run_name="smoke_gnn")
    cnn_summary = train_experiment(cnn_config, output_root=tmp_path, run_name="smoke_cnn")
    assert Path(gnn_summary["final_checkpoint"]).is_file()
    assert Path(cnn_summary["final_checkpoint"]).is_file()
    head_to_head = evaluate_checkpoint_pair(
        cnn_summary["final_checkpoint"],
        cnn_summary["final_checkpoint"],
        n_games=2,
        n_sim=2,
        max_plies=8,
        seed=0,
        output_path=tmp_path / "head_to_head.json",
    )
    assert head_to_head["wins"] + head_to_head["draws"] + head_to_head["losses"] == 2
