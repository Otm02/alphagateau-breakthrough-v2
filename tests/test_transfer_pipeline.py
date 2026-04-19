from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "src")

from alphagateau_breakthrough.configs import EXPERIMENT_PRESETS
from alphagateau_breakthrough.training import build_config_from_preset, train_experiment


def test_transfer_pipeline_skips_retraining_when_checkpoint_is_provided(tmp_path: Path) -> None:
    pretrain_root = tmp_path / "pretrained"
    pretrain_config = build_config_from_preset(
        EXPERIMENT_PRESETS["gnn_5x5_pretrain"],
        num_iterations=1,
        selfplay_games=2,
        num_simulations=2,
        max_plies=8,
        eval_games=2,
    )
    pretrain_summary = train_experiment(pretrain_config, output_root=pretrain_root, run_name="gnn_5x5_pretrain")
    assert Path(pretrain_summary["final_checkpoint"]).is_file()

    pipeline_root = tmp_path / "transfer_pipeline"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_transfer_pipeline.py",
            "--output-root",
            str(pipeline_root),
            "--pretrained-checkpoint",
            str(pretrain_summary["final_checkpoint"]),
            "--iterations-finetune",
            "1",
            "--selfplay-games",
            "2",
            "--num-simulations",
            "2",
            "--max-plies-5x5",
            "8",
            "--max-plies-8x8",
            "8",
            "--eval-games",
            "2",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert not (pipeline_root / "gnn_5x5_pretrain").exists()
    assert (pipeline_root / "gnn_8x8_finetune" / "summary.json").is_file()
    assert (pipeline_root / "gnn_transfer_summary.json").is_file()
    zero_shot_path = pipeline_root / "gnn_transfer_zero_shot.json"
    assert zero_shot_path.is_file()
    zero_shot_summary = json.loads(zero_shot_path.read_text(encoding="utf-8"))
    assert {"wins", "draws", "losses"} <= zero_shot_summary.keys()
