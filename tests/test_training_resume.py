from __future__ import annotations

import csv
from pathlib import Path

import sys

sys.path.insert(0, "src")

from alphagateau_breakthrough.configs import EXPERIMENT_PRESETS
from alphagateau_breakthrough.training import build_config_from_preset, train_experiment


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_train_experiment_resume_appends_metrics(tmp_path: Path) -> None:
    config_one = build_config_from_preset(
        EXPERIMENT_PRESETS["gnn_5x5_pretrain"],
        num_iterations=1,
        selfplay_games=2,
        num_simulations=2,
        max_plies=8,
        eval_games=2,
    )
    summary_one = train_experiment(config_one, output_root=tmp_path, run_name="resume_case")
    assert summary_one["status"] == "completed"
    rows_after_first_run = read_csv_rows(tmp_path / "resume_case" / "metrics.csv")
    assert [int(row["iteration"]) for row in rows_after_first_run] == [1]

    config_two = build_config_from_preset(
        EXPERIMENT_PRESETS["gnn_5x5_pretrain"],
        num_iterations=2,
        selfplay_games=2,
        num_simulations=2,
        max_plies=8,
        eval_games=2,
    )
    summary_two = train_experiment(config_two, output_root=tmp_path, run_name="resume_case", resume=True)
    assert summary_two["status"] == "completed"
    assert summary_two["final_iteration"] == 2

    rows_after_resume = read_csv_rows(tmp_path / "resume_case" / "metrics.csv")
    assert [int(row["iteration"]) for row in rows_after_resume] == [1, 2]
    status_path = tmp_path / "resume_case" / "status.json"
    assert status_path.is_file()
    final_checkpoint = tmp_path / "resume_case" / "checkpoints" / "final.pkl"
    assert final_checkpoint.is_file()
