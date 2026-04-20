from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, "src")

from alphagateau_breakthrough.plotting import _read_greedy_eval_points, _read_zero_shot_win_rate


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_read_greedy_eval_points_prefers_sparse_evaluation_csv(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_csv(
        run_dir / "metrics.csv",
        [
            {"iteration": 1, "policy_loss": 3.0, "value_loss": 1.0, "greedy_win_rate": ""},
            {"iteration": 2, "policy_loss": 2.5, "value_loss": 0.8, "greedy_win_rate": ""},
            {"iteration": 3, "policy_loss": 2.0, "value_loss": 0.6, "greedy_win_rate": 0.75},
        ],
    )
    _write_csv(
        run_dir / "evaluation.csv",
        [
            {
                "iteration": 3,
                "wins": 9,
                "draws": 0,
                "losses": 3,
                "win_rate": 0.75,
                "draw_rate": 0.0,
                "loss_rate": 0.25,
            }
        ],
    )

    assert _read_greedy_eval_points(run_dir) == ([3], [0.75])


def test_read_zero_shot_win_rate_uses_actual_transfer_json(tmp_path: Path) -> None:
    zero_shot_path = tmp_path / "gnn_transfer_zero_shot.json"
    zero_shot_path.write_text('{"win_rate": 0.125}', encoding="utf-8")
    summary_path = tmp_path / "gnn_transfer_summary.json"
    summary_path.write_text('{"zero_shot_8x8": {"win_rate": 0.5}}', encoding="utf-8")

    assert _read_zero_shot_win_rate(
        transfer_zero_shot_path=zero_shot_path,
        transfer_summary_path=summary_path,
    ) == 0.125
