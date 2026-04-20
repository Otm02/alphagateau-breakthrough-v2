from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_mimi_project_full_rejects_tiny_overrides() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.update(
        {
            "NUM_ITERATIONS": "1",
            "SELFPLAY_GAMES": "2",
            "NUM_SIMULATIONS": "2",
            "EVAL_GAMES": "2",
            "MAX_PLIES_5X5": "8",
            "MAX_PLIES_8X8": "8",
            "HEAD_TO_HEAD_GAMES": "2",
            "HEAD_TO_HEAD_SIMULATIONS": "2",
            "HEAD_TO_HEAD_MAX_PLIES": "8",
        }
    )

    completed = subprocess.run(
        ["bash", "scripts/mimi_project.sh", "full"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "refusing to launch scripts/mimi_project.sh full with tiny overrides" in completed.stderr
    assert "mimi_full_validation.sh" in completed.stderr
