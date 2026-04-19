from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_submit_mimi_pipeline_dry_run_writes_expected_manifest(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = tmp_path / "pipeline_manifest.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/submit_mimi_pipeline.py",
            "--dry-run",
            "--manifest-path",
            str(manifest_path),
            "--max-attempts",
            "2",
            "--num-iterations",
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
            "--head-to-head-games",
            "2",
            "--head-to-head-simulations",
            "2",
            "--head-to-head-max-plies",
            "8",
        ],
        check=True,
        cwd=repo_root,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dry_run"] is True
    assert len(manifest["jobs"]) == 9
    lanes = {}
    for job in manifest["jobs"]:
        lanes.setdefault(job["lane"], []).append(job)

    assert len(lanes["gnn-8x8-scratch"]) == 2
    assert len(lanes["cnn-8x8-scratch"]) == 2
    assert len(lanes["gnn-5x5-pretrain"]) == 2
    assert len(lanes["gnn-transfer"]) == 2
    assert len(lanes["postprocess"]) == 1
    assert lanes["gnn-transfer"][0]["dependency"].startswith("afterok:")
    assert lanes["gnn-transfer"][1]["dependency"].startswith("afterany:")
    assert lanes["postprocess"][0]["dependency"].startswith("afterok:")
    assert manifest["lane_terminals"]["postprocess"] == lanes["postprocess"][0]["job_id"]
    assert any(path.endswith("head_to_head_gnn_vs_cnn.json") for path in manifest["expected_outputs"])
