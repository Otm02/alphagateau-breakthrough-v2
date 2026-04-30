#!/usr/bin/env python3
from __future__ import annotations

import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "dist" / "alphagateau-breakthrough-training.zip"
INCLUDE_PATHS = [
    README := REPO_ROOT / "README.md",
    RUN_GUIDE := REPO_ROOT / "RUN_GUIDE.md",
    PYPROJECT := REPO_ROOT / "pyproject.toml",
    PIXI := REPO_ROOT / "pixi.toml",
    GITIGNORE := REPO_ROOT / ".gitignore",
    REPO_ROOT / "src" / "alphagateau_breakthrough",
    REPO_ROOT / "scripts",
    REPO_ROOT / "slurm",
]
OPTIONAL_INCLUDE_PATHS = [
    REPO_ROOT / "artifacts" / "experiments" / "gnn_8x8_scratch" / "config.json",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_8x8_scratch" / "evaluation.csv",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_8x8_scratch" / "metrics.csv",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_8x8_scratch" / "summary.json",
    REPO_ROOT / "artifacts" / "experiments" / "cnn_8x8_scratch" / "config.json",
    REPO_ROOT / "artifacts" / "experiments" / "cnn_8x8_scratch" / "evaluation.csv",
    REPO_ROOT / "artifacts" / "experiments" / "cnn_8x8_scratch" / "metrics.csv",
    REPO_ROOT / "artifacts" / "experiments" / "cnn_8x8_scratch" / "summary.json",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_5x5_pretrain_cosine" / "config.json",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_5x5_pretrain_cosine" / "evaluation.csv",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_5x5_pretrain_cosine" / "metrics.csv",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_5x5_pretrain_cosine" / "summary.json",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_8x8_scratch_cosine" / "config.json",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_8x8_scratch_cosine" / "evaluation.csv",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_8x8_scratch_cosine" / "metrics.csv",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_8x8_scratch_cosine" / "summary.json",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_8x8_finetune_cosine" / "config.json",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_8x8_finetune_cosine" / "evaluation.csv",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_8x8_finetune_cosine" / "metrics.csv",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_8x8_finetune_cosine" / "summary.json",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_5x5_pretrain_cosine_cat_forg" / "config.json",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_5x5_pretrain_cosine_cat_forg" / "evaluation.csv",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_5x5_pretrain_cosine_cat_forg" / "metrics.csv",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_5x5_pretrain_cosine_cat_forg" / "summary.json",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_6x6_finetune_cosine_cat_forg" / "config.json",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_6x6_finetune_cosine_cat_forg" / "evaluation.csv",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_6x6_finetune_cosine_cat_forg" / "metrics.csv",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_6x6_finetune_cosine_cat_forg" / "summary.json",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_8x8_finetune_cosine_cat_forg" / "config.json",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_8x8_finetune_cosine_cat_forg" / "evaluation.csv",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_8x8_finetune_cosine_cat_forg" / "metrics.csv",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_8x8_finetune_cosine_cat_forg" / "summary.json",
    REPO_ROOT / "artifacts" / "experiments" / "td_5x5_scratch" / "config.json",
    REPO_ROOT / "artifacts" / "experiments" / "td_5x5_scratch" / "evaluation.csv",
    REPO_ROOT / "artifacts" / "experiments" / "td_5x5_scratch" / "metrics.csv",
    REPO_ROOT / "artifacts" / "experiments" / "td_5x5_scratch" / "summary.json",
    REPO_ROOT / "artifacts" / "experiments" / "td_8x8_scratch" / "config.json",
    REPO_ROOT / "artifacts" / "experiments" / "td_8x8_scratch" / "evaluation.csv",
    REPO_ROOT / "artifacts" / "experiments" / "td_8x8_scratch" / "metrics.csv",
    REPO_ROOT / "artifacts" / "experiments" / "td_8x8_scratch" / "summary.json",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_transfer_summary.json",
    REPO_ROOT / "artifacts" / "experiments" / "gnn_transfer_zero_shot.json",
    REPO_ROOT / "artifacts" / "experiments" / "head_to_head_gnn_5x5_6x6_8x8_finetune_cosine_vs_cnn.json",
    REPO_ROOT / "artifacts" / "experiments" / "head_to_head_gnn_5x5_to_8x8_vs_cnn.json",
    REPO_ROOT / "artifacts" / "experiments" / "head_to_head_gnn_cosine_vs_cnn.json",
    REPO_ROOT / "report" / "figures" / "breakthrough_rules.png",
    REPO_ROOT / "report" / "figures" / "encoding_visualisation.png",
    REPO_ROOT / "report" / "figures" / "gnn_8x8_scratch_curve.png",
    REPO_ROOT / "report" / "figures" / "transfer_curve.png",
    REPO_ROOT / "report" / "figures" / "gnn_5x5_6x6_8x8_progressive_winrate.png",
]
EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".pixi",
    ".pytest_cache",
    ".venv",
    "artifacts",
    "dist",
    "logs",
    "notebooks",
    "presentation",
    "report",
}
EXCLUDED_FILE_NAMES = {".DS_Store"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
EXCLUDED_RELATIVE_PATHS = {
    Path("scripts/bootstrap_mimi_shell.sh"),
    Path("scripts/evaluate_models.py"),
    Path("scripts/mimi_bootstrap_shared_env.sh"),
    Path("scripts/mimi_collect_outputs.sh"),
    Path("scripts/mimi_dry_run.sh"),
    Path("scripts/mimi_full_dag.sh"),
    Path("scripts/mimi_full_validation.sh"),
    Path("scripts/mimi_project.sh"),
    Path("scripts/mimi_queue.sh"),
    Path("scripts/mimi_runtime_check.sh"),
    Path("scripts/mimi_smoke_dag.sh"),
    Path("scripts/plot_catastrophic_forgetting.py"),
    Path("scripts/plot_scratch_comparison.py"),
    Path("scripts/plot_submission_figures.py"),
    Path("scripts/postprocess_experiments.py"),
    Path("scripts/run_smoke_suite.py"),
    Path("scripts/setup_mimi_env.sh"),
    Path("scripts/show_runtime_info.py"),
    Path("scripts/submit_mimi_pipeline.py"),
    Path("scripts/zip_training_bundle.py"),
}


def ensure_required_paths_exist(paths: list[Path]) -> None:
    missing = [
        path.relative_to(REPO_ROOT).as_posix() for path in paths if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Required bundle paths are missing: {', '.join(missing)}"
        )


def should_include(path: Path) -> bool:
    relative = path.relative_to(REPO_ROOT)
    if path in OPTIONAL_INCLUDE_PATHS:
        return path.is_file()
    if relative in EXCLUDED_RELATIVE_PATHS:
        return False
    if any(part in EXCLUDED_DIR_NAMES for part in relative.parts[:-1]):
        return False
    if path.name in EXCLUDED_DIR_NAMES:
        return False
    if path.name in EXCLUDED_FILE_NAMES:
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    return path.is_file()


def iter_bundle_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path.is_file():
            candidates = [path]
        else:
            candidates = sorted(
                candidate for candidate in path.rglob("*") if candidate.is_file()
            )
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen or not should_include(candidate):
                continue
            seen.add(resolved)
            files.append(candidate)
    return sorted(files)


def write_zip(output_path: Path, files: list[Path]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in files:
            archive.write(file_path, arcname=file_path.relative_to(REPO_ROOT))


def main() -> None:
    ensure_required_paths_exist(INCLUDE_PATHS)
    files = iter_bundle_files(INCLUDE_PATHS + [path for path in OPTIONAL_INCLUDE_PATHS if path.exists()])
    write_zip(OUTPUT_PATH, files)
    size_bytes = OUTPUT_PATH.stat().st_size
    print(
        f"Created {OUTPUT_PATH.relative_to(REPO_ROOT)} with {len(files)} files ({size_bytes} bytes)"
    )


if __name__ == "__main__":
    main()
