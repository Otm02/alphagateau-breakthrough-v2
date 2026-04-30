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
    files = iter_bundle_files(INCLUDE_PATHS)
    write_zip(OUTPUT_PATH, files)
    size_bytes = OUTPUT_PATH.stat().st_size
    print(
        f"Created {OUTPUT_PATH.relative_to(REPO_ROOT)} with {len(files)} files ({size_bytes} bytes)"
    )


if __name__ == "__main__":
    main()
