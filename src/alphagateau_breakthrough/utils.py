import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp

from .env import action_to_notation


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_jsonable(data: Any) -> Any:
    if is_dataclass(data):
        return to_jsonable(asdict(data))
    if isinstance(data, dict):
        return {str(key): to_jsonable(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [to_jsonable(value) for value in data]
    if isinstance(data, Path):
        return str(data)
    if hasattr(data, "tolist"):
        return data.tolist()
    return data


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(data), file, indent=2)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(to_jsonable(row)) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def broadcast_where(
    mask: jnp.ndarray, left: jnp.ndarray, right: jnp.ndarray
) -> jnp.ndarray:
    while mask.ndim < left.ndim:
        mask = mask[..., None]
    return jnp.where(mask, left, right)


def tree_select(mask: jnp.ndarray, left: Any, right: Any) -> Any:
    return jax.tree.map(lambda x, y: broadcast_where(mask, x, y), left, right)


def action_string(
    board: jnp.ndarray, player: jnp.ndarray, action: jnp.ndarray, board_size: int
) -> str:
    from_row, from_col, to_row, to_col, capture = action_to_notation(
        board, player, action, board_size
    )

    def square(row: int, col: int) -> str:
        return f"{chr(ord('a') + col)}{row + 1}"

    sep = "x" if bool(capture) else "-"
    return (
        f"{square(int(from_row), int(from_col))}{sep}{square(int(to_row), int(to_col))}"
    )
