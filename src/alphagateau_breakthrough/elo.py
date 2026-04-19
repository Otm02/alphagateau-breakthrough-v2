from __future__ import annotations

from collections import defaultdict

import numpy as np


def _match_row(i1: int, i2: int, n_players: int) -> np.ndarray:
    row = np.zeros(n_players, dtype=np.float64)
    row[i1] = 1.0
    row[i2] = -1.0
    return row


def _score_from_results(results: tuple[int, int, int] | list[int]) -> float:
    wins, draws, losses = results
    return (wins - losses) / max(1, wins + draws + losses)


def compute_wls_elo(results: dict[str, dict[str, list[int]]], average: int = 1000) -> tuple[dict[str, int], dict[str, float]]:
    players = sorted(results.keys())
    if not players:
        return {}, {}
    player_to_id = {player: idx for idx, player in enumerate(players)}
    rows = []
    targets = []
    weights = []
    for player_a in players:
        for player_b, outcome in results[player_a].items():
            if player_a >= player_b:
                continue
            score = _score_from_results(outcome)
            p_bar = np.clip((score + 1.0) / 2.0, 1e-4, 1.0 - 1e-4)
            rows.append(_match_row(player_to_id[player_a], player_to_id[player_b], len(players)))
            targets.append(-(400.0 / np.log(10.0)) * np.log(1.0 / p_bar - 1.0))
            n_games = max(1, sum(outcome))
            weights.append(n_games * p_bar * (1.0 - p_bar))
    rows.append(np.ones(len(players), dtype=np.float64))
    targets.append(float(average * len(players)))
    weights.append(1e6)
    x = np.stack(rows, axis=0)
    y = np.array(targets, dtype=np.float64)
    w = np.sqrt(np.array(weights, dtype=np.float64))
    xw = x * w[:, None]
    yw = y * w
    solution, _, _, _ = np.linalg.lstsq(xw, yw, rcond=None)
    covariance = np.linalg.pinv(xw.T @ xw)
    std = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    elo = {player: int(round(solution[player_to_id[player]])) for player in players}
    elo_std = {player: float(std[player_to_id[player]]) for player in players}
    return elo, elo_std


def update_results_table(
    table: dict[str, dict[str, list[int]]],
    player_a: str,
    player_b: str,
    outcome: int,
) -> dict[str, dict[str, list[int]]]:
    results = defaultdict(dict, {k: dict(v) for k, v in table.items()})
    results.setdefault(player_a, {})
    results.setdefault(player_b, {})
    results[player_a].setdefault(player_b, [0, 0, 0])
    results[player_b].setdefault(player_a, [0, 0, 0])
    index = {1: 0, 0: 1, -1: 2}[outcome]
    inv_index = {0: 2, 1: 1, 2: 0}[index]
    results[player_a][player_b][index] += 1
    results[player_b][player_a][inv_index] += 1
    return {k: dict(v) for k, v in results.items()}
