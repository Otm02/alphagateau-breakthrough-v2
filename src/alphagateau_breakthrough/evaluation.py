from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .elo import compute_wls_elo, update_results_table
from .env import BreakthroughEnv
from .mcts import gumbel_policy
from .models import build_model_manager, load_checkpoint
from .utils import action_string, write_json


def _legal_indices(mask: jnp.ndarray) -> jnp.ndarray:
    return jnp.argwhere(mask, size=mask.shape[0], fill_value=-1).reshape(-1)


def greedy_action(state, board_size: int) -> int:
    legal = np.where(np.array(state.legal_action_mask))[0]
    if legal.size == 0:
        return 0
    board = np.array(state._board)
    player = int(state.current_player)
    best_score = None
    best_action = int(legal[0])
    for action in legal:
        from_square = action // 3
        move_type = action % 3
        from_row = from_square // board_size
        from_col = from_square % board_size
        to_row = from_row + 1
        to_col = from_col + [0, -1, 1][move_type]
        if player == 1:
            abs_to_row = board_size - 1 - to_row
            abs_to_col = board_size - 1 - to_col
        else:
            abs_to_row = to_row
            abs_to_col = to_col
        target = board[abs_to_row, abs_to_col]
        goal = to_row == board_size - 1
        capture = target != 0
        score = (
            1000 * int(goal),
            100 * int(capture),
            int(to_row),
            -abs([0, -1, 1][move_type]),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_action = int(action)
    return best_action


def _select_model_action(env, model, params, state, rng_key, n_sim):
    batched_state = jax.tree.map(lambda x: x[jnp.newaxis, ...], state)
    policy = gumbel_policy(
        state=batched_state,
        model=model,
        params=params,
        env=env,
        rng_key=rng_key,
        n_sim=n_sim,
    )
    return int(policy.action[0])


def play_game(
    *,
    env: BreakthroughEnv,
    player0: tuple[str, Any],
    player1: tuple[str, Any],
    max_plies: int,
    seed: int,
    log_path: str | Path | None = None,
) -> dict:
    rng = jax.random.PRNGKey(seed)
    state = env.init(rng)
    transcript: list[str] = []
    agent_lookup = {0: player0, 1: player1}
    for ply in range(max_plies):
        rng, subkey = jax.random.split(rng)
        agent_type, agent_payload = agent_lookup[int(state.current_player)]
        if agent_type == "model":
            model, params, n_sim = agent_payload
            action = _select_model_action(env, model, params, state, subkey, n_sim)
        elif agent_type == "greedy":
            action = greedy_action(state, env.board_size)
        else:
            legal = np.where(np.array(state.legal_action_mask))[0]
            if legal.size == 0:
                action = 0
            else:
                action = int(legal[int(jax.random.randint(subkey, (), 0, legal.size))])
        transcript.append(action_string(state._board, state.current_player, jnp.int32(action), env.board_size))
        state = env.step(state, jnp.int32(action))
        if bool(state.terminated):
            break
    result = int(np.array(state.rewards[0]).round())
    summary = {
        "result": result,
        "winner": int(state.winner),
        "plies": len(transcript),
        "moves": transcript,
    }
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(log_path).open("w", encoding="utf-8") as file:
            file.write("\n".join(transcript))
    return summary


def evaluate_against_greedy(
    *,
    env: BreakthroughEnv,
    model,
    params,
    n_games: int,
    n_sim: int,
    max_plies: int,
    seed: int,
    log_path: str | Path | None = None,
) -> dict:
    wins = draws = losses = 0
    sample_log = None
    for game_id in range(n_games):
        if game_id % 2 == 0:
            player0 = ("model", (model, params, n_sim))
            player1 = ("greedy", None)
        else:
            player0 = ("greedy", None)
            player1 = ("model", (model, params, n_sim))
        summary = play_game(
            env=env,
            player0=player0,
            player1=player1,
            max_plies=max_plies,
            seed=seed + game_id,
            log_path=None,
        )
        result = summary["result"] if game_id % 2 == 0 else -summary["result"]
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1
        if sample_log is None:
            sample_log = summary["moves"]
    if log_path is not None and sample_log is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(log_path).open("w", encoding="utf-8") as file:
            file.write("\n".join(sample_log))
    total = max(1, n_games)
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / total,
        "draw_rate": draws / total,
        "loss_rate": losses / total,
    }


def evaluate_checkpoint_pair(
    checkpoint_a: str | Path,
    checkpoint_b: str | Path,
    *,
    n_games: int = 12,
    n_sim: int = 32,
    max_plies: int = 256,
    seed: int = 0,
    output_path: str | Path | None = None,
) -> dict:
    payload_a = load_checkpoint(checkpoint_a)
    payload_b = load_checkpoint(checkpoint_b)
    config_a = payload_a["config"]
    config_b = payload_b["config"]
    env = BreakthroughEnv(8)
    model_a = build_model_manager(
        model_id=Path(checkpoint_a).stem,
        model_type=config_a["model_type"],
        board_size=8,
        inner_size=config_a["hidden_size"],
        n_res_layers=config_a["n_res_layers"],
        attention_pooling=config_a["attention_pooling"],
        mix_edge_node=config_a["mix_edge_node"],
        add_features=config_a["add_features"],
        self_edges=config_a["self_edges"],
        simple_update=config_a["simple_update"],
        sync_updates=config_a["sync_updates"],
    )
    model_b = build_model_manager(
        model_id=Path(checkpoint_b).stem,
        model_type=config_b["model_type"],
        board_size=8,
        inner_size=config_b["hidden_size"],
        n_res_layers=config_b["n_res_layers"],
        attention_pooling=config_b["attention_pooling"],
        mix_edge_node=config_b["mix_edge_node"],
        add_features=config_b["add_features"],
        self_edges=config_b["self_edges"],
        simple_update=config_b["simple_update"],
        sync_updates=config_b["sync_updates"],
    )
    wins = draws = losses = 0
    move_log = None
    for game_id in range(n_games):
        if game_id % 2 == 0:
            player0 = ("model", (model_a, {"params": payload_a["params"], "batch_stats": payload_a["batch_stats"]}, n_sim))
            player1 = ("model", (model_b, {"params": payload_b["params"], "batch_stats": payload_b["batch_stats"]}, n_sim))
        else:
            player0 = ("model", (model_b, {"params": payload_b["params"], "batch_stats": payload_b["batch_stats"]}, n_sim))
            player1 = ("model", (model_a, {"params": payload_a["params"], "batch_stats": payload_a["batch_stats"]}, n_sim))
        summary = play_game(
            env=env,
            player0=player0,
            player1=player1,
            max_plies=max_plies,
            seed=seed + game_id,
        )
        result = summary["result"] if game_id % 2 == 0 else -summary["result"]
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1
        if move_log is None:
            move_log = summary["moves"]
    total = max(1, n_games)
    summary = {
        "checkpoint_a": str(checkpoint_a),
        "checkpoint_b": str(checkpoint_b),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / total,
        "draw_rate": draws / total,
        "loss_rate": losses / total,
        "sample_moves": move_log or [],
    }
    if output_path is not None:
        write_json(output_path, summary)
    return summary


def run_tournament(
    checkpoints: list[str | Path],
    *,
    n_games: int = 12,
    n_sim: int = 32,
    max_plies: int = 256,
    seed: int = 0,
    output_dir: str | Path = "artifacts/tournament",
) -> dict:
    results_table: dict[str, dict[str, list[int]]] = {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, checkpoint_a in enumerate(checkpoints):
        for checkpoint_b in checkpoints[idx + 1 :]:
            summary = evaluate_checkpoint_pair(
                checkpoint_a,
                checkpoint_b,
                n_games=n_games,
                n_sim=n_sim,
                max_plies=max_plies,
                seed=seed + idx,
            )
            key_a = Path(checkpoint_a).stem
            key_b = Path(checkpoint_b).stem
            for _ in range(summary["wins"]):
                results_table = update_results_table(results_table, key_a, key_b, 1)
            for _ in range(summary["draws"]):
                results_table = update_results_table(results_table, key_a, key_b, 0)
            for _ in range(summary["losses"]):
                results_table = update_results_table(results_table, key_a, key_b, -1)
    elo, elo_std = compute_wls_elo(results_table)
    summary = {"results": results_table, "elo": elo, "elo_std": elo_std}
    write_json(output_dir / "tournament_summary.json", summary)
    return summary
