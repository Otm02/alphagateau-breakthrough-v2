from __future__ import annotations

import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, "src")

from alphagateau_breakthrough.env import BreakthroughEnv, canonical_board, decode_action, encode_action


def test_initial_legal_move_counts() -> None:
    env_5 = BreakthroughEnv(5)
    env_8 = BreakthroughEnv(8)
    state_5 = env_5.init(jax.random.PRNGKey(0))
    state_8 = env_8.init(jax.random.PRNGKey(1))
    assert int(state_5.legal_action_mask.sum()) == 13
    assert int(state_8.legal_action_mask.sum()) == 22


def test_action_encode_decode_round_trip() -> None:
    action = encode_action(1, 2, 2, 5)
    assert decode_action(action, 5) == (1, 2, 2, 3, 2)


def test_canonicalisation_for_second_player() -> None:
    env = BreakthroughEnv(5)
    board = jnp.zeros((5, 5), dtype=jnp.int8)
    board = board.at[0, 0].set(1)
    board = board.at[4, 4].set(-1)
    state = env._make_state(
        board=board,
        current_player=jnp.int32(1),
        rewards=jnp.zeros(2, dtype=jnp.float32),
        terminated=jnp.bool_(False),
        winner=jnp.int32(-1),
        turn_count=jnp.int32(0),
    )
    canon = canonical_board(state._board, state.current_player)
    assert int(canon[0, 0]) == 1
    assert int(canon[4, 4]) == -1


def test_diagonal_empty_move_and_capture_are_legal() -> None:
    env = BreakthroughEnv(5)
    board = jnp.zeros((5, 5), dtype=jnp.int8)
    board = board.at[1, 2].set(1)
    board = board.at[2, 3].set(-1)
    state = env._make_state(
        board=board,
        current_player=jnp.int32(0),
        rewards=jnp.zeros(2, dtype=jnp.float32),
        terminated=jnp.bool_(False),
        winner=jnp.int32(-1),
        turn_count=jnp.int32(0),
    )
    assert bool(state.legal_action_mask[encode_action(1, 2, 1, 5)])
    assert bool(state.legal_action_mask[encode_action(1, 2, 2, 5)])


def test_edge_file_offboard_diagonal_is_illegal() -> None:
    env = BreakthroughEnv(5)
    board = jnp.zeros((5, 5), dtype=jnp.int8)
    board = board.at[1, 0].set(1)
    state = env._make_state(
        board=board,
        current_player=jnp.int32(0),
        rewards=jnp.zeros(2, dtype=jnp.float32),
        terminated=jnp.bool_(False),
        winner=jnp.int32(-1),
        turn_count=jnp.int32(0),
    )
    assert not bool(state.legal_action_mask[encode_action(1, 0, 1, 5)])
    assert bool(state.legal_action_mask[encode_action(1, 0, 0, 5)])
    assert bool(state.legal_action_mask[encode_action(1, 0, 2, 5)])


def test_goal_rank_and_elimination_wins() -> None:
    env = BreakthroughEnv(5)

    goal_board = jnp.zeros((5, 5), dtype=jnp.int8)
    goal_board = goal_board.at[3, 2].set(1)
    goal_board = goal_board.at[4, 4].set(-1)
    goal_state = env._make_state(
        board=goal_board,
        current_player=jnp.int32(0),
        rewards=jnp.zeros(2, dtype=jnp.float32),
        terminated=jnp.bool_(False),
        winner=jnp.int32(-1),
        turn_count=jnp.int32(0),
    )
    next_goal_state = env.step(goal_state, jnp.int32(encode_action(3, 2, 0, 5)))
    assert bool(next_goal_state.terminated)
    assert int(next_goal_state.winner) == 0
    assert tuple(map(float, next_goal_state.rewards)) == (1.0, -1.0)

    elim_board = jnp.zeros((5, 5), dtype=jnp.int8)
    elim_board = elim_board.at[1, 2].set(1)
    elim_board = elim_board.at[2, 3].set(-1)
    elim_state = env._make_state(
        board=elim_board,
        current_player=jnp.int32(0),
        rewards=jnp.zeros(2, dtype=jnp.float32),
        terminated=jnp.bool_(False),
        winner=jnp.int32(-1),
        turn_count=jnp.int32(0),
    )
    next_elim_state = env.step(elim_state, jnp.int32(encode_action(1, 2, 2, 5)))
    assert bool(next_elim_state.terminated)
    assert int(next_elim_state.winner) == 0
