from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from flax import struct


FORWARD = 0
DIAG_LEFT = 1
DIAG_RIGHT = 2
MOVE_DELTAS = jnp.array([0, -1, 1], dtype=jnp.int32)


@struct.dataclass
class BreakthroughState:
    _board: jnp.ndarray
    observation: jnp.ndarray
    legal_action_mask: jnp.ndarray
    current_player: jnp.ndarray
    rewards: jnp.ndarray
    terminated: jnp.ndarray
    winner: jnp.ndarray
    turn_count: jnp.ndarray


def canonical_board(board: jnp.ndarray, player: jnp.ndarray) -> jnp.ndarray:
    flipped = -jnp.flip(board, axis=(0, 1))
    player_mask = player == 0
    while player_mask.ndim < board.ndim:
        player_mask = player_mask[..., None]
    return jnp.where(player_mask, board, flipped)


def observation_from_canonical(board: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack(
        [
            (board == 1).astype(jnp.float32),
            (board == -1).astype(jnp.float32),
            (board == 0).astype(jnp.float32),
        ],
        axis=-1,
    )


def _legal_action_mask_from_canonical(board: jnp.ndarray) -> jnp.ndarray:
    size = board.shape[0]
    from_square = jnp.arange(size * size, dtype=jnp.int32)
    rows = from_square // size
    cols = from_square % size
    deltas = jnp.take(MOVE_DELTAS, jnp.arange(3, dtype=jnp.int32))

    def move_mask(delta_col: jnp.ndarray) -> jnp.ndarray:
        to_rows = rows + 1
        to_cols = cols + delta_col
        on_board = (
            (to_rows >= 0)
            & (to_rows < size)
            & (to_cols >= 0)
            & (to_cols < size)
        )
        safe_rows = jnp.clip(to_rows, 0, size - 1)
        safe_cols = jnp.clip(to_cols, 0, size - 1)
        target = board[safe_rows, safe_cols]
        forward_legal = (delta_col == 0) & (target == 0)
        diagonal_legal = (delta_col != 0) & (target != 1)
        return (board.reshape(-1) == 1) & on_board & (forward_legal | diagonal_legal)

    mask = jax.vmap(move_mask)(deltas).transpose(1, 0)
    return mask.reshape(-1)


def _action_to_canonical_coords(action: jnp.ndarray, size: int) -> tuple[jnp.ndarray, ...]:
    from_square = action // 3
    move_type = action % 3
    from_row = from_square // size
    from_col = from_square % size
    to_row = from_row + 1
    to_col = from_col + MOVE_DELTAS[move_type]
    return from_row, from_col, to_row, to_col, move_type


def encode_action(from_row: int, from_col: int, move_type: int, board_size: int) -> int:
    return (from_row * board_size + from_col) * 3 + move_type


def decode_action(action: int, board_size: int) -> tuple[int, int, int, int, int]:
    action_jnp = jnp.int32(action)
    from_row, from_col, to_row, to_col, move_type = _action_to_canonical_coords(action_jnp, board_size)
    return int(from_row), int(from_col), int(to_row), int(to_col), int(move_type)


def _canonical_to_absolute(row: jnp.ndarray, col: jnp.ndarray, size: int, player: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    abs_row = jnp.where(player == 0, row, size - 1 - row)
    abs_col = jnp.where(player == 0, col, size - 1 - col)
    return abs_row, abs_col


class BreakthroughEnv:
    def __init__(self, board_size: int):
        self.board_size = int(board_size)
        self.num_actions = self.board_size * self.board_size * 3
        self.id = f"breakthrough_{self.board_size}x{self.board_size}"

    def initial_board(self) -> jnp.ndarray:
        board = jnp.zeros((self.board_size, self.board_size), dtype=jnp.int8)
        board = board.at[:2, :].set(1)
        board = board.at[-2:, :].set(-1)
        return board

    def observe(self, state: BreakthroughState, player: jnp.ndarray) -> jnp.ndarray:
        return observation_from_canonical(canonical_board(state._board, player))

    def _make_state(
        self,
        board: jnp.ndarray,
        current_player: jnp.ndarray,
        rewards: jnp.ndarray,
        terminated: jnp.ndarray,
        winner: jnp.ndarray,
        turn_count: jnp.ndarray,
    ) -> BreakthroughState:
        board_current = canonical_board(board, current_player)
        observation = observation_from_canonical(board_current)
        legal_action_mask = _legal_action_mask_from_canonical(board_current)
        legal_action_mask = jnp.where(terminated, jnp.zeros_like(legal_action_mask), legal_action_mask)
        return BreakthroughState(
            _board=board,
            observation=observation,
            legal_action_mask=legal_action_mask,
            current_player=current_player,
            rewards=rewards,
            terminated=terminated,
            winner=winner,
            turn_count=turn_count,
        )

    def init(self, key: jnp.ndarray) -> BreakthroughState:
        del key
        return self._make_state(
            board=self.initial_board(),
            current_player=jnp.int32(0),
            rewards=jnp.zeros(2, dtype=jnp.float32),
            terminated=jnp.bool_(False),
            winner=jnp.int32(-1),
            turn_count=jnp.int32(0),
        )

    def _terminal_rewards(self, winner: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(
            winner == 0,
            jnp.array([1.0, -1.0], dtype=jnp.float32),
            jnp.array([-1.0, 1.0], dtype=jnp.float32),
        )

    def _step_illegal(self, state: BreakthroughState) -> BreakthroughState:
        winner = jnp.int32(1) - state.current_player
        next_player = winner
        board = state._board
        return self._make_state(
            board=board,
            current_player=next_player,
            rewards=self._terminal_rewards(winner),
            terminated=jnp.bool_(True),
            winner=winner,
            turn_count=state.turn_count + 1,
        )

    def _step_legal(self, state: BreakthroughState, action: jnp.ndarray) -> BreakthroughState:
        size = self.board_size
        from_row, from_col, to_row, to_col, _ = _action_to_canonical_coords(action, size)
        abs_from_row, abs_from_col = _canonical_to_absolute(from_row, from_col, size, state.current_player)
        abs_to_row, abs_to_col = _canonical_to_absolute(to_row, to_col, size, state.current_player)
        moving_piece = jnp.where(state.current_player == 0, jnp.int8(1), jnp.int8(-1))
        board = state._board
        board = board.at[abs_from_row, abs_from_col].set(0)
        board = board.at[abs_to_row, abs_to_col].set(moving_piece)

        next_player = jnp.int32(1) - state.current_player
        win_by_goal = to_row == size - 1
        opponent_piece = -moving_piece
        win_by_elimination = jnp.sum(board == opponent_piece) == 0
        preview_state = self._make_state(
            board=board,
            current_player=next_player,
            rewards=jnp.zeros(2, dtype=jnp.float32),
            terminated=jnp.bool_(False),
            winner=jnp.int32(-1),
            turn_count=state.turn_count + 1,
        )
        opponent_stuck = ~preview_state.legal_action_mask.any()
        actor_wins = win_by_goal | win_by_elimination | opponent_stuck
        winner = jnp.where(actor_wins, state.current_player, jnp.int32(-1))
        rewards = jnp.where(
            actor_wins,
            self._terminal_rewards(winner),
            jnp.zeros(2, dtype=jnp.float32),
        )
        return self._make_state(
            board=board,
            current_player=next_player,
            rewards=rewards,
            terminated=actor_wins,
            winner=winner,
            turn_count=state.turn_count + 1,
        )

    def step(self, state: BreakthroughState, action: jnp.ndarray) -> BreakthroughState:
        def do_step(_: None) -> BreakthroughState:
            legal = state.legal_action_mask[action]
            return jax.lax.cond(
                legal,
                lambda __: self._step_legal(state, action),
                lambda __: self._step_illegal(state),
                operand=None,
            )

        return jax.lax.cond(state.terminated, lambda _: state, do_step, operand=None)


@partial(jax.jit, static_argnames=("board_size",))
def action_to_notation(
    board: jnp.ndarray,
    player: jnp.ndarray,
    action: jnp.ndarray,
    board_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    from_row, from_col, to_row, to_col, move_type = _action_to_canonical_coords(action, board_size)
    abs_from_row, abs_from_col = _canonical_to_absolute(from_row, from_col, board_size, player)
    abs_to_row, abs_to_col = _canonical_to_absolute(to_row, to_col, board_size, player)
    capture = board[abs_to_row, abs_to_col] != 0
    del move_type
    return abs_from_row, abs_from_col, abs_to_row, abs_to_col, capture
