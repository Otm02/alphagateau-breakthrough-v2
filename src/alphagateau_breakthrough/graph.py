from typing import NamedTuple

import jax
import jax.numpy as jnp


class BreakthroughGraphsTuple(NamedTuple):
    nodes: jnp.ndarray
    edges: jnp.ndarray
    receivers: jnp.ndarray
    senders: jnp.ndarray
    n_node: jnp.ndarray
    n_edge: jnp.ndarray
    action_edge_indices: jnp.ndarray


def _current_player_edge_features(
    board: jnp.ndarray, legal_action_mask: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    size = board.shape[0]
    n_nodes = size * size
    senders = jnp.repeat(jnp.arange(n_nodes, dtype=jnp.int32), 3)
    move_types = jnp.tile(jnp.arange(3, dtype=jnp.int32), n_nodes)
    rows = senders // size
    cols = senders % size
    delta_cols = jnp.take(jnp.array([0, -1, 1], dtype=jnp.int32), move_types)
    to_rows = rows + 1
    to_cols = cols + delta_cols
    on_board = (to_rows >= 0) & (to_rows < size) & (to_cols >= 0) & (to_cols < size)
    safe_rows = jnp.clip(to_rows, 0, size - 1)
    safe_cols = jnp.clip(to_cols, 0, size - 1)
    receivers = jnp.where(on_board, safe_rows * size + safe_cols, senders)
    target = board[safe_rows, safe_cols]
    edge_features = jnp.stack(
        [
            jnp.ones_like(legal_action_mask, dtype=jnp.float32),
            jnp.zeros_like(legal_action_mask, dtype=jnp.float32),
            legal_action_mask.astype(jnp.float32),
            jnp.ones_like(legal_action_mask, dtype=jnp.float32),
            (move_types == 0).astype(jnp.float32),
            (move_types != 0).astype(jnp.float32),
            ((move_types != 0) & (target == -1) & on_board).astype(jnp.float32),
            (on_board & (to_rows == size - 1)).astype(jnp.float32),
        ],
        axis=-1,
    )
    return senders, receivers, edge_features


def _opponent_edge_features(
    board: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    size = board.shape[0]
    n_nodes = size * size
    senders = jnp.repeat(jnp.arange(n_nodes, dtype=jnp.int32), 3)
    move_types = jnp.tile(jnp.arange(3, dtype=jnp.int32), n_nodes)
    rows = senders // size
    cols = senders % size
    delta_cols = jnp.take(jnp.array([0, -1, 1], dtype=jnp.int32), move_types)
    to_rows = rows - 1
    to_cols = cols + delta_cols
    on_board = (to_rows >= 0) & (to_rows < size) & (to_cols >= 0) & (to_cols < size)
    safe_rows = jnp.clip(to_rows, 0, size - 1)
    safe_cols = jnp.clip(to_cols, 0, size - 1)
    receivers = jnp.where(on_board, safe_rows * size + safe_cols, senders)
    target = board[safe_rows, safe_cols]
    piece_present = board.reshape(-1)[senders] == -1
    forward_legal = (move_types == 0) & (target == 0)
    diagonal_legal = (move_types != 0) & (target != -1)
    legal = piece_present & on_board & (forward_legal | diagonal_legal)
    edge_features = jnp.stack(
        [
            jnp.zeros_like(legal, dtype=jnp.float32),
            jnp.ones_like(legal, dtype=jnp.float32),
            legal.astype(jnp.float32),
            -jnp.ones_like(legal, dtype=jnp.float32),
            (move_types == 0).astype(jnp.float32),
            (move_types != 0).astype(jnp.float32),
            ((move_types != 0) & (target == 1) & on_board).astype(jnp.float32),
            (on_board & (to_rows == 0)).astype(jnp.float32),
        ],
        axis=-1,
    )
    return senders, receivers, edge_features


def _single_state_to_graph(
    board: jnp.ndarray, legal_action_mask: jnp.ndarray
) -> tuple[jnp.ndarray, ...]:
    size = board.shape[0]
    n_nodes = size * size
    rows = jnp.repeat(jnp.arange(size, dtype=jnp.float32), size)
    cols = jnp.tile(jnp.arange(size, dtype=jnp.float32), size)
    size_scale = jnp.float32(max(size - 1, 1))
    board_flat = board.reshape(-1)
    nodes = jnp.stack(
        [
            (board_flat == 1).astype(jnp.float32),
            (board_flat == -1).astype(jnp.float32),
            rows / size_scale,
            cols / size_scale,
            (rows == 0).astype(jnp.float32),
            (rows == size - 1).astype(jnp.float32),
            (cols == 0).astype(jnp.float32),
            (cols == size - 1).astype(jnp.float32),
        ],
        axis=-1,
    )

    current_senders, current_receivers, current_edges = _current_player_edge_features(
        board, legal_action_mask
    )
    opponent_senders, opponent_receivers, opponent_edges = _opponent_edge_features(
        board
    )
    senders = jnp.concatenate([current_senders, opponent_senders], axis=0)
    receivers = jnp.concatenate([current_receivers, opponent_receivers], axis=0)
    edges = jnp.concatenate([current_edges, opponent_edges], axis=0)
    action_edge_indices = jnp.arange(n_nodes * 3, dtype=jnp.int32)
    return nodes, edges, senders, receivers, action_edge_indices


@jax.jit
def state_to_graph(
    board: jnp.ndarray, legal_action_mask: jnp.ndarray
) -> BreakthroughGraphsTuple:
    if board.ndim == 2:
        board = board[jnp.newaxis, ...]
        legal_action_mask = legal_action_mask[jnp.newaxis, ...]
    nodes, edges, senders, receivers, action_edge_indices = jax.vmap(
        _single_state_to_graph
    )(board, legal_action_mask)
    batch_size, n_nodes, node_dim = nodes.shape
    _, n_edges, edge_dim = edges.shape
    node_offsets = (jnp.arange(batch_size, dtype=jnp.int32) * n_nodes)[:, None]
    edge_offsets = (jnp.arange(batch_size, dtype=jnp.int32) * n_edges)[:, None]
    return BreakthroughGraphsTuple(
        nodes=nodes.reshape(batch_size * n_nodes, node_dim),
        edges=edges.reshape(batch_size * n_edges, edge_dim),
        senders=(senders + node_offsets).reshape(-1),
        receivers=(receivers + node_offsets).reshape(-1),
        n_node=jnp.full((batch_size,), n_nodes, dtype=jnp.int32),
        n_edge=jnp.full((batch_size,), n_edges, dtype=jnp.int32),
        action_edge_indices=action_edge_indices + edge_offsets,
    )
