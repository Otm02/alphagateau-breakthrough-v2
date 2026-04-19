from __future__ import annotations

import sys

import jax

sys.path.insert(0, "src")

from alphagateau_breakthrough.env import BreakthroughEnv, canonical_board
from alphagateau_breakthrough.graph import state_to_graph
from alphagateau_breakthrough.models import build_model_manager


def test_graph_node_count_and_action_edges() -> None:
    env = BreakthroughEnv(5)
    state = env.init(jax.random.PRNGKey(0))
    graph = state_to_graph(canonical_board(state._board, state.current_player), state.legal_action_mask)
    assert tuple(graph.nodes.shape) == (25, 8)
    assert tuple(graph.action_edge_indices.shape) == (1, 75)
    edge_indices = graph.action_edge_indices[0]
    edge_legal = graph.edges[edge_indices, 2]
    assert int((edge_legal > 0.5).sum()) == int(state.legal_action_mask.sum())


def test_gnn_checkpoint_runs_across_board_sizes() -> None:
    env_5 = BreakthroughEnv(5)
    env_8 = BreakthroughEnv(8)
    model = build_model_manager(
        model_id="transfer_test",
        model_type="gnn",
        board_size=5,
        inner_size=32,
        n_res_layers=2,
    )
    state_5 = env_5.init(jax.random.PRNGKey(0))
    variables = model.init(jax.random.PRNGKey(1), model.format_data(state=state_5))
    state_8 = env_8.init(jax.random.PRNGKey(2))
    logits, value = model(
        model.format_data(state=state_8),
        legal_action_mask=state_8.legal_action_mask,
        params={"params": variables["params"], "batch_stats": variables["batch_stats"]},
    )
    assert tuple(logits.shape) == (1, env_8.num_actions)
    assert tuple(value.shape) == (1,)
