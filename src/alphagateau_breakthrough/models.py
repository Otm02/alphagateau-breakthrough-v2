from __future__ import annotations

import pickle
from pathlib import Path
from typing import Mapping, NamedTuple, Optional, Tuple, cast

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph

from .env import BreakthroughState, canonical_board
from .graph import BreakthroughGraphsTuple, state_to_graph


class AttentionPooling(nn.Module):
    @nn.compact
    def __call__(
        self,
        *,
        x: jnp.ndarray,
        segment_ids: jnp.ndarray,
        num_segments: int,
    ) -> jnp.ndarray:
        att = cast(
            jnp.ndarray,
            jraph.segment_softmax(
                nn.Dense(1)(x).squeeze(-1),
                segment_ids,
                num_segments,
            ),
        )
        att = jnp.tile(att, (x.shape[1], 1)).transpose()
        return jraph.segment_sum(x * att, segment_ids, num_segments)


class BNR(nn.Module):
    momentum: float = 0.9

    @nn.compact
    def __call__(self, *, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = nn.BatchNorm(momentum=self.momentum)(x, use_running_average=not training)
        return jax.nn.relu(x)


class GATEAU(nn.Module):
    out_dim: int = 64
    mix_edge_node: bool = False
    add_features: bool = True
    self_edges: bool = True
    simple_update: bool = True
    sync_updates: Optional[bool] = None

    @nn.compact
    def __call__(self, *, graph: BreakthroughGraphsTuple) -> BreakthroughGraphsTuple:
        sum_n_node = graph.nodes.shape[0]
        sync_updates = (not self.simple_update) if self.sync_updates is None else self.sync_updates
        node_features = graph.nodes
        edge_features = graph.edges

        sent_attributes_1 = nn.Dense(self.out_dim)(node_features)[graph.senders]
        if self.simple_update:
            sent_attributes_2 = node_features[graph.senders]
        else:
            sent_attributes_2 = nn.Dense(self.out_dim)(node_features)[graph.senders]
        received_attributes = nn.Dense(self.out_dim)(node_features)[graph.receivers]
        if sync_updates:
            edge_features_0 = nn.Dense(self.out_dim)(edge_features)
        else:
            edge_features_0 = None
        edge_features = nn.Dense(self.out_dim)(edge_features)
        if self.add_features:
            edge_features = sent_attributes_1 + edge_features + received_attributes
        else:
            edge_features = sent_attributes_1 * edge_features * received_attributes
        attention_coeffs = nn.leaky_relu(nn.Dense(1)(edge_features))
        attention_weights = jraph.segment_softmax(
            attention_coeffs,
            segment_ids=graph.receivers,
            num_segments=sum_n_node,
        )
        if self.mix_edge_node:
            if self.add_features:
                message = sent_attributes_2 + (edge_features_0 if sync_updates else edge_features)
            else:
                message = sent_attributes_2 * (edge_features_0 if sync_updates else edge_features)
        else:
            message = sent_attributes_2
        if self.simple_update:
            message = nn.Dense(self.out_dim)(message)
        message = attention_weights * message
        node_features = jraph.segment_sum(
            message,
            segment_ids=graph.receivers,
            num_segments=sum_n_node,
        )
        if self.self_edges:
            node_features = node_features + nn.Dense(self.out_dim)(graph.nodes)
        return graph._replace(nodes=node_features, edges=edge_features)


class EGNN3(nn.Module):
    out_dim: int = 64
    mix_edge_node: bool = False
    add_features: bool = True
    self_edges: bool = True
    simple_update: bool = True
    sync_updates: Optional[bool] = None

    @nn.compact
    def __call__(self, *, graph: BreakthroughGraphsTuple, training: bool = False) -> BreakthroughGraphsTuple:
        node_skip = graph.nodes
        edge_skip = graph.edges
        graph = GATEAU(
            out_dim=self.out_dim,
            mix_edge_node=self.mix_edge_node,
            add_features=self.add_features,
            self_edges=self.self_edges,
            simple_update=self.simple_update,
            sync_updates=self.sync_updates,
        )(
            graph=graph._replace(
                nodes=BNR()(x=graph.nodes, training=training),
                edges=BNR()(x=graph.edges, training=training),
            )
        )
        graph = GATEAU(
            out_dim=self.out_dim,
            mix_edge_node=self.mix_edge_node,
            add_features=self.add_features,
            self_edges=self.self_edges,
            simple_update=self.simple_update,
            sync_updates=self.sync_updates,
        )(
            graph=graph._replace(
                nodes=BNR()(x=graph.nodes, training=training),
                edges=BNR()(x=graph.edges, training=training),
            )
        )
        return graph._replace(nodes=graph.nodes + node_skip, edges=graph.edges + edge_skip)


class AlphaGateauModel(nn.Module):
    inner_size: int = 64
    n_res_layers: int = 4
    attention_pooling: bool = True
    mix_edge_node: bool = False
    add_features: bool = True
    self_edges: bool = True
    simple_update: bool = True
    sync_updates: Optional[bool] = None

    @nn.compact
    def __call__(self, *, graphs: BreakthroughGraphsTuple, training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        graphs = graphs._replace(
            nodes=nn.Dense(self.inner_size)(graphs.nodes),
            edges=nn.Dense(self.inner_size)(graphs.edges),
        )
        for _ in range(self.n_res_layers):
            graphs = EGNN3(
                out_dim=self.inner_size,
                mix_edge_node=self.mix_edge_node,
                add_features=self.add_features,
                self_edges=self.self_edges,
                simple_update=self.simple_update,
                sync_updates=self.sync_updates,
            )(graph=graphs, training=training)

        node_features = BNR()(x=graphs.nodes, training=training)
        edge_features = BNR()(x=graphs.edges, training=training)
        edge_logits = nn.Dense(self.inner_size)(edge_features)
        edge_logits = BNR()(x=edge_logits, training=training)
        edge_logits = nn.Dense(1)(edge_logits).squeeze(-1)
        logits = edge_logits[graphs.action_edge_indices]

        segment_ids = jnp.repeat(
            jnp.arange(graphs.n_node.shape[0], dtype=jnp.int32),
            graphs.n_node,
            total_repeat_length=node_features.shape[0],
        )
        value_hidden = nn.Dense(self.inner_size)(node_features)
        value_hidden = BNR()(x=value_hidden, training=training)
        if self.attention_pooling:
            value_hidden = AttentionPooling()(
                x=value_hidden,
                segment_ids=segment_ids,
                num_segments=graphs.n_node.shape[0],
            )
        value = nn.Dense(1)(jax.nn.relu(value_hidden))
        return logits, jnp.tanh(value).reshape((-1,))


class ResidualBlock(nn.Module):
    num_channels: int

    @nn.compact
    def __call__(self, *, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        skip = x
        x = nn.BatchNorm(momentum=0.9)(x, use_running_average=not training)
        x = jax.nn.relu(x)
        x = nn.Conv(self.num_channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(momentum=0.9)(x, use_running_average=not training)
        x = jax.nn.relu(x)
        x = nn.Conv(self.num_channels, kernel_size=(3, 3), padding="SAME")(x)
        return x + skip


class AlphaZeroBaseline(nn.Module):
    n_actions: int
    inner_size: int = 64
    n_res_layers: int = 4

    @nn.compact
    def __call__(self, *, x: jnp.ndarray, training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = x.astype(jnp.float32)
        x = nn.Conv(self.inner_size, kernel_size=(3, 3), padding="SAME")(x)
        for _ in range(self.n_res_layers):
            x = ResidualBlock(num_channels=self.inner_size)(x=x, training=training)
        x = nn.BatchNorm(momentum=0.9)(x, use_running_average=not training)
        x = jax.nn.relu(x)
        logits = nn.Conv(features=2, kernel_size=(1, 1), padding="SAME")(x)
        logits = nn.BatchNorm(momentum=0.9)(logits, use_running_average=not training)
        logits = jax.nn.relu(logits)
        logits = logits.reshape((logits.shape[0], -1))
        logits = nn.Dense(self.n_actions)(logits)

        value = nn.Conv(features=1, kernel_size=(1, 1), padding="SAME")(x)
        value = nn.BatchNorm(momentum=0.9)(value, use_running_average=not training)
        value = jax.nn.relu(value)
        value = value.reshape((value.shape[0], -1))
        value = nn.Dense(self.inner_size)(value)
        value = jax.nn.relu(value)
        value = nn.Dense(1)(value)
        return logits, jnp.tanh(value).reshape((-1,))

class TDValueNet(nn.Module):
    board_size: int
    inner_size: int = 64
    n_res_layers: int = 4
    

    @nn.compact
    def __call__(self, *, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = x.astype(jnp.float32)
        x = nn.Conv(self.inner_size, kernel_size=(3, 3), padding="SAME")(x)
        for _ in range(self.n_res_layers):
            x = ResidualBlock(num_channels=self.inner_size)(x=x, training=training)
        x = nn.BatchNorm(momentum=0.9)(x, use_running_average=not training)
        x = jax.nn.relu(x)
        value = nn.Conv(features=1, kernel_size=(1, 1), padding="SAME")(x)
        value = nn.BatchNorm(momentum=0.9)(value, use_running_average=not training)
        value = jax.nn.relu(value)
        value = value.reshape((value.shape[0], -1))
        value = nn.Dense(self.inner_size)(value)
        value = jax.nn.relu(value)
        value = nn.Dense(1)(value)
        batch_size = value.shape[0]
        dummy_logits = jnp.zeros((batch_size, self.board_size * self.board_size * 3))
        return dummy_logits, jnp.tanh(value).reshape((-1,))
    
class ModelManager(NamedTuple):
    id: str
    model: nn.Module
    use_graph: bool
    board_size: int
    model_type: str

    def init(self, key: chex.PRNGKey, x: jnp.ndarray | BreakthroughGraphsTuple) -> Mapping[str, chex.ArrayTree]:
        if self.use_graph:
            return self.model.init(key, graphs=cast(BreakthroughGraphsTuple, x))
        return self.model.init(key, x=cast(jnp.ndarray, x))

    def __call__(
        self,
        x: jnp.ndarray | BreakthroughGraphsTuple,
        legal_action_mask: jnp.ndarray,
        params: chex.ArrayTree,
        training: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray] | Tuple[Tuple[jnp.ndarray, jnp.ndarray], chex.ArrayTree]:
        if self.use_graph:
            outputs, batch_stats = self.model.apply(
                cast(Mapping, params),
                graphs=cast(BreakthroughGraphsTuple, x),
                mutable=["batch_stats"],
                training=training,
            )
        else:
            outputs, batch_stats = self.model.apply(
                cast(Mapping, params),
                x=cast(jnp.ndarray, x),
                mutable=["batch_stats"],
                training=training,
            )
        logits, value = outputs
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        if training:
            return (logits, value), batch_stats["batch_stats"]
        return logits, value

    def format_data(
        self,
        *,
        state: BreakthroughState | None = None,
        board: jnp.ndarray | None = None,
        observation: jnp.ndarray | None = None,
        legal_action_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray | BreakthroughGraphsTuple:
        if state is not None:
            board = canonical_board(state._board, state.current_player)
            observation = state.observation
            legal_action_mask = state.legal_action_mask
        if self.use_graph:
            assert board is not None and legal_action_mask is not None
            return state_to_graph(board, legal_action_mask)
        assert observation is not None
        if observation.ndim == 3:
            observation = observation[jnp.newaxis, ...]
        return observation


def build_model_manager(
    *,
    model_id: str,
    model_type: str,
    board_size: int,
    inner_size: int,
    n_res_layers: int,
    attention_pooling: bool = True,
    mix_edge_node: bool = False,
    add_features: bool = True,
    self_edges: bool = True,
    simple_update: bool = True,
    sync_updates: Optional[bool] = None,
) -> ModelManager:
    if model_type == "gnn":
        model = AlphaGateauModel(
            inner_size=inner_size,
            n_res_layers=n_res_layers,
            attention_pooling=attention_pooling,
            mix_edge_node=mix_edge_node,
            add_features=add_features,
            self_edges=self_edges,
            simple_update=simple_update,
            sync_updates=sync_updates,
        )
        return ModelManager(
            id=model_id,
            model=model,
            use_graph=True,
            board_size=board_size,
            model_type=model_type,
        )
    if model_type == "cnn":
        model = AlphaZeroBaseline(
            n_actions=board_size * board_size * 3,
            inner_size=inner_size,
            n_res_layers=n_res_layers,
        )
        return ModelManager(
            id=model_id,
            model=model,
            use_graph=False,
            board_size=board_size,
            model_type=model_type,
        )
    if model_type == "td":
        model = TDValueNet(
            board_size = board_size,
            inner_size=inner_size,
            n_res_layers=n_res_layers,
        )
        return ModelManager(
            id=model_id,
            model=model,
            use_graph=False,
            board_size=board_size,
            model_type=model_type,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def save_checkpoint(
    path: str | Path,
    *,
    config: dict,
    params: chex.ArrayTree,
    batch_stats: chex.ArrayTree,
    iteration: int,
) -> None:
    payload = {
        "config": config,
        "params": params,
        "batch_stats": batch_stats,
        "iteration": iteration,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: str | Path) -> dict:
    with Path(path).open("rb") as file:
        return pickle.load(file)
