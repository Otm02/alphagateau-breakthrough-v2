from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TrainConfig:
    experiment_name: str
    board_size: int
    model_type: str
    hidden_size: int = 64
    n_res_layers: int = 4
    learning_rate: float = 1e-3
    num_iterations: int = 40
    selfplay_games: int = 64
    max_plies: int = 192
    num_simulations: int = 32
    replay_window: int = 50_000
    batch_size: int = 32
    training_passes: int = 1
    eval_interval: int = 5
    checkpoint_interval: int = 5
    eval_games: int = 12
    eval_simulations: int = 32
    seed: int = 0
    attention_pooling: bool = True
    mix_edge_node: bool = False
    add_features: bool = True
    self_edges: bool = True
    simple_update: bool = True
    sync_updates: bool | None = None
    initial_checkpoint: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


EXPERIMENT_PRESETS = {
    "gnn_8x8_scratch": TrainConfig(
        experiment_name="gnn_8x8_scratch",
        board_size=8,
        model_type="gnn",
        selfplay_games=64,
        max_plies=192,
        num_iterations=40,
    ),
    "gnn_5x5_pretrain": TrainConfig(
        experiment_name="gnn_5x5_pretrain",
        board_size=5,
        model_type="gnn",
        selfplay_games=128,
        max_plies=96,
        num_iterations=40,
    ),
    "gnn_8x8_finetune": TrainConfig(
        experiment_name="gnn_8x8_finetune",
        board_size=8,
        model_type="gnn",
        selfplay_games=64,
        max_plies=192,
        num_iterations=30,
    ),
    "cnn_8x8_scratch": TrainConfig(
        experiment_name="cnn_8x8_scratch",
        board_size=8,
        model_type="cnn",
        selfplay_games=64,
        max_plies=192,
        num_iterations=40,
    ),
}
