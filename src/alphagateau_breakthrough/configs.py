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
    lr_schedule: str = "constant"   # "constant", "cosine", "step"
    lr_warmup_steps: int = 0
    lr_decay_factor: float = 0.1    # only when lr_schedule is "step"
    num_iterations: int = 40
    selfplay_games: int = 64
    max_plies: int = 256
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
        max_plies=256,
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
        max_plies=256,
        num_iterations=30,
    ),
    "cnn_8x8_scratch": TrainConfig(
        experiment_name="cnn_8x8_scratch",
        board_size=8,
        model_type="cnn",
        selfplay_games=64,
        max_plies=256,
        num_iterations=40,
    ),
    "gnn_8x8_scratch_cosine": TrainConfig(
        experiment_name="gnn_8x8_scratch_cosine",
        board_size=8,
        model_type="gnn",
        selfplay_games=64,
        max_plies=256,
        num_iterations=40,
        lr_schedule="cosine",
    ),
    "gnn_5x5_pretrain_cosine": TrainConfig(
        experiment_name="gnn_5x5_pretrain_cosine",
        board_size=5,
        model_type="gnn",
        selfplay_games=128,
        max_plies=96,
        num_iterations=40,
        lr_schedule="cosine",
    ),
    "gnn_6x6_finetune_cosine": TrainConfig(
        experiment_name="gnn_6x6_finetune_cosine",
        board_size=6,
        model_type="gnn",
        selfplay_games=64,
        max_plies=256,
        num_iterations=30,
        lr_schedule="cosine", 
    ),
    "gnn_6x6_finetune": TrainConfig(
        experiment_name="gnn_6x6_finetune",
        board_size=6,
        model_type="gnn",
        selfplay_games=64,
        max_plies=128,
        num_iterations=30,
    ),
    "gnn_8x8_finetune_cosine": TrainConfig(
        experiment_name="gnn_8x8_finetune_cosine",
        board_size=8,
        model_type="gnn",
        selfplay_games=64,
        max_plies=256,
        num_iterations=30,
        lr_schedule="cosine", 
    ),
    
}

@dataclass
class TDTrainConfig:
    experiment_name: str
    board_size: int
    model_type: str
    hidden_size: int = 64
    n_res_layers: int = 4
    learning_rate: float = 1e-3
    discount_factor: float = 1.0
    num_iterations: int = 40
    episodes_per_iteration: int = 64
    lambda_: float = 0.0
    eval_interval: int = 5
    checkpoint_interval: int = 5
    eval_games: int = 12
    seed: int = 0
    initial_checkpoint: str | None = None
    max_plies: int = 256,
    training_passes: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

TD_PRESETS = {
    "td_5x5_scratch": TDTrainConfig(
        model_type="td",
        experiment_name="td_5x5_scratch",
        board_size=5,
        seed=42,
        max_plies=96,
        episodes_per_iteration=128,
        discount_factor=0.99,
        lambda_=0.7,
    ),
    "td_8x8_scratch": TDTrainConfig(
        model_type="td",
        experiment_name="td_8x8_scratch",
        board_size=8,
        seed=42,
        discount_factor=0.99,
        max_plies=256,
        lambda_=0.7,
    )
}