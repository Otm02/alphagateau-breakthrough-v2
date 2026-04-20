"""AlphaGateau Breakthrough package."""

from .configs import EXPERIMENT_PRESETS, TrainConfig

__all__ = [
    "BreakthroughEnv",
    "BreakthroughState",
    "EXPERIMENT_PRESETS",
    "ModelManager",
    "TrainConfig",
    "build_model_manager",
    "evaluate_checkpoint_pair",
    "generate_submission_figures",
    "load_checkpoint",
    "run_tournament",
    "save_checkpoint",
    "train_experiment",
]


def __getattr__(name):
    if name in {"BreakthroughEnv", "BreakthroughState"}:
        from .env import BreakthroughEnv, BreakthroughState

        return {"BreakthroughEnv": BreakthroughEnv, "BreakthroughState": BreakthroughState}[name]
    if name in {"evaluate_checkpoint_pair", "run_tournament"}:
        from .evaluation import evaluate_checkpoint_pair, run_tournament

        return {"evaluate_checkpoint_pair": evaluate_checkpoint_pair, "run_tournament": run_tournament}[name]
    if name in {"ModelManager", "build_model_manager", "load_checkpoint", "save_checkpoint"}:
        from .models import ModelManager, build_model_manager, load_checkpoint, save_checkpoint

        return {
            "ModelManager": ModelManager,
            "build_model_manager": build_model_manager,
            "load_checkpoint": load_checkpoint,
            "save_checkpoint": save_checkpoint,
        }[name]
    if name == "train_experiment":
        from .training import train_experiment

        return train_experiment
    if name == "generate_submission_figures":
        from .plotting import generate_submission_figures

        return generate_submission_figures
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
