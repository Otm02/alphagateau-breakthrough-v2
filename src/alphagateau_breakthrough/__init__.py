"""AlphaGateau Breakthrough package."""

from .configs import EXPERIMENT_PRESETS, TrainConfig
from .env import BreakthroughEnv, BreakthroughState
from .evaluation import evaluate_checkpoint_pair, run_tournament
from .models import ModelManager, build_model_manager, load_checkpoint, save_checkpoint
from .training import train_experiment


def generate_submission_figures(*args, **kwargs):
    # Import plotting lazily so package import does not require matplotlib
    # unless figure-generation functionality is actually used.
    from .plotting import generate_submission_figures as _generate_submission_figures

    return _generate_submission_figures(*args, **kwargs)

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
