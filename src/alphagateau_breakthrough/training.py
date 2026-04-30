import pickle
import signal
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .configs import TrainConfig
from .env import BreakthroughEnv
from .evaluation import evaluate_against_greedy
from .mcts import SelfPlayBatch, selfplay
from .models import ModelManager, build_model_manager, load_checkpoint, save_checkpoint
from .utils import ensure_dir, write_csv, write_json, write_jsonl

RESUME_STATE_NAME = "latest_resume.pkl"
STATUS_NAME = "status.json"


class SampleBatch(NamedTuple):
    board: np.ndarray
    obs: np.ndarray
    lam: np.ndarray
    policy_tgt: np.ndarray
    value_tgt: np.ndarray
    mask: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data: SampleBatch | None = None

    def __len__(self) -> int:
        if self.data is None:
            return 0
        return int(self.data.value_tgt.shape[0])

    def extend(self, samples: SampleBatch) -> None:
        if self.data is None:
            self.data = samples
            return
        merged = []
        for old, new in zip(self.data, samples):
            merged.append(np.concatenate([old, new], axis=0)[-self.capacity :])
        self.data = SampleBatch(*merged)

    def sample(self, rng: np.random.Generator, batch_size: int) -> SampleBatch:
        if self.data is None:
            raise ValueError("Replay buffer is empty.")
        indices = rng.integers(0, len(self), size=batch_size)
        return SampleBatch(*(field[indices] for field in self.data))

    def to_payload(self) -> tuple[np.ndarray, ...] | None:
        if self.data is None:
            return None
        return tuple(np.asarray(field) for field in self.data)

    @classmethod
    def from_payload(
        cls, capacity: int, payload: tuple[np.ndarray, ...] | None
    ) -> ReplayBuffer:
        replay = cls(capacity)
        if payload is not None:
            replay.data = SampleBatch(*(np.asarray(field) for field in payload))
        return replay


class SignalCheckpointState:
    def __init__(self) -> None:
        self.requested = False
        self.signal_name: str | None = None
        self._handlers: dict[int, object] = {}

    def __enter__(self) -> SignalCheckpointState:
        for sig in [getattr(signal, "SIGUSR1", None), getattr(signal, "SIGTERM", None)]:
            if sig is None:
                continue
            self._handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for sig, handler in self._handlers.items():
            signal.signal(sig, handler)

    def _handle_signal(self, signum, _frame) -> None:
        self.requested = True
        if self.signal_name is None:
            try:
                self.signal_name = signal.Signals(signum).name
            except ValueError:
                self.signal_name = str(signum)
        print(
            f"Received {self.signal_name}; checkpointing after the current safe point.",
            flush=True,
        )


def selfplay_to_samples(data: SelfPlayBatch) -> SampleBatch:
    value_mask = np.cumsum(np.array(data.terminated)[::-1], axis=0)[::-1] >= 1
    reward = np.array(data.reward)
    discount = np.array(data.discount)
    value_tgt = np.zeros_like(reward)
    carry = np.zeros(reward.shape[1], dtype=np.float32)
    for t in range(reward.shape[0] - 1, -1, -1):
        carry = reward[t] + discount[t] * carry
        value_tgt[t] = carry
    board = np.array(data.board).reshape((-1,) + data.board.shape[2:])
    obs = np.array(data.obs).reshape((-1,) + data.obs.shape[2:])
    lam = np.array(data.lam).reshape((-1,) + data.lam.shape[2:])
    policy_tgt = np.array(data.action_weights).reshape(
        (-1,) + data.action_weights.shape[2:]
    )
    value_tgt = value_tgt.reshape(-1)
    mask = value_mask.reshape(-1)
    return SampleBatch(
        board=board,
        obs=obs,
        lam=lam,
        policy_tgt=policy_tgt,
        value_tgt=value_tgt,
        mask=mask.astype(np.float32),
    )


def make_train_step(model: ModelManager, optimizer: optax.GradientTransformation):
    def loss_fn(params, batch_stats, batch):
        inputs = model.format_data(
            board=jnp.asarray(batch.board),
            observation=jnp.asarray(batch.obs),
            legal_action_mask=jnp.asarray(batch.lam),
        )
        (logits, value), new_batch_stats = model(
            inputs,
            legal_action_mask=jnp.asarray(batch.lam),
            params={"params": params, "batch_stats": batch_stats},
            training=True,
        )
        policy_loss = optax.softmax_cross_entropy(
            logits, jnp.asarray(batch.policy_tgt)
        ).mean()
        value_loss = (
            optax.l2_loss(value, jnp.asarray(batch.value_tgt)) * jnp.asarray(batch.mask)
        ).mean()
        return policy_loss + value_loss, (new_batch_stats, policy_loss, value_loss)

    @jax.jit
    def train_step(params, batch_stats, opt_state, batch):
        grads, (new_batch_stats, policy_loss, value_loss) = jax.grad(
            loss_fn, has_aux=True
        )(params, batch_stats, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, new_batch_stats, opt_state, policy_loss, value_loss

    return train_step


def initialise_model(
    config: TrainConfig, run_name: str
) -> tuple[ModelManager, chex.ArrayTree, chex.ArrayTree]:
    model = build_model_manager(
        model_id=run_name,
        model_type=config.model_type,
        board_size=config.board_size,
        inner_size=config.hidden_size,
        n_res_layers=config.n_res_layers,
        attention_pooling=config.attention_pooling,
        mix_edge_node=config.mix_edge_node,
        add_features=config.add_features,
        self_edges=config.self_edges,
        simple_update=config.simple_update,
        sync_updates=config.sync_updates,
    )
    env = BreakthroughEnv(config.board_size)
    dummy_state = env.init(jax.random.PRNGKey(0))
    inputs = model.format_data(state=dummy_state)
    variables = model.init(jax.random.PRNGKey(config.seed), inputs)
    params = variables["params"]
    batch_stats = variables["batch_stats"]
    if config.initial_checkpoint:
        payload = load_checkpoint(config.initial_checkpoint)
        params = payload["params"]
        batch_stats = payload["batch_stats"]
    return model, params, batch_stats


def _save_pickle(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def _load_pickle(path: str | Path) -> dict:
    with Path(path).open("rb") as file:
        return pickle.load(file)


def _resume_state_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / RESUME_STATE_NAME


def _summary_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "summary.json"


def _status_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / STATUS_NAME


def _read_json(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    import json

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _checkpoint_state(
    *,
    run_dir: Path,
    config: TrainConfig,
    run_name: str,
    params: chex.ArrayTree,
    batch_stats: chex.ArrayTree,
    opt_state: chex.ArrayTree,
    replay: ReplayBuffer,
    rng: chex.PRNGKey,
    numpy_rng: np.random.Generator,
    metrics_rows: list[dict],
    eval_rows: list[dict],
    completed_iterations: int,
    latest_checkpoint: str | None,
    final_checkpoint: str | None,
    status: str,
    interrupt_signal: str | None = None,
) -> dict:
    resume_payload = {
        "config": config.to_dict(),
        "run_name": run_name,
        "completed_iterations": completed_iterations,
        "params": params,
        "batch_stats": batch_stats,
        "opt_state": opt_state,
        "replay_buffer": replay.to_payload(),
        "rng_key": np.asarray(rng),
        "numpy_rng_state": numpy_rng.bit_generator.state,
        "metrics_rows": metrics_rows,
        "eval_rows": eval_rows,
        "latest_checkpoint": latest_checkpoint,
        "final_checkpoint": final_checkpoint,
        "status": status,
        "interrupt_signal": interrupt_signal,
    }
    _save_pickle(_resume_state_path(run_dir), resume_payload)
    write_csv(run_dir / "metrics.csv", metrics_rows)
    write_jsonl(run_dir / "metrics.jsonl", metrics_rows)
    write_csv(run_dir / "evaluation.csv", eval_rows)
    write_json(run_dir / "config.json", config.to_dict())
    summary = {
        "experiment_name": config.experiment_name,
        "run_name": run_name,
        "board_size": config.board_size,
        "model_type": config.model_type,
        "final_checkpoint": final_checkpoint,
        "final_iteration": completed_iterations,
        "target_iterations": config.num_iterations,
        "status": status,
        "resume_state": str(_resume_state_path(run_dir)),
    }
    if metrics_rows:
        summary["final_metrics"] = metrics_rows[-1]
    if eval_rows:
        summary["latest_eval"] = eval_rows[-1]
    write_json(_summary_path(run_dir), summary)
    write_json(
        _status_path(run_dir),
        {
            "experiment_name": config.experiment_name,
            "run_name": run_name,
            "board_size": config.board_size,
            "model_type": config.model_type,
            "target_iterations": config.num_iterations,
            "completed_iterations": completed_iterations,
            "remaining_iterations": max(
                0, config.num_iterations - completed_iterations
            ),
            "status": status,
            "is_complete": status == "completed",
            "interrupt_signal": interrupt_signal,
            "latest_checkpoint": latest_checkpoint,
            "final_checkpoint": final_checkpoint,
            "resume_state": str(_resume_state_path(run_dir)),
        },
    )
    return summary


def _validate_resume_config(config: TrainConfig, resume_payload: dict) -> None:
    saved = resume_payload["config"]
    for key in [
        "experiment_name",
        "board_size",
        "model_type",
        "hidden_size",
        "n_res_layers",
        "learning_rate",
        "replay_window",
        "batch_size",
        "training_passes",
        "attention_pooling",
        "mix_edge_node",
        "add_features",
        "self_edges",
        "simple_update",
        "sync_updates",
        "seed",
        "lr_schedule",
        "lr_decay_factor",
        "lr_warmup_steps",
    ]:
        if saved.get(key) != config.to_dict().get(key):
            raise ValueError(
                f"Resume config mismatch for {key}: {saved.get(key)!r} != {config.to_dict().get(key)!r}"
            )


def build_optimizer(config: TrainConfig) -> optax.GradientTransformation:
    if config.lr_schedule == "cosine":
        estimated_updates = (
            config.replay_window // config.batch_size
        ) * config.training_passes
        schedule = optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=config.num_iterations * estimated_updates,
        )
    elif config.lr_schedule == "step":
        schedule = optax.piecewise_constant_schedule(
            init_value=config.learning_rate,
            boundaries_and_scales={
                config.num_iterations // 2: config.lr_decay_factor,
            },
        )
    else:
        schedule = config.learning_rate  # constant, existing behavior

    if config.lr_warmup_steps > 0:
        warmup = optax.linear_schedule(
            init_value=0.0,
            end_value=config.learning_rate,
            transition_steps=config.lr_warmup_steps,
        )
        schedule = optax.join_schedules(
            schedules=[warmup, schedule],
            boundaries=[config.lr_warmup_steps],
        )

    return optax.adam(schedule)


def train_experiment(
    config: TrainConfig,
    *,
    output_root: str | Path = "artifacts/experiments",
    run_name: str | None = None,
    resume: bool = False,
) -> dict:
    run_name = run_name or config.experiment_name
    env = BreakthroughEnv(config.board_size)
    run_dir = ensure_dir(Path(output_root) / run_name)
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    eval_dir = ensure_dir(run_dir / "evaluation")
    optimizer = build_optimizer(config=config)
    resume_payload = None
    latest_checkpoint: str | None = None
    if resume and _resume_state_path(run_dir).is_file():
        resume_payload = _load_pickle(_resume_state_path(run_dir))
        _validate_resume_config(config, resume_payload)
    elif resume:
        existing_summary = _read_json(_summary_path(run_dir))
        if (
            existing_summary is not None
            and existing_summary.get("status") == "completed"
            and int(existing_summary.get("final_iteration", 0)) >= config.num_iterations
        ):
            return existing_summary

    if resume_payload is not None:
        model = build_model_manager(
            model_id=run_name,
            model_type=config.model_type,
            board_size=config.board_size,
            inner_size=config.hidden_size,
            n_res_layers=config.n_res_layers,
            attention_pooling=config.attention_pooling,
            mix_edge_node=config.mix_edge_node,
            add_features=config.add_features,
            self_edges=config.self_edges,
            simple_update=config.simple_update,
            sync_updates=config.sync_updates,
        )
        params = resume_payload["params"]
        batch_stats = resume_payload["batch_stats"]
        opt_state = resume_payload["opt_state"]
        replay = ReplayBuffer.from_payload(
            config.replay_window, resume_payload["replay_buffer"]
        )
        rng = jnp.asarray(resume_payload["rng_key"])
        numpy_rng = np.random.default_rng()
        numpy_rng.bit_generator.state = resume_payload["numpy_rng_state"]
        metrics_rows = list(resume_payload["metrics_rows"])
        eval_rows = list(resume_payload["eval_rows"])
        completed_iterations = int(resume_payload["completed_iterations"])
        latest_checkpoint = resume_payload.get("latest_checkpoint")
        if completed_iterations >= config.num_iterations:
            existing_summary = _read_json(_summary_path(run_dir))
            if existing_summary is not None:
                return existing_summary
            return _checkpoint_state(
                run_dir=run_dir,
                config=config,
                run_name=run_name,
                params=params,
                batch_stats=batch_stats,
                opt_state=opt_state,
                replay=replay,
                rng=rng,
                numpy_rng=numpy_rng,
                metrics_rows=metrics_rows,
                eval_rows=eval_rows,
                completed_iterations=completed_iterations,
                latest_checkpoint=latest_checkpoint,
                final_checkpoint=resume_payload.get("final_checkpoint"),
                status="completed",
            )
    else:
        model, params, batch_stats = initialise_model(config, run_name)
        opt_state = optimizer.init(params)
        replay = ReplayBuffer(config.replay_window)
        rng = jax.random.PRNGKey(config.seed)
        numpy_rng = np.random.default_rng(config.seed)
        metrics_rows = []
        eval_rows = []
        completed_iterations = 0

    train_step = make_train_step(model, optimizer)
    interrupted = False
    interrupt_signal = None
    final_checkpoint = checkpoints_dir / "final.pkl"

    with SignalCheckpointState() as signal_state:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                f"Training {run_name}",
                total=config.num_iterations,
                completed=completed_iterations,
            )
            for iteration in range(completed_iterations + 1, config.num_iterations + 1):
                rng, selfplay_key = jax.random.split(rng)
                selfplay_data = selfplay(
                    env=env,
                    model=model,
                    params={"params": params, "batch_stats": batch_stats},
                    rng_key=selfplay_key,
                    n_games=config.selfplay_games,
                    max_plies=config.max_plies,
                    n_sim=config.num_simulations,
                )
                replay.extend(selfplay_to_samples(selfplay_data))
                num_updates = max(
                    1, (len(replay) // config.batch_size) * config.training_passes
                )
                policy_losses = []
                value_losses = []
                for _ in range(num_updates):
                    batch = replay.sample(numpy_rng, config.batch_size)
                    params, batch_stats, opt_state, policy_loss, value_loss = (
                        train_step(params, batch_stats, opt_state, batch)
                    )
                    policy_losses.append(float(policy_loss))
                    value_losses.append(float(value_loss))

                metrics_row = {
                    "iteration": iteration,
                    "replay_size": len(replay),
                    "policy_loss": float(np.mean(policy_losses)),
                    "value_loss": float(np.mean(value_losses)),
                }
                metrics_rows.append(metrics_row)

                if (
                    iteration % config.checkpoint_interval == 0
                    or iteration == config.num_iterations
                ):
                    checkpoint_path = checkpoints_dir / f"iter_{iteration:04d}.pkl"
                    save_checkpoint(
                        checkpoint_path,
                        config=config.to_dict(),
                        params=params,
                        batch_stats=batch_stats,
                        iteration=iteration,
                    )
                    latest_checkpoint = str(checkpoint_path)

                if (
                    iteration % config.eval_interval == 0
                    or iteration == config.num_iterations
                ):
                    eval_summary = evaluate_against_greedy(
                        env=env,
                        model=model,
                        params={"params": params, "batch_stats": batch_stats},
                        n_games=config.eval_games,
                        n_sim=config.eval_simulations,
                        max_plies=config.max_plies,
                        seed=config.seed + iteration,
                        log_path=eval_dir / f"greedy_eval_iter_{iteration:04d}.txt",
                    )
                    eval_row = {"iteration": iteration, **eval_summary}
                    eval_rows.append(eval_row)
                    metrics_rows[-1].update(
                        {
                            "greedy_win_rate": eval_summary["win_rate"],
                            "greedy_draw_rate": eval_summary["draw_rate"],
                            "greedy_loss_rate": eval_summary["loss_rate"],
                        }
                    )

                completed_iterations = iteration
                progress.advance(task)

                status = "running"
                if (
                    signal_state.requested
                    and completed_iterations < config.num_iterations
                ):
                    interrupted = True
                    interrupt_signal = signal_state.signal_name
                    status = "interrupted"

                _checkpoint_state(
                    run_dir=run_dir,
                    config=config,
                    run_name=run_name,
                    params=params,
                    batch_stats=batch_stats,
                    opt_state=opt_state,
                    replay=replay,
                    rng=rng,
                    numpy_rng=numpy_rng,
                    metrics_rows=metrics_rows,
                    eval_rows=eval_rows,
                    completed_iterations=completed_iterations,
                    latest_checkpoint=latest_checkpoint,
                    final_checkpoint=str(final_checkpoint)
                    if final_checkpoint.is_file()
                    else None,
                    status=status,
                    interrupt_signal=interrupt_signal,
                )
                if interrupted:
                    break

    if interrupted:
        return _read_json(_summary_path(run_dir)) or {
            "experiment_name": config.experiment_name,
            "run_name": run_name,
            "status": "interrupted",
            "final_iteration": completed_iterations,
        }

    save_checkpoint(
        final_checkpoint,
        config=config.to_dict(),
        params=params,
        batch_stats=batch_stats,
        iteration=config.num_iterations,
    )
    latest_checkpoint = str(final_checkpoint)
    return _checkpoint_state(
        run_dir=run_dir,
        config=config,
        run_name=run_name,
        params=params,
        batch_stats=batch_stats,
        opt_state=opt_state,
        replay=replay,
        rng=rng,
        numpy_rng=numpy_rng,
        metrics_rows=metrics_rows,
        eval_rows=eval_rows,
        completed_iterations=config.num_iterations,
        latest_checkpoint=latest_checkpoint,
        final_checkpoint=str(final_checkpoint),
        status="completed",
    )


def build_config_from_preset(
    preset: TrainConfig,
    *,
    initial_checkpoint: str | None = None,
    num_iterations: int | None = None,
    selfplay_games: int | None = None,
    num_simulations: int | None = None,
    max_plies: int | None = None,
    eval_games: int | None = None,
    lr_schedule: str | None = None,
    lr_decay_factor: float | None = None,
    lr_warmup_steps: int | None = None,
) -> TrainConfig:
    config = preset
    if initial_checkpoint is not None:
        config = replace(config, initial_checkpoint=initial_checkpoint)
    if num_iterations is not None:
        config = replace(config, num_iterations=num_iterations)
    if selfplay_games is not None:
        config = replace(config, selfplay_games=selfplay_games)
    if num_simulations is not None:
        config = replace(config, num_simulations=num_simulations)
    if max_plies is not None:
        config = replace(config, max_plies=max_plies)
    if eval_games is not None:
        config = replace(config, eval_games=eval_games)
    if lr_schedule is not None:
        config = replace(config, lr_schedule=lr_schedule)
    if lr_decay_factor is not None:
        config = replace(config, lr_decay_factor=lr_decay_factor)
    if lr_warmup_steps is not None:
        config = replace(config, lr_warmup_steps=lr_warmup_steps)
    return config
