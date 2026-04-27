from __future__ import annotations

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
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from .configs import TDTrainConfig
from .env import BreakthroughEnv, BreakthroughState
from .evaluation import evaluate_td_against_greedy
from .models import ModelManager, build_model_manager, load_checkpoint, save_checkpoint
from .utils import ensure_dir, write_csv, write_json, write_jsonl

RESUME_STATE_NAME = "latest_resume.pkl"
STATUS_NAME = "status.json"

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
        print(f"Received {self.signal_name}; checkpointing after the current safe point.", flush=True)

def make_batch_successors(env):
    """
    Returns a JIT-compiled function that steps ALL possible actions from a state
    in a single vmapped GPU call. Shape is always (num_actions,) so JAX compiles once.
    Illegal actions are handled gracefully by env.step (returns same state).
    """
    all_actions = jnp.arange(env.num_actions, dtype=jnp.int32)
 
    @jax.jit
    def batch_successors(state: BreakthroughState) -> BreakthroughState:
        return jax.vmap(lambda a: env.step(state, a))(all_actions)
 
    return batch_successors
 
 
def td_selfplay_episode(env, params, batch_stats, model, rng, max_plies, batch_successors):
    """
    One self-play episode for TD(λ) training.
 
    Returns a trajectory of (obs, next_obs, reward, done) tuples.
    """
    state = env.init(rng)
    trajectory = []
    n_plies = 0
 
    while not state.terminated and n_plies < max_plies:
        legal_mask = np.array(state.legal_action_mask)
        legal_indices = np.where(legal_mask)[0]
        current_player = int(state.current_player)
 
        # Single vmapped call over ALL actions — fixed shape, no recompilation
        all_next_states = batch_successors(state)
 
        # Gather only the legal successor observations
        obs_list = []
        lam_list = []
        for i in legal_indices:
            ns_i = jax.tree_util.tree_map(lambda x: x[i], all_next_states)
            obs_list.append(model.format_data(state=ns_i))
            lam_list.append(ns_i.legal_action_mask)
 
        obs_batch = jnp.concatenate(obs_list, axis=0)
        lam_batch = jnp.stack(lam_list, axis=0)
 
        _, vals = model(
            obs_batch, lam_batch,
            params={"params": params, "batch_stats": batch_stats},
        )
        vals = np.array(vals)
 
        next_players = np.array(all_next_states.current_player)
        for k, i in enumerate(legal_indices):
            if next_players[i] != current_player:
                vals[k] = -vals[k]
 
        best_k = int(np.argmax(vals))
        best_action = int(legal_indices[best_k])
        next_state = env.step(state, best_action)
 
        obs_cur = np.array(model.format_data(state=state))
        obs_nxt = np.array(model.format_data(state=next_state))
        trajectory.append((
            obs_cur,
            obs_nxt,
            float(next_state.rewards[current_player]),
            float(next_state.terminated),
        ))
 
        state = next_state
        n_plies += 1
 
    return trajectory

def compute_lambda_returns(
    trajectory: list[tuple],
    model: ModelManager,
    params,
    batch_stats,
    gamma: float,
    lambda_: float,
) -> list[tuple]:
    """
    Compute offline λ-returns for a trajectory and return (obs, lambda_return) pairs.
 
    G_t^λ = (1 - λ) * Σ_{n=1}^{T-t-1} λ^{n-1} * G_t^{(n)}  +  λ^{T-t-1} * G_t^{T}
 
    Equivalently, the backward recursive form (used here, O(T) time):
        G_T = r_T                                      (terminal)
        G_t = r_t + gamma * [(1-λ) * V(s_{t+1}) + λ * G_{t+1}]
 
    This is exact and avoids storing eligibility traces.
 
    Args:
        trajectory: list of (obs, next_obs, reward, done) from td_selfplay_episode
        model: ModelManager for computing V(s_{t+1})
        params, batch_stats: current model parameters
        gamma: discount factor
        lambda_: TD(λ) mixing parameter (0 = TD(0), 1 = Monte Carlo)
 
    Returns:
        list of (obs, lambda_return) pairs ready for training
    """
    if not trajectory:
        return []
 
    T = len(trajectory)
 
    # Batch all next_obs into one forward pass to get V(s_{t+1}) for all t
    all_next_obs = jnp.array([t[1] for t in trajectory])   # (T, *obs_shape)
    n_actions = model.board_size * model.board_size * 3
    dummy_mask = jnp.ones((T, n_actions), dtype=bool)
 
    _, v_next_all = model(
        all_next_obs, dummy_mask,
        params={"params": params, "batch_stats": batch_stats},
        training=False,
    )
    v_next_all = np.array(v_next_all)   # (T,)
 
    # Backward sweep to compute λ-returns
    lambda_returns = np.zeros(T, dtype=np.float32)
    g = 0.0  # G_{T} initialised; will be set at first (last) step
 
    for t in reversed(range(T)):
        _, _, reward, done = trajectory[t]
        v_next = float(v_next_all[t])
 
        if done:
            # Terminal transition: no bootstrap, G_t = reward
            g = reward
        else:
            # G_t = r_t + gamma * [(1 - lambda_) * V(s_{t+1}) + lambda_ * G_{t+1}]
            g = reward + gamma * ((1.0 - lambda_) * v_next + lambda_ * g)
 
        lambda_returns[t] = g
 
    return [(trajectory[t][0], lambda_returns[t]) for t in range(T)]
 
 
def make_td_lambda_train_step(model: ModelManager, optimizer: optax.GradientTransformation):
    """
    Train step for TD(λ): fits V(s) directly to precomputed λ-returns.
 
    The λ-returns are computed offline before this call, so the loss is a
    simple MSE between the model's value prediction and the target.
    No bootstrapping happens inside this function.
    """
    def loss_fn(params, batch_stats, obs, targets):
        n_actions = model.board_size * model.board_size * 3
        dummy_mask = jnp.ones((obs.shape[0], n_actions), dtype=bool)
 
        (_, v), new_batch_stats = model(
            obs, dummy_mask,
            params={"params": params, "batch_stats": batch_stats},
            training=True,
        )
        loss = jnp.mean((v - targets) ** 2)
        return loss, new_batch_stats
 
    @jax.jit
    def train_step(params, batch_stats, opt_state, obs, targets):
        (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch_stats, obs, targets
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_batch_stats, new_opt_state, loss
 
    return train_step

# def make_td_train_step(model: ModelManager, optimizer: optax.GradientTransformation, gamma: float):
#     def loss_fn(params, batch_stats, obs, next_obs, reward, done):
#         n_actions = model.board_size * model.board_size * 3
#         dummy_mask = jnp.ones((obs.shape[0], n_actions), dtype=bool)

#         # training=True → returns ((logits, value), batch_stats)
#         (_, v), new_batch_stats = model(
#             obs, dummy_mask,
#             params={"params": params, "batch_stats": batch_stats},
#             training=True,
#         )

#         # training=False → returns (logits, value) directly, no batch_stats
#         _, v_next = model(
#             next_obs, dummy_mask,
#             params={"params": params, "batch_stats": batch_stats},
#             training=False,
#         )

#         v_next = jax.lax.stop_gradient(v_next)
#         target = reward + gamma * v_next * (1.0 - done)
#         loss = jnp.mean((v - target) ** 2)
#         return loss, new_batch_stats

#     @jax.jit
#     def train_step(params, batch_stats, opt_state, obs, next_obs, reward, done):
#         (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(
#             params, batch_stats, obs, next_obs, reward, done
#         )
#         updates, new_opt_state = optimizer.update(grads, opt_state, params)
#         new_params = optax.apply_updates(params, updates)
#         return new_params, new_batch_stats, new_opt_state, loss

#     return train_step


def initialise_td_model(config: TDTrainConfig, run_name: str) -> tuple[ModelManager, chex.ArrayTree, chex.ArrayTree]:
    model = build_model_manager(
        model_id=run_name,
        model_type="td",
        board_size=config.board_size,
        inner_size=config.hidden_size,
        n_res_layers=config.n_res_layers,
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


def _td_checkpoint_state(
    *,
    run_dir: Path,
    config: TDTrainConfig,
    run_name: str,
    params: chex.ArrayTree,
    batch_stats: chex.ArrayTree,
    opt_state: chex.ArrayTree,
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
            "remaining_iterations": max(0, config.num_iterations - completed_iterations),
            "status": status,
            "is_complete": status == "completed",
            "interrupt_signal": interrupt_signal,
            "latest_checkpoint": latest_checkpoint,
            "final_checkpoint": final_checkpoint,
            "resume_state": str(_resume_state_path(run_dir)),
        },
    )
    return summary


def _validate_resume_config(config: TDTrainConfig, resume_payload: dict) -> None:
    saved = resume_payload["config"]
    for key in [
        "experiment_name",
        "board_size",
        "model_type",
        "hidden_size",
        "n_res_layers",
        "learning_rate",
        "discount_factor",
        "lambda_",
        "episodes_per_iteration",
        "seed",
    ]:
        if saved.get(key) != config.to_dict().get(key):
            raise ValueError(f"Resume config mismatch for {key}: {saved.get(key)!r} != {config.to_dict().get(key)!r}")


def train_experiment(
    config: TDTrainConfig,
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
    optimizer = optax.adam(config.learning_rate)
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
            model_type="td",
            board_size=config.board_size,
            inner_size=config.hidden_size,
            n_res_layers=config.n_res_layers,
        )
        params = resume_payload["params"]
        batch_stats = resume_payload["batch_stats"]
        opt_state = resume_payload["opt_state"]
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
            return _td_checkpoint_state(
                run_dir=run_dir,
                config=config,
                run_name=run_name,
                params=params,
                batch_stats=batch_stats,
                opt_state=opt_state,
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
        model, params, batch_stats = initialise_td_model(config, run_name)
        opt_state = optimizer.init(params)
        rng = jax.random.PRNGKey(config.seed)
        numpy_rng = np.random.default_rng(config.seed)
        metrics_rows = []
        eval_rows = []
        completed_iterations = 0

    train_step = make_td_lambda_train_step(model, optimizer)
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
            task = progress.add_task(f"Training {run_name}", total=config.num_iterations, completed=completed_iterations)
            batch_successors = make_batch_successors(env) 
            for iteration in range(completed_iterations + 1, config.num_iterations + 1):
                rng, *episode_keys = jax.random.split(rng, config.episodes_per_iteration + 1)
 
                # Collect trajectories
                all_samples = []  # list of (obs, lambda_return)
                for ep_key in episode_keys:
                    trajectory = td_selfplay_episode(
                        env, params, batch_stats, model, ep_key, config.max_plies, batch_successors
                    )
                    # Compute λ-returns offline for this episode
                    samples = compute_lambda_returns(
                        trajectory,
                        model=model,
                        params=params,
                        batch_stats=batch_stats,
                        gamma=config.discount_factor,
                        lambda_=config.lambda_,
                    )
                    all_samples.extend(samples)
 
                # Build training batch from (obs, lambda_return) pairs
                obs     = jnp.array([s[0] for s in all_samples])
                targets = jnp.array([s[1] for s in all_samples], dtype=jnp.float32)
 
                params, batch_stats, opt_state, loss = train_step(
                    params, batch_stats, opt_state, obs, targets
                )
 
                metrics_row = {
                    "iteration": iteration,
                    "n_transitions": len(all_samples),
                    "value_loss": float(loss),
                }
                metrics_rows.append(metrics_row)

                if iteration % config.checkpoint_interval == 0 or iteration == config.num_iterations:
                    checkpoint_path = checkpoints_dir / f"iter_{iteration:04d}.pkl"
                    save_checkpoint(
                        checkpoint_path,
                        config=config.to_dict(),
                        params=params,
                        batch_stats=batch_stats,
                        iteration=iteration,
                    )
                    latest_checkpoint = str(checkpoint_path)

                if iteration % config.eval_interval == 0 or iteration == config.num_iterations:
                    eval_summary = evaluate_td_against_greedy(
                        env=env,
                        model=model,
                        params={"params": params, "batch_stats": batch_stats},
                        n_games=config.eval_games,
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
                if signal_state.requested and completed_iterations < config.num_iterations:
                    interrupted = True
                    interrupt_signal = signal_state.signal_name
                    status = "interrupted"

                _td_checkpoint_state(
                    run_dir=run_dir,
                    config=config,
                    run_name=run_name,
                    params=params,
                    batch_stats=batch_stats,
                    opt_state=opt_state,
                    rng=rng,
                    numpy_rng=numpy_rng,
                    metrics_rows=metrics_rows,
                    eval_rows=eval_rows,
                    completed_iterations=completed_iterations,
                    latest_checkpoint=latest_checkpoint,
                    final_checkpoint=str(final_checkpoint) if final_checkpoint.is_file() else None,
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
    return _td_checkpoint_state(
        run_dir=run_dir,
        config=config,
        run_name=run_name,
        params=params,
        batch_stats=batch_stats,
        opt_state=opt_state,
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
    preset: TDTrainConfig,
    *,
    initial_checkpoint: str | None = None,
    num_iterations: int | None = None,
    episodes_per_iteration: int | None = None,
    discount_factor: float | None = None,
    max_plies: int | None = None,
    eval_games: int | None = None,
) -> TDTrainConfig:
    config = preset
    if initial_checkpoint is not None:
        config = replace(config, initial_checkpoint=initial_checkpoint)
    if num_iterations is not None:
        config = replace(config, num_iterations=num_iterations)
    if episodes_per_iteration is not None:
        config = replace(config, episodes_per_iteration=episodes_per_iteration)
    if discount_factor is not None:
        config = replace(config, discount_factor=discount_factor)
    if max_plies is not None:
        config = replace(config, max_plies=max_plies)
    if eval_games is not None:
        config = replace(config, eval_games=eval_games)
    return config
