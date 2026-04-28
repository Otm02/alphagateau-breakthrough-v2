from __future__ import annotations

import pickle
import signal
from dataclasses import replace
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Signal handling (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# JAX-native episode collection
# ---------------------------------------------------------------------------

def make_collect_episodes(env: BreakthroughEnv, model: ModelManager, max_plies: int, epsilon: float = 0.3):
    """
    Returns a JIT-compiled function that collects a batch of self-play episodes
    entirely on-device using lax.while_loop + vmap.
 
    All rewards are recorded from the perspective of the STARTING player of
    each episode (always player 0 for normal episodes, player 1 for swapped).
    This gives a consistent frame of reference so compute_lambda_returns needs
    no perspective flipping.
 
    Half the episodes start with a flipped board (player 1 to move) so both
    players experience winning and losing.
 
    Returns:
        obs          : (B, T, *obs_shape)  float32
        rewards      : (B, T)              float32  — from starting player's POV
        dones        : (B, T)              bool
        lengths      : (B,)                int32
        start_players: (B,)                int32    — 0 or 1
    """
    n_actions = env.num_actions
    dummy_state = env.init(jax.random.PRNGKey(0))
    obs_shape = model.format_data(state=dummy_state).squeeze(0).shape
 
    from .env import canonical_board, _legal_action_mask_from_canonical, observation_from_canonical
 
    def _greedy_action(state: BreakthroughState, params: chex.ArrayTree, rng: chex.PRNGKey) -> jnp.ndarray:
        all_next = jax.vmap(lambda a: env.step(state, a))(
            jnp.arange(n_actions, dtype=jnp.int32)
        )
        all_obs = jax.vmap(lambda s: model.format_data(state=s).squeeze(0))(all_next)
        dummy_mask = jnp.ones((n_actions, n_actions), dtype=bool)
        _, vals = model(all_obs, dummy_mask, params=params, training=False)
        # negate values for successor states where opponent is to move
        vals = jnp.where(
            jnp.not_equal(all_next.current_player, state.current_player), -vals, vals
        )
        vals = jnp.where(state.legal_action_mask, vals, jnp.finfo(vals.dtype).min)
 
        rng, eps_rng, rand_rng = jax.random.split(rng, 3)
        greedy_action = jnp.argmax(vals)
        legal_mask = state.legal_action_mask.astype(jnp.float32)
        rand_action = jax.random.choice(rand_rng, n_actions, p=legal_mask / legal_mask.sum())
        use_random = jax.random.uniform(eps_rng) < epsilon
        return jnp.where(use_random, rand_action, greedy_action)
 
    def _single_episode(rng: chex.PRNGKey, params: chex.ArrayTree, swap: jnp.ndarray):
        """
        swap=False: player 0 starts, rewards from player 0's POV
        swap=True:  player 1 starts, rewards from player 1's POV
        Either way, the starting player's reward is positive when they win.
        """
        obs_buf     = jnp.zeros((max_plies, *obs_shape), dtype=jnp.float32)
        rewards_buf = jnp.zeros((max_plies,),             dtype=jnp.float32)
        dones_buf   = jnp.zeros((max_plies,),             dtype=bool)
 
        rng, init_rng = jax.random.split(rng)
        state = env.init(init_rng)
 
        # Build swapped initial state using field-level jnp.where (JAX-traceable)
        board_normal  = state._board
        board_flipped = (-jnp.flip(state._board, axis=(0, 1))).astype(state._board.dtype)
        board         = jnp.where(swap, board_flipped, board_normal)
        start_player  = jnp.where(swap, jnp.int32(1), jnp.int32(0))
 
        canon    = canonical_board(board, start_player)
        obs_init = observation_from_canonical(canon)
        lam_init = _legal_action_mask_from_canonical(canon)
        state = state.replace(
            _board=board,
            current_player=start_player,
            observation=obs_init,
            legal_action_mask=lam_init,
        )
 
        def cond(carry):
            state, _, _, _, t, _ = carry
            return jnp.logical_and(~state.terminated, t < max_plies)
 
        def body(carry):
            state, obs_buf, rewards_buf, dones_buf, t, rng = carry
            rng, action_rng = jax.random.split(rng)
 
            obs    = model.format_data(state=state).squeeze(0).astype(jnp.float32)
            action = _greedy_action(state, params, action_rng)
            next_state = env.step(state, action)
 
            # Always record reward from starting player's perspective
            reward = next_state.rewards[start_player]
            done   = next_state.terminated
 
            obs_buf     = obs_buf.at[t].set(obs)
            rewards_buf = rewards_buf.at[t].set(reward)
            dones_buf   = dones_buf.at[t].set(done)
 
            return next_state, obs_buf, rewards_buf, dones_buf, t + 1, rng
 
        init = (state, obs_buf, rewards_buf, dones_buf, jnp.int32(0), rng)
        _, obs_buf, rewards_buf, dones_buf, length, _ = jax.lax.while_loop(
            cond, body, init
        )
        return obs_buf, rewards_buf, dones_buf, length, start_player
 
    def collect_episodes(rng_keys: chex.PRNGKey, params: chex.ArrayTree):
        B = rng_keys.shape[0]
        # First half: player 0 starts. Second half: player 1 starts.
        swap_flags = jnp.arange(B) >= (B // 2)
        return jax.vmap(lambda k, swap: _single_episode(k, params, swap))(rng_keys, swap_flags)
 
    return jax.jit(collect_episodes)
 
 
def make_compute_lambda_returns(model: ModelManager, gamma: float, lambda_: float):
    """
    Computes TD(λ) returns for a batch of episodes.
 
    Since all rewards are already in the starting player's fixed frame
    (from make_collect_episodes), no perspective flipping is needed here.
    The backward scan is straightforward:
 
        G_T = r_T                          (terminal)
        G_t = r_t + gamma * [(1-λ)*V(s_{t+1}) + λ*G_{t+1}]
 
    V(s_{t+1}) is from the model's perspective of whoever acts at t+1.
    Since rewards alternate sign naturally (starting player gets +1 when
    they win, -1 when they lose), the value estimates need to be negated
    when the current player at t+1 differs from the starting player.
 
    Args:
        obs          : (B, T, *obs_shape)
        rewards      : (B, T)   — fixed starting-player frame
        dones        : (B, T)
        lengths      : (B,)
        start_players: (B,)     — 0 or 1, which player started each episode
        params       : model parameters
 
    Returns:
        flat_obs     : (B*T, *obs_shape)
        flat_targets : (B*T,)
        valid_mask   : (B*T,)  bool
    """
    n_actions = model.board_size * model.board_size * 3
 
    def _episode_returns(
        rewards:      jnp.ndarray,   # (T,)
        dones:        jnp.ndarray,   # (T,)
        v_next:       jnp.ndarray,   # (T,) — model value at s_{t+1}, current-player frame
        start_player: jnp.ndarray,   # scalar int32
        all_obs:      jnp.ndarray,   # (T, *obs_shape) — to infer current player per step
        # We infer who acts at each t+1 from the parity of the step index
        # (Breakthrough always alternates, starting from start_player)
    ) -> jnp.ndarray:
        T = rewards.shape[0]
 
        def scan_fn(g_next, t):
            r    = rewards[t]
            done = dones[t]
            vn   = v_next[t]
 
            # Player acting at t+1: start_player XOR ((t+1) % 2)
            # If different from start_player, negate vn to get it in start_player's frame
            player_at_t1 = (start_player + t + 1) % 2
            vn_corrected = jnp.where(
                jnp.equal(player_at_t1, start_player), vn, -vn
            )
 
            g = jnp.where(
                done,
                r,
                r + gamma * ((1.0 - lambda_) * vn_corrected + lambda_ * g_next),
            )
            return g, g
 
        _, returns = jax.lax.scan(
            scan_fn,
            init=jnp.float32(0.0),
            xs=jnp.arange(T - 1, -1, -1, dtype=jnp.int32),
        )
        return returns[::-1]
 
    def compute_lambda_returns(
        obs:           jnp.ndarray,   # (B, T, *obs_shape)
        rewards:       jnp.ndarray,   # (B, T)
        dones:         jnp.ndarray,   # (B, T)
        lengths:       jnp.ndarray,   # (B,)
        start_players: jnp.ndarray,   # (B,)
        params:        chex.ArrayTree,
    ):
        B, T = obs.shape[:2]
        obs_flat   = obs.reshape(B * T, *obs.shape[2:])
        dummy_mask = jnp.ones((B * T, n_actions), dtype=bool)
 
        # One batched forward pass: V(s_t) for all t, in current-player's frame
        _, v_all = model(obs_flat, dummy_mask, params=params, training=False)
        v_all = v_all.reshape(B, T)
 
        # v_next[b, t] = V(s_{t+1}), shift left and pad with 0
        v_next = jnp.concatenate([v_all[:, 1:], jnp.zeros((B, 1))], axis=1)
 
        # Compute λ-returns for all episodes in parallel
        returns = jax.vmap(_episode_returns)(
            rewards, dones, v_next, start_players, obs
        )  # (B, T)
 
        timesteps = jnp.arange(T, dtype=jnp.int32)[None, :]
        valid = timesteps < lengths[:, None]  # (B, T)
 
        return obs_flat, returns.reshape(B * T), valid.reshape(B * T)
 
    return jax.jit(compute_lambda_returns)

# ---------------------------------------------------------------------------
# Training step — loss weighted by valid_mask, no dynamic indexing
# ---------------------------------------------------------------------------

def make_td_lambda_train_step(model: ModelManager, optimizer: optax.GradientTransformation):
    n_actions = model.board_size * model.board_size * 3

    def loss_fn(params, batch_stats, obs, targets, valid_mask):
        dummy_mask = jnp.ones((obs.shape[0], n_actions), dtype=bool)
        (_, v), new_batch_stats = model(
            obs, dummy_mask,
            params={"params": params, "batch_stats": batch_stats},
            training=True,
        )
        loss = jnp.sum(((v - targets) ** 2) * valid_mask) / jnp.maximum(valid_mask.sum(), 1)
        return loss, new_batch_stats

    @jax.jit
    def train_step(params, batch_stats, opt_state, obs, targets, valid_mask):
        (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch_stats, obs, targets, valid_mask
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_batch_stats, new_opt_state, loss

    return train_step


# ---------------------------------------------------------------------------
# Model initialisation (unchanged)
# ---------------------------------------------------------------------------

def initialise_td_model(
    config: TDTrainConfig, run_name: str
) -> tuple[ModelManager, chex.ArrayTree, chex.ArrayTree]:
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
    batch_stats = variables.get("batch_stats", {})
    if config.initial_checkpoint:
        payload = load_checkpoint(config.initial_checkpoint)
        params = payload["params"]
        batch_stats = payload["batch_stats"]
    return model, params, batch_stats


# ---------------------------------------------------------------------------
# Persistence helpers (unchanged)
# ---------------------------------------------------------------------------

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
        "experiment_name", "board_size", "model_type", "hidden_size",
        "n_res_layers", "learning_rate", "discount_factor", "lambda_",
        "episodes_per_iteration", "seed", "training_passes",
    ]:
        if saved.get(key) != config.to_dict().get(key):
            raise ValueError(
                f"Resume config mismatch for {key}: "
                f"{saved.get(key)!r} != {config.to_dict().get(key)!r}"
            )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

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
                run_dir=run_dir, config=config, run_name=run_name,
                params=params, batch_stats=batch_stats, opt_state=opt_state,
                rng=rng, numpy_rng=numpy_rng, metrics_rows=metrics_rows,
                eval_rows=eval_rows, completed_iterations=completed_iterations,
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

    collect_episodes = make_collect_episodes(env, model, config.max_plies)
    compute_lambda_returns = make_compute_lambda_returns(
        model, config.discount_factor, config.lambda_
    )
    train_step = make_td_lambda_train_step(model, optimizer)
    training_passes = config.training_passes

    interrupted = False
    interrupt_signal = None
    final_checkpoint = checkpoints_dir / "final.pkl"

    # Track next checkpoint iteration to stay on clean multiples after resume
    next_checkpoint_iter = (
        (completed_iterations // config.checkpoint_interval + 1)
        * config.checkpoint_interval
    )
    next_eval_iter = (
        (completed_iterations // config.eval_interval + 1)
        * config.eval_interval
    )

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
                # --- Episode collection (fully on-device) ---
                rng, episodes_rng = jax.random.split(rng)
                episode_keys = jax.random.split(episodes_rng, config.episodes_per_iteration)

                obs_buf, rewards_buf, dones_buf, lengths, start_players = collect_episodes(
                    episode_keys,
                    {"params": params, "batch_stats": batch_stats},
                )

                # --- λ-return computation (on-device, perspective-corrected) ---
                flat_obs, flat_targets, valid_mask = compute_lambda_returns(
                    obs_buf, rewards_buf, dones_buf, lengths, start_players,
                    {"params": params, "batch_stats": batch_stats},
                )
                n_transitions = int(valid_mask.sum())

                # --- Training (multiple passes, loss weighted by valid_mask) ---
                total_loss = 0.0
                for _ in range(training_passes):
                    params, batch_stats, opt_state, loss = train_step(
                        params, batch_stats, opt_state,
                        flat_obs, flat_targets, valid_mask,
                    )
                    total_loss += float(loss)
                avg_loss = total_loss / training_passes

                metrics_row = {
                    "iteration": iteration,
                    "n_transitions": n_transitions,
                    "value_loss": avg_loss,
                }
                metrics_rows.append(metrics_row)

                if iteration >= next_checkpoint_iter or iteration == config.num_iterations:
                    checkpoint_path = checkpoints_dir / f"iter_{iteration:04d}.pkl"
                    save_checkpoint(
                        checkpoint_path,
                        config=config.to_dict(),
                        params=params,
                        batch_stats=batch_stats,
                        iteration=iteration,
                    )
                    latest_checkpoint = str(checkpoint_path)
                    next_checkpoint_iter += config.checkpoint_interval

                if iteration >= next_eval_iter or iteration == config.num_iterations:
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
                    metrics_rows[-1].update({
                        "greedy_win_rate":  eval_summary["win_rate"],
                        "greedy_draw_rate": eval_summary["draw_rate"],
                        "greedy_loss_rate": eval_summary["loss_rate"],
                    })
                    next_eval_iter += config.eval_interval

                completed_iterations = iteration
                progress.advance(task)

                # Lightweight status update every iteration
                status = "running"
                if signal_state.requested and completed_iterations < config.num_iterations:
                    interrupted = True
                    interrupt_signal = signal_state.signal_name
                    status = "interrupted"

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
                        "latest_checkpoint": latest_checkpoint,
                    },
                )

                # Heavy resume checkpoint only at checkpoint intervals
                if iteration >= (next_checkpoint_iter - config.checkpoint_interval) or interrupted:
                    _td_checkpoint_state(
                        run_dir=run_dir, config=config, run_name=run_name,
                        params=params, batch_stats=batch_stats, opt_state=opt_state,
                        rng=rng, numpy_rng=numpy_rng, metrics_rows=metrics_rows,
                        eval_rows=eval_rows, completed_iterations=completed_iterations,
                        latest_checkpoint=latest_checkpoint,
                        final_checkpoint=str(final_checkpoint) if final_checkpoint.is_file() else None,
                        status=status, interrupt_signal=interrupt_signal,
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
        run_dir=run_dir, config=config, run_name=run_name,
        params=params, batch_stats=batch_stats, opt_state=opt_state,
        rng=rng, numpy_rng=numpy_rng, metrics_rows=metrics_rows,
        eval_rows=eval_rows, completed_iterations=config.num_iterations,
        latest_checkpoint=latest_checkpoint,
        final_checkpoint=str(final_checkpoint),
        status="completed",
    )


# ---------------------------------------------------------------------------
# Config builder (unchanged)
# ---------------------------------------------------------------------------

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
