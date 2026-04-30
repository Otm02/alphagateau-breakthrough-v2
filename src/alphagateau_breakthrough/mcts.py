from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import mctx

from .env import BreakthroughEnv, BreakthroughState, canonical_board
from .models import ModelManager
from .utils import tree_select


class SelfPlayBatch(NamedTuple):
    board: jnp.ndarray
    obs: jnp.ndarray
    lam: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray


def recurrent_fn(
    params: chex.ArrayTree,
    rng_key: chex.PRNGKey,
    action: chex.Array,
    state: BreakthroughState,
    env: BreakthroughEnv,
    model: ModelManager,
) -> tuple[mctx.RecurrentFnOutput, BreakthroughState]:
    del rng_key
    actor = state.current_player
    next_state = jax.vmap(env.step)(state, action)
    logits, value = model(
        model.format_data(state=next_state),
        legal_action_mask=next_state.legal_action_mask,
        params=params,
    )
    reward = next_state.rewards[jnp.arange(next_state.rewards.shape[0]), actor]
    value = jnp.where(next_state.terminated, 0.0, value)
    discount = jnp.where(next_state.terminated, 0.0, -jnp.ones_like(value))
    output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return output, next_state


def gumbel_policy(
    *,
    state: BreakthroughState,
    model: ModelManager,
    params: chex.ArrayTree,
    env: BreakthroughEnv,
    rng_key: chex.PRNGKey,
    n_sim: int,
) -> mctx.PolicyOutput:
    logits, value = model(
        model.format_data(state=state),
        legal_action_mask=state.legal_action_mask,
        params=params,
    )
    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
    return mctx.gumbel_muzero_policy(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=partial(recurrent_fn, env=env, model=model),
        num_simulations=n_sim,
        invalid_actions=~state.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=1.0,
    )


def _auto_reset_step(
    *,
    state: BreakthroughState,
    action: jnp.ndarray,
    env: BreakthroughEnv,
    reset_keys: chex.PRNGKey,
) -> tuple[BreakthroughState, BreakthroughState]:
    stepped = jax.vmap(env.step)(state, action)
    resets = jax.vmap(env.init)(reset_keys)
    next_state = tree_select(stepped.terminated, resets, stepped)
    return stepped, next_state


def selfplay(
    *,
    env: BreakthroughEnv,
    model: ModelManager,
    params: chex.ArrayTree,
    rng_key: chex.PRNGKey,
    n_games: int,
    max_plies: int,
    n_sim: int,
) -> SelfPlayBatch:
    keys = jax.random.split(rng_key, n_games + max_plies + 1)
    state = jax.vmap(env.init)(keys[:n_games])
    rollout_keys = keys[n_games:]

    def body_fn(
        carry: BreakthroughState, key: chex.PRNGKey
    ) -> tuple[BreakthroughState, SelfPlayBatch]:
        key_policy, key_reset = jax.random.split(key)
        policy = gumbel_policy(
            state=carry,
            model=model,
            params=params,
            env=env,
            rng_key=key_policy,
            n_sim=n_sim,
        )
        reset_keys = jax.random.split(key_reset, carry.current_player.shape[0])
        stepped, next_state = _auto_reset_step(
            state=carry,
            action=policy.action,
            env=env,
            reset_keys=reset_keys,
        )
        actor = carry.current_player
        reward = stepped.rewards[jnp.arange(stepped.rewards.shape[0]), actor]
        discount = jnp.where(stepped.terminated, 0.0, -jnp.ones_like(reward))
        batch = SelfPlayBatch(
            board=canonical_board(carry._board, carry.current_player),
            obs=carry.observation,
            lam=carry.legal_action_mask,
            reward=reward,
            terminated=stepped.terminated,
            action_weights=policy.action_weights,
            discount=discount,
        )
        return next_state, batch

    _, data = jax.lax.scan(body_fn, state, rollout_keys[:max_plies])
    return data
