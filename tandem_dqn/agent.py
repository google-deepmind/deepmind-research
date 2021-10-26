# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tandem DQN agent class."""

import typing
from typing import Any, Callable, Mapping, Set, Text

from absl import logging
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax

from tandem_dqn import losses
from tandem_dqn import parts
from tandem_dqn import processors
from tandem_dqn import replay as replay_lib


class TandemTuple(typing.NamedTuple):
  active: Any
  passive: Any


def tandem_map(fn: Callable[..., Any], *args):
  return TandemTuple(
      active=fn(*[a.active for a in args]),
      passive=fn(*[a.passive for a in args]))


def replace_module_params(source, target, modules):
  """Replace selected module params in target by corresponding source values."""
  source, _ = hk.data_structures.partition(
      lambda module, name, value: module in modules,
      source)
  return hk.data_structures.merge(target, source)


class TandemDqn(parts.Agent):
  """Tandem DQN agent."""

  def __init__(
      self,
      preprocessor: processors.Processor,
      sample_network_input: jnp.ndarray,
      network: TandemTuple,
      optimizer: TandemTuple,
      loss: TandemTuple,
      transition_accumulator: Any,
      replay: replay_lib.TransitionReplay,
      batch_size: int,
      exploration_epsilon: Callable[[int], float],
      min_replay_capacity_fraction: float,
      learn_period: int,
      target_network_update_period: int,
      tied_layers: Set[str],
      rng_key: parts.PRNGKey,
  ):
    self._preprocessor = preprocessor
    self._replay = replay
    self._transition_accumulator = transition_accumulator
    self._batch_size = batch_size
    self._exploration_epsilon = exploration_epsilon
    self._min_replay_capacity = min_replay_capacity_fraction * replay.capacity
    self._learn_period = learn_period
    self._target_network_update_period = target_network_update_period

    # Initialize network parameters and optimizer.
    self._rng_key, network_rng_key_active, network_rng_key_passive = (
        jax.random.split(rng_key, 3))
    active_params = network.active.init(
        network_rng_key_active, sample_network_input[None, ...])
    passive_params = network.passive.init(
        network_rng_key_passive, sample_network_input[None, ...])
    self._online_params = TandemTuple(
        active=active_params, passive=passive_params)
    self._target_params = self._online_params
    self._opt_state = tandem_map(
        lambda optim, params: optim.init(params),
        optimizer, self._online_params)

    # Other agent state: last action, frame count, etc.
    self._action = None
    self._frame_t = -1  # Current frame index.

    # Stats.
    stats = [
        'loss_active',
        'loss_passive',
        'frac_diff_argmax',
        'mc_error_active',
        'mc_error_passive',
        'mc_error_abs_active',
        'mc_error_abs_passive',
    ]
    self._statistics = {k: np.nan for k in stats}

    # Define jitted loss, update, and policy functions here instead of as
    # class methods, to emphasize that these are meant to be pure functions
    # and should not access the agent object's state via `self`.

    def network_outputs(rng_key, online_params, target_params, transitions):
      """Compute all potentially needed outputs of active and passive net."""
      _, *apply_keys = jax.random.split(rng_key, 4)
      outputs_tm1 = tandem_map(
          lambda net, param: net.apply(param, apply_keys[0], transitions.s_tm1),
          network, online_params)
      outputs_t = tandem_map(
          lambda net, param: net.apply(param, apply_keys[1], transitions.s_t),
          network, online_params)
      outputs_target_t = tandem_map(
          lambda net, param: net.apply(param, apply_keys[2], transitions.s_t),
          network, target_params)
      return outputs_tm1, outputs_t, outputs_target_t

    # Helper functions to define active and passive losses.
    # Active and passive losses are allowed to depend on all active and passive
    # outputs, but stop-gradient is used to prevent gradients from flowing
    # from active loss to passive network params and vice versa.
    def sg_active(x):
      return TandemTuple(
          active=jax.lax.stop_gradient(x.active), passive=x.passive)

    def sg_passive(x):
      return TandemTuple(
          active=x.active, passive=jax.lax.stop_gradient(x.passive))

    def compute_loss(online_params, target_params, transitions, rng_key):
      rng_key, apply_key = jax.random.split(rng_key)
      outputs_tm1, outputs_t, outputs_target_t = network_outputs(
          apply_key, online_params, target_params, transitions)

      _, loss_key_active, loss_key_passive = jax.random.split(rng_key, 3)
      loss_active = loss.active(
          sg_passive(outputs_tm1), sg_passive(outputs_t), outputs_target_t,
          transitions, loss_key_active)
      loss_passive = loss.passive(
          sg_active(outputs_tm1), sg_active(outputs_t), outputs_target_t,
          transitions, loss_key_passive)

      # Logging stuff.
      a_tm1 = transitions.a_tm1
      mc_return_tm1 = transitions.mc_return_tm1
      q_values = TandemTuple(
          active=outputs_tm1.active.q_values,
          passive=outputs_tm1.passive.q_values)
      mc_error = jax.tree_map(
          lambda q: losses.batch_mc_learning(q, a_tm1, mc_return_tm1),
          q_values)
      mc_error_abs = jax.tree_map(jnp.abs, mc_error)
      q_argmax = jax.tree_map(lambda q: jnp.argmax(q, axis=-1), q_values)
      argmax_diff = jnp.not_equal(q_argmax.active, q_argmax.passive)

      batch_mean = lambda x: jnp.mean(x, axis=0)
      logs = {
          'loss_active': loss_active,
          'loss_passive': loss_passive
      }
      logs.update(jax.tree_map(batch_mean, {
          'frac_diff_argmax': argmax_diff,
          'mc_error_active': mc_error.active,
          'mc_error_passive': mc_error.passive,
          'mc_error_abs_active': mc_error_abs.active,
          'mc_error_abs_passive': mc_error_abs.passive,
      }))
      return loss_active + loss_passive, logs

    def optim_update(optim, online_params, d_loss_d_params, opt_state):
      updates, new_opt_state = optim.update(d_loss_d_params, opt_state)
      new_online_params = optax.apply_updates(online_params, updates)
      return new_opt_state, new_online_params

    def compute_loss_grad(rng_key, online_params, target_params, transitions):
      rng_key, grad_key = jax.random.split(rng_key)
      (_, logs), d_loss_d_params = jax.value_and_grad(
          compute_loss, has_aux=True)(
              online_params, target_params, transitions, grad_key)
      return rng_key, logs, d_loss_d_params

    def update_active(rng_key, opt_state, online_params, target_params,
                      transitions):
      """Applies learning update for active network only."""
      rng_key, logs, d_loss_d_params = compute_loss_grad(
          rng_key, online_params, target_params, transitions)
      new_opt_state_active, new_online_params_active = optim_update(
          optimizer.active, online_params.active, d_loss_d_params.active,
          opt_state.active)
      new_opt_state = opt_state._replace(
          active=new_opt_state_active)
      new_online_params = online_params._replace(
          active=new_online_params_active)
      return rng_key, new_opt_state, new_online_params, logs

    self._update_active = jax.jit(update_active)

    def update_passive(rng_key, opt_state, online_params, target_params,
                       transitions):
      """Applies learning update for passive network only."""
      rng_key, logs, d_loss_d_params = compute_loss_grad(
          rng_key, online_params, target_params, transitions)
      new_opt_state_passive, new_online_params_passive = optim_update(
          optimizer.passive, online_params.passive, d_loss_d_params.passive,
          opt_state.passive)
      new_opt_state = opt_state._replace(
          passive=new_opt_state_passive)
      new_online_params = online_params._replace(
          passive=new_online_params_passive)
      return rng_key, new_opt_state, new_online_params, logs

    self._update_passive = jax.jit(update_passive)

    def update_active_passive(rng_key, opt_state, online_params,
                              target_params, transitions):
      """Applies learning update for both active & passive networks."""
      rng_key, logs, d_loss_d_params = compute_loss_grad(
          rng_key, online_params, target_params, transitions)

      new_opt_state_active, new_online_params_active = optim_update(
          optimizer.active, online_params.active, d_loss_d_params.active,
          opt_state.active)
      new_opt_state_passive, new_online_params_passive = optim_update(
          optimizer.passive, online_params.passive, d_loss_d_params.passive,
          opt_state.passive)
      new_opt_state = TandemTuple(active=new_opt_state_active,
                                  passive=new_opt_state_passive)
      new_online_params = TandemTuple(active=new_online_params_active,
                                      passive=new_online_params_passive)
      return rng_key, new_opt_state, new_online_params, logs

    self._update_active_passive = jax.jit(update_active_passive)

    self._update = None  # set_training_mode needs to be called to set this.

    def sync_tied_layers(online_params):
      """Set tied layer params of passive to respective values of active."""
      new_online_params_passive = replace_module_params(
          source=online_params.active, target=online_params.passive,
          modules=tied_layers)
      return online_params._replace(passive=new_online_params_passive)

    self._sync_tied_layers = jax.jit(sync_tied_layers)

    def select_action(rng_key, network_params, s_t, exploration_epsilon):
      """Samples action from eps-greedy policy wrt Q-values at given state."""
      rng_key, apply_key, policy_key = jax.random.split(rng_key, 3)
      q_t = network.active.apply(network_params, apply_key,
                                 s_t[None, ...]).q_values[0]
      a_t = rlax.epsilon_greedy().sample(policy_key, q_t, exploration_epsilon)
      return rng_key, a_t

    self._select_action = jax.jit(select_action)

  def step(self, timestep: dm_env.TimeStep) -> parts.Action:
    """Selects action given timestep and potentially learns."""
    self._frame_t += 1

    timestep = self._preprocessor(timestep)

    if timestep is None:  # Repeat action.
      action = self._action
    else:
      action = self._action = self._act(timestep)

      for transition in self._transition_accumulator.step(timestep, action):
        self._replay.add(transition)

    if self._replay.size < self._min_replay_capacity:
      return action

    if self._frame_t % self._learn_period == 0:
      self._learn()

    if self._frame_t % self._target_network_update_period == 0:
      self._target_params = self._online_params

    return action

  def reset(self) -> None:
    """Resets the agent's episodic state such as frame stack and action repeat.

    This method should be called at the beginning of every episode.
    """
    self._transition_accumulator.reset()
    processors.reset(self._preprocessor)
    self._action = None

  def _act(self, timestep) -> parts.Action:
    """Selects action given timestep, according to epsilon-greedy policy."""
    s_t = timestep.observation
    network_params = self._online_params.active
    self._rng_key, a_t = self._select_action(
        self._rng_key, network_params, s_t, self.exploration_epsilon)
    return parts.Action(jax.device_get(a_t))

  def _learn(self) -> None:
    """Samples a batch of transitions from replay and learns from it."""
    logging.log_first_n(logging.INFO, 'Begin learning', 1)
    transitions = self._replay.sample(self._batch_size)
    self._rng_key, self._opt_state, self._online_params, logs = self._update(
        self._rng_key,
        self._opt_state,
        self._online_params,
        self._target_params,
        transitions,
    )
    self._online_params = self._sync_tied_layers(self._online_params)
    self._statistics.update(jax.device_get(logs))

  def set_training_mode(self, mode: str):
    """Sets training mode to one of 'active', 'passive', or 'active_passive'."""
    if mode == 'active':
      self._update = self._update_active
    elif mode == 'passive':
      self._update = self._update_passive
    elif mode == 'active_passive':
      self._update = self._update_active_passive

  @property
  def online_params(self) -> TandemTuple:
    """Returns current parameters of Q-network."""
    return self._online_params

  @property
  def statistics(self) -> Mapping[Text, float]:
    """Returns current agent statistics as a dictionary."""
    # Check for DeviceArrays in values as this can be very slow.
    assert all(
        not isinstance(x, jnp.DeviceArray) for x in self._statistics.values())
    return self._statistics

  @property
  def exploration_epsilon(self) -> float:
    """Returns epsilon value currently used by (eps-greedy) behavior policy."""
    return self._exploration_epsilon(self._frame_t)

  def get_state(self) -> Mapping[Text, Any]:
    """Retrieves agent state as a dictionary (e.g. for serialization)."""
    state = {
        'rng_key': self._rng_key,
        'frame_t': self._frame_t,
        'opt_state_active': self._opt_state.active,
        'online_params_active': self._online_params.active,
        'target_params_active': self._target_params.active,
        'opt_state_passive': self._opt_state.passive,
        'online_params_passive': self._online_params.passive,
        'target_params_passive': self._target_params.passive,
        'replay': self._replay.get_state(),
    }
    return state

  def set_state(self, state: Mapping[Text, Any]) -> None:
    """Sets agent state from a (potentially de-serialized) dictionary."""
    self._rng_key = state['rng_key']
    self._frame_t = state['frame_t']
    self._opt_state = TandemTuple(
        active=jax.device_put(state['opt_state_active']),
        passive=jax.device_put(state['opt_state_passive']))
    self._online_params = TandemTuple(
        active=jax.device_put(state['online_params_active']),
        passive=jax.device_put(state['online_params_passive']))
    self._target_params = TandemTuple(
        active=jax.device_put(state['target_params_active']),
        passive=jax.device_put(state['target_params_passive']))
    self._replay.set_state(state['replay'])
