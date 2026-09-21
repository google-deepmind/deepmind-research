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
"""Losses for TandemDQN."""

from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import rlax

from tandem_dqn import networks

# Batch variants of double_q_learning and SARSA.
batch_double_q_learning = jax.vmap(rlax.double_q_learning)
batch_sarsa_learning = jax.vmap(rlax.sarsa)

# Batch variant of quantile_q_learning with fixed tau input across batch.
batch_quantile_q_learning = jax.vmap(
    rlax.quantile_q_learning, in_axes=(0, None, 0, 0, 0, 0, 0, None))


def _mc_learning(
    q_tm1: chex.Array,
    a_tm1: chex.Numeric,
    mc_return_tm1: chex.Array,
) -> chex.Numeric:
  """Calculates the MC return error."""
  chex.assert_rank([q_tm1, a_tm1], [1, 0])
  chex.assert_type([q_tm1, a_tm1], [float, int])
  return mc_return_tm1 - q_tm1[a_tm1]

# Batch variant of MC learning.
batch_mc_learning = jax.vmap(_mc_learning)


def _qr_loss(q_tm1, q_t, q_target_t, transitions, rng_key):
  """Calculates QR-Learning loss from network outputs and transitions."""
  del q_t, rng_key  # Unused.
  # Compute Q value distributions.
  huber_param = 1.
  quantiles = networks.make_quantiles()
  losses = batch_quantile_q_learning(
      q_tm1.q_dist,
      quantiles,
      transitions.a_tm1,
      transitions.r_t,
      transitions.discount_t,
      q_target_t.q_dist,  # No double Q-learning here.
      q_target_t.q_dist,
      huber_param,
  )
  loss = jnp.mean(losses)
  return loss


def _sarsa_loss(q_tm1, q_t, transitions, rng_key):
  """Calculates SARSA loss from network outputs and transitions."""
  del rng_key  # Unused.
  grad_error_bound = 1. / 32
  td_errors = batch_sarsa_learning(
      q_tm1.q_values,
      transitions.a_tm1,
      transitions.r_t,
      transitions.discount_t,
      q_t.q_values,
      transitions.a_t
  )
  td_errors = rlax.clip_gradient(td_errors, -grad_error_bound, grad_error_bound)
  losses = rlax.l2_loss(td_errors)
  loss = jnp.mean(losses)
  return loss


def _mc_loss(q_tm1, transitions, rng_key):
  """Calculates Monte-Carlo return loss, i.e. regression towards MC return."""
  del rng_key  # Unused.
  errors = batch_mc_learning(q_tm1.q_values, transitions.a_tm1,
                             transitions.mc_return_tm1)
  loss = jnp.mean(rlax.l2_loss(errors))
  return loss


def _double_q_loss(q_tm1, q_t, q_target_t, transitions, rng_key):
  """Calculates Double Q-Learning loss from network outputs and transitions."""
  del rng_key  # Unused.
  grad_error_bound = 1. / 32
  td_errors = batch_double_q_learning(
      q_tm1.q_values,
      transitions.a_tm1,
      transitions.r_t,
      transitions.discount_t,
      q_target_t.q_values,
      q_t.q_values,
  )
  td_errors = rlax.clip_gradient(td_errors, -grad_error_bound, grad_error_bound)
  losses = rlax.l2_loss(td_errors)
  loss = jnp.mean(losses)
  return loss


def _q_regression_loss(q_tm1, q_tm1_target):
  """Loss for regression of all action values towards targets."""
  errors = q_tm1.q_values - jax.lax.stop_gradient(q_tm1_target.q_values)
  loss = jnp.mean(rlax.l2_loss(errors))
  return loss


def make_loss_fn(loss_type: str, active: bool) -> Callable[..., Any]:
  """Create active or passive loss function of given type."""

  if active:
    primary = lambda x: x.active
    secondary = lambda x: x.passive
  else:
    primary = lambda x: x.passive
    secondary = lambda x: x.active

  def sarsa_loss_fn(q_tm1, q_t, q_target_t, transitions, rng_key):
    """SARSA loss using own networks."""
    del q_t  # Unused.
    return _sarsa_loss(primary(q_tm1), primary(q_target_t), transitions,
                       rng_key)

  def mc_loss_fn(q_tm1, q_t, q_target_t, transitions, rng_key):
    """MonteCarlo loss."""
    del q_t, q_target_t
    return _mc_loss(primary(q_tm1), transitions, rng_key)

  def double_q_loss_fn(q_tm1, q_t, q_target_t, transitions, rng_key):
    """Regular DoubleQ loss using own networks."""
    return _double_q_loss(primary(q_tm1), primary(q_t), primary(q_target_t),
                          transitions, rng_key)

  def double_q_loss_v_fn(q_tm1, q_t, q_target_t, transitions, rng_key):
    """DoubleQ loss using other network's (target) value function."""
    return _double_q_loss(primary(q_tm1), primary(q_t), secondary(q_target_t),
                          transitions, rng_key)

  def double_q_loss_p_fn(q_tm1, q_t, q_target_t, transitions, rng_key):
    """DoubleQ loss using other network's (online) argmax policy."""
    return _double_q_loss(primary(q_tm1), secondary(q_t), primary(q_target_t),
                          transitions, rng_key)

  def double_q_loss_pv_fn(q_tm1, q_t, q_target_t, transitions, rng_key):
    """DoubleQ loss using other network's argmax policy & target value fn."""
    return _double_q_loss(primary(q_tm1), secondary(q_t), secondary(q_target_t),
                          transitions, rng_key)

  # Pure regression.
  def q_regression_loss_fn(q_tm1, q_t, q_target_t, transitions, rng_key):
    """Pure regression of q_tm1(self) towards q_tm1(other)."""
    del q_t, q_target_t, transitions, rng_key   # Unused.
    return _q_regression_loss(primary(q_tm1), secondary(q_tm1))

  # QR loss.
  def qr_loss_fn(q_tm1, q_t, q_target_t, transitions, rng_key):
    """QR-Q loss using own networks."""
    return _qr_loss(primary(q_tm1), primary(q_t), primary(q_target_t),
                    transitions, rng_key)

  if loss_type == 'double_q':
    return double_q_loss_fn
  elif loss_type == 'sarsa':
    return sarsa_loss_fn
  elif loss_type == 'mc_return':
    return mc_loss_fn
  elif loss_type == 'double_q_v':
    return double_q_loss_v_fn
  elif loss_type == 'double_q_p':
    return double_q_loss_p_fn
  elif loss_type == 'double_q_pv':
    return double_q_loss_pv_fn
  elif loss_type == 'q_regression':
    return q_regression_loss_fn
  elif loss_type == 'qr':
    return qr_loss_fn
  else:
    raise ValueError('Unknown loss "{}"'.format(loss_type))
