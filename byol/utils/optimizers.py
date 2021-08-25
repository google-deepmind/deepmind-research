# Copyright 2020 DeepMind Technologies Limited.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of LARS Optimizer with optax."""

from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
import tree as nest

# A filter function takes a path and a value as input and outputs True for
# variable to apply update and False not to apply the update
FilterFn = Callable[[Tuple[Any], jnp.ndarray], jnp.ndarray]


def exclude_bias_and_norm(path: Tuple[Any], val: jnp.ndarray) -> jnp.ndarray:
  """Filter to exclude biaises and normalizations weights."""
  del val
  if path[-1] == "b" or "norm" in path[-2]:
    return False
  return True


def _partial_update(updates: optax.Updates,
                    new_updates: optax.Updates,
                    params: optax.Params,
                    filter_fn: Optional[FilterFn] = None) -> optax.Updates:
  """Returns new_update for params which filter_fn is True else updates."""

  if filter_fn is None:
    return new_updates

  wrapped_filter_fn = lambda x, y: jnp.array(filter_fn(x, y))
  params_to_filter = nest.map_structure_with_path(wrapped_filter_fn, params)

  def _update_fn(g: jnp.ndarray, t: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
    m = m.astype(g.dtype)
    return g * (1. - m) + t * m

  return jax.tree_multimap(_update_fn, updates, new_updates, params_to_filter)


class ScaleByLarsState(NamedTuple):
  mu: jnp.ndarray


def scale_by_lars(
    momentum: float = 0.9,
    eta: float = 0.001,
    filter_fn: Optional[FilterFn] = None) -> optax.GradientTransformation:
  """Rescales updates according to the LARS algorithm.

  Does not include weight decay.
  References:
    [You et al, 2017](https://arxiv.org/abs/1708.03888)

  Args:
    momentum: momentum coeficient.
    eta: LARS coefficient.
    filter_fn: an optional filter function.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params: optax.Params) -> ScaleByLarsState:
    mu = jax.tree_multimap(jnp.zeros_like, params)  # momentum
    return ScaleByLarsState(mu=mu)

  def update_fn(updates: optax.Updates, state: ScaleByLarsState,
                params: optax.Params) -> Tuple[optax.Updates, ScaleByLarsState]:

    def lars_adaptation(
        update: jnp.ndarray,
        param: jnp.ndarray,
    ) -> jnp.ndarray:
      param_norm = jnp.linalg.norm(param)
      update_norm = jnp.linalg.norm(update)
      return update * jnp.where(
          param_norm > 0.,
          jnp.where(update_norm > 0,
                    (eta * param_norm / update_norm), 1.0), 1.0)

    adapted_updates = jax.tree_multimap(lars_adaptation, updates, params)
    adapted_updates = _partial_update(updates, adapted_updates, params,
                                      filter_fn)
    mu = jax.tree_multimap(lambda g, t: momentum * g + t,
                           state.mu, adapted_updates)
    return mu, ScaleByLarsState(mu=mu)

  return optax.GradientTransformation(init_fn, update_fn)


class AddWeightDecayState(NamedTuple):
  """Stateless transformation."""


def add_weight_decay(
    weight_decay: float,
    filter_fn: Optional[FilterFn] = None) -> optax.GradientTransformation:
  """Adds a weight decay to the update.

  Args:
    weight_decay: weight_decay coeficient.
    filter_fn: an optional filter function.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_) -> AddWeightDecayState:
    return AddWeightDecayState()

  def update_fn(
      updates: optax.Updates,
      state: AddWeightDecayState,
      params: optax.Params,
  ) -> Tuple[optax.Updates, AddWeightDecayState]:
    new_updates = jax.tree_multimap(lambda g, p: g + weight_decay * p, updates,
                                    params)
    new_updates = _partial_update(updates, new_updates, params, filter_fn)
    return new_updates, state

  return optax.GradientTransformation(init_fn, update_fn)


LarsState = List  # Type for the lars optimizer


def lars(
    learning_rate: float,
    weight_decay: float = 0.,
    momentum: float = 0.9,
    eta: float = 0.001,
    weight_decay_filter: Optional[FilterFn] = None,
    lars_adaptation_filter: Optional[FilterFn] = None,
) -> optax.GradientTransformation:
  """Creates lars optimizer with weight decay.

  References:
    [You et al, 2017](https://arxiv.org/abs/1708.03888)

  Args:
    learning_rate: learning rate coefficient.
    weight_decay: weight decay coefficient.
    momentum: momentum coefficient.
    eta: LARS coefficient.
    weight_decay_filter: optional filter function to only apply the weight
      decay on a subset of parameters. The filter function takes as input the
      parameter path (as a tuple) and its associated update, and return a True
      for params to apply the weight decay and False for params to not apply
      the weight decay. When weight_decay_filter is set to None, the weight
      decay is not applied to the bias, i.e. when the variable name is 'b', and
      the weight decay is not applied to nornalization params, i.e. the
      panultimate path contains 'norm'.
    lars_adaptation_filter: similar to weight decay filter but for lars
      adaptation

  Returns:
    An optax.GradientTransformation, i.e. a (init_fn, update_fn) tuple.
  """

  if weight_decay_filter is None:
    weight_decay_filter = lambda *_: True
  if lars_adaptation_filter is None:
    lars_adaptation_filter = lambda *_: True

  return optax.chain(
      add_weight_decay(
          weight_decay=weight_decay, filter_fn=weight_decay_filter),
      scale_by_lars(
          momentum=momentum, eta=eta, filter_fn=lars_adaptation_filter),
      optax.scale(-learning_rate),
  )
