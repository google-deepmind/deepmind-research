# Lint as: python3
# Copyright 2020 DeepMind Technologies Limited.
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
"""Bernoulli Gated Linear Network."""

from typing import List, Text, Tuple

import chex
import jax
import jax.numpy as jnp
import rlax
import tensorflow_probability as tfp

from gated_linear_networks import base

tfp = tfp.experimental.substrates.jax
tfd = tfp.distributions

Array = chex.Array

GLN_EPS = 0.01
MAX_WEIGHT = 200.


class GatedLinearNetwork(base.GatedLinearNetwork):
  """Bernoulli Gated Linear Network."""

  def __init__(self,
               output_sizes: List[int],
               context_dim: int,
               name: Text = "bernoulli_gln"):
    """Initialize a Bernoulli GLN."""
    super(GatedLinearNetwork, self).__init__(
        output_sizes,
        context_dim,
        inference_fn=GatedLinearNetwork._inference_fn,
        update_fn=GatedLinearNetwork._update_fn,
        init=jnp.zeros,
        dtype=jnp.float32,
        name=name)

  def _add_bias(self, inputs):
    return jnp.append(inputs, rlax.sigmoid(1.))

  @staticmethod
  def _inference_fn(
      inputs: Array,           # [input_size]
      side_info: Array,        # [side_info_size]
      weights: Array,          # [2**context_dim, input_size]
      hyperplanes: Array,      # [context_dim, side_info_size]
      hyperplane_bias: Array,  # [context_dim]
  ) -> Array:
    """Inference step for a single Beurnolli neuron."""

    weight_index = GatedLinearNetwork._compute_context(side_info, hyperplanes,
                                                       hyperplane_bias)
    used_weights = weights[weight_index]
    inputs = rlax.logit(jnp.clip(inputs, GLN_EPS, 1. - GLN_EPS))
    prediction = rlax.sigmoid(jnp.dot(used_weights, inputs))

    return prediction

  @staticmethod
  def _update_fn(
      inputs: Array,           # [input_size]
      side_info: Array,        # [side_info_size]
      weights: Array,          # [2**context_dim, num_features]
      hyperplanes: Array,      # [context_dim, side_info_size]
      hyperplane_bias: Array,  # [context_dim]
      target: Array,           # []
      learning_rate: float,
  ) -> Tuple[Array, Array, Array]:
    """Update step for a single Bernoulli neuron."""

    def log_loss_fn(inputs, side_info, weights, hyperplanes, hyperplane_bias,
                    target):
      """Log loss for a single Bernoulli neuron."""
      prediction = GatedLinearNetwork._inference_fn(inputs, side_info, weights,
                                                    hyperplanes,
                                                    hyperplane_bias)
      prediction = jnp.clip(prediction, GLN_EPS, 1. - GLN_EPS)
      return rlax.log_loss(prediction, target), prediction

    grad_log_loss = jax.value_and_grad(log_loss_fn, argnums=2, has_aux=True)
    ((log_loss, prediction),
     dloss_dweights) = grad_log_loss(inputs, side_info, weights, hyperplanes,
                                     hyperplane_bias, target)

    delta_weights = learning_rate * dloss_dweights
    new_weights = jnp.clip(weights - delta_weights, -MAX_WEIGHT, MAX_WEIGHT)
    return new_weights, prediction, log_loss


class LastNeuronAggregator(base.LastNeuronAggregator):
  """Bernoulli last neuron aggregator, implemented by the super class."""
  pass
