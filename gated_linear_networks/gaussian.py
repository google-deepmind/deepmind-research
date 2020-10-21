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
"""Gaussian Gated Linear Network."""

from typing import Callable, List, Text, Tuple

import chex
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp

from gated_linear_networks import base

tfp = tfp.experimental.substrates.jax
tfd = tfp.distributions

Array = chex.Array

MIN_SIGMA_SQ_AGGREGATOR = 0.5
MAX_SIGMA_SQ = 1e5
MAX_WEIGHT = 1e3
MIN_WEIGHT = -1e3


def _unpack_inputs(inputs: Array) -> Tuple[Array, Array]:
  inputs = jnp.atleast_2d(inputs)
  chex.assert_rank(inputs, 2)
  (mu, sigma_sq) = [jnp.squeeze(x, 1) for x in jnp.hsplit(inputs, 2)]
  return mu, sigma_sq


def _pack_inputs(mu: Array, sigma_sq: Array) -> Array:
  mu = jnp.atleast_1d(mu)
  sigma_sq = jnp.atleast_1d(sigma_sq)
  chex.assert_rank([mu, sigma_sq], 1)
  return jnp.vstack([mu, sigma_sq]).T


class GatedLinearNetwork(base.GatedLinearNetwork):
  """Gaussian Gated Linear Network."""

  def __init__(
      self,
      output_sizes: List[int],
      context_dim: int,
      bias_len: int = 3,
      bias_max_mu: float = 1.,
      bias_sigma_sq: float = 1.,
      name: Text = "gaussian_gln"):
    """Initialize a Gaussian GLN."""
    super(GatedLinearNetwork, self).__init__(
        output_sizes,
        context_dim,
        inference_fn=GatedLinearNetwork._inference_fn,
        update_fn=GatedLinearNetwork._update_fn,
        init=base.ShapeScaledConstant(),
        dtype=jnp.float64,
        name=name)

    self._bias_len = bias_len
    self._bias_max_mu = bias_max_mu
    self._bias_sigma_sq = bias_sigma_sq

  def _add_bias(self, inputs):
    mu = jnp.linspace(-1. * self._bias_max_mu, self._bias_max_mu,
                      self._bias_len)
    sigma_sq = self._bias_sigma_sq * jnp.ones_like(mu)
    bias = _pack_inputs(mu, sigma_sq)
    return jnp.concatenate([inputs, bias], axis=0)

  @staticmethod
  def _inference_fn(
      inputs: Array,           # [input_size, 2]
      side_info: Array,        # [side_info_size]
      weights: Array,          # [2**context_dim, input_size]
      hyperplanes: Array,      # [context_dim, side_info_size]
      hyperplane_bias: Array,  # [context_dim]
      min_sigma_sq: float,
  ) -> Array:
    """Inference step for a single Gaussian neuron."""

    mu_in, sigma_sq_in = _unpack_inputs(inputs)
    weight_index = GatedLinearNetwork._compute_context(side_info, hyperplanes,
                                                       hyperplane_bias)
    used_weights = weights[weight_index]

    # This projection operation is differentiable and affects the gradients.
    used_weights = GatedLinearNetwork._project_weights(inputs, used_weights,
                                                       min_sigma_sq)

    sigma_sq_out = 1. / jnp.sum(used_weights / sigma_sq_in)
    mu_out = sigma_sq_out * jnp.sum((used_weights * mu_in) / sigma_sq_in)
    prediction = jnp.hstack((mu_out, sigma_sq_out))
    return prediction

  @staticmethod
  def _project_weights(inputs: Array,     # [input_size]
                       weights: Array,    # [2**context_dim, num_features]
                       min_sigma_sq: float) -> Array:
    """Implements hard projection."""

    # This projection should be performed before the sigma related ones.
    weights = jnp.minimum(jnp.maximum(MIN_WEIGHT, weights), MAX_WEIGHT)
    _, sigma_sq_in = _unpack_inputs(inputs)

    lambda_in = 1. / sigma_sq_in
    sigma_sq_out = 1. / weights.dot(lambda_in)

    # If w.dot(x) < U, linearly project w such that w.dot(x) = U.
    weights = jnp.where(
        sigma_sq_out < min_sigma_sq, weights - lambda_in *
        (1. / sigma_sq_out - 1. / min_sigma_sq) / jnp.sum(lambda_in**2),
        weights)

    # If w.dot(x) > U, linearly project w such that w.dot(x) = U.
    weights = jnp.where(
        sigma_sq_out > MAX_SIGMA_SQ, weights - lambda_in *
        (1. / sigma_sq_out - 1. / MAX_SIGMA_SQ) / jnp.sum(lambda_in**2),
        weights)

    return weights

  @staticmethod
  def _update_fn(
      inputs: Array,           # [input_size]
      side_info: Array,        # [side_info_size]
      weights: Array,          # [2**context_dim, num_features]
      hyperplanes: Array,      # [context_dim, side_info_size]
      hyperplane_bias: Array,  # [context_dim]
      target: Array,           # []
      learning_rate: float,
      min_sigma_sq: float,     # needed for inference (weight projection)
      ) -> Tuple[Array, Array, Array]:
    """Update step for a single Gaussian neuron."""

    def log_loss_fn(inputs, side_info, weights, hyperplanes, hyperplane_bias,
                    target):
      """Log loss for a single Gaussian neuron."""
      prediction = GatedLinearNetwork._inference_fn(inputs, side_info, weights,
                                                    hyperplanes,
                                                    hyperplane_bias,
                                                    min_sigma_sq)
      mu, sigma_sq = prediction.T
      loss = -tfd.Normal(mu, jnp.sqrt(sigma_sq)).log_prob(target)
      return loss, prediction

    grad_log_loss = jax.value_and_grad(log_loss_fn, argnums=2, has_aux=True)
    (log_loss,
     prediction), dloss_dweights = grad_log_loss(inputs, side_info, weights,
                                                 hyperplanes, hyperplane_bias,
                                                 target)

    delta_weights = learning_rate * dloss_dweights
    return weights - delta_weights, prediction, log_loss


class ConstantInputSigma(base.Mutator):
  """Input pre-processing by concatenating a constant sigma^2."""

  def __init__(
      self,
      network_factory: Callable[..., GatedLinearNetwork],
      input_sigma_sq: float,
      name: Text = "constant_input_sigma",
  ):
    super(ConstantInputSigma, self).__init__(network_factory, name)
    self._input_sigma_sq = input_sigma_sq

  def inference(self, inputs, *args, **kwargs):
    """ConstantInputSigma inference."""
    chex.assert_rank(inputs, 1)
    sigma_sq = self._input_sigma_sq * jnp.ones_like(inputs)
    return self._network.inference(_pack_inputs(inputs, sigma_sq), *args,
                                   **kwargs)

  def update(self, inputs, *args, **kwargs):
    """ConstantInputSigma update."""
    chex.assert_rank(inputs, 1)
    sigma_sq = self._input_sigma_sq * jnp.ones_like(inputs)
    return self._network.update(_pack_inputs(inputs, sigma_sq), *args, **kwargs)


class LastNeuronAggregator(base.LastNeuronAggregator):
  """Gaussian last neuron aggregator, implemented by the super class."""
  pass
