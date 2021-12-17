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
"""Encoder architectures to be used with VAE."""

import abc
from typing import Generic, TypeVar

import haiku as hk
import jax
import jax.numpy as jnp

from avae import types

_Params = TypeVar('_Params')


class EncoderBase(hk.Module, Generic[_Params]):
  """Abstract class for encoder architectures."""

  def __init__(self, latent_dim: int):
    """Class initializer.

    Args:
     latent_dim: Latent dimensions of the model.
    """
    super().__init__()
    self._latent_dim = latent_dim

  @abc.abstractmethod
  def __call__(self, input_data: jnp.ndarray) -> _Params:
    """Return posterior distribution over latents.

    Args:
     input_data: Input batch of shape (batch_size, ...).

    Returns:
     Parameters of the posterior distribution over the latents.
    """

  @abc.abstractmethod
  def sample(self, posterior: _Params, key: jnp.ndarray) -> jnp.ndarray:
    """Sample from the given posterior distribution.

    Args:
     posterior: Parameters of posterior distribution over the latents.
     key: Random number generator key.

    Returns:
     Sample from the posterior distribution over latents,
     shape[batch_size, latent_dim]
    """


class ColorMnistMLPEncoder(EncoderBase[types.NormalParams]):
  """MLP encoder for ColorMnist."""

  _hidden_units = (200, 200, 200, 200)

  def __call__(
      self, input_data: jnp.ndarray) -> types.NormalParams:
    """Return posterior distribution over latents.

    Args:
     input_data: Input batch of shape (batch_size, ...).

    Returns:
     Posterior distribution over the latents.
    """
    out = hk.Flatten()(input_data)
    for units in self._hidden_units:
      out = hk.Linear(units)(out)
      out = jax.nn.relu(out)
    out = hk.Linear(2 * self._latent_dim)(out)
    return _normal_params_from_logits(out)

  def sample(
      self,
      posterior: types.NormalParams,
      key: jnp.ndarray,
  ) -> jnp.ndarray:
    """Sample from the given normal posterior (mean, var) distribution.

    Args:
     posterior: Posterior over the latents.
     key: Random number generator key.
    Returns:
     Sample from the posterior distribution over latents,
     shape[batch_size, latent_dim]
    """
    eps = jax.random.normal(
        key, shape=(posterior.mean.shape[0], self._latent_dim))
    return posterior.mean + eps * posterior.variance


def _normal_params_from_logits(
    logits: jnp.ndarray) -> types.NormalParams:
  """Construct mean and variance of normal distribution from given logits."""
  mean, log_variance = jnp.split(logits, 2, axis=1)
  variance = jnp.exp(log_variance)
  return types.NormalParams(mean=mean, variance=variance)
