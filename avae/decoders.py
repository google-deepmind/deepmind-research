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

"""Decoder architectures to be used with VAE."""

import abc

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class DecoderBase(hk.Module):
  """Base class for decoder network classes."""

  def __init__(self, obs_var: float):
    """Class initializer.

    Args:
     obs_var: oversation variance of the dataset.
    """
    super().__init__()
    self._obs_var = obs_var

  @abc.abstractmethod
  def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
    """Reconstruct from a given latent sample.

    Args:
     z: latent samples of shape (batch_size, latent_dim)
    Returns:
     Reconstruction with shape (batch_size, ...).
    """

  def data_fidelity(
      self,
      input_data: jnp.ndarray,
      recons: jnp.ndarray,
  ) -> jnp.ndarray:
    """Compute Data fidelity (recons loss) for given input and recons.

    Args:
     input_data: Input batch of shape (batch_size, ...).
     recons: Reconstruction of the input data. An array with the same shape as
       `input_data.data`.
    Returns:
     Computed data fidelity term across batch of data. An array of shape
     `(batch_size,)`.
    """
    error = (input_data - recons).reshape(input_data.shape[0], -1)
    return -0.5 * jnp.sum(jnp.square(error), axis=1) / self._obs_var


class ColorMnistMLPDecoder(DecoderBase):
  """MLP decoder for Color Mnist."""

  _hidden_units = (200, 200, 200, 200)
  _image_dims = (28, 28, 3)  # Dimensions of a single MNIST image.

  def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
    """Reconstruct with given latent sample.

    Args:
     z: latent samples of shape (batch_size, latent_dim)
    Returns:
     Reconstructions data of shape (batch_size, 28, 28, 3).
    """
    out = z
    for units in self._hidden_units:
      out = hk.Linear(units)(out)
      out = jax.nn.relu(out)
    out = hk.Linear(np.product(self._image_dims))(out)
    out = jax.nn.sigmoid(out)
    return jnp.reshape(out, (-1,) + self._image_dims)

