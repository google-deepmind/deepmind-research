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

"""Standard VAE class."""

from typing import Optional

import jax
import jax.numpy as jnp

from avae import decoders
from avae import encoders
from avae import kl
from avae import types


class VAE:
  """VAE class.

  This class defines the ELBO used in training VAE models. It also adds function
  for forward passing data through VAE.
  """

  def __init__(self, encoder: encoders.EncoderBase,
               decoder: decoders.DecoderBase, rho: Optional[float] = None):
    """Class initializer.

    Args:
     encoder: Encoder network architecture.
     decoder: Decoder network architecture.
     rho: Rho parameter used in AVAE training.
    """
    self._encoder = encoder
    self._decoder = decoder
    self._rho = rho

  def vae_elbo(
      self, input_data: jnp.ndarray,
      key: jnp.ndarray) -> types.ELBOOutputs:
    """ELBO for training VAE.

    Args:
     input_data: Input batch of shape (batch_size, ...).
     key: Key for random number generator.
    Returns:

     Computed VAE Elbo as type util_dataclasses.ELBOOutputs
    """
    posterior = self._encoder(input_data)
    samples = self._encoder.sample(posterior, key)
    kls = jax.vmap(kl.kl_p_with_uniform_normal, [0])(
        posterior.mean, posterior.variance)
    recons = self._decoder(samples)
    data_fidelity = self._decoder.data_fidelity(input_data, recons)
    elbo = data_fidelity - kls
    return types.ELBOOutputs(elbo, data_fidelity, kls)

  def avae_elbo(
      self, input_data: jnp.ndarray,
      key: jnp.ndarray) -> types.ELBOOutputs:
    """ELBO for training AVAE model.

    Args:
     input_data: Input batch of shape (batch_size, ...).
     key: Key for random number generator.
    Returns:
     Computed AVAE Elbo in nested tuple (Elbo, (data_fidelity, KL)). All arrays
     have batch dimension intact.
    """
    aux_images = jax.lax.stop_gradient(self(input_data, key))

    posterior = self._encoder(input_data)
    samples = self._encoder.sample(posterior, key)
    kls = jax.vmap(kl.kl_p_with_uniform_normal, [0, 0])(
        posterior.mean, posterior.variance)
    recons = self._decoder(samples)
    data_fidelity = self._decoder.data_fidelity(input_data, recons)
    elbo = data_fidelity - kls

    aux_posterior = self._encoder(aux_images)
    latent_mean = posterior.mean
    latent_var = posterior.variance
    aux_latent_mean = aux_posterior.mean
    aux_latent_var = aux_posterior.variance
    latent_dim = latent_mean.shape[1]

    def _reduce(x):
      return jnp.mean(jnp.sum(x, axis=1))

    # Computation of <log p(Z_aux | Z)>.
    expected_log_conditional = (
        aux_latent_var + jnp.square(self._rho) * latent_var +
        jnp.square(aux_latent_mean - self._rho * latent_mean))
    expected_log_conditional = _reduce(expected_log_conditional)
    expected_log_conditional /= 2.0 * (1.0 - jnp.square(self._rho))
    expected_log_conditional = (latent_dim *
                                jnp.log(1.0 / (2 * jnp.pi)) -
                                expected_log_conditional)
    elbo += expected_log_conditional
    # Entropy of Z_aux
    elbo += _reduce(0.5 * jnp.log(2 * jnp.pi * jnp.e * aux_latent_var))

    return types.ELBOOutputs(elbo, data_fidelity, kls)

  def __call__(
      self, input_data: jnp.ndarray,
      key: jnp.ndarray) -> jnp.ndarray:
    """Reconstruction of the input data.

    Args:
     input_data: Input batch of shape (batch_size, ...).
     key: Key for random number generator.
    Returns:
     Reconstruction of the input data as jnp.ndarray of shape
     [batch_dim, observation_dims].
    """
    posterior = self._encoder(input_data)
    samples = self._encoder.sample(posterior, key)
    recons = self._decoder(samples)
    return recons

