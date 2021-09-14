# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Functions and classes for performing variational inference."""

from typing import Callable, Iterable, Optional

import haiku as hk
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class Variational(hk.Module):
  """A module representing the variational distribution q(H | *O).

  H is assumed to be a continuous variable.
  """

  def __init__(self,
               common_layer_sizes: Iterable[int],
               activation: Callable[[jnp.DeviceArray],
                                    jnp.DeviceArray] = jnp.tanh,
               output_dim: int = 1,
               name: Optional[str] = None):
    """Initialises a `Variational` instance.

    Args:
      common_layer_sizes: The number of hidden units in the shared dense
        network layers.
      activation: Nonlinearity function to apply to each of the
        common layers.
      output_dim: The dimensionality of `H`.
      name: A name to assign to the module instance.
    """
    super().__init__(name=name)
    self._common_layer_sizes = common_layer_sizes
    self._activation = activation
    self._output_dim = output_dim

    self._linear_layers = [
        hk.Linear(layer_size)
        for layer_size in self._common_layer_sizes
    ]

    self._mean_output = hk.Linear(self._output_dim)
    self._log_var_output = hk.Linear(self._output_dim)

  def __call__(self, *args) -> tfd.Distribution:
    """Create a distribution for q(H | *O).

    Args:
      *args: `List[DeviceArray]`. Corresponds to the values of whatever
        variables are in the conditional set *O.

    Returns:
      `tfp.distributions.NormalDistribution` instance.
    """
    # Stack all inputs, ensuring that shapes are consistent and that they are
    # all of dtype float32.
    input_ = [hk.Flatten()(arg) for arg in args]
    input_ = jnp.concatenate(input_, axis=1)

    # Create a common set of layers, then final layer separates mean & log_var
    for layer in self._linear_layers:
      input_ = layer(input_)
      input_ = self._activation(input_)

    # input_ now represents a tensor of shape (batch_size, final_layer_size).
    # This is now put through two final layers, one for the computation of each
    # of the mean and standard deviation of the resultant distribution.
    mean = self._mean_output(input_)
    log_var = self._log_var_output(input_)
    std = jnp.sqrt(jnp.exp(log_var))
    return tfd.Normal(mean, std)
