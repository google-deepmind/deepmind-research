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
"""Common functions used across more than one test."""
import jax
import jax.numpy as jnp
import jax.random as jnr

from kfac_ferminet_alpha import loss_functions


def fully_connected_layer(params, x):
  w, b = params
  return jnp.matmul(x, w) + b[None]


def init_autoencoder(key, data_shape):
  """Initialize the standard autoencoder."""
  assert len(data_shape) == 1
  x_size = data_shape[0]
  sizes = [x_size, 1000, 500, 250, 30, 250, 500, 1000, x_size]
  keys = jnr.split(key, len(sizes) - 1)
  params = []
  for key, dim_in, dim_out in zip(keys, sizes, sizes[1:]):
    # Glorot uniform initialization
    c = jnp.sqrt(6 / (dim_in + dim_out))
    w = jax.random.uniform(key, shape=(dim_in, dim_out), minval=-c, maxval=c)
    b = jnp.zeros([dim_out])
    params.append((w, b))
  return params


def autoencoder(all_params, x_in):
  """Evaluate the standard autoencoder.

  Note that the objective of this autoencoder is not standard, bur rather a sum
  of the standard sigmoid crossentropy and squared loss. The reason for this is
  to test on handling multiple losses.

  Args:
    all_params: All parameter values.
    x_in: Inputs to the network.

  Returns:
      The value of the two losses and intermediate layer values.
  """
  h_in = x_in
  layers_values = []
  for i, params in enumerate(all_params):
    h_out = fully_connected_layer(params, h_in)
    layers_values.append((h_out, h_in))
    # Last layer does not have a nonlinearity
    if i % 4 != 3:
      # h_in = nn.leaky_relu(h_out)
      h_in = jnp.tanh(h_out)
    else:
      h_in = h_out
  h1, _ = loss_functions.register_normal_predictive_distribution(h_in, x_in)
  h2, _ = loss_functions.register_normal_predictive_distribution(
      h_in, targets=x_in, weight=0.1)
  l1 = (h1 - x_in)**2 + jnp.log(jnp.pi) / 2
  l1 = jnp.sum(l1, axis=-1)
  l2 = (h2 - x_in)**2 + jnp.log(jnp.pi) / 2
  l2 = jnp.sum(l2, axis=-1)
  return [l1, l2 * 0.1], layers_values
