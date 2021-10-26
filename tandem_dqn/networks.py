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
"""DQN agent network components and implementation."""

import typing
from typing import Any, Callable, Tuple, Union

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

Network = hk.Transformed
Params = hk.Params
NetworkFn = Callable[..., Any]


class QNetworkOutputs(typing.NamedTuple):
  q_values: jnp.ndarray


class QRNetworkOutputs(typing.NamedTuple):
  q_values: jnp.ndarray
  q_dist: jnp.ndarray


NUM_QUANTILES = 201


def _dqn_default_initializer(
    num_input_units: int) -> hk.initializers.Initializer:
  """Default initialization scheme inherited from past implementations of DQN.

  This scheme was historically used to initialize all weights and biases
  in convolutional and linear layers of DQN-type agents' networks.
  It initializes each weight as an independent uniform sample from [`-c`, `c`],
  where `c = 1 / np.sqrt(num_input_units)`, and `num_input_units` is the number
  of input units affecting a single output unit in the given layer, i.e. the
  total number of inputs in the case of linear (dense) layers, and
  `num_input_channels * kernel_width * kernel_height` in the case of
  convolutional layers.

  Args:
    num_input_units: number of input units to a single output unit of the layer.

  Returns:
    Haiku weight initializer.
  """
  max_val = np.sqrt(1 / num_input_units)
  return hk.initializers.RandomUniform(-max_val, max_val)


def make_quantiles():
  """Quantiles for QR-DQN."""
  return (jnp.arange(0, NUM_QUANTILES) + 0.5) / float(NUM_QUANTILES)


def conv(
    num_features: int,
    kernel_shape: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    name=None,
) -> NetworkFn:
  """Convolutional layer with DQN's legacy weight initialization scheme."""

  def net_fn(inputs):
    """Function representing conv layer with DQN's legacy initialization."""
    num_input_units = inputs.shape[-1] * kernel_shape[0] * kernel_shape[1]
    initializer = _dqn_default_initializer(num_input_units)
    layer = hk.Conv2D(
        num_features,
        kernel_shape=kernel_shape,
        stride=stride,
        w_init=initializer,
        b_init=initializer,
        padding='VALID',
        name=name)
    return layer(inputs)

  return net_fn


def linear(num_outputs: int, with_bias=True, name=None) -> NetworkFn:
  """Linear layer with DQN's legacy weight initialization scheme."""

  def net_fn(inputs):
    """Function representing linear layer with DQN's legacy initialization."""
    initializer = _dqn_default_initializer(inputs.shape[-1])
    layer = hk.Linear(
        num_outputs,
        with_bias=with_bias,
        w_init=initializer,
        b_init=initializer,
        name=name)
    return layer(inputs)

  return net_fn


def linear_with_shared_bias(num_outputs: int, name=None) -> NetworkFn:
  """Linear layer with single shared bias instead of one bias per output."""

  def layer_fn(inputs):
    """Function representing a linear layer with single shared bias."""
    initializer = _dqn_default_initializer(inputs.shape[-1])
    bias_free_linear = hk.Linear(
        num_outputs, with_bias=False, w_init=initializer, name=name)
    linear_output = bias_free_linear(inputs)
    bias = hk.get_parameter('b', [1], inputs.dtype, init=initializer)
    bias = jnp.broadcast_to(bias, linear_output.shape)
    return linear_output + bias

  return layer_fn


def dqn_torso() -> NetworkFn:
  """DQN convolutional torso.

  Includes scaling from [`0`, `255`] (`uint8`) to [`0`, `1`] (`float32`)`.

  Returns:
    Network function that `haiku.transform` can be called on.
  """

  def net_fn(inputs):
    """Function representing convolutional torso for a DQN Q-network."""
    network = hk.Sequential([
        lambda x: x.astype(jnp.float32) / 255.,
        conv(32, kernel_shape=(8, 8), stride=(4, 4), name='conv1'),
        jax.nn.relu,
        conv(64, kernel_shape=(4, 4), stride=(2, 2), name='conv2'),
        jax.nn.relu,
        conv(64, kernel_shape=(3, 3), stride=(1, 1), name='conv3'),
        jax.nn.relu,
        hk.Flatten(),
    ])
    return network(inputs)

  return net_fn


def dqn_value_head(num_actions: int, shared_bias: bool = False) -> NetworkFn:
  """Regular DQN Q-value head with single hidden layer."""

  last_layer = linear_with_shared_bias if shared_bias else linear

  def net_fn(inputs):
    """Function representing value head for a DQN Q-network."""
    network = hk.Sequential([
        linear(512, name='linear1'),
        jax.nn.relu,
        last_layer(num_actions, name='output'),
    ])
    return network(inputs)

  return net_fn


def qr_atari_network(num_actions: int, quantiles: jnp.ndarray) -> NetworkFn:
  """QR-DQN network, expects `uint8` input."""

  chex.assert_rank(quantiles, 1)
  num_quantiles = len(quantiles)

  def net_fn(inputs):
    """Function representing QR-DQN Q-network."""
    network = hk.Sequential([
        dqn_torso(),
        dqn_value_head(num_quantiles * num_actions),
    ])
    network_output = network(inputs)
    q_dist = jnp.reshape(network_output, (-1, num_quantiles, num_actions))
    q_values = jnp.mean(q_dist, axis=1)
    q_values = jax.lax.stop_gradient(q_values)
    return QRNetworkOutputs(q_dist=q_dist, q_values=q_values)

  return net_fn


def double_dqn_atari_network(num_actions: int) -> NetworkFn:
  """DQN network with shared bias in final layer, expects `uint8` input."""

  def net_fn(inputs):
    """Function representing DQN Q-network with shared bias output layer."""
    network = hk.Sequential([
        dqn_torso(),
        dqn_value_head(num_actions, shared_bias=True),
    ])
    return QNetworkOutputs(q_values=network(inputs))

  return net_fn


def make_network(network_type: str, num_actions: int) -> Network:
  """Constructs network."""
  if network_type == 'double_q':
    network_fn = double_dqn_atari_network(num_actions)
  elif network_type == 'qr':
    quantiles = make_quantiles()
    network_fn = qr_atari_network(num_actions, quantiles)
  else:
    raise ValueError('Unknown network "{}"'.format(network_type))

  return hk.transform(network_fn)
