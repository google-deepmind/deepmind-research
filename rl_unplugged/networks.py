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
"""Networks used for training agents.
"""

from acme.tf import networks as acme_networks
from acme.tf import utils as tf2_utils
import numpy as np
import sonnet as snt
import tensorflow as tf


def instance_norm_and_elu(x):
  mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
  x_ = x - mean
  var = tf.reduce_mean(x_**2, axis=[1, 2], keepdims=True)
  x_norm = x_ / (var + 1e-6)
  return tf.nn.elu(x_norm)


class ControlNetwork(snt.Module):
  """Image, proprio and optionally action encoder used for actors and critics.
  """

  def __init__(self,
               proprio_encoder_size: int,
               proprio_keys=None,
               activation=tf.nn.elu):
    """Creates a ControlNetwork.

    Args:
      proprio_encoder_size: Size of the linear layer for the proprio encoder.
      proprio_keys: Optional list of names of proprioceptive observations.
        Defaults to all observations. Note that if this is specified, any
        observation not contained in proprio_keys will be ignored by the agent.
      activation: Linear layer activation function.
    """
    super().__init__(name='control_network')
    self._activation = activation
    self._proprio_keys = proprio_keys

    self._proprio_encoder = acme_networks.LayerNormMLP([proprio_encoder_size])

  def __call__(self, inputs, action: tf.Tensor = None, task=None):
    """Evaluates the ControlNetwork.

    Args:
      inputs:  A dictionary of agent observation tensors.
      action:  Agent actions.
      task:    Optional encoding of the task.

    Raises:
      ValueError: if neither proprio_input is provided.
      ValueError: if some proprio input looks suspiciously like pixel inputs.

    Returns:
      Processed network output.
    """
    if not isinstance(inputs, dict):
      inputs = {'inputs': inputs}

    proprio_input = []
    # By default, treat all observations as proprioceptive.
    if self._proprio_keys is None:
      self._proprio_keys = list(sorted(inputs.keys()))
    for key in self._proprio_keys:
      proprio_input.append(snt.Flatten()(inputs[key]))
      if np.prod(inputs[key].shape[1:]) > 32*32*3:
        raise ValueError(
            'This input does not resemble a proprioceptive '
            'state: {} with shape {}'.format(
                key, inputs[key].shape))

    # Append optional action input (i.e. for critic networks).
    if action is not None:
      proprio_input.append(action)

    proprio_input = tf2_utils.batch_concat(proprio_input)
    proprio_state = self._proprio_encoder(proprio_input)

    return proprio_state
