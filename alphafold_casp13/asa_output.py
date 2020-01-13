# Lint as: python3.
# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Class for predicting Accessible Surface Area."""

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import


class ASAOutputLayer(object):
  """An output layer to predict Accessible Surface Area."""

  def __init__(self, name='asa'):
    self.name = name

  def compute_asa_output(self, activations):
    """Just compute the logits and outputs given activations."""
    asa_logits = tf.contrib.layers.linear(
        activations, 1,
        weights_initializer=tf.random_uniform_initializer(-0.01, 0.01),
        scope='ASALogits')
    self.asa_output = tf.nn.relu(asa_logits, name='ASA_output_relu')

    return asa_logits
