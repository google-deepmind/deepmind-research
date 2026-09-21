# pylint: disable=g-bad-file-header
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
"""Online data normalization."""

import sonnet as snt
import tensorflow.compat.v1 as tf


class Normalizer(snt.AbstractModule):
  """Feature normalizer that accumulates statistics online."""

  def __init__(self, size, max_accumulations=10**6, std_epsilon=1e-8,
               name='Normalizer'):
    super(Normalizer, self).__init__(name=name)
    self._max_accumulations = max_accumulations
    self._std_epsilon = std_epsilon
    with self._enter_variable_scope():
      self._acc_count = tf.Variable(0, dtype=tf.float32, trainable=False)
      self._num_accumulations = tf.Variable(0, dtype=tf.float32,
                                            trainable=False)
      self._acc_sum = tf.Variable(tf.zeros(size, tf.float32), trainable=False)
      self._acc_sum_squared = tf.Variable(tf.zeros(size, tf.float32),
                                          trainable=False)

  def _build(self, batched_data, accumulate=True):
    """Normalizes input data and accumulates statistics."""
    update_op = tf.no_op()
    if accumulate:
      # stop accumulating after a million updates, to prevent accuracy issues
      update_op = tf.cond(self._num_accumulations < self._max_accumulations,
                          lambda: self._accumulate(batched_data),
                          tf.no_op)
    with tf.control_dependencies([update_op]):
      return (batched_data - self._mean()) / self._std_with_epsilon()

  @snt.reuse_variables
  def inverse(self, normalized_batch_data):
    """Inverse transformation of the normalizer."""
    return normalized_batch_data * self._std_with_epsilon() + self._mean()

  def _accumulate(self, batched_data):
    """Function to perform the accumulation of the batch_data statistics."""
    count = tf.cast(tf.shape(batched_data)[0], tf.float32)
    data_sum = tf.reduce_sum(batched_data, axis=0)
    squared_data_sum = tf.reduce_sum(batched_data**2, axis=0)
    return tf.group(
        tf.assign_add(self._acc_sum, data_sum),
        tf.assign_add(self._acc_sum_squared, squared_data_sum),
        tf.assign_add(self._acc_count, count),
        tf.assign_add(self._num_accumulations, 1.))

  def _mean(self):
    safe_count = tf.maximum(self._acc_count, 1.)
    return self._acc_sum / safe_count

  def _std_with_epsilon(self):
    safe_count = tf.maximum(self._acc_count, 1.)
    std = tf.sqrt(self._acc_sum_squared / safe_count - self._mean()**2)
    return tf.math.maximum(std, self._std_epsilon)
