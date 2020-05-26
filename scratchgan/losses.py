# Lint as: python3
# Copyright 2019 DeepMind Technologies Limited and Google LLC
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
"""Losses for sequential GANs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf


def sequential_cross_entropy_loss(logits, expected):
  """The cross entropy loss for binary classification.

  Used to train the discriminator when not using WGAN loss.
  Assume logits is the log probability of classifying as 1. (real).

  Args:
    logits: a `tf.Tensor`, the model produced logits, shape [batch_size,
      sequence_length].
    expected: a `tf.Tensor`, the expected output, shape [batch_size,
      sequence_length].

  Returns:
    A scalar `tf.Tensor`, the average loss obtained on the given inputs.
  """
  batch_size, sequence_length = logits.shape.as_list()
  expected = tf.cast(expected, tf.float32)

  ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=expected, logits=logits)
  return tf.reshape(ce, [batch_size, sequence_length])


def reinforce_loss(disc_logits, gen_logprobs, gamma, decay):
  """The REINFORCE loss.

  Args:
      disc_logits: float tensor, shape [batch_size, sequence_length].
      gen_logprobs: float32 tensor, shape [batch_size, sequence_length]
      gamma: a float, discount factor for cumulative reward.
      decay: a float, decay rate for the EWMA baseline of REINFORCE.

  Returns:
    Float tensor, shape [batch_size, sequence_length], the REINFORCE loss for
    each timestep.
  """
  # Assume 1 logit for each timestep.
  batch_size, sequence_length = disc_logits.shape.as_list()
  gen_logprobs.shape.assert_is_compatible_with([batch_size, sequence_length])

  disc_predictions = tf.nn.sigmoid(disc_logits)

  # MaskGAN uses log(D), but this is more stable empirically.
  rewards = 2.0 * disc_predictions - 1

  # Compute cumulative rewards.
  rewards_list = tf.unstack(rewards, axis=1)
  cumulative_rewards = []
  for t in range(sequence_length):
    cum_value = tf.zeros(shape=[batch_size])
    for s in range(t, sequence_length):
      cum_value += np.power(gamma, (s - t)) * rewards_list[s]
    cumulative_rewards.append(cum_value)
  cumulative_rewards = tf.stack(cumulative_rewards, axis=1)

  cumulative_rewards.shape.assert_is_compatible_with(
      [batch_size, sequence_length])

  with tf.variable_scope("reinforce", reuse=tf.AUTO_REUSE):
    ewma_reward = tf.get_variable("ewma_reward", initializer=0.0)

  mean_reward = tf.reduce_mean(cumulative_rewards)
  new_ewma_reward = decay * ewma_reward + (1.0 - decay) * mean_reward
  update_op = tf.assign(ewma_reward, new_ewma_reward)

  # REINFORCE
  with tf.control_dependencies([update_op]):
    advantage = cumulative_rewards - ewma_reward
    loss = -tf.stop_gradient(advantage) * gen_logprobs

  loss.shape.assert_is_compatible_with([batch_size, sequence_length])
  return loss, cumulative_rewards, ewma_reward
