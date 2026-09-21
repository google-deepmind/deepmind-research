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
"""Regressed agent."""

import numpy as np
import tensorflow.compat.v1 as tf


class Agent():
  """A DQN Agent."""

  def __init__(
      self,
      batch_size,
      optimizer_name,
      optimizer_kwargs,
      init_w,
  ):
    """A simple DQN agent.

    Args:
      batch_size: Size of update batch.
      optimizer_name: Name of an optimizer from tf.train
      optimizer_kwargs: Keyword arguments for the optimizer.
      init_w: The initial cumulant weight.
    """
    self._batch_size = batch_size
    self._init_w = np.array(init_w)
    self._replay = []

    # Regress w by gradient descent, could also use closed-form solution.
    self._n_cumulants = len(init_w)
    self._regressed_w = tf.get_variable(
        "regressed_w",
        dtype=tf.float32,
        initializer=lambda: tf.to_float(init_w))
    cumulants_ph = tf.placeholder(
        shape=(None, self._n_cumulants), dtype=tf.float32)
    rewards_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    predicted_rewards = tf.reduce_sum(
        tf.multiply(self._regressed_w, cumulants_ph), axis=-1)
    loss = tf.reduce_sum(tf.square(predicted_rewards - rewards_ph))

    with tf.variable_scope("optimizer"):
      self._optimizer = getattr(tf.train, optimizer_name)(**optimizer_kwargs)
      train_op = self._optimizer.minimize(loss)

    # Make session and callables.
    session = tf.Session()
    self._update_fn = session.make_callable(train_op,
                                            [cumulants_ph, rewards_ph])
    self._action = session.make_callable(self._regressed_w.read_value(), [])
    session.run(tf.global_variables_initializer())

  def step(self, timestep, is_training=False):
    """Select actions according to epsilon-greedy policy."""
    del timestep

    if is_training:
      # Can also just use random actions at environment level.
      return np.random.uniform(low=-1.0, high=1.0, size=(self._n_cumulants,))

    return self._action()

  def update(self, step_tm1, action, step_t):
    """Takes in a transition from the environment."""
    del step_tm1, action

    transition = [
        step_t.observation["cumulants"],
        step_t.reward,
    ]
    self._replay.append(transition)

    if len(self._replay) == self._batch_size:
      batch = list(zip(*self._replay))
      self._update_fn(*batch)
      self._replay = []  # Just a queue.

  def get_logs(self):
    return dict(regressed=self._action())
