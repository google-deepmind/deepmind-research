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
"""DQN agent."""

import numpy as np
import sonnet as snt
import tensorflow.compat.v1 as tf


class Agent():
  """A DQN Agent."""

  def __init__(
      self,
      obs_spec,
      action_spec,
      network_kwargs,
      epsilon,
      additional_discount,
      batch_size,
      optimizer_name,
      optimizer_kwargs,
  ):
    """A simple DQN agent.

    Args:
      obs_spec: The observation spec.
      action_spec: The action spec.
      network_kwargs: Keyword arguments for snt.nets.MLP
      epsilon: Exploration probability.
      additional_discount: Discount on returns used by the agent.
      batch_size: Size of update batch.
      optimizer_name: Name of an optimizer from tf.train
      optimizer_kwargs: Keyword arguments for the optimizer.
    """

    self._epsilon = epsilon
    self._additional_discount = additional_discount
    self._batch_size = batch_size

    self._n_actions = action_spec.num_values
    self._network = ValueNet(self._n_actions, network_kwargs=network_kwargs)

    self._replay = []

    obs_spec = self._extract_observation(obs_spec)

    # Placeholders for policy
    o = tf.placeholder(shape=obs_spec.shape, dtype=obs_spec.dtype)
    q = self._network(tf.expand_dims(o, axis=0))

    # Placeholders for update.
    o_tm1 = tf.placeholder(shape=(None,) + obs_spec.shape, dtype=obs_spec.dtype)
    a_tm1 = tf.placeholder(shape=(None,), dtype=tf.int32)
    r_t = tf.placeholder(shape=(None,), dtype=tf.float32)
    d_t = tf.placeholder(shape=(None,), dtype=tf.float32)
    o_t = tf.placeholder(shape=(None,) + obs_spec.shape, dtype=obs_spec.dtype)

    # Compute values over all options.
    q_tm1 = self._network(o_tm1)
    q_t = self._network(o_t)

    a_t = tf.cast(tf.argmax(q_t, axis=-1), tf.int32)
    qa_tm1 = _batched_index(q_tm1, a_tm1)
    qa_t = _batched_index(q_t, a_t)

    # TD error
    g = additional_discount * d_t
    td_error = tf.stop_gradient(r_t + g * qa_t) - qa_tm1
    loss = tf.reduce_sum(tf.square(td_error) / 2)

    with tf.variable_scope("optimizer"):
      self._optimizer = getattr(tf.train, optimizer_name)(**optimizer_kwargs)
      train_op = self._optimizer.minimize(loss)

    # Make session and callables.
    session = tf.Session()
    self._update_fn = session.make_callable(train_op,
                                            [o_tm1, a_tm1, r_t, d_t, o_t])
    self._value_fn = session.make_callable(q, [o])
    session.run(tf.global_variables_initializer())

  def _extract_observation(self, obs):
    return obs["arena"]

  def step(self, timestep, is_training=False):
    """Select actions according to epsilon-greedy policy."""

    if is_training and np.random.rand() < self._epsilon:
      return np.random.randint(self._n_actions)

    q_values = self._value_fn(
        self._extract_observation(timestep.observation))
    return int(np.argmax(q_values))

  def update(self, step_tm1, action, step_t):
    """Takes in a transition from the environment."""

    transition = [
        self._extract_observation(step_tm1.observation),
        action,
        step_t.reward,
        step_t.discount,
        self._extract_observation(step_t.observation),
    ]
    self._replay.append(transition)

    if len(self._replay) == self._batch_size:
      batch = list(zip(*self._replay))
      self._update_fn(*batch)
      self._replay = []  # Just a queue.


class ValueNet(snt.AbstractModule):
  """Value Network."""

  def __init__(self,
               n_actions,
               network_kwargs,
               name="value_network"):
    """Construct a value network sonnet module.

    Args:
      n_actions: Number of actions.
      network_kwargs: Network arguments.
      name: Name
    """
    super(ValueNet, self).__init__(name=name)
    self._n_actions = n_actions
    self._network_kwargs = network_kwargs

  def _build(self, observation):
    flat_obs = snt.BatchFlatten()(observation)
    net = snt.nets.MLP(**self._network_kwargs)(flat_obs)
    net = snt.Linear(output_size=self._n_actions)(net)

    return net

  @property
  def num_actions(self):
    return self._n_actions


def _batched_index(values, indices):
  one_hot_indices = tf.one_hot(indices, values.shape[-1], dtype=values.dtype)
  return tf.reduce_sum(values * one_hot_indices, axis=-1)
