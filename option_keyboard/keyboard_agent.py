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
"""Keyboard agent."""

import os

import numpy as np
import sonnet as snt
import tensorflow.compat.v1 as tf

from option_keyboard import smart_module


class Agent():
  """An Option Keyboard Agent."""

  def __init__(
      self,
      obs_spec,
      action_spec,
      policy_weights,
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
      policy_weights: A list of vectors each representing the cumulant weights
        for that particular option/policy.
      network_kwargs: Keyword arguments for snt.nets.MLP
      epsilon: Exploration probability.
      additional_discount: Discount on returns used by the agent.
      batch_size: Size of update batch.
      optimizer_name: Name of an optimizer from tf.train
      optimizer_kwargs: Keyword arguments for the optimizer.
    """

    tf.logging.info(policy_weights)
    self._policy_weights = tf.convert_to_tensor(
        policy_weights, dtype=tf.float32)
    self._current_policy = None

    self._epsilon = epsilon
    self._additional_discount = additional_discount
    self._batch_size = batch_size

    self._n_actions = action_spec.num_values
    self._n_policies, self._n_cumulants = policy_weights.shape

    def create_network():
      return OptionValueNet(
          self._n_policies,
          self._n_cumulants,
          self._n_actions,
          network_kwargs=network_kwargs,
      )

    self._network = smart_module.SmartModuleExport(create_network)
    self._replay = []

    obs_spec = self._extract_observation(obs_spec)

    def option_values(values, policy):
      return tf.tensordot(
          values[:, policy, ...], self._policy_weights[policy], axes=[1, 0])

    # Placeholders for policy.
    o = tf.placeholder(shape=obs_spec.shape, dtype=obs_spec.dtype)
    p = tf.placeholder(shape=(), dtype=tf.int32)
    q = self._network(tf.expand_dims(o, axis=0))
    qo = option_values(q, p)

    # Placeholders for update.
    o_tm1 = tf.placeholder(shape=(None,) + obs_spec.shape, dtype=obs_spec.dtype)
    a_tm1 = tf.placeholder(shape=(None,), dtype=tf.int32)
    c_t = tf.placeholder(shape=(None, self._n_cumulants), dtype=tf.float32)
    d_t = tf.placeholder(shape=(None,), dtype=tf.float32)
    o_t = tf.placeholder(shape=(None,) + obs_spec.shape, dtype=obs_spec.dtype)

    # Compute values over all options.
    q_tm1 = self._network(o_tm1)
    q_t = self._network(o_t)
    qo_t = option_values(q_t, p)

    a_t = tf.cast(tf.argmax(qo_t, axis=-1), tf.int32)
    qa_tm1 = _batched_index(q_tm1[:, p, ...], a_tm1)
    qa_t = _batched_index(q_t[:, p, ...], a_t)

    # TD error
    g = additional_discount * tf.expand_dims(d_t, axis=-1)
    td_error = tf.stop_gradient(c_t + g * qa_t) - qa_tm1
    loss = tf.reduce_sum(tf.square(td_error) / 2)

    # Dummy calls to keyboard for SmartModule
    _ = self._network.gpi(o_tm1[0], c_t[0])
    _ = self._network.num_cumulants
    _ = self._network.num_policies
    _ = self._network.num_actions

    with tf.variable_scope("optimizer"):
      self._optimizer = getattr(tf.train, optimizer_name)(**optimizer_kwargs)
      train_op = self._optimizer.minimize(loss)

    # Make session and callables.
    session = tf.Session()
    self._session = session
    self._update_fn = session.make_callable(
        train_op, [o_tm1, a_tm1, c_t, d_t, o_t, p])
    self._value_fn = session.make_callable(qo, [o, p])
    session.run(tf.global_variables_initializer())

    self._saver = tf.train.Saver(var_list=self._network.variables)

  @property
  def keyboard(self):
    return self._network

  def _extract_observation(self, obs):
    return obs["arena"]

  def step(self, timestep, is_training=False):
    """Select actions according to epsilon-greedy policy."""
    if timestep.first():
      self._current_policy = np.random.randint(self._n_policies)

    if is_training and np.random.rand() < self._epsilon:
      return np.random.randint(self._n_actions)

    q_values = self._value_fn(
        self._extract_observation(timestep.observation), self._current_policy)
    return int(np.argmax(q_values))

  def update(self, step_tm1, action, step_t):
    """Takes in a transition from the environment."""

    transition = [
        self._extract_observation(step_tm1.observation),
        action,
        step_t.observation["cumulants"],
        step_t.discount,
        self._extract_observation(step_t.observation),
    ]
    self._replay.append(transition)

    if len(self._replay) == self._batch_size:
      batch = list(zip(*self._replay)) + [self._current_policy]
      self._update_fn(*batch)
      self._replay = []  # Just a queue.

  def export(self, path):
    tf.logging.info("Exporting keyboard to %s", path)
    self._network.export(
        os.path.join(path, "tfhub"), self._session, overwrite=True)
    self._saver.save(self._session, os.path.join(path, "checkpoints"))


class OptionValueNet(snt.AbstractModule):
  """Option Value net."""

  def __init__(self,
               n_policies,
               n_cumulants,
               n_actions,
               network_kwargs,
               name="option_keyboard"):
    """Construct an Option Value Net sonnet module.

    Args:
      n_policies: Number of policies.
      n_cumulants: Number of cumulants.
      n_actions: Number of actions.
      network_kwargs: Network arguments.
      name: Name
    """
    super(OptionValueNet, self).__init__(name=name)
    self._n_policies = n_policies
    self._n_cumulants = n_cumulants
    self._n_actions = n_actions
    self._network_kwargs = network_kwargs

  def _build(self, observation):
    values = []

    flat_obs = snt.BatchFlatten()(observation)
    for _ in range(self._n_cumulants):
      net = snt.nets.MLP(**self._network_kwargs)(flat_obs)
      net = snt.Linear(output_size=self._n_policies * self._n_actions)(net)
      net = snt.BatchReshape([self._n_policies, self._n_actions])(net)
      values.append(net)
    values = tf.stack(values, axis=2)
    return values

  def gpi(self, observation, cumulant_weights):
    q_values = self.__call__(tf.expand_dims(observation, axis=0))[0]
    q_w = tf.tensordot(q_values, cumulant_weights, axes=[1, 0])  # [P,a]
    q_w_actions = tf.reduce_max(q_w, axis=0)

    action = tf.cast(tf.argmax(q_w_actions), tf.int32)

    return action

  @property
  def num_cumulants(self):
    return self._n_cumulants

  @property
  def num_policies(self):
    return self._n_policies

  @property
  def num_actions(self):
    return self._n_actions


def _batched_index(values, indices):
  one_hot_indices = tf.one_hot(indices, values.shape[-1], dtype=values.dtype)
  one_hot_indices = tf.expand_dims(one_hot_indices, axis=1)
  return tf.reduce_sum(values * one_hot_indices, axis=-1)
