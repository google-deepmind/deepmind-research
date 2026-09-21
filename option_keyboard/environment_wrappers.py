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
"""Environment with keyboard."""

import itertools

from absl import logging

import dm_env

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import tree

from option_keyboard import smart_module


class EnvironmentWithLogging(dm_env.Environment):
  """Wraps an environment with additional logging."""

  def __init__(self, env):
    self._env = env
    self._episode_return = 0

  def reset(self):
    self._episode_return = 0
    return self._env.reset()

  def step(self, action):
    """Take action in the environment and do some logging."""

    step = self._env.step(action)
    if step.first():
      step = self._env.step(action)
      self._episode_return = 0

    self._episode_return += step.reward
    return step

  @property
  def episode_return(self):
    return self._episode_return

  def action_spec(self):
    return self._env.action_spec()

  def observation_spec(self):
    return self._env.observation_spec()

  def __getattr__(self, name):
    return getattr(self._env, name)


class EnvironmentWithKeyboard(dm_env.Environment):
  """Wraps an environment with a keyboard."""

  def __init__(self,
               env,
               keyboard,
               keyboard_ckpt_path,
               n_actions_per_dim,
               additional_discount,
               call_and_return=False):
    self._env = env
    self._keyboard = keyboard
    self._discount = additional_discount
    self._call_and_return = call_and_return

    options = _discretize_actions(n_actions_per_dim, keyboard.num_cumulants)
    self._options_np = options
    options = tf.convert_to_tensor(options, dtype=tf.float32)
    self._options = options

    obs_spec = self._extract_observation(env.observation_spec())
    obs_ph = tf.placeholder(shape=obs_spec.shape, dtype=obs_spec.dtype)
    option_ph = tf.placeholder(shape=(), dtype=tf.int32)
    gpi_action = self._keyboard.gpi(obs_ph, options[option_ph])

    session = tf.Session()
    self._gpi_action = session.make_callable(gpi_action, [obs_ph, option_ph])
    self._keyboard_action = session.make_callable(
        self._keyboard(tf.expand_dims(obs_ph, axis=0))[0], [obs_ph])
    session.run(tf.global_variables_initializer())

    if keyboard_ckpt_path:
      saver = tf.train.Saver(var_list=keyboard.variables)
      saver.restore(session, keyboard_ckpt_path)

  def _compute_reward(self, option, obs):
    return np.sum(self._options_np[option] * obs["cumulants"])

  def reset(self):
    return self._env.reset()

  def step(self, option):
    """Take a step in the keyboard, then the environment."""

    step_count = 0
    option_step = None
    while True:
      obs = self._extract_observation(self._env.observation())
      action = self._gpi_action(obs, option)
      action_step = self._env.step(action)
      step_count += 1

      if option_step is None:
        option_step = action_step
      else:
        new_discount = (
            option_step.discount * self._discount * action_step.discount)
        new_reward = (
            option_step.reward + new_discount * action_step.reward)
        option_step = option_step._replace(
            observation=action_step.observation,
            reward=new_reward,
            discount=new_discount,
            step_type=action_step.step_type)

      if action_step.last():
        break

      # Terminate option.
      if self._should_terminate(option, action_step.observation):
        break

      if not self._call_and_return:
        break

    return option_step

  def _should_terminate(self, option, obs):
    if self._compute_reward(option, obs) > 0:
      return True
    elif np.all(self._options_np[option] <= 0):
      # TODO(shaobohou) A hack ensure option with non-positive weights
      # terminates after one step
      return True
    else:
      return False

  def action_spec(self):
    return dm_env.specs.DiscreteArray(
        num_values=self._options_np.shape[0], name="action")

  def _extract_observation(self, obs):
    return obs["arena"]

  def observation_spec(self):
    return self._env.observation_spec()

  def __getattr__(self, name):
    return getattr(self._env, name)


class EnvironmentWithKeyboardDirect(dm_env.Environment):
  """Wraps an environment with a keyboard.

  This is different from EnvironmentWithKeyboard as the actions space is not
  discretized.

  TODO(shaobohou) Merge the two implementations.
  """

  def __init__(self,
               env,
               keyboard,
               keyboard_ckpt_path,
               additional_discount,
               call_and_return=False):
    self._env = env
    self._keyboard = keyboard
    self._discount = additional_discount
    self._call_and_return = call_and_return

    obs_spec = self._extract_observation(env.observation_spec())
    obs_ph = tf.placeholder(shape=obs_spec.shape, dtype=obs_spec.dtype)
    option_ph = tf.placeholder(
        shape=(keyboard.num_cumulants,), dtype=tf.float32)
    gpi_action = self._keyboard.gpi(obs_ph, option_ph)

    session = tf.Session()
    self._gpi_action = session.make_callable(gpi_action, [obs_ph, option_ph])
    self._keyboard_action = session.make_callable(
        self._keyboard(tf.expand_dims(obs_ph, axis=0))[0], [obs_ph])
    session.run(tf.global_variables_initializer())

    if keyboard_ckpt_path:
      saver = tf.train.Saver(var_list=keyboard.variables)
      saver.restore(session, keyboard_ckpt_path)

  def _compute_reward(self, option, obs):
    assert option.shape == obs["cumulants"].shape
    return np.sum(option * obs["cumulants"])

  def reset(self):
    return self._env.reset()

  def step(self, option):
    """Take a step in the keyboard, then the environment."""

    step_count = 0
    option_step = None
    while True:
      obs = self._extract_observation(self._env.observation())
      action = self._gpi_action(obs, option)
      action_step = self._env.step(action)
      step_count += 1

      if option_step is None:
        option_step = action_step
      else:
        new_discount = (
            option_step.discount * self._discount * action_step.discount)
        new_reward = (
            option_step.reward + new_discount * action_step.reward)
        option_step = option_step._replace(
            observation=action_step.observation,
            reward=new_reward,
            discount=new_discount,
            step_type=action_step.step_type)

      if action_step.last():
        break

      # Terminate option.
      if self._should_terminate(option, action_step.observation):
        break

      if not self._call_and_return:
        break

    return option_step

  def _should_terminate(self, option, obs):
    if self._compute_reward(option, obs) > 0:
      return True
    elif np.all(option <= 0):
      # TODO(shaobohou) A hack ensure option with non-positive weights
      # terminates after one step
      return True
    else:
      return False

  def action_spec(self):
    return dm_env.specs.BoundedArray(shape=(self._keyboard.num_cumulants,),
                                     dtype=np.float32,
                                     minimum=-1.0,
                                     maximum=1.0,
                                     name="action")

  def _extract_observation(self, obs):
    return obs["arena"]

  def observation_spec(self):
    return self._env.observation_spec()

  def __getattr__(self, name):
    return getattr(self._env, name)


def _discretize_actions(num_actions_per_dim,
                        action_space_dim,
                        min_val=-1.0,
                        max_val=1.0):
  """Discrete action space."""
  if num_actions_per_dim > 1:
    discretized_dim_action = np.linspace(
        min_val, max_val, num_actions_per_dim, endpoint=True)
    discretized_actions = [discretized_dim_action] * action_space_dim
    discretized_actions = itertools.product(*discretized_actions)
    discretized_actions = list(discretized_actions)
  elif num_actions_per_dim == 1:
    discretized_actions = [
        max_val * np.eye(action_space_dim),
        min_val * np.eye(action_space_dim),
    ]
    discretized_actions = np.concatenate(discretized_actions, axis=0)
  elif num_actions_per_dim == 0:
    discretized_actions = np.eye(action_space_dim)
  else:
    raise ValueError(
        "Unsupported num_actions_per_dim {}".format(num_actions_per_dim))

  discretized_actions = np.array(discretized_actions)

  # Remove options with all zeros.
  non_zero_entries = np.sum(np.square(discretized_actions), axis=-1) != 0.0
  discretized_actions = discretized_actions[non_zero_entries]
  logging.info("Total number of discretized actions: %s",
               len(discretized_actions))
  logging.info("Discretized actions: %s", discretized_actions)

  return discretized_actions


class EnvironmentWithLearnedPhi(dm_env.Environment):
  """Wraps an environment with learned phi model."""

  def __init__(self, env, model_path):
    self._env = env

    create_ph = lambda x: tf.placeholder(shape=x.shape, dtype=x.dtype)
    add_batch = lambda x: tf.expand_dims(x, axis=0)

    # Make session and callables.
    with tf.Graph().as_default():
      model = smart_module.SmartModuleImport(hub.Module(model_path))

      obs_spec = env.observation_spec()
      obs_ph = tree.map_structure(create_ph, obs_spec)
      action_ph = tf.placeholder(shape=(), dtype=tf.int32)
      phis = model(tree.map_structure(add_batch, obs_ph), add_batch(action_ph))

      self.num_phis = phis.shape.as_list()[-1]
      self._last_phis = np.zeros((self.num_phis,), dtype=np.float32)

      session = tf.Session()
      self._session = session
      self._phis_fn = session.make_callable(
          phis[0], tree.flatten([obs_ph, action_ph]))
      self._session.run(tf.global_variables_initializer())

  def reset(self):
    self._last_phis = np.zeros((self.num_phis,), dtype=np.float32)
    return self._env.reset()

  def step(self, action):
    """Take action in the environment and do some logging."""

    phis = self._phis_fn(*tree.flatten([self._env.observation(), action]))
    step = self._env.step(action)

    if step.first():
      phis = self._phis_fn(*tree.flatten([self._env.observation(), action]))
      step = self._env.step(action)

    step.observation["cumulants"] = phis
    self._last_phis = phis

    return step

  def action_spec(self):
    return self._env.action_spec()

  def observation(self):
    obs = self._env.observation()
    obs["cumulants"] = self._last_phis
    return obs

  def observation_spec(self):
    obs_spec = self._env.observation_spec()
    obs_spec["cumulants"] = dm_env.specs.BoundedArray(
        shape=(self.num_phis,),
        dtype=np.float32,
        minimum=-1e9,
        maximum=1e9,
        name="collected_resources")
    return obs_spec

  def __getattr__(self, name):
    return getattr(self._env, name)
