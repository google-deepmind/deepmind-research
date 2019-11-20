# Lint as: python2, python3
# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Threaded batch environment wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from concurrent import futures

from six.moves import range
from six.moves import zip

from tvt import nest_utils


class BatchEnv(object):
  """Wrapper that steps multiple environments in separate threads.

  The threads are stepped in lock step, so all threads progress by one step
  before any move to the next step.
  """

  def __init__(self, batch_size, env_builder, **env_kwargs):
    self.batch_size = batch_size
    self._envs = [env_builder(**env_kwargs) for _ in range(batch_size)]
    self._num_actions = self._envs[0].num_actions
    self._observation_shape = self._envs[0].observation_shape
    self._episode_length = self._envs[0].episode_length

    self._executor = futures.ThreadPoolExecutor(max_workers=self.batch_size)

  def reset(self):
    """Reset the entire batch of environments."""

    def reset_environment(env):
      return env.reset()

    try:
      output_list = []
      for env in self._envs:
        output_list.append(self._executor.submit(reset_environment, env))
      output_list = [env_output.result() for env_output in output_list]
    except KeyboardInterrupt:
      self._executor.shutdown(wait=True)
      raise

    observations, rewards = nest_utils.nest_stack(output_list)
    return observations, rewards

  def step(self, action_list):
    """Step batch of envs.

    Args:
      action_list: A list of actions, one per environment in the batch. Each one
        should be a scalar int or a numpy scaler int.

    Returns:
      A tuple (observations, rewards):
        observations: A nest of observations, each one a numpy array where the
          first dimension has size equal to the number of environments in the
          batch.
        rewards: An array of rewards with size equal to the number of
          environments in the batch.
    """

    def step_environment(env, action):
      return env.step(action)

    try:
      output_list = []
      for env, action in zip(self._envs, action_list):
        output_list.append(self._executor.submit(step_environment, env, action))
      output_list = [env_output.result() for env_output in output_list]
    except KeyboardInterrupt:
      self._executor.shutdown(wait=True)
      raise

    observations, rewards = nest_utils.nest_stack(output_list)
    return observations, rewards

  @property
  def observation_shape(self):
    """Observation shape per environment, i.e. with no batch dimension."""
    return self._observation_shape

  @property
  def num_actions(self):
    return self._num_actions

  @property
  def episode_length(self):
    return self._episode_length

  def last_phase_rewards(self):
    return [env.last_phase_reward() for env in self._envs]
