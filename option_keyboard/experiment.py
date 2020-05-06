# Lint as: python3
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
"""A simple training loop."""

from absl import logging

import numpy as np


def run(environment, agent, num_episodes, report_every=200):
  """Runs an agent on an environment.

  Args:
    environment: The environment.
    agent: The agent.
    num_episodes: Number of episodes to train for.
    report_every: Frequency at which training progress are reported (episodes).
  """

  train_returns = []
  eval_returns = []
  for episode_id in range(num_episodes):
    # Run a training episode.
    train_episode_return = run_episode(environment, agent, is_training=True)
    train_returns.append(train_episode_return)

    # Run an evaluation episode.
    eval_episode_return = run_episode(environment, agent, is_training=False)
    eval_returns.append(eval_episode_return)

    if ((episode_id + 1) % report_every) == 0:
      logging.info(
          "Episode %s, avg train return %.3f, avg eval return %.3f",
          episode_id + 1,
          np.mean(train_returns[-report_every:]),
          np.mean(eval_returns[-report_every:]),
      )


def run_episode(environment, agent, is_training=False):
  """Run a single episode."""

  timestep = environment.reset()

  while not timestep.last():
    action = agent.step(timestep, is_training)
    new_timestep = environment.step(action)

    if is_training:
      agent.update(timestep, action, new_timestep)

    timestep = new_timestep

  episode_return = environment.episode_return

  return episode_return
