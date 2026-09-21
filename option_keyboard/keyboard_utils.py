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
"""Keyboard utils."""

import numpy as np

from option_keyboard import configs
from option_keyboard import environment_wrappers
from option_keyboard import experiment
from option_keyboard import keyboard_agent
from option_keyboard import scavenger


def create_and_train_keyboard(num_episodes,
                              policy_weights=None,
                              export_path=None):
  """Train an option keyboard."""
  if policy_weights is None:
    policy_weights = np.eye(2, dtype=np.float32)

  env_config = configs.get_pretrain_config()
  env = scavenger.Scavenger(**env_config)
  env = environment_wrappers.EnvironmentWithLogging(env)

  agent = keyboard_agent.Agent(
      obs_spec=env.observation_spec(),
      action_spec=env.action_spec(),
      policy_weights=policy_weights,
      network_kwargs=dict(
          output_sizes=(64, 128),
          activate_final=True,
      ),
      epsilon=0.1,
      additional_discount=0.9,
      batch_size=10,
      optimizer_name="AdamOptimizer",
      optimizer_kwargs=dict(learning_rate=3e-4,))

  if num_episodes:
    experiment.run(env, agent, num_episodes=num_episodes)
    agent.export(export_path)

  return agent


def create_and_train_keyboard_with_phi(num_episodes,
                                       phi_model_path,
                                       policy_weights,
                                       export_path=None):
  """Train an option keyboard."""
  env_config = configs.get_pretrain_config()
  env = scavenger.Scavenger(**env_config)
  env = environment_wrappers.EnvironmentWithLogging(env)
  env = environment_wrappers.EnvironmentWithLearnedPhi(env, phi_model_path)

  agent = keyboard_agent.Agent(
      obs_spec=env.observation_spec(),
      action_spec=env.action_spec(),
      policy_weights=policy_weights,
      network_kwargs=dict(
          output_sizes=(64, 128),
          activate_final=True,
      ),
      epsilon=0.1,
      additional_discount=0.9,
      batch_size=10,
      optimizer_name="AdamOptimizer",
      optimizer_kwargs=dict(learning_rate=3e-4,))

  if num_episodes:
    experiment.run(env, agent, num_episodes=num_episodes)
    agent.export(export_path)

  return agent
