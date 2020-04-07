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
"""Run an experiment."""

from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v1 as tf

from option_keyboard import configs
from option_keyboard import dqn_agent
from option_keyboard import environment_wrappers
from option_keyboard import experiment
from option_keyboard import keyboard_agent
from option_keyboard import scavenger

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 10000, "Number of training episodes.")
flags.DEFINE_integer("num_pretrain_episodes", 20000,
                     "Number of pretraining episodes.")


def _train_keyboard(num_episodes):
  """Train an option keyboard."""
  env_config = configs.get_pretrain_config()
  env = scavenger.Scavenger(**env_config)
  env = environment_wrappers.EnvironmentWithLogging(env)

  agent = keyboard_agent.Agent(
      obs_spec=env.observation_spec(),
      action_spec=env.action_spec(),
      policy_weights=np.array([
          [1.0, 0.0],
          [0.0, 1.0],
      ]),
      network_kwargs=dict(
          output_sizes=(64, 128),
          activate_final=True,
      ),
      epsilon=0.1,
      additional_discount=0.9,
      batch_size=10,
      optimizer_name="AdamOptimizer",
      optimizer_kwargs=dict(learning_rate=3e-4,))

  experiment.run(env, agent, num_episodes=num_episodes)

  return agent


def main(argv):
  del argv

  # Pretrain the keyboard and save a checkpoint.
  pretrain_agent = _train_keyboard(num_episodes=FLAGS.num_pretrain_episodes)
  keyboard_ckpt_path = "/tmp/option_keyboard/keyboard.ckpt"
  pretrain_agent.export(keyboard_ckpt_path)

  # Create the task environment.
  base_env_config = configs.get_task_config()
  base_env = scavenger.Scavenger(**base_env_config)
  base_env = environment_wrappers.EnvironmentWithLogging(base_env)

  # Wrap the task environment with the keyboard.
  additional_discount = 0.9
  env = environment_wrappers.EnvironmentWithKeyboard(
      env=base_env,
      keyboard=pretrain_agent.keyboard,
      keyboard_ckpt_path=keyboard_ckpt_path,
      n_actions_per_dim=3,
      additional_discount=additional_discount,
      call_and_return=True)

  # Create the player agent.
  agent = dqn_agent.Agent(
      obs_spec=env.observation_spec(),
      action_spec=env.action_spec(),
      network_kwargs=dict(
          output_sizes=(64, 128),
          activate_final=True,
      ),
      epsilon=0.1,
      additional_discount=additional_discount,
      batch_size=10,
      optimizer_name="AdamOptimizer",
      optimizer_kwargs=dict(learning_rate=3e-4,))

  experiment.run(env, agent, num_episodes=FLAGS.num_episodes)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  app.run(main)
