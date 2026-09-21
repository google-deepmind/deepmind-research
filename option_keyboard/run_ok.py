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

import os

from absl import app
from absl import flags

import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

from option_keyboard import configs
from option_keyboard import dqn_agent
from option_keyboard import environment_wrappers
from option_keyboard import experiment
from option_keyboard import keyboard_utils
from option_keyboard import scavenger
from option_keyboard import smart_module

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 10000, "Number of training episodes.")
flags.DEFINE_integer("num_pretrain_episodes", 20000,
                     "Number of pretraining episodes.")
flags.DEFINE_integer("report_every", 200,
                     "Frequency at which metrics are reported.")
flags.DEFINE_string("keyboard_path", None, "Path to pretrained keyboard model.")
flags.DEFINE_string("output_path", None, "Path to write out training curves.")


def main(argv):
  del argv

  # Pretrain the keyboard and save a checkpoint.
  if FLAGS.keyboard_path:
    keyboard_path = FLAGS.keyboard_path
  else:
    with tf.Graph().as_default():
      export_path = "/tmp/option_keyboard/keyboard"
      _ = keyboard_utils.create_and_train_keyboard(
          num_episodes=FLAGS.num_pretrain_episodes, export_path=export_path)
      keyboard_path = os.path.join(export_path, "tfhub")

  # Load the keyboard.
  keyboard = smart_module.SmartModuleImport(hub.Module(keyboard_path))

  # Create the task environment.
  base_env_config = configs.get_task_config()
  base_env = scavenger.Scavenger(**base_env_config)
  base_env = environment_wrappers.EnvironmentWithLogging(base_env)

  # Wrap the task environment with the keyboard.
  additional_discount = 0.9
  env = environment_wrappers.EnvironmentWithKeyboard(
      env=base_env,
      keyboard=keyboard,
      keyboard_ckpt_path=None,
      n_actions_per_dim=3,
      additional_discount=additional_discount,
      call_and_return=False)

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

  _, ema_returns = experiment.run(
      env,
      agent,
      num_episodes=FLAGS.num_episodes,
      report_every=FLAGS.report_every)
  if FLAGS.output_path:
    experiment.write_returns_to_file(FLAGS.output_path, ema_returns)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  app.run(main)
