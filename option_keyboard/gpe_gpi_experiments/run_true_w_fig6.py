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
r"""Run an experiment.

Run GPE/GPI on the "balancing" task with a fixed w


For example, first train a keyboard:

python3 train_keyboard.py -- --logtostderr --policy_weights_name=12


Then, evaluate the keyboard with a fixed w.

python3 run_true_w_fig6.py -- --logtostderr \
  --keyboard_path=/tmp/option_keyboard/keyboard_12/tfhub
"""

import csv

from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.io import gfile
import tensorflow_hub as hub

from option_keyboard import configs
from option_keyboard import environment_wrappers
from option_keyboard import experiment
from option_keyboard import scavenger
from option_keyboard import smart_module

from option_keyboard.gpe_gpi_experiments import regressed_agent

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 1000, "Number of training episodes.")
flags.DEFINE_string("keyboard_path", None, "Path to keyboard model.")
flags.DEFINE_list("test_w", None, "The w to test.")
flags.DEFINE_string("output_path", None, "Path to write out returns.")


def main(argv):
  del argv

  # Load the keyboard.
  keyboard = smart_module.SmartModuleImport(hub.Module(FLAGS.keyboard_path))

  # Create the task environment.
  base_env_config = configs.get_task_config()
  base_env = scavenger.Scavenger(**base_env_config)
  base_env = environment_wrappers.EnvironmentWithLogging(base_env)

  # Wrap the task environment with the keyboard.
  additional_discount = 0.9
  env = environment_wrappers.EnvironmentWithKeyboardDirect(
      env=base_env,
      keyboard=keyboard,
      keyboard_ckpt_path=None,
      additional_discount=additional_discount,
      call_and_return=False)

  # Create the player agent.
  agent = regressed_agent.Agent(
      batch_size=10,
      optimizer_name="AdamOptimizer",
      # Disable training.
      optimizer_kwargs=dict(learning_rate=0.0,),
      init_w=[float(x) for x in FLAGS.test_w])

  returns = []
  for _ in range(FLAGS.num_episodes):
    returns.append(experiment.run_episode(env, agent))
  tf.logging.info("#" * 80)
  tf.logging.info(
      f"Avg. return over {FLAGS.num_episodes} episodes is {np.mean(returns)}")
  tf.logging.info("#" * 80)

  if FLAGS.output_path:
    with gfile.GFile(FLAGS.output_path, "w") as file:
      writer = csv.writer(file, delimiter=" ", quoting=csv.QUOTE_MINIMAL)
      writer.writerow(["return"])
      for val in returns:
        writer.writerow([val])


if __name__ == "__main__":
  tf.disable_v2_behavior()
  app.run(main)
