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

Run GPE/GPI on task (1, -1) with w obtained by regression.


For example, first train a keyboard:

python3 train_keyboard.py -- --logtostderr --policy_weights_name=12 \
  --export_path=/tmp/option_keyboard/keyboard


Then, evaluate the keyboard with w by regression.

python3 run_regressed_w_fig4b.py -- --logtostderr \
  --keyboard_path=/tmp/option_keyboard/keyboard_12/tfhub
"""

from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

from option_keyboard import configs
from option_keyboard import environment_wrappers
from option_keyboard import experiment
from option_keyboard import scavenger
from option_keyboard import smart_module

from option_keyboard.gpe_gpi_experiments import regressed_agent

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 4000, "Number of training episodes.")
flags.DEFINE_integer("report_every", 5,
                     "Frequency at which metrics are reported.")
flags.DEFINE_string("keyboard_path", None, "Path to keyboard model.")
flags.DEFINE_string("output_path", None, "Path to write out training curves.")


def main(argv):
  del argv

  # Load the keyboard.
  keyboard = smart_module.SmartModuleImport(hub.Module(FLAGS.keyboard_path))

  # Create the task environment.
  base_env_config = configs.get_fig4_task_config()
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
      optimizer_kwargs=dict(learning_rate=3e-2,),
      init_w=np.random.normal(size=keyboard.num_cumulants) * 0.1,
  )

  _, ema_returns = experiment.run(
      env,
      agent,
      num_episodes=FLAGS.num_episodes,
      report_every=FLAGS.report_every,
      num_eval_reps=20)
  if FLAGS.output_path:
    experiment.write_returns_to_file(FLAGS.output_path, ema_returns)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  app.run(main)
