# pylint: disable=g-bad-file-header
# pylint: disable=line-too-long
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

This script generates the raw data for the polar plots used to visualise how
well a trained keyboard covers the space of w.


For example, train 3 separate keyboards with different base policies:

python3 train_keyboard.py --logtostderr --policy_weights_name=12
python3 train_keyboard.py --logtostderr --policy_weights_name=34
python3 train_keyboard.py --logtostderr --policy_weights_name=5


Then generate the polar plot data as follows:

python3 eval_keyboard_fig5.py --logtostderr \
  --keyboard_paths=/tmp/option_keyboard/keyboard_12/tfhub,/tmp/option_keyboard/keyboard_34/tfhub,/tmp/option_keyboard/keyboard_5/tfhub \
  --num_episodes=1000


Example outout:
[[ 0.11        0.261      -0.933     ]
 [ 1.302       3.955       0.54      ]
 [ 2.398       4.434       1.2105359 ]
 [ 3.459       4.606       2.087     ]
 [ 4.09026795  4.60911325  3.06106882]
 [ 4.55499485  4.71947818  3.8123229 ]
 [ 4.715       4.835       4.395     ]
 [ 4.75743564  4.64095528  4.46330207]
 [ 4.82518207  4.71232378  4.56190708]
 [ 4.831       4.7155      4.5735    ]
 [ 4.78074425  4.6754641   4.58312762]
 [ 4.70154374  4.5416429   4.47850417]
 [ 4.694       4.631       4.427     ]
 [ 4.25085125  4.56606664  3.68157677]
 [ 3.61726795  4.4838453   2.68154403]
 [ 2.714       4.43        1.554     ]
 [ 1.69        4.505       0.9635359 ]
 [ 0.894       4.043       0.424     ]
 [ 0.099       0.349       0.055     ]]
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
flags.DEFINE_list("keyboard_paths", [], "Path to keyboard model.")
flags.DEFINE_string("output_path", None, "Path to write out returns.")


def evaluate_keyboard(keyboard_path, weights_to_sweep):
  """Evaluate a keyboard."""

  # Load the keyboard.
  keyboard = smart_module.SmartModuleImport(hub.Module(keyboard_path))

  # Create the task environment.
  all_returns = []
  for w_to_sweep in weights_to_sweep.tolist():
    base_env_config = configs.get_fig5_task_config(w_to_sweep)
    base_env = scavenger.Scavenger(**base_env_config)
    base_env = environment_wrappers.EnvironmentWithLogging(base_env)

    # Wrap the task environment with the keyboard.
    with tf.variable_scope(None, default_name="inner_loop"):
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
          init_w=w_to_sweep)

    returns = []
    for _ in range(FLAGS.num_episodes):
      returns.append(experiment.run_episode(env, agent))
    tf.logging.info(f"Task: {w_to_sweep}, mean returns over "
                    f"{FLAGS.num_episodes} episodes is {np.mean(returns)}")
    all_returns.append(returns)

  return all_returns


def main(argv):
  del argv

  angles_to_sweep = np.deg2rad(np.linspace(-90, 180, num=19, endpoint=True))
  weights_to_sweep = np.stack(
      [np.sin(angles_to_sweep),
       np.cos(angles_to_sweep)], axis=-1)
  weights_to_sweep /= np.sum(
      np.maximum(weights_to_sweep, 0.0), axis=-1, keepdims=True)
  weights_to_sweep = np.clip(weights_to_sweep, -1000, 1000)
  tf.logging.info(weights_to_sweep)

  all_returns = []
  for keyboard_path in FLAGS.keyboard_paths:
    returns = evaluate_keyboard(keyboard_path, weights_to_sweep)
    all_returns.append(returns)

  print("Results:")
  print(np.mean(all_returns, axis=-1).T)

  if FLAGS.output_path:
    with gfile.GFile(FLAGS.output_path, "w") as file:
      writer = csv.writer(file, delimiter=" ", quoting=csv.QUOTE_MINIMAL)
      writer.writerow(["angle", "return", "idx"])
      for idx, returns in enumerate(all_returns):
        for row in np.array(returns).T.tolist():
          assert len(angles_to_sweep) == len(row)
          for ang, val in zip(angles_to_sweep, row):
            ang = "{:.4g}".format(ang)
            val = "{:.4g}".format(val)
            writer.writerow([ang, val, idx])


if __name__ == "__main__":
  tf.disable_v2_behavior()
  app.run(main)
