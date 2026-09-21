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
"""Run an experiment.

Run a q-learning agent on a task.
"""

from absl import app
from absl import flags

import tensorflow.compat.v1 as tf

from option_keyboard import configs
from option_keyboard import dqn_agent
from option_keyboard import environment_wrappers
from option_keyboard import experiment
from option_keyboard import scavenger

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 10000, "Number of training episodes.")
flags.DEFINE_list("test_w", None, "The w to test.")
flags.DEFINE_integer("report_every", 200,
                     "Frequency at which metrics are reported.")
flags.DEFINE_string("output_path", None, "Path to write out training curves.")


def main(argv):
  del argv

  # Create the task environment.
  test_w = [float(x) for x in FLAGS.test_w]
  env_config = configs.get_fig5_task_config(test_w)
  env = scavenger.Scavenger(**env_config)
  env = environment_wrappers.EnvironmentWithLogging(env)

  # Create the flat agent.
  agent = dqn_agent.Agent(
      obs_spec=env.observation_spec(),
      action_spec=env.action_spec(),
      network_kwargs=dict(
          output_sizes=(64, 128),
          activate_final=True,
      ),
      epsilon=0.1,
      additional_discount=0.9,
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
