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
"""Train a keyboard."""

from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v1 as tf

from option_keyboard import keyboard_utils

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_pretrain_episodes", 20000,
                     "Number of pretraining episodes.")
flags.DEFINE_string("export_path", None,
                    "Where to save the keyboard checkpoints.")
flags.DEFINE_string("policy_weights_name", None,
                    "A string repsenting the policy weights.")


def main(argv):
  del argv

  all_policy_weights = {
      "1": [1., 0.],
      "2": [0., 1.],
      "3": [1., -1.],
      "4": [-1., 1.],
      "5": [1., 1.],
  }
  if FLAGS.policy_weights_name:
    policy_weights = np.array(
        [all_policy_weights[v] for v in FLAGS.policy_weights_name])
    num_episodes = ((FLAGS.num_pretrain_episodes // 2) *
                    max(2, len(policy_weights)))
    export_path = FLAGS.export_path + "_" + FLAGS.policy_weights_name
  else:
    policy_weights = None
    num_episodes = FLAGS.num_pretrain_episodes
    export_path = FLAGS.export_path

  keyboard_utils.create_and_train_keyboard(
      num_episodes=num_episodes,
      policy_weights=policy_weights,
      export_path=export_path)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  app.run(main)
