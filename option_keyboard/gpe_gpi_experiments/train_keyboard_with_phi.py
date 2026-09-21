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
flags.DEFINE_integer("num_phis", None, "Size of phi")
flags.DEFINE_string("phi_model_path", None,
                    "Where to load the phi model checkpoints.")
flags.DEFINE_string("export_path", None,
                    "Where to save the keyboard checkpoints.")


def main(argv):
  del argv

  keyboard_utils.create_and_train_keyboard_with_phi(
      num_episodes=FLAGS.num_pretrain_episodes,
      phi_model_path=FLAGS.phi_model_path,
      policy_weights=np.eye(FLAGS.num_phis, dtype=np.float32),
      export_path=FLAGS.export_path)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  app.run(main)
