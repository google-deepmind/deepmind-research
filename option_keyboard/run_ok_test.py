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
"""Tests for training a keyboard and then running a DQN agent on top of it."""

from absl import flags
from absl.testing import absltest

import tensorflow.compat.v1 as tf

from option_keyboard import run_ok

FLAGS = flags.FLAGS


class RunDQNTest(absltest.TestCase):

  def test_run(self):
    FLAGS.num_episodes = 200
    FLAGS.num_pretrain_episodes = 200
    run_ok.main(None)


if __name__ == '__main__':
  tf.disable_v2_behavior()
  absltest.main()
