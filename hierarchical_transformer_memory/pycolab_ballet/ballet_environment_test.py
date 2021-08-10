# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for pycolab_ballet.ballet_environment_wrapper."""
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from hierarchical_transformer_memory.pycolab_ballet import ballet_environment
from hierarchical_transformer_memory.pycolab_ballet import ballet_environment_core


class BalletEnvironmentTest(parameterized.TestCase):

  def test_full_wrapper(self):
    env = ballet_environment.BalletEnvironment(
        num_dancers=1, dance_delay=16, max_steps=200,
        rng=np.random.default_rng(seed=0))
    result = env.reset()
    self.assertIsNone(result.reward)
    level_size = ballet_environment_core.ROOM_SIZE
    upsample_size = ballet_environment.UPSAMPLE_SIZE
    # wait for dance to complete
    for i in range(30):
      result = env.step(0).observation
      self.assertEqual(result[0].shape,
                       (level_size[0] * upsample_size,
                        level_size[1] * upsample_size,
                        3))
      self.assertEqual(str(result[1])[:5],
                       np.array("watch"))
    for i in [1, 1, 1, 1]:  # first gets eaten before agent can move
      result = env.step(i)
      self.assertEqual(result.observation[0].shape,
                       (level_size[0] * upsample_size,
                        level_size[1] * upsample_size,
                        3))
      self.assertEqual(str(result.observation[1])[:11],
                       np.array("up_and_down"))
    self.assertEqual(result.reward, 1.)
    # check egocentric scrolling is working, by checking object is in center
    np.testing.assert_array_almost_equal(
        result.observation[0][45:54, 45:54],
        ballet_environment._generate_template("orange plus") / 255.)

  @parameterized.parameters(
      "2_delay16",
      "4_delay16",
      "8_delay48",
  )
  def test_simple_builder(self, level_name):
    dance_delay = int(level_name[-2:])
    np.random.seed(0)
    env = ballet_environment.simple_builder(level_name)
    # check max steps are set to match paper settings
    self.assertEqual(env._max_steps,
                     320 if dance_delay == 16 else 1024)
    # test running a few steps of each
    env.reset()
    level_size = ballet_environment_core.ROOM_SIZE
    upsample_size = ballet_environment.UPSAMPLE_SIZE
    for i in range(8):
      result = env.step(i)  # check all 8 movements work
      self.assertEqual(result.observation[0].shape,
                       (level_size[0] * upsample_size,
                        level_size[1] * upsample_size,
                        3))
      self.assertEqual(str(result.observation[1])[:5],
                       np.array("watch"))
      self.assertEqual(result.reward, 0.)

if __name__ == "__main__":
  absltest.main()
