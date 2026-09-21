# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for experiments."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_env import test_utils

from fusion_tcv import agent
from fusion_tcv import experiments
from fusion_tcv import run_loop


class FundamentalCapabilityTest(test_utils.EnvironmentTestMixin,
                                absltest.TestCase):

  def make_object_under_test(self):
    return experiments.fundamental_capability()


class ElongationTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def make_object_under_test(self):
    return experiments.elongation()


class IterTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def make_object_under_test(self):
    return experiments.iter()


class NegativeTriangularityTest(test_utils.EnvironmentTestMixin,
                                absltest.TestCase):

  def make_object_under_test(self):
    return experiments.negative_triangularity()


class SnowflakeTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def make_object_under_test(self):
    return experiments.snowflake()


class DropletTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def make_object_under_test(self):
    return experiments.droplet()


class ExperimentsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("fundamental_capability", experiments.fundamental_capability),
      ("elongation", experiments.elongation),
      ("iter", experiments.iter),
      ("negative_triangularity", experiments.negative_triangularity),
      ("snowflake", experiments.snowflake),
      ("droplet", experiments.droplet),
  )
  def test_env(self, env_fn):
    traj = run_loop.run_loop(env_fn(), agent.ZeroAgent(), max_steps=10)
    self.assertGreaterEqual(len(traj.reward), 1)


if __name__ == "__main__":
  absltest.main()
