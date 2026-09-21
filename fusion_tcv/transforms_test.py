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
"""Tests for transforms."""

import math

from absl.testing import absltest
from fusion_tcv import transforms


NAN = float("nan")


class TransformsTest(absltest.TestCase):

  def assertNan(self, value: float):
    self.assertTrue(math.isnan(value))

  def test_clip(self):
    self.assertEqual(transforms.clip(-1, 0, 1), 0)
    self.assertEqual(transforms.clip(5, 0, 1), 1)
    self.assertEqual(transforms.clip(0.5, 0, 1), 0.5)
    self.assertNan(transforms.clip(NAN, 0, 1))

  def test_scale(self):
    self.assertEqual(transforms.scale(0, 0, 0.5, 0, 1), 0)
    self.assertEqual(transforms.scale(0.125, 0, 0.5, 0, 1), 0.25)
    self.assertEqual(transforms.scale(0.25, 0, 0.5, 0, 1), 0.5)
    self.assertEqual(transforms.scale(0.5, 0, 0.5, 0, 1), 1)
    self.assertEqual(transforms.scale(1, 0, 0.5, 0, 1), 2)
    self.assertEqual(transforms.scale(-1, 0, 0.5, 0, 1), -2)
    self.assertEqual(transforms.scale(0.5, 1, 0, 0, 1), 0.5)
    self.assertEqual(transforms.scale(0.25, 1, 0, 0, 1), 0.75)

    self.assertEqual(transforms.scale(0, 0, 1, -4, 4), -4)
    self.assertEqual(transforms.scale(0.25, 0, 1, -4, 4), -2)
    self.assertEqual(transforms.scale(0.5, 0, 1, -4, 4), 0)
    self.assertEqual(transforms.scale(0.75, 0, 1, -4, 4), 2)
    self.assertEqual(transforms.scale(1, 0, 1, -4, 4), 4)

    self.assertNan(transforms.scale(NAN, 0, 1, -4, 4))

  def test_logistic(self):
    self.assertLess(transforms.logistic(-50), 0.000001)
    self.assertLess(transforms.logistic(-5), 0.01)
    self.assertEqual(transforms.logistic(0), 0.5)
    self.assertGreater(transforms.logistic(5), 0.99)
    self.assertGreater(transforms.logistic(50), 0.999999)
    self.assertAlmostEqual(transforms.logistic(0.8), math.tanh(0.4) / 2 + 0.5)
    self.assertNan(transforms.logistic(NAN))

  def test_exp_scaled(self):
    t = transforms.NegExp(good=0, bad=1)
    self.assertNan(t([NAN])[0])
    self.assertAlmostEqual(t([0])[0], 1)
    self.assertAlmostEqual(t([1])[0], 0.1)
    self.assertLess(t([50])[0], 0.000001)

    t = transforms.NegExp(good=10, bad=30)
    self.assertAlmostEqual(t([0])[0], 1)
    self.assertAlmostEqual(t([10])[0], 1)
    self.assertLess(t([3000])[0], 0.000001)

    t = transforms.NegExp(good=30, bad=10)
    self.assertAlmostEqual(t([50])[0], 1)
    self.assertAlmostEqual(t([30])[0], 1)
    self.assertAlmostEqual(t([10])[0], 0.1)
    self.assertLess(t([-90])[0], 0.00001)

  def test_neg(self):
    t = transforms.Neg()
    self.assertEqual(t([-5, -3, 0, 1, 4]), [5, 3, 0, -1, -4])
    self.assertNan(t([NAN])[0])

  def test_abs(self):
    t = transforms.Abs()
    self.assertEqual(t([-5, -3, 0, 1, 4]), [5, 3, 0, 1, 4])
    self.assertNan(t([NAN])[0])

  def test_pow(self):
    t = transforms.Pow(2)
    self.assertEqual(t([-5, -3, 0, 1, 4]), [25, 9, 0, 1, 16])
    self.assertNan(t([NAN])[0])

  def test_log(self):
    t = transforms.Log()
    self.assertAlmostEqual(t([math.exp(2)])[0], 2, 4)  # Low precision from eps.
    self.assertNan(t([NAN])[0])

  def test_clipped_linear(self):
    t = transforms.ClippedLinear(good=0.1, bad=0.3)
    self.assertAlmostEqual(t([0])[0], 1)
    self.assertAlmostEqual(t([0.05])[0], 1)
    self.assertAlmostEqual(t([0.1])[0], 1)
    self.assertAlmostEqual(t([0.15])[0], 0.75)
    self.assertAlmostEqual(t([0.2])[0], 0.5)
    self.assertAlmostEqual(t([0.25])[0], 0.25)
    self.assertAlmostEqual(t([0.3])[0], 0)
    self.assertAlmostEqual(t([0.4])[0], 0)
    self.assertNan(t([NAN])[0])

    t = transforms.ClippedLinear(good=1, bad=0.5)
    self.assertAlmostEqual(t([1.5])[0], 1)
    self.assertAlmostEqual(t([1])[0], 1)
    self.assertAlmostEqual(t([0.75])[0], 0.5)
    self.assertAlmostEqual(t([0.5])[0], 0)
    self.assertAlmostEqual(t([0.25])[0], 0)

  def test_softplus(self):
    t = transforms.SoftPlus(good=0.1, bad=0.3)
    self.assertEqual(t([0])[0], 1)
    self.assertEqual(t([0.1])[0], 1)
    self.assertAlmostEqual(t([0.3])[0], 0.1)
    self.assertLess(t([0.5])[0], 0.01)
    self.assertNan(t([NAN])[0])

    t = transforms.SoftPlus(good=1, bad=0.5)
    self.assertEqual(t([1.5])[0], 1)
    self.assertEqual(t([1])[0], 1)
    self.assertAlmostEqual(t([0.5])[0], 0.1)
    self.assertLess(t([0.1])[0], 0.01)

  def test_sigmoid(self):
    t = transforms.Sigmoid(good=0.1, bad=0.3)
    self.assertGreater(t([0])[0], 0.99)
    self.assertAlmostEqual(t([0.1])[0], 0.95)
    self.assertAlmostEqual(t([0.2])[0], 0.5)
    self.assertAlmostEqual(t([0.3])[0], 0.05)
    self.assertLess(t([0.4])[0], 0.01)
    self.assertNan(t([NAN])[0])

    t = transforms.Sigmoid(good=1, bad=0.5)
    self.assertGreater(t([1.5])[0], 0.99)
    self.assertAlmostEqual(t([1])[0], 0.95)
    self.assertAlmostEqual(t([0.75])[0], 0.5)
    self.assertAlmostEqual(t([0.5])[0], 0.05)
    self.assertLess(t([0.25])[0], 0.01)

  def test_equal(self):
    t = transforms.Equal()
    self.assertEqual(t([0])[0], 1)
    self.assertEqual(t([0.001])[0], 0)
    self.assertNan(t([NAN])[0])

    t = transforms.Equal(not_equal_val=0.5)
    self.assertEqual(t([0])[0], 1)
    self.assertEqual(t([0.001])[0], 0.5)


if __name__ == "__main__":
  absltest.main()
