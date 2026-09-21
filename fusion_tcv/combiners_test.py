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
"""Tests for combiners."""

import math

from absl.testing import absltest
from fusion_tcv import combiners


NAN = float("nan")


class CombinersTest(absltest.TestCase):

  def assertNan(self, value):
    self.assertLen(value, 1)
    self.assertTrue(math.isnan(value[0]))

  def test_errors(self):
    c = combiners.Mean()
    with self.assertRaises(ValueError):
      c([0, 1], [1])
    with self.assertRaises(ValueError):
      c([0, 1], [1, 2, 3])
    with self.assertRaises(ValueError):
      c([0, 1], [-1, 2])

  def test_mean(self):
    c = combiners.Mean()
    self.assertEqual(c([0, 2, 4]), [2])
    self.assertEqual(c([0, 0.5, 1]), [0.5])
    self.assertEqual(c([0, 0.5, 1], [0, 0, 1]), [1])
    self.assertEqual(c([0, 1], [1, 3]), [0.75])
    self.assertEqual(c([0, NAN], [1, 3]), [0])
    self.assertNan(c([NAN, NAN], [1, 3]))

  def test_geometric_mean(self):
    c = combiners.GeometricMean()
    self.assertEqual(c([0.5, 0]), [0])
    self.assertEqual(c([0.3]), [0.3])
    self.assertEqual(c([4, 4]), [4])
    self.assertEqual(c([0.5, 0.5]), [0.5])
    self.assertEqual(c([0.5, 0.5], [1, 3]), [0.5])
    self.assertEqual(c([0.5, 1], [1, 2]), [0.5**(1/3)])
    self.assertEqual(c([0.5, 1], [2, 1]), [0.5**(2/3)])
    self.assertEqual(c([0.5, 0], [2, 0]), [0.5])
    self.assertEqual(c([0.5, 0, 0], [2, 1, 0]), [0])
    self.assertEqual(c([0.5, NAN, 0], [2, 1, 0]), [0.5])
    self.assertNan(c([NAN, NAN], [1, 3]))

  def test_multiply(self):
    c = combiners.Multiply()
    self.assertEqual(c([0.5, 0]), [0])
    self.assertEqual(c([0.3]), [0.3])
    self.assertEqual(c([0.5, 0.5]), [0.25])
    self.assertEqual(c([0.5, 0.5], [1, 3]), [0.0625])
    self.assertEqual(c([0.5, 1], [1, 2]), [0.5])
    self.assertEqual(c([0.5, 1], [2, 1]), [0.25])
    self.assertEqual(c([0.5, 0], [2, 0]), [0.25])
    self.assertEqual(c([0.5, 0, 0], [2, 1, 0]), [0])
    self.assertEqual(c([0.5, NAN], [1, 1]), [0.5])
    self.assertNan(c([NAN, NAN], [1, 3]))

  def test_min(self):
    c = combiners.Min()
    self.assertEqual(c([0, 1]), [0])
    self.assertEqual(c([0.5, 1]), [0.5])
    self.assertEqual(c([1, 0.75]), [0.75])
    self.assertEqual(c([1, 3]), [1])
    self.assertEqual(c([1, 1, 3], [0, 1, 1]), [1])
    self.assertEqual(c([NAN, 3]), [3])
    self.assertNan(c([NAN, NAN], [1, 3]))

  def test_max(self):
    c = combiners.Max()
    self.assertEqual(c([0, 1]), [1])
    self.assertEqual(c([0.5, 1]), [1])
    self.assertEqual(c([1, 0.75]), [1])
    self.assertEqual(c([1, 3]), [3])
    self.assertEqual(c([1, 1, 3], [0, 1, 1]), [3])
    self.assertEqual(c([NAN, 3]), [3])
    self.assertNan(c([NAN, NAN], [1, 3]))

  def test_lnorm(self):
    c = combiners.LNorm(1)
    self.assertEqual(c([0, 2, 4]), [2])
    self.assertEqual(c([0, 0.5, 1]), [0.5])
    self.assertEqual(c([3, 4]), [7 / 2])
    self.assertEqual(c([0, 2, 4], [1, 1, 0]), [1])
    self.assertEqual(c([0, 2, NAN]), [1])
    self.assertNan(c([NAN, NAN], [1, 3]))

    c = combiners.LNorm(1, normalized=False)
    self.assertEqual(c([0, 2, 4]), [6])
    self.assertEqual(c([0, 0.5, 1]), [1.5])
    self.assertEqual(c([3, 4]), [7])

    c = combiners.LNorm(2)
    self.assertEqual(c([3, 4]), [5 / 2**0.5])

    c = combiners.LNorm(2, normalized=False)
    self.assertEqual(c([3, 4]), [5])

    c = combiners.LNorm(math.inf)
    self.assertAlmostEqual(c([3, 4])[0], 4)

    c = combiners.LNorm(math.inf, normalized=False)
    self.assertAlmostEqual(c([3, 4])[0], 4)

  def test_smoothmax(self):
    # Max
    c = combiners.SmoothMax(math.inf)
    self.assertEqual(c([0, 1]), [1])
    self.assertEqual(c([0.5, 1]), [1])
    self.assertEqual(c([1, 0.75]), [1])
    self.assertEqual(c([1, 3]), [3])

    # Smooth Max
    c = combiners.SmoothMax(1)
    self.assertAlmostEqual(c([0, 1])[0], 0.7310585786300049)

    # Mean
    c = combiners.SmoothMax(0)
    self.assertEqual(c([0, 2, 4]), [2])
    self.assertEqual(c([0, 0.5, 1]), [0.5])
    self.assertEqual(c([0, 0.5, 1], [0, 0, 1]), [1])
    self.assertEqual(c([0, 2, NAN]), [1])
    self.assertEqual(c([0, 2, NAN], [0, 1, 1]), [2])
    self.assertAlmostEqual(c([0, 1], [1, 3])[0], 0.75)
    self.assertNan(c([NAN, NAN], [1, 3]))

    # Smooth Min
    c = combiners.SmoothMax(-1)
    self.assertEqual(c([0, 1])[0], 0.2689414213699951)

    # Min
    c = combiners.SmoothMax(-math.inf)
    self.assertEqual(c([0, 1]), [0])
    self.assertEqual(c([0.5, 1]), [0.5])
    self.assertEqual(c([1, 0.75]), [0.75])
    self.assertEqual(c([1, 3]), [1])


if __name__ == "__main__":
  absltest.main()
