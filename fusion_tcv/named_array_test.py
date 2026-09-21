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
"""Tests for named_array."""

from absl.testing import absltest
import numpy as np

from fusion_tcv import named_array


class NamedRangesTest(absltest.TestCase):

  def test_lengths_to_ranges(self):
    self.assertEqual(named_array.lengths_to_ranges({"a": 2, "b": 3}),
                     {"a": [0, 1], "b": [2, 3, 4]})

  def test_named_ranges(self):
    action_counts = {"E": 8, "F": 8, "OH": 2, "DUMMY": 1, "G": 1}
    actions = named_array.NamedRanges(action_counts)
    self.assertEqual(actions.range("E"), list(range(8)))
    self.assertEqual(actions["F"], list(range(8, 16)))
    self.assertEqual(actions.range("G"), [19])
    self.assertEqual(actions.index("G"), 19)
    with self.assertRaises(ValueError):
      actions.index("F")
    for k, v in action_counts.items():
      self.assertEqual(actions.count(k), v)
    self.assertEqual(actions.counts(), action_counts)
    self.assertEqual(list(actions.names()), list(action_counts.keys()))
    self.assertEqual(actions.size, sum(action_counts.values()))

    refs = actions.new_named_array()
    self.assertEqual(refs.array.shape, (actions.size,))
    np.testing.assert_array_equal(refs.array, np.zeros((actions.size,)))

    refs = actions.new_random_named_array()
    self.assertEqual(refs.array.shape, (actions.size,))
    self.assertFalse(np.array_equal(refs.array, np.zeros((actions.size,))))


class NamedArrayTest(absltest.TestCase):

  def test_name_array(self):
    action_counts = {"E": 8, "F": 8, "OH": 2, "DUMMY": 1, "G": 1}
    actions_ranges = named_array.NamedRanges(action_counts)
    actions_array = np.arange(actions_ranges.size) + 100
    actions = named_array.NamedArray(actions_array, actions_ranges)
    for k in action_counts:
      self.assertEqual(list(actions[k]), [v + 100 for v in actions_ranges[k]])
    actions["G"] = -5
    self.assertEqual(list(actions["G"]), [-5])
    self.assertEqual(actions_array[19], -5)

    for i in range(action_counts["E"]):
      actions.names.set_range(f"E_{i}", [i])

    actions["E_3"] = 53
    self.assertEqual(list(actions["E_1"]), [101])
    self.assertEqual(list(actions["E_3"]), [53])
    self.assertEqual(actions_array[3], 53)

    actions["F", 2] = 72
    self.assertEqual(actions_array[10], 72)

    actions["F", [4, 5]] = 74
    self.assertEqual(actions_array[12], 74)
    self.assertEqual(actions_array[13], 74)

    actions["F", 0:2] = 78
    self.assertEqual(actions_array[8], 78)
    self.assertEqual(actions_array[9], 78)

    self.assertEqual(list(actions["F"]), [78, 78, 72, 111, 74, 74, 114, 115])

    with self.assertRaises(ValueError):
      actions["F"][5] = 85


if __name__ == "__main__":
  absltest.main()
