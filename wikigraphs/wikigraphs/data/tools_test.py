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
#
# WikiGraphs is licensed under the terms of the Creative Commons
# Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license.
#
# WikiText-103 data (unchanged) is licensed by Salesforce.com, Inc. under the
# terms of the Creative Commons Attribution-ShareAlike 4.0 International
# (CC BY-SA 4.0) license. You can find details about CC BY-SA 4.0 at:
#
#     https://creativecommons.org/licenses/by-sa/4.0/legalcode
#
# Freebase data is licensed by Google LLC under the terms of the Creative
# Commons CC BY 4.0 license. You may obtain a copy of the License at:
#
#     https://creativecommons.org/licenses/by/4.0/legalcode
#
# ==============================================================================
"""Tests for wikigraphs.data.tools."""
from absl.testing import absltest
import numpy as np
from wikigraphs.data import tools


class ToolsTest(absltest.TestCase):

  def test_padding(self):
    np.testing.assert_array_equal(
        tools.pad_to(np.arange(3), 5),
        [0, 1, 2, 0, 0])
    np.testing.assert_array_equal(
        tools.pad_to(np.arange(3), 5, pad_value=-1),
        [0, 1, 2, -1, -1])
    np.testing.assert_array_equal(
        tools.pad_to(np.arange(6).reshape(2, 3), 4, axis=0, pad_value=-1),
        [[0, 1, 2],
         [3, 4, 5],
         [-1, -1, -1],
         [-1, -1, -1]])
    np.testing.assert_array_equal(
        tools.pad_to(np.arange(6).reshape(2, 3), 4, axis=-1, pad_value=-1),
        [[0, 1, 2, -1],
         [3, 4, 5, -1]])

  def test_dynamic_batch(self):
    def dataset():
      data = [[1, 2, 2, 2],
              [1, 3, 3],
              [1, 4]]
      for d in data:
        yield np.array(d, dtype=np.int32)
    batches = list(tools.dynamic_batch(
        dataset(), batch_size=2, timesteps=3, return_incomplete_batch=False))
    self.assertLen(batches, 1)
    np.testing.assert_array_equal(
        batches[0]['obs'],
        [[1, 2, 2], [1, 3, 3]])
    np.testing.assert_array_equal(
        batches[0]['should_reset'],
        [[1, 0, 0], [1, 0, 0]])

    batches = list(tools.dynamic_batch(
        dataset(), batch_size=2, timesteps=3, return_incomplete_batch=True,
        pad=True, pad_value=0))
    # Note `return_incomplete_batch=False` drops all the incomplete batches,
    # and this can be more than just the last batch.
    self.assertLen(batches, 3)
    np.testing.assert_array_equal(
        batches[0]['obs'],
        [[1, 2, 2], [1, 3, 3]])
    np.testing.assert_array_equal(
        batches[0]['should_reset'],
        [[1, 0, 0], [1, 0, 0]])

    np.testing.assert_array_equal(
        batches[1]['obs'],
        [[2, 2, 1], [3, 0, 0]])
    np.testing.assert_array_equal(
        batches[1]['should_reset'],
        [[0, 0, 1], [0, 1, 0]])

    np.testing.assert_array_equal(
        batches[2]['obs'],
        [[1, 4, 0], [0, 0, 0]])
    np.testing.assert_array_equal(
        batches[2]['should_reset'],
        [[1, 0, 1], [1, 0, 0]])

    with self.assertRaises(ValueError):
      batches = list(tools.dynamic_batch(
          dataset(), batch_size=2, timesteps=3, return_incomplete_batch=True,
          pad=False))

  def test_batch_graph_text_pairs(self):
    def source():
      yield (1, np.array([1, 1, 1, 1, 1], dtype=np.int32))
      yield (2, np.array([2, 2], dtype=np.int32))
      yield (3, np.array([3, 3, 3, 3, 3, 3], dtype=np.int32))

    data_iter = tools.batch_graph_text_pairs(
        source(), batch_size=2, timesteps=3, pad_value=0)

    batches = list(data_iter)
    self.assertLen(batches, 4)

    batch = batches[0]
    np.testing.assert_array_equal(
        batch['obs'],
        [[1, 1, 1],
         [2, 2, 0]])
    self.assertEqual(batch['graphs'], [1, 2])
    np.testing.assert_array_equal(
        batch['should_reset'],
        [[1, 0, 0],
         [1, 0, 0]])

    batch = batches[1]
    np.testing.assert_array_equal(
        batch['obs'],
        [[1, 1, 1],
         [3, 3, 3]])
    self.assertEqual(batch['graphs'], [1, 3])
    np.testing.assert_array_equal(
        batch['should_reset'],
        [[0, 0, 0],
         [1, 0, 0]])

    batch = batches[2]
    np.testing.assert_array_equal(
        batch['obs'],
        [[1, 0, 0],
         [3, 3, 3]])
    self.assertEqual(batch['graphs'], [1, 3])
    np.testing.assert_array_equal(
        batch['should_reset'],
        [[0, 0, 0],
         [0, 0, 0]])

    batch = batches[3]
    np.testing.assert_array_equal(
        batch['obs'],
        [[0, 0, 0],
         [3, 3, 0]])
    self.assertEqual(batch['graphs'], [None, 3])
    np.testing.assert_array_equal(
        batch['should_reset'],
        [[1, 0, 0],
         [0, 0, 0]])

  def test_batch_graph_text_pairs_batch_size1(self):
    def source():
      yield (0, np.array([1, 2], dtype=np.int32))
      yield (1, np.array([1, 2, 3, 4, 5, 6], dtype=np.int32))

    data_iter = tools.batch_graph_text_pairs(
        source(), batch_size=1, timesteps=3, pad_value=0)

    batches = list(data_iter)

    batch = batches[0]
    np.testing.assert_array_equal(batch['obs'], [[1, 2, 0]])
    self.assertEqual(batch['graphs'], [0])
    np.testing.assert_array_equal(batch['should_reset'], [[1, 0, 0]])

    batch = batches[1]
    np.testing.assert_array_equal(batch['obs'], [[1, 2, 3]])
    self.assertEqual(batch['graphs'], [1])
    np.testing.assert_array_equal(batch['should_reset'], [[1, 0, 0]])

    batch = batches[2]
    np.testing.assert_array_equal(batch['obs'], [[3, 4, 5]])
    self.assertEqual(batch['graphs'], [1])
    np.testing.assert_array_equal(batch['should_reset'], [[0, 0, 0]])

    batch = batches[3]
    np.testing.assert_array_equal(batch['obs'], [[5, 6, 0]])
    self.assertEqual(batch['graphs'], [1])
    np.testing.assert_array_equal(batch['should_reset'], [[0, 0, 0]])

    self.assertLen(batches, 4)


if __name__ == '__main__':
  absltest.main()
