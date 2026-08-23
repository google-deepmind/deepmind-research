# pylint: disable=g-bad-file-header
# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for MeshGraphNets dataset parsing."""

import numpy as np
import tensorflow.compat.v1 as tf

from meshgraphnets import dataset


class DatasetTest(tf.test.TestCase):

  def _parse_sizing_field(self, values, lengths):
    example = tf.train.Example(features=tf.train.Features(feature={
        'sizing_field': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[values.tobytes()])),
        'length_sizing_field': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[lengths.tobytes()])),
    })).SerializeToString()
    meta = {
        'field_names': ['sizing_field', 'length_sizing_field'],
        'features': {
            'sizing_field': {
                'type': 'dynamic_varlen',
                'shape': [-1, 4],
                'dtype': "<dtype: 'float32'>",
            },
        },
    }
    return dataset._parse(tf.convert_to_tensor(example), meta)['sizing_field']

  def test_dynamic_varlen_tolerates_legacy_metadata(self):
    values = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float32)
    lengths = np.array([2, 1], dtype=np.int32)

    parsed = self._parse_sizing_field(values, lengths)

    with self.session() as sess:
      result = sess.run(parsed)

    self.assertAllClose(result.values, values)
    self.assertAllEqual(result.row_splits, [0, 2, 3])

  def test_dynamic_varlen_allows_empty_rows(self):
    values = np.empty((0, 3), dtype=np.float32)
    lengths = np.array([0, 0], dtype=np.int32)

    parsed = self._parse_sizing_field(values, lengths)

    with self.session() as sess:
      result = sess.run(parsed)

    self.assertEqual(result.values.shape[0], 0)
    self.assertAllEqual(result.row_splits, [0, 0, 0])


if __name__ == '__main__':
  tf.test.main()
