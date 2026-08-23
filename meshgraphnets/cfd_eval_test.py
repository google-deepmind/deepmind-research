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
"""Tests for CFD rollout evaluation."""

import numpy as np
import tensorflow.compat.v1 as tf

from meshgraphnets import cfd_eval
from meshgraphnets.common import NodeType


class _IncrementVelocityModel:

  def __call__(self, inputs):
    return inputs['velocity'] + tf.ones_like(inputs['velocity'])


class CfdEvalTest(tf.test.TestCase):

  def _inputs(self):
    return {
        'cells': tf.constant(
            np.zeros((3, 1, 3), dtype=np.int32)),
        'mesh_pos': tf.constant(
            np.zeros((3, 2, 2), dtype=np.float32)),
        'velocity': tf.constant(
            np.zeros((3, 2, 2), dtype=np.float32)),
        'node_type': tf.constant(
            np.full((3, 2, 1), NodeType.NORMAL, dtype=np.int32)),
    }

  def test_default_rollout_uses_ground_truth_length(self):
    _, trajectory = cfd_eval.evaluate(
        _IncrementVelocityModel(), self._inputs())

    with self.session() as sess:
      prediction = sess.run(trajectory['pred_velocity'])

    self.assertEqual(prediction.shape[0], 3)

  def test_long_rollout_extends_prediction_and_static_mesh(self):
    _, trajectory = cfd_eval.evaluate(
        _IncrementVelocityModel(), self._inputs(), num_steps=5)

    with self.session() as sess:
      result = sess.run(trajectory)

    self.assertEqual(result['pred_velocity'].shape[0], 5)
    self.assertEqual(result['faces'].shape[0], 5)
    self.assertEqual(result['mesh_pos'].shape[0], 5)
    self.assertEqual(result['gt_velocity'].shape[0], 3)
    self.assertAllClose(result['pred_velocity'][:, 0, 0], [0, 1, 2, 3, 4])


if __name__ == '__main__':
  tf.test.main()
