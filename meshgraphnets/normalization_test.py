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
"""Tests for MeshGraphNets online normalization."""

import tensorflow.compat.v1 as tf

from meshgraphnets import normalization


class NormalizerTest(tf.test.TestCase):

  def test_roundoff_negative_variance_does_not_produce_nan(self):
    normalizer = normalization.Normalizer(size=1, std_epsilon=1e-8)
    std = normalizer._std_with_epsilon()

    # These accumulated statistics produce a small negative value for
    # E[x^2] - E[x]^2, as can happen from floating-point roundoff.
    set_stats = tf.group(
        tf.assign(normalizer._acc_count, 1.),
        tf.assign(normalizer._acc_sum, [1.0000001]),
        tf.assign(normalizer._acc_sum_squared, [1.]))

    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(set_stats)
      result = sess.run(std)

    self.assertAllClose(result, [1e-8])


if __name__ == '__main__':
  tf.test.main()
