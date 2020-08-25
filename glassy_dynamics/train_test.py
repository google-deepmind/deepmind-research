# Copyright 2019 Deepmind Technologies Limited.
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

"""Tests for train."""

import os

import numpy as np
import tensorflow.compat.v1 as tf

from glassy_dynamics import train


class TrainTest(tf.test.TestCase):

  def test_get_targets(self):
    initial_positions = np.array([[0, 0, 0], [1, 2, 3]])
    trajectory_target_positions = [
        np.array([[1, 0, 0], [1, 2, 4]]),
        np.array([[0, 1, 0], [1, 0, 3]]),
        np.array([[0, 0, 5], [1, 2, 3]]),
    ]
    expected_targets = np.array([7.0 / 3.0, 1.0])
    targets = train.get_targets(initial_positions, trajectory_target_positions)
    np.testing.assert_almost_equal(expected_targets, targets)

  def test_load_data(self):
    file_pattern = os.path.join(os.path.dirname(__file__), 'testdata',
                                'test_small.pickle')

    with self.subTest('ContentAndShapesAreAsExpected'):
      data = train.load_data(file_pattern, 0)
      self.assertEqual(len(data), 1)
      element = data[0]
      self.assertTupleEqual(element.positions.shape, (20, 3))
      self.assertTupleEqual(element.box.shape, (3,))
      self.assertTupleEqual(element.targets.shape, (20,))
      self.assertTupleEqual(element.types.shape, (20,))

    with self.subTest('TargetsGrowAsAFunctionOfTime'):
      previous_mean_target = 0.0
      # Time index 9 refers to 1/e = 0.36 in the IS, and therefore it is between
      # Time index 5 (0.4) and  time index 6 (0.3).
      for time_index in [0, 1, 2, 3, 4, 5, 9, 6, 7, 8]:
        data = train.load_data(file_pattern, time_index)[0]
        current_mean_target = data.targets.mean()
        self.assertGreater(current_mean_target, previous_mean_target)
        previous_mean_target = current_mean_target


class TensorflowTrainTest(tf.test.TestCase):

  def test_get_loss_op(self):
    """Tests the correct calculation of the loss operations."""
    prediction = tf.constant([0.0, 1.0, 2.0, 1.0, 2.0], dtype=tf.float32)
    target = tf.constant([1.0, 25.0, 0.0, 4.0, 2.0], dtype=tf.float32)
    types = tf.constant([0, 1, 0, 0, 0], dtype=tf.int32)
    loss_ops = train.get_loss_ops(prediction, target, types)
    loss = self.evaluate(loss_ops)
    self.assertAlmostEqual(loss.l1_loss, 1.5)
    self.assertAlmostEqual(loss.l2_loss, 14.0 / 4.0)
    self.assertAlmostEqual(loss.correlation, -0.15289416)

  def test_get_minimize_op(self):
    """Tests the minimize operation by minimizing a single variable."""
    var = tf.Variable([1.0], name='test')
    loss = var**2
    minimize = train.get_minimize_op(loss, 1e-1)
    with self.session():
      tf.global_variables_initializer().run()
      for _ in range(100):
        minimize.run()
      value = var.eval()
      self.assertLess(abs(value[0]), 0.01)

  def test_train_model(self):
    """Tests if we can overfit to a small test dataset."""
    file_pattern = os.path.join(os.path.dirname(__file__), 'testdata',
                                'test_small.pickle')
    best_correlation_value = train.train_model(
        train_file_pattern=file_pattern,
        test_file_pattern=file_pattern,
        n_epochs=1000,
        augment_data_using_rotations=False,
        learning_rate=1e-4,
        n_recurrences=2,
        edge_threshold=5,
        mlp_sizes=(32, 32),
        measurement_store_interval=1000)
    # The test dataset contains only a single sample with 20 particles.
    # Therefore we expect the model to be able to memorize the targets perfectly
    # if the model works correctly.
    self.assertGreater(best_correlation_value, 0.99)

  def test_apply_model(self):
    """Tests if we can apply a model to a small test dataset."""
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints',
                                   't044_s09.ckpt')
    file_pattern = os.path.join(os.path.dirname(__file__), 'testdata',
                                'test_large.pickle')
    predictions = train.apply_model(checkpoint_path=checkpoint_path,
                                    file_pattern=file_pattern,
                                    time_index=0)
    data = train.load_data(file_pattern, 0)
    targets = data[0].targets
    correlation_value = np.corrcoef(predictions[0], targets)[0, 1]
    self.assertGreater(correlation_value, 0.5)


if __name__ == '__main__':
  tf.test.main()
