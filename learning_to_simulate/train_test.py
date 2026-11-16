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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for particle simulation evaluation metrics."""

from unittest import mock

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from learning_to_simulate import train


# Empty-weight batches before and after valid data, unequal valid particle
# counts, and a batch with no kinematic particles.
_BATCHES = (
    ((3,), (100.0,)),
    ((0, 3), (2.0, 100.0)),
    ((0, 0, 3, 3), (4.0, 6.0, -100.0, 100.0)),
    ((0, 0), (1.0, 3.0)),
    ((3, 3), (200.0, -200.0)),
)


class EvaluationMetricsTest(tf.test.TestCase, parameterized.TestCase):
  """Checks estimator metrics against errors on learned particle dynamics."""

  @parameterized.named_parameters(('2d', 2), ('3d', 3))
  def test_one_step_metrics_exclude_kinematic_particles(self, num_dimensions):
    with tf.Graph().as_default() as graph:
      particle_types = tf.placeholder(tf.int32, shape=[None])
      errors = tf.placeholder(tf.float32, shape=[None, num_dimensions])
      # A trainable scalar lets the estimator construct its training operation.
      acceleration = errors + tf.get_variable('offset', initializer=0.0)
      target = tf.zeros_like(errors)
      simulator = mock.Mock(return_value=2.0 * acceleration)
      simulator.get_predicted_and_target_normalized_accelerations.return_value = (
          acceleration, target)
      positions = tf.zeros([tf.shape(particle_types)[0],
                            train.INPUT_SEQUENCE_LENGTH, num_dimensions])
      features = {
          'particle_type': particle_types,
          'position': positions,
          'n_particles_per_example': tf.shape(particle_types),
      }
      tf.train.get_or_create_global_step()
      with mock.patch.object(train, '_read_metadata', return_value={}), \
           mock.patch.object(train, '_get_simulator', return_value=simulator):
        estimator_fn = train.get_one_step_estimator_fn('', noise_std=0.0)
        spec = estimator_fn(features, target, tf.estimator.ModeKeys.EVAL)

      values, updates = zip(*spec.eval_metric_ops.values())
      with self.session(graph=graph) as session:
        session.run([
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        ])
        squared_error_sum = 0.0
        coordinate_count = 0
        for types, magnitudes in _BATCHES:
          batch_errors = (np.asarray(magnitudes, dtype=np.float32)[:, None] +
                          np.arange(num_dimensions, dtype=np.float32))
          session.run(updates, feed_dict={
              particle_types: types, errors: batch_errors})
          valid_errors = batch_errors[np.asarray(types) != 3]
          squared_error_sum += np.sum(valid_errors**2)
          coordinate_count += valid_errors.size
          expected_mse = (squared_error_sum / coordinate_count
                          if coordinate_count else 0.0)
          metrics = dict(zip(spec.eval_metric_ops, session.run(values)))
          self.assertAllClose(metrics['loss_mse'], expected_mse)
          self.assertAllClose(
              metrics['one_step_position_mse'], 4.0 * expected_mse)

  @parameterized.named_parameters(('2d', 2), ('3d', 3))
  def test_rollout_metrics_exclude_kinematic_particles(self, num_dimensions):
    num_steps = 3
    with tf.Graph().as_default() as graph:
      particle_types = tf.placeholder(tf.int32, shape=[None])
      increments = tf.placeholder(tf.float32, shape=[None, num_dimensions])
      positions = tf.zeros([tf.shape(particle_types)[0],
                            train.INPUT_SEQUENCE_LENGTH + num_steps,
                            num_dimensions])
      features = {
          'particle_type': particle_types,
          'position': positions,
          'n_particles_per_example': tf.shape(particle_types),
      }

      def simulator(position_sequence, **unused_kwargs):
        return position_sequence[:, -1] + increments

      metadata = {'sequence_length': train.INPUT_SEQUENCE_LENGTH + num_steps}
      with mock.patch.object(train, '_read_metadata', return_value=metadata), \
           mock.patch.object(train, '_get_simulator', return_value=simulator):
        estimator_fn = train.get_rollout_estimator_fn('', noise_std=0.0)
        spec = estimator_fn(features, None, tf.estimator.ModeKeys.EVAL)

      value, update = spec.eval_metric_ops['rollout_error_mse']
      with self.session(graph=graph) as session:
        session.run(tf.local_variables_initializer())
        squared_error_sum = 0.0
        coordinate_count = 0
        for types, magnitudes in _BATCHES:
          batch_increments = (
              np.asarray(magnitudes, dtype=np.float32)[:, None] +
              np.arange(num_dimensions, dtype=np.float32))
          _, loss, predictions = session.run(
              [update, spec.loss, spec.predictions['predicted_rollout']],
              feed_dict={particle_types: types, increments: batch_increments})
          non_kinematic = np.asarray(types) != 3
          # The real rollout keeps prescribed particles at ground truth.
          self.assertAllEqual(predictions[0, :, ~non_kinematic],
                              np.zeros_like(predictions[0, :, ~non_kinematic]))
          valid_errors = (np.arange(1, num_steps + 1)[:, None, None] *
                          batch_increments[None, non_kinematic])
          expected_loss = (np.mean(valid_errors**2)
                           if valid_errors.size else 0.0)
          self.assertAllClose(loss, expected_loss)
          squared_error_sum += np.sum(valid_errors**2)
          coordinate_count += valid_errors.size
          expected_mse = (squared_error_sum / coordinate_count
                          if coordinate_count else 0.0)
          self.assertAllClose(session.run(value), expected_mse)


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
