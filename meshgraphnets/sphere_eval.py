# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
"""Functions to build evaluation metrics for sphere dynamic data."""

import tensorflow.compat.v1 as tf

from meshgraphnets.common import NodeType


def _rollout(model, initial_state, num_steps):
  """Rolls out a model trajectory for sphere dynamics."""
  mask = tf.equal(initial_state['node_type'][:, 0], NodeType.NORMAL)

  def step_fn(step, prev_pos, cur_pos, trajectory):
    """Single step of sphere dynamics simulation."""
    prediction = model({**initial_state,
                        'prev|world_pos': prev_pos,
                        'world_pos': cur_pos})
    # Don't update kinematic/boundary nodes - they remain fixed
    next_pos = tf.where(mask, prediction, cur_pos)
    trajectory = trajectory.write(step, cur_pos)
    return step+1, cur_pos, next_pos, trajectory

  _, _, _, output = tf.while_loop(
      cond=lambda step, last, cur, traj: tf.less(step, num_steps),
      body=step_fn,
      loop_vars=(0, initial_state['prev|world_pos'], initial_state['world_pos'],
                 tf.TensorArray(tf.float32, num_steps)),
      parallel_iterations=1)
  return output.stack()


def evaluate(model, inputs):
  """Performs model rollouts and create evaluation statistics for sphere dynamics."""
  initial_state = {k: v[0] for k, v in inputs.items()}
  num_steps = inputs['cells'].shape[0]
  prediction = _rollout(model, initial_state, num_steps)

  # Calculate mean squared error between prediction and ground truth
  error = tf.reduce_mean((prediction - inputs['world_pos'])**2, axis=-1)
  
  # Create evaluation metrics for different prediction horizons
  scalars = {'mse_%d_steps' % horizon: tf.reduce_mean(error[1:horizon+1])
             for horizon in [1, 5, 10, 20, 50, 100, 200]}
  
  # Additional sphere-specific metrics
  # Calculate error in terms of surface displacement
  center_of_mass_gt = tf.reduce_mean(inputs['world_pos'], axis=1, keepdims=True)
  center_of_mass_pred = tf.reduce_mean(prediction, axis=1, keepdims=True)
  com_error = tf.reduce_mean((center_of_mass_pred - center_of_mass_gt)**2)
  scalars['center_of_mass_error'] = com_error
  
  # Calculate volume preservation error (for sphere dynamics)
  # This is approximated by the variance in distance from center
  def calc_volume_variance(positions):
    com = tf.reduce_mean(positions, axis=1, keepdims=True)
    distances = tf.norm(positions - com, axis=-1)
    return tf.reduce_var(distances, axis=1)
  
  vol_var_gt = calc_volume_variance(inputs['world_pos'])
  vol_var_pred = calc_volume_variance(prediction)
  volume_error = tf.reduce_mean((vol_var_pred - vol_var_gt)**2)
  scalars['volume_preservation_error'] = volume_error
  
  # Trajectory operations for visualization and further analysis
  traj_ops = {
      'faces': inputs['cells'],
      'mesh_pos': inputs['mesh_pos'],
      'gt_pos': inputs['world_pos'],
      'pred_pos': prediction,
      'node_type': inputs['node_type']
  }
  
  return scalars, traj_ops
