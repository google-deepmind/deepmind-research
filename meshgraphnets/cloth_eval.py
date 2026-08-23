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
"""Functions to build evaluation metrics for cloth data."""

import tensorflow.compat.v1 as tf

from meshgraphnets.common import NodeType


def _rollout(model, initial_state, num_steps):
  """Rolls out a model trajectory."""
  mask = tf.equal(initial_state['node_type'][:, 0], NodeType.NORMAL)

  def step_fn(step, prev_pos, cur_pos, trajectory):
    prediction = model({**initial_state,
                        'prev|world_pos': prev_pos,
                        'world_pos': cur_pos})
    # don't update kinematic nodes
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


def _resize_static_trajectory_field(field, num_steps):
  """Trims or extends a static mesh field to match a rollout length."""
  field_steps = field.shape.as_list()[0]
  if num_steps <= field_steps:
    return field[:num_steps]
  extra_steps = num_steps - field_steps
  return tf.concat(
      [field, tf.tile(field[-1:], [extra_steps, 1, 1])], axis=0)


def evaluate(model, inputs, num_steps=None):
  """Performs model rollouts and create stats."""
  initial_state = {k: v[0] for k, v in inputs.items()}
  ground_truth_steps = inputs['cells'].shape.as_list()[0]
  if num_steps is None:
    num_steps = ground_truth_steps
  if num_steps <= 0:
    raise ValueError('num_steps must be positive.')
  prediction = _rollout(model, initial_state, num_steps)

  comparison_steps = min(num_steps, ground_truth_steps)
  error = tf.reduce_mean(
      (prediction[:comparison_steps] -
       inputs['world_pos'][:comparison_steps])**2,
      axis=-1)
  scalars = {'mse_%d_steps' % horizon: tf.reduce_mean(error[1:horizon+1])
             for horizon in [1, 10, 20, 50, 100, 200]}
  traj_ops = {
      'faces': _resize_static_trajectory_field(inputs['cells'], num_steps),
      'mesh_pos': _resize_static_trajectory_field(inputs['mesh_pos'], num_steps),
      'gt_pos': inputs['world_pos'][:comparison_steps],
      'pred_pos': prediction
  }
  return scalars, traj_ops
