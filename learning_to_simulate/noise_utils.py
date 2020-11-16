# Lint as: python3
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
"""Methods to calculate input noise."""

import tensorflow.compat.v1 as tf

from learning_to_simulate import learned_simulator


def get_random_walk_noise_for_position_sequence(
    position_sequence, noise_std_last_step):
  """Returns random-walk noise in the velocity applied to the position."""

  velocity_sequence = learned_simulator.time_diff(position_sequence)

  # We want the noise scale in the velocity at the last step to be fixed.
  # Because we are going to compose noise at each step using a random_walk:
  # std_last_step**2 = num_velocities * std_each_step**2
  # so to keep `std_last_step` fixed, we apply at each step:
  # std_each_step `std_last_step / np.sqrt(num_input_velocities)`
  # TODO(alvarosg): Make sure this is consistent with the value and
  # description provided in the paper.
  num_velocities = velocity_sequence.shape.as_list()[1]
  velocity_sequence_noise = tf.random.normal(
      tf.shape(velocity_sequence),
      stddev=noise_std_last_step / num_velocities ** 0.5,
      dtype=position_sequence.dtype)

  # Apply the random walk.
  velocity_sequence_noise = tf.cumsum(velocity_sequence_noise, axis=1)

  # Integrate the noise in the velocity to the positions, assuming
  # an Euler intergrator and a dt = 1, and adding no noise to the very first
  # position (since that will only be used to calculate the first position
  # change).
  position_sequence_noise = tf.concat([
      tf.zeros_like(velocity_sequence_noise[:, 0:1]),
      tf.cumsum(velocity_sequence_noise, axis=1)], axis=1)

  return position_sequence_noise
