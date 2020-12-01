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

"""Example script accompanying ICML 2020 submission.

   "Learning to Simulate Complex Physics with Graph Networks"

   Alvaro Sanchez-Gonzalez*, Jonathan Godwin*, Tobias Pfaff*, Rex Ying,
   Jure Leskovec, Peter W. Battaglia

   https://arxiv.org/abs/2002.09405

Here we provide the utility function `sample_random_position_sequence()` which
returns a sequence of positions for a variable number of particles, similar to
what a real dataset would provide, and connect the model to it, in both,
single step inference and training mode.

Dependencies include Tensorflow 1.x, Sonnet 1.x and the Graph Nets 1.1 library.
"""

import collections

from learning_to_simulate import learned_simulator
from learning_to_simulate import noise_utils
import numpy as np
import tensorflow.compat.v1 as tf

INPUT_SEQUENCE_LENGTH = 6
SEQUENCE_LENGTH = INPUT_SEQUENCE_LENGTH + 1  # add one target position.
NUM_DIMENSIONS = 3
NUM_PARTICLE_TYPES = 6
BATCH_SIZE = 5
GLOBAL_CONTEXT_SIZE = 6

Stats = collections.namedtuple("Stats", ["mean", "std"])

DUMMY_STATS = Stats(
    mean=np.zeros([NUM_DIMENSIONS], dtype=np.float32),
    std=np.ones([NUM_DIMENSIONS], dtype=np.float32))
DUMMY_CONTEXT_STATS = Stats(
    mean=np.zeros([GLOBAL_CONTEXT_SIZE], dtype=np.float32),
    std=np.ones([GLOBAL_CONTEXT_SIZE], dtype=np.float32))
DUMMY_BOUNDARIES = [(-1., 1.)] * NUM_DIMENSIONS


def sample_random_position_sequence():
  """Returns mock data mimicking the input features collected by the encoder."""
  num_particles = tf.random_uniform(
      shape=(), minval=50, maxval=1000, dtype=tf.int32)
  position_sequence = tf.random.normal(
      shape=[num_particles, SEQUENCE_LENGTH, NUM_DIMENSIONS])
  return position_sequence


def main():

  # Build the model.
  learnable_model = learned_simulator.LearnedSimulator(
      num_dimensions=NUM_DIMENSIONS,
      connectivity_radius=0.05,
      graph_network_kwargs=dict(
          latent_size=128,
          mlp_hidden_size=128,
          mlp_num_hidden_layers=2,
          num_message_passing_steps=10,
      ),
      boundaries=DUMMY_BOUNDARIES,
      normalization_stats={"acceleration": DUMMY_STATS,
                           "velocity": DUMMY_STATS,
                           "context": DUMMY_CONTEXT_STATS,},
      num_particle_types=NUM_PARTICLE_TYPES,
      particle_type_embedding_size=16,
    )

  # Sample a batch of particle sequences with shape:
  # [TOTAL_NUM_PARTICLES, SEQUENCE_LENGTH, NUM_DIMENSIONS]
  sampled_position_sequences = [
      sample_random_position_sequence() for _ in range(BATCH_SIZE)]
  position_sequence_batch = tf.concat(sampled_position_sequences, axis=0)

  # Count how many particles are present in each element in the batch.
  # [BATCH_SIZE]
  n_particles_per_example = tf.stack(
      [tf.shape(seq)[0] for seq in sampled_position_sequences], axis=0)

  # Sample particle types.
  # [TOTAL_NUM_PARTICLES]
  particle_types = tf.random_uniform(
      [tf.shape(position_sequence_batch)[0]],
      0, NUM_PARTICLE_TYPES, dtype=tf.int32)

  # Sample global context.
  global_context = tf.random_uniform(
      [BATCH_SIZE, GLOBAL_CONTEXT_SIZE], -1., 1., dtype=tf.float32)

  # Separate input sequence from target sequence.
  # [TOTAL_NUM_PARTICLES, INPUT_SEQUENCE_LENGTH, NUM_DIMENSIONS]
  input_position_sequence = position_sequence_batch[:, :-1]
  # [TOTAL_NUM_PARTICLES, NUM_DIMENSIONS]
  target_next_position = position_sequence_batch[:, -1]

  # Single step of inference with the model to predict next position for each
  # particle [TOTAL_NUM_PARTICLES, NUM_DIMENSIONS].
  predicted_next_position = learnable_model(
      input_position_sequence, n_particles_per_example, global_context,
      particle_types)
  print(f"Per-particle output tensor: {predicted_next_position}")

  # Obtaining predicted and target normalized accelerations for training.
  position_sequence_noise = (
      noise_utils.get_random_walk_noise_for_position_sequence(
          input_position_sequence, noise_std_last_step=6.7e-4))

  # Both with shape [TOTAL_NUM_PARTICLES, NUM_DIMENSIONS]
  predicted_normalized_acceleration, target_normalized_acceleration = (
      learnable_model.get_predicted_and_target_normalized_accelerations(
          target_next_position, position_sequence_noise,
          input_position_sequence, n_particles_per_example, global_context,
          particle_types))
  print(f"Predicted norm. acceleration: {predicted_normalized_acceleration}")
  print(f"Target norm. acceleration: {target_normalized_acceleration}")

  with tf.train.SingularMonitoredSession() as sess:
    sess.run([predicted_next_position,
              predicted_normalized_acceleration,
              target_normalized_acceleration])


if __name__ == "__main__":
  tf.disable_v2_behavior()
  main()
