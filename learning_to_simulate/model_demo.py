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

Here we provide the utility function `get_sample_random_graph()` which
returns a random graph with the same characteristics as the graphs that would
be constructed by an encoder preprocessor. This simulates the output of an
encoder preprocessor: a graph with connectivity and features as described in the
paper, with features normalized to zero-mean unit-variance.


Dependencies include Tensorflow 1.x, Sonnet 1.x and the Graph Nets 1.1 library.
"""

from typing import Mapping

import graph_nets as gn
from learning_to_simulate import model

import tensorflow as tf


def get_sample_random_graph():
  """Returns mock data mimicking the input features collected by the encoder."""
  num_particles = tf.random_uniform(
      shape=(), minval=1, maxval=1000, dtype=tf.int32)
  average_num_neighbors = tf.random_uniform(
      shape=(), minval=10, maxval=30, dtype=tf.int32)
  num_edges = average_num_neighbors * num_particles

  graph_dict = {
      # 37 node features including:
      # * Previous 5 velocities (5*3)
      # * Node type embeddings (16)
      # * Clipped distance to 6 walls (6)
      "nodes": tf.random.normal(shape=[num_particles, 37]),

      # 4 edge features including:
      # * Relative displacement (3)
      # * Relative distance (1)
      "edges": tf.random.normal(shape=[num_edges, 4]),

      # Global (e.g. including global friction angle):
      # * Friction angle (1)
      "globals": tf.random.normal(shape=[1]),

      # Senders are receiver node indices for each edge.
      "senders": tf.random_uniform(
          shape=[num_edges], minval=0, maxval=num_particles, dtype=tf.int32),
      "receivers": tf.random_uniform(
          shape=[num_edges], minval=0, maxval=num_particles, dtype=tf.int32)
  }
  return graph_dict


def main():
  # Build the model.
  learnable_model = model.EncodeProcessDecode(
      latent_size=128,
      mlp_hidden_size=128,
      mlp_num_hidden_layers=2,
      num_message_passing_steps=10,
      output_size=3)

  # Fetch a batch of graphs put them into a `gn.graphs.GraphsTuple`.
  batch_size = 2
  input_graphs = gn.utils_tf.data_dicts_to_graphs_tuple(
      [get_sample_random_graph() for _ in range(batch_size)])

  per_particle_output = learnable_model(input_graphs)
  # Tensor of shape [total_num_particles, 3], with 3 outputs for each particle.
  print(f"Per-particle output tensor: {per_particle_output}")

  # All variables for the MLPs inside the GraphNetworks.
  print(f"All variables: {learnable_model.variables}")

  # Evaluate the output of the neural network.
  with tf.train.SingularMonitoredSession() as sess:
    per_particle_output_array = sess.run(per_particle_output)

  print(f"Per-particle output array: {per_particle_output_array}")


if __name__ == "__main__":
  main()
