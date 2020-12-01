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
"""Full model implementation accompanying ICML 2020 submission.

   "Learning to Simulate Complex Physics with Graph Networks"

   Alvaro Sanchez-Gonzalez*, Jonathan Godwin*, Tobias Pfaff*, Rex Ying,
   Jure Leskovec, Peter W. Battaglia

   https://arxiv.org/abs/2002.09405

"""

import graph_nets as gn
import sonnet as snt
import tensorflow.compat.v1 as tf

from learning_to_simulate import connectivity_utils
from learning_to_simulate import graph_network

STD_EPSILON = 1e-8


class LearnedSimulator(snt.AbstractModule):
  """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

  def __init__(
      self,
      num_dimensions,
      connectivity_radius,
      graph_network_kwargs,
      boundaries,
      normalization_stats,
      num_particle_types,
      particle_type_embedding_size,
      name="LearnedSimulator"):
    """Inits the model.

    Args:
      num_dimensions: Dimensionality of the problem.
      connectivity_radius: Scalar with the radius of connectivity.
      graph_network_kwargs: Keyword arguments to pass to the learned part
        of the graph network `model.EncodeProcessDecode`.
      boundaries: List of 2-tuples, containing the lower and upper boundaries of
        the cuboid containing the particles along each dimensions, matching
        the dimensionality of the problem.
      normalization_stats: Dictionary with statistics with keys "acceleration"
        and "velocity", containing a named tuple for each with mean and std
        fields, matching the dimensionality of the problem.
      num_particle_types: Number of different particle types.
      particle_type_embedding_size: Embedding size for the particle type.
      name: Name of the Sonnet module.

    """
    super().__init__(name=name)

    self._connectivity_radius = connectivity_radius
    self._num_particle_types = num_particle_types
    self._boundaries = boundaries
    self._normalization_stats = normalization_stats
    with self._enter_variable_scope():
      self._graph_network = graph_network.EncodeProcessDecode(
          output_size=num_dimensions, **graph_network_kwargs)

      if self._num_particle_types > 1:
        self._particle_type_embedding = tf.get_variable(
            "particle_embedding",
            [self._num_particle_types, particle_type_embedding_size],
            trainable=True, use_resource=True)

  def _build(self, position_sequence, n_particles_per_example,
             global_context=None, particle_types=None):
    """Produces a model step, outputting the next position for each particle.

    Args:
      position_sequence: Sequence of positions for each node in the batch,
        with shape [num_particles_in_batch, sequence_length, num_dimensions]
      n_particles_per_example: Number of particles for each graph in the batch
        with shape [batch_size]
      global_context: Tensor of shape [batch_size, context_size], with global
        context.
      particle_types: Integer tensor of shape [num_particles_in_batch] with
        the integer types of the particles, from 0 to `num_particle_types - 1`.
        If None, we assume all particles are the same type.

    Returns:
      Next position with shape [num_particles_in_batch, num_dimensions] for one
      step into the future from the input sequence.
    """
    input_graphs_tuple = self._encoder_preprocessor(
        position_sequence, n_particles_per_example, global_context,
        particle_types)

    normalized_acceleration = self._graph_network(input_graphs_tuple)

    next_position = self._decoder_postprocessor(
        normalized_acceleration, position_sequence)

    return next_position

  def _encoder_preprocessor(
      self, position_sequence, n_node, global_context, particle_types):
    # Extract important features from the position_sequence.
    most_recent_position = position_sequence[:, -1]
    velocity_sequence = time_diff(position_sequence)  # Finite-difference.

    # Get connectivity of the graph.
    (senders, receivers, n_edge
     ) = connectivity_utils.compute_connectivity_for_batch_pyfunc(
         most_recent_position, n_node, self._connectivity_radius)

    # Collect node features.
    node_features = []

    # Normalized velocity sequence, merging spatial an time axis.
    velocity_stats = self._normalization_stats["velocity"]
    normalized_velocity_sequence = (
        velocity_sequence - velocity_stats.mean) / velocity_stats.std

    flat_velocity_sequence = snt.MergeDims(start=1, size=2)(
        normalized_velocity_sequence)
    node_features.append(flat_velocity_sequence)

    # Normalized clipped distances to lower and upper boundaries.
    # boundaries are an array of shape [num_dimensions, 2], where the second
    # axis, provides the lower/upper boundaries.
    boundaries = tf.constant(self._boundaries, dtype=tf.float32)
    distance_to_lower_boundary = (
        most_recent_position - tf.expand_dims(boundaries[:, 0], 0))
    distance_to_upper_boundary = (
        tf.expand_dims(boundaries[:, 1], 0) - most_recent_position)
    distance_to_boundaries = tf.concat(
        [distance_to_lower_boundary, distance_to_upper_boundary], axis=1)
    normalized_clipped_distance_to_boundaries = tf.clip_by_value(
        distance_to_boundaries / self._connectivity_radius, -1., 1.)
    node_features.append(normalized_clipped_distance_to_boundaries)

    # Particle type.
    if self._num_particle_types > 1:
      particle_type_embeddings = tf.nn.embedding_lookup(
          self._particle_type_embedding, particle_types)
      node_features.append(particle_type_embeddings)

    # Collect edge features.
    edge_features = []

    # Relative displacement and distances normalized to radius
    normalized_relative_displacements = (
        tf.gather(most_recent_position, senders) -
        tf.gather(most_recent_position, receivers)) / self._connectivity_radius
    edge_features.append(normalized_relative_displacements)

    normalized_relative_distances = tf.norm(
        normalized_relative_displacements, axis=-1, keepdims=True)
    edge_features.append(normalized_relative_distances)

    # Normalize the global context.
    if global_context is not None:
      context_stats = self._normalization_stats["context"]
      # Context in some datasets are all zero, so add an epsilon for numerical
      # stability.
      global_context = (global_context - context_stats.mean) / tf.math.maximum(
          context_stats.std, STD_EPSILON)

    return gn.graphs.GraphsTuple(
        nodes=tf.concat(node_features, axis=-1),
        edges=tf.concat(edge_features, axis=-1),
        globals=global_context,  # self._graph_net will appending this to nodes.
        n_node=n_node,
        n_edge=n_edge,
        senders=senders,
        receivers=receivers,
        )

  def _decoder_postprocessor(self, normalized_acceleration, position_sequence):

    # The model produces the output in normalized space so we apply inverse
    # normalization.
    acceleration_stats = self._normalization_stats["acceleration"]
    acceleration = (
        normalized_acceleration * acceleration_stats.std
        ) + acceleration_stats.mean

    # Use an Euler integrator to go from acceleration to position, assuming
    # a dt=1 corresponding to the size of the finite difference.
    most_recent_position = position_sequence[:, -1]
    most_recent_velocity = most_recent_position - position_sequence[:, -2]

    new_velocity = most_recent_velocity + acceleration  # * dt = 1
    new_position = most_recent_position + new_velocity  # * dt = 1
    return new_position

  def get_predicted_and_target_normalized_accelerations(
      self, next_position, position_sequence_noise, position_sequence,
      n_particles_per_example, global_context=None, particle_types=None):  # pylint: disable=g-doc-args
    """Produces normalized and predicted acceleration targets.

    Args:
      next_position: Tensor of shape [num_particles_in_batch, num_dimensions]
        with the positions the model should output given the inputs.
      position_sequence_noise: Tensor of the same shape as `position_sequence`
        with the noise to apply to each particle.
      position_sequence, n_node, global_context, particle_types: Inputs to the
        model as defined by `_build`.

    Returns:
      Tensors of shape [num_particles_in_batch, num_dimensions] with the
        predicted and target normalized accelerations.
    """

    # Add noise to the input position sequence.
    noisy_position_sequence = position_sequence + position_sequence_noise

    # Perform the forward pass with the noisy position sequence.
    input_graphs_tuple = self._encoder_preprocessor(
        noisy_position_sequence, n_particles_per_example, global_context,
        particle_types)
    predicted_normalized_acceleration = self._graph_network(input_graphs_tuple)

    # Calculate the target acceleration, using an `adjusted_next_position `that
    # is shifted by the noise in the last input position.
    next_position_adjusted = next_position + position_sequence_noise[:, -1]
    target_normalized_acceleration = self._inverse_decoder_postprocessor(
        next_position_adjusted, noisy_position_sequence)
    # As a result the inverted Euler update in the `_inverse_decoder` produces:
    # * A target acceleration that does not explicitly correct for the noise in
    #   the input positions, as the `next_position_adjusted` is different
    #   from the true `next_position`.
    # * A target acceleration that exactly corrects noise in the input velocity
    #   since the target next velocity calculated by the inverse Euler update
    #   as `next_position_adjusted - noisy_position_sequence[:,-1]`
    #   matches the ground truth next velocity (noise cancels out).

    return predicted_normalized_acceleration, target_normalized_acceleration

  def _inverse_decoder_postprocessor(self, next_position, position_sequence):
    """Inverse of `_decoder_postprocessor`."""

    previous_position = position_sequence[:, -1]
    previous_velocity = previous_position - position_sequence[:, -2]
    next_velocity = next_position - previous_position
    acceleration = next_velocity - previous_velocity

    acceleration_stats = self._normalization_stats["acceleration"]
    normalized_acceleration = (
        acceleration - acceleration_stats.mean) / acceleration_stats.std
    return normalized_acceleration


def time_diff(input_sequence):
  return input_sequence[:, 1:] - input_sequence[:, :-1]

