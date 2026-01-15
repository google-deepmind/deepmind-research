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

============================================================================
BEGINNER'S GUIDE TO THIS FILE
============================================================================

This file implements the "LearnedSimulator" - a neural network that predicts
how particles move in physical simulations (water, sand, gas, etc.).

KEY CONCEPTS:

1. LEARNED SIMULATOR
   - Instead of physics equations, we train a neural network on simulation data
   - Input: particle positions over the last few timesteps
   - Output: predicted position at the next timestep

2. WHY GRAPHS?
   - Particles interact with nearby particles (like water molecules pushing)
   - Graph representation: particles = nodes, interactions = edges
   - Edges connect particles within a "connectivity radius"

3. THE PIPELINE (Encode-Process-Decode):
   
   Positions → [Encoder] → Graph → [Message Passing] → [Decoder] → Acceleration

4. PHYSICS INSIGHT:
   - Network predicts ACCELERATION (not position)
   - Then: new_velocity = old_velocity + acceleration
           new_position = old_position + new_velocity

5. NORMALIZATION:
   - All features normalized to zero-mean, unit-variance
   - Helps neural network train more stably

============================================================================
"""

import graph_nets as gn
import sonnet as snt
import tensorflow.compat.v1 as tf

from learning_to_simulate import connectivity_utils
from learning_to_simulate import graph_network

STD_EPSILON = 1e-8  # Small value to prevent division by zero


class LearnedSimulator(snt.AbstractModule):
  """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf.
  
  BEGINNER NOTE:
  This class is the main model. It takes particle positions as input and
  predicts where they'll be in the next timestep through:
  
  1. Building a graph connecting nearby particles
  2. Message-passing to let particles "communicate"
  3. Predicting acceleration for each particle
  4. Using physics to convert acceleration → position
  """

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
        # BEGINNER: Usually 2 (2D) or 3 (3D simulations)
      connectivity_radius: Scalar with the radius of connectivity.
        # BEGINNER: Particles within this distance are connected by an edge
      graph_network_kwargs: Keyword arguments to pass to the learned part
        of the graph network `model.EncodeProcessDecode`.
        # BEGINNER: Neural network configuration (layer sizes, etc.)
      boundaries: List of 2-tuples, containing the lower and upper boundaries of
        the cuboid containing the particles along each dimensions, matching
        the dimensionality of the problem.
        # BEGINNER: The "walls" of the simulation box
      normalization_stats: Dictionary with statistics with keys "acceleration"
        and "velocity", containing a named tuple for each with mean and std
        fields, matching the dimensionality of the problem.
        # BEGINNER: Pre-computed mean/std for normalizing inputs/outputs
      num_particle_types: Number of different particle types.
        # BEGINNER: Different materials (water, sand) can have different types
      particle_type_embedding_size: Embedding size for the particle type.
        # BEGINNER: Vector size to represent each particle type
      name: Name of the Sonnet module.

    """
    super().__init__(name=name)

    self._connectivity_radius = connectivity_radius
    self._num_particle_types = num_particle_types
    self._boundaries = boundaries
    self._normalization_stats = normalization_stats
    with self._enter_variable_scope():
      # Create the core graph neural network (the "brain" of the simulator)
      self._graph_network = graph_network.EncodeProcessDecode(
          output_size=num_dimensions, **graph_network_kwargs)

      # Create embeddings for particle types (like word embeddings for materials)
      if self._num_particle_types > 1:
        self._particle_type_embedding = tf.get_variable(
            "particle_embedding",
            [self._num_particle_types, particle_type_embedding_size],
            trainable=True, use_resource=True)

  def _build(self, position_sequence, n_particles_per_example,
             global_context=None, particle_types=None):
    """Produces a model step, outputting the next position for each particle.
    
    BEGINNER WALKTHROUGH - The Forward Pass:
    
    Step 1: _encoder_preprocessor
            - Convert positions → graph structure
            - Compute velocities from position history
            - Find nearby particles to connect
            
    Step 2: _graph_network (message passing)
            - Particles exchange information with neighbors
            - Network learns what info to send/receive
            
    Step 3: _decoder_postprocessor
            - Convert normalized acceleration → real acceleration
            - Use physics: position = position + velocity + acceleration

    Args:
      position_sequence: Sequence of positions for each node in the batch,
        with shape [num_particles_in_batch, sequence_length, num_dimensions]
        # BEGINNER: History of particle positions (e.g., last 5 timesteps)
      n_particles_per_example: Number of particles for each graph in the batch
        with shape [batch_size]
        # BEGINNER: How many particles belong to each simulation in the batch
      global_context: Tensor of shape [batch_size, context_size], with global
        context.
        # BEGINNER: Global info like external forces (gravity)
      particle_types: Integer tensor of shape [num_particles_in_batch] with
        the integer types of the particles, from 0 to `num_particle_types - 1`.
        If None, we assume all particles are the same type.
        # BEGINNER: Material labels (0=water, 1=sand, etc.)

    Returns:
      Next position with shape [num_particles_in_batch, num_dimensions] for one
      step into the future from the input sequence.
    """
    # STEP 1: Build input graph with features
    input_graphs_tuple = self._encoder_preprocessor(
        position_sequence, n_particles_per_example, global_context,
        particle_types)

    # STEP 2: Run graph neural network (message passing)
    normalized_acceleration = self._graph_network(input_graphs_tuple)

    # STEP 3: Convert acceleration to position using physics
    next_position = self._decoder_postprocessor(
        normalized_acceleration, position_sequence)

    return next_position

  def _encoder_preprocessor(
      self, position_sequence, n_node, global_context, particle_types):
    """Builds the input graph from raw positions.
    
    BEGINNER NOTE - This function:
    1. Computes velocities via finite difference
    2. Finds particle pairs within connectivity_radius (graph edges)
    3. Builds normalized node features (velocity, boundary distance, type)
    4. Builds normalized edge features (relative displacement, distance)
    """
    # Get most recent position for building connectivity
    most_recent_position = position_sequence[:, -1]
    
    # Compute velocities: velocity[t] = position[t+1] - position[t]
    velocity_sequence = time_diff(position_sequence)  # Finite-difference.

    # Find connected particle pairs using KD-tree spatial query
    (senders, receivers, n_edge
     ) = connectivity_utils.compute_connectivity_for_batch_pyfunc(
         most_recent_position, n_node, self._connectivity_radius)

    # ===== BUILD NODE FEATURES =====
    node_features = []

    # Feature 1: Normalized velocity history
    # Formula: normalized = (value - mean) / std
    velocity_stats = self._normalization_stats["velocity"]
    normalized_velocity_sequence = (
        velocity_sequence - velocity_stats.mean) / velocity_stats.std

    # Flatten: [particles, time, dims] → [particles, time*dims]
    flat_velocity_sequence = snt.MergeDims(start=1, size=2)(
        normalized_velocity_sequence)
    node_features.append(flat_velocity_sequence)

    # Feature 2: Distance to boundaries (helps learn wall bouncing)
    boundaries = tf.constant(self._boundaries, dtype=tf.float32)
    distance_to_lower_boundary = (
        most_recent_position - tf.expand_dims(boundaries[:, 0], 0))
    distance_to_upper_boundary = (
        tf.expand_dims(boundaries[:, 1], 0) - most_recent_position)
    distance_to_boundaries = tf.concat(
        [distance_to_lower_boundary, distance_to_upper_boundary], axis=1)
    # Clip to [-1, 1] so distant boundaries don't dominate
    normalized_clipped_distance_to_boundaries = tf.clip_by_value(
        distance_to_boundaries / self._connectivity_radius, -1., 1.)
    node_features.append(normalized_clipped_distance_to_boundaries)

    # Feature 3: Particle type embedding (like word embeddings for materials)
    if self._num_particle_types > 1:
      particle_type_embeddings = tf.nn.embedding_lookup(
          self._particle_type_embedding, particle_types)
      node_features.append(particle_type_embeddings)

    # ===== BUILD EDGE FEATURES =====
    edge_features = []

    # Feature 1: Relative displacement (direction between particles)
    # Crucial for physics - tells "how far and which direction?"
    normalized_relative_displacements = (
        tf.gather(most_recent_position, senders) -
        tf.gather(most_recent_position, receivers)) / self._connectivity_radius
    edge_features.append(normalized_relative_displacements)

    # Feature 2: Distance between particles (scalar magnitude)
    # Many physics forces depend on distance (e.g., 1/r² for gravity)
    normalized_relative_distances = tf.norm(
        normalized_relative_displacements, axis=-1, keepdims=True)
    edge_features.append(normalized_relative_distances)

    # Normalize global context if provided
    if global_context is not None:
      context_stats = self._normalization_stats["context"]
      # Add epsilon to prevent division by zero
      global_context = (global_context - context_stats.mean) / tf.math.maximum(
          context_stats.std, STD_EPSILON)

    # Package into GraphsTuple (the graph_nets library data structure)
    return gn.graphs.GraphsTuple(
        nodes=tf.concat(node_features, axis=-1),      # All node features
        edges=tf.concat(edge_features, axis=-1),      # All edge features
        globals=global_context,                        # Global features
        n_node=n_node,                                 # Nodes per graph
        n_edge=n_edge,                                 # Edges per graph
        senders=senders,                               # Edge source nodes
        receivers=receivers,                           # Edge target nodes
        )

  def _decoder_postprocessor(self, normalized_acceleration, position_sequence):
    """Converts normalized acceleration to next position.
    
    BEGINNER NOTE - The Physics:
    
    1. UN-NORMALIZE: acceleration = normalized * std + mean
    
    2. EULER INTEGRATION (basic physics):
       new_velocity = old_velocity + acceleration * dt
       new_position = old_position + new_velocity * dt
       (We use dt=1 for simplicity)
    
    WHY PREDICT ACCELERATION?
    - It's the "natural" physics quantity (F = ma)
    - Smoother and easier to learn than position
    - Network focuses on learning FORCES
    """

    # Un-normalize: value = normalized * std + mean
    acceleration_stats = self._normalization_stats["acceleration"]
    acceleration = (
        normalized_acceleration * acceleration_stats.std
        ) + acceleration_stats.mean

    # Euler integration (physics!)
    # v_new = v_old + a, then x_new = x_old + v_new
    most_recent_position = position_sequence[:, -1]
    most_recent_velocity = most_recent_position - position_sequence[:, -2]

    new_velocity = most_recent_velocity + acceleration  # * dt = 1
    new_position = most_recent_position + new_velocity  # * dt = 1
    return new_position

  def get_predicted_and_target_normalized_accelerations(
      self, next_position, position_sequence_noise, position_sequence,
      n_particles_per_example, global_context=None, particle_types=None):
    """Produces normalized predicted and target accelerations for training.
    
    BEGINNER NOTE - Training with Noise:
    
    We add noise to inputs during training. Why?
    1. REGULARIZATION: Prevents memorizing exact trajectories
    2. ROBUSTNESS: Model handles prediction errors better
    3. DISTRIBUTION SHIFT: Simulates error accumulation during rollout
    
    The clever trick: shift target by noise at last position.
    This makes model correct velocity errors but NOT position errors.
    (Position errors can't be fixed, but velocity errors can!)
    
    Args:
      next_position: Ground truth next position for training target
      position_sequence_noise: Noise to add to input positions
      position_sequence, n_particles_per_example, global_context, particle_types: 
        Same as _build method

    Returns:
      Predicted and target normalized accelerations for loss computation
    """

    # Add noise to input positions (training augmentation)
    noisy_position_sequence = position_sequence + position_sequence_noise

    # Forward pass with noisy inputs
    input_graphs_tuple = self._encoder_preprocessor(
        noisy_position_sequence, n_particles_per_example, global_context,
        particle_types)
    predicted_normalized_acceleration = self._graph_network(input_graphs_tuple)

    # Compute target: shift by noise at last position
    # This creates a "fair" target accounting for noisy input
    next_position_adjusted = next_position + position_sequence_noise[:, -1]
    target_normalized_acceleration = self._inverse_decoder_postprocessor(
        next_position_adjusted, noisy_position_sequence)
    # Result: model corrects velocity noise but not position noise

    return predicted_normalized_acceleration, target_normalized_acceleration

  def _inverse_decoder_postprocessor(self, next_position, position_sequence):
    """Inverse of _decoder_postprocessor - computes target acceleration.
    
    BEGINNER NOTE:
    This reverses the decoder: position → velocity → acceleration → normalized
    Used during training to create ground truth targets.
    """

    previous_position = position_sequence[:, -1]
    previous_velocity = previous_position - position_sequence[:, -2]
    next_velocity = next_position - previous_position
    acceleration = next_velocity - previous_velocity  # Rearranged Euler equation

    # Normalize for training
    acceleration_stats = self._normalization_stats["acceleration"]
    normalized_acceleration = (
        acceleration - acceleration_stats.mean) / acceleration_stats.std
    return normalized_acceleration


def time_diff(input_sequence):
  """Computes velocity via finite difference: v[t] = x[t+1] - x[t].
  
  BEGINNER NOTE:
  This is numerical differentiation - approximating the derivative.
  With dt=1: velocity ≈ (position_next - position_current) / 1
  Returns one fewer timestep (need pairs to compute differences).
  """
  return input_sequence[:, 1:] - input_sequence[:, :-1]

