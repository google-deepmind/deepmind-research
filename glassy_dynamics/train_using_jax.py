# Lint as: python3
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

"""Training pipeline for the prediction of particle mobilities in glasses."""

import enum
import functools
import logging
import pickle
import random
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax

# Only used for file operations.
# You can use glob.glob and python's open function to replace the tf usage below
# on most platforms.
import tensorflow.compat.v1 as tf


class ParticleType(enum.IntEnum):
  """The simulation contains two particle types, identified as type A and B.

  The dataset encodes the particle type in an integer.
    - 0 corresponds to particle type A.
    - 1 corresponds to particle type B.
  """
  A = 0
  B = 1


def make_graph_from_static_structure(positions, types, box, edge_threshold):
  """Returns graph representing the static structure of the glass.

  Each particle is represented by a node in the graph. The particle type is
  stored as a node feature.
  Two particles at a distance less than the threshold are connected by an edge.
  The relative distance vector is stored as an edge feature.

  Args:
    positions: particle positions with shape [n_particles, 3].
    types: particle types with shape [n_particles].
    box: dimensions of the cubic box that contains the particles with shape [3].
    edge_threshold: particles at distance less than threshold are connected by
      an edge.
  """
  # Calculate pairwise relative distances between particles: shape [n, n, 3].
  cross_positions = positions[None, :, :] - positions[:, None, :]
  # Enforces periodic boundary conditions.
  box_ = box[None, None, :]
  cross_positions += (cross_positions < -box_ / 2.).astype(np.float32) * box_
  cross_positions -= (cross_positions > box_ / 2.).astype(np.float32) * box_
  # Calculates adjacency matrix in a sparse format (indices), based on the given
  # distances and threshold.
  distances = np.linalg.norm(cross_positions, axis=-1)
  indices = np.where(distances < edge_threshold)
  # Defines graph.
  nodes = types[:, None]
  senders = indices[0]
  receivers = indices[1]
  edges = cross_positions[indices]

  return jraph.pad_with_graphs(jraph.GraphsTuple(
      nodes=nodes.astype(np.float32),
      n_node=np.reshape(nodes.shape[0], [1]),
      edges=edges.astype(np.float32),
      n_edge=np.reshape(edges.shape[0], [1]),
      globals=np.zeros((1, 1), dtype=np.float32),
      receivers=receivers.astype(np.int32),
      senders=senders.astype(np.int32)
      ), n_node=4097, n_edge=200000)


def get_targets(initial_positions, trajectory_target_positions):
  """Returns the averaged particle mobilities from the sampled trajectories.

  Args:
    initial_positions: the initial positions of the particles with shape
      [n_particles, 3].
    trajectory_target_positions: the absolute positions of the particles at the
      target time for all sampled trajectories, each with shape
      [n_particles, 3].
  """
  targets = np.mean([np.linalg.norm(t - initial_positions, axis=-1)
                     for t in trajectory_target_positions], axis=0)
  return targets.astype(np.float32)


def load_data(file_pattern, time_index, max_files_to_load=None):
  """Returns a graphs and targets of the training or test dataset.

  Args:
    file_pattern: pattern matching the files with the simulation data.
    time_index: the time index of the targets.
    max_files_to_load: the maximum number of files to load.
  """
  filenames = tf.io.gfile.glob(file_pattern)
  if max_files_to_load:
    filenames = filenames[:max_files_to_load]

  graphs_and_targets = []
  for filename in filenames:
    with tf.io.gfile.GFile(filename, 'rb') as f:
      data = pickle.load(f)
    mask = (data['types'] == ParticleType.A).astype(np.int32)
    # Mask dummy node due to padding
    mask = np.concatenate([mask, np.zeros((1,), dtype=np.int32)], axis=-1)
    targets = get_targets(
        data['positions'], data['trajectory_target_positions'][time_index])
    targets = np.concatenate(
        [targets, np.zeros((1,), dtype=np.float32)], axis=-1)
    graphs_and_targets.append(
        (make_graph_from_static_structure(
            data['positions'].astype(np.float32),
            data['types'].astype(np.int32),
            data['box'].astype(np.float32),
            edge_threshold=2.0),
         targets,
         mask))
  return graphs_and_targets


def apply_random_rotation(graph):
  """Returns randomly rotated graph representation.

  The rotation is an element of O(3) with rotation angles multiple of pi/2.
  This function assumes that the relative particle distances are stored in
  the edge features.

  Args:
    graph: The graphs tuple as defined in `graph_nets.graphs`.
  """
  # Transposes edge features, so that the axes are in the first dimension.
  # Outputs a tensor of shape [3, n_particles].
  xyz = np.transpose(graph.edges)
  # Random pi/2 rotation(s)
  permutation = np.array([0, 1, 2], dtype=np.int32)
  np.random.shuffle(permutation)
  xyz = xyz[permutation]
  # Random reflections.
  symmetry = np.random.randint(0, 2, [3])
  symmetry = 1 - 2 * np.reshape(symmetry, [3, 1]).astype(np.float32)
  xyz = xyz * symmetry
  edges = np.transpose(xyz)
  return graph._replace(edges=edges)


def network_definition(graph):
  """Defines a graph neural network.

  Args:
    graph: Graphstuple the network processes.

  Returns:
    Decoded nodes.
  """
  model_fn = functools.partial(
      hk.nets.MLP,
      w_init=hk.initializers.VarianceScaling(1.0),
      b_init=hk.initializers.VarianceScaling(1.0))
  mlp_sizes = (64, 64)
  num_message_passing_steps = 7

  node_encoder = model_fn(output_sizes=mlp_sizes, activate_final=True)
  edge_encoder = model_fn(output_sizes=mlp_sizes, activate_final=True)
  node_decoder = model_fn(output_sizes=mlp_sizes + (1,), activate_final=False)

  node_encoding = node_encoder(graph.nodes)
  edge_encoding = edge_encoder(graph.edges)
  graph = graph._replace(nodes=node_encoding, edges=edge_encoding)

  update_edge_fn = jraph.concatenated_args(
      model_fn(output_sizes=mlp_sizes, activate_final=True))
  update_node_fn = jraph.concatenated_args(
      model_fn(output_sizes=mlp_sizes, activate_final=True))
  gn = jraph.InteractionNetwork(
      update_edge_fn=update_edge_fn,
      update_node_fn=update_node_fn,
      include_sent_messages_in_node_update=True)

  for _ in range(num_message_passing_steps):
    graph = graph._replace(
        nodes=jnp.concatenate([graph.nodes, node_encoding], axis=-1),
        edges=jnp.concatenate([graph.edges, edge_encoding], axis=-1))
    graph = gn(graph)

  return jnp.squeeze(node_decoder(graph.nodes), axis=-1)


def train_model(train_file_pattern,
                test_file_pattern,
                max_files_to_load=None,
                n_epochs=1000,
                time_index=9,
                learning_rate=1e-4,
                grad_clip=1.0,
                measurement_store_interval=1000,
                checkpoint_path=None):
  """Trains GraphModel using tensorflow.

  Args:
    train_file_pattern: pattern matching the files with the training data.
    test_file_pattern: pattern matching the files with the test data.
    max_files_to_load: the maximum number of train and test files to load.
      If None, all files will be loaded.
    n_epochs: the number of passes through the training dataset (epochs).
    time_index: the time index (0-9) of the target mobilities.
    learning_rate: the learning rate used by the optimizer.
    grad_clip: all gradients are clipped to the given value.
    measurement_store_interval: number of steps between storing objective values
      (loss and correlation).
    checkpoint_path: ignored by this implementation.
  """
  if checkpoint_path:
    logging.warning('The checkpoint_path argument is ignored.')
  random.seed(42)
  np.random.seed(42)
  # Loads train and test dataset.
  dataset_kwargs = dict(
      time_index=time_index,
      max_files_to_load=max_files_to_load)
  logging.info('Load training data')
  training_data = load_data(train_file_pattern, **dataset_kwargs)
  logging.info('Load test data')
  test_data = load_data(test_file_pattern, **dataset_kwargs)
  logging.info('Finished loading data')

  network = hk.without_apply_rng(hk.transform(network_definition))
  params = network.init(jax.random.PRNGKey(42), training_data[0][0])

  opt_init, opt_update = optax.chain(
      optax.clip_by_global_norm(grad_clip),
      optax.scale_by_adam(0.9, 0.999, 1e-8),
      optax.scale(-learning_rate))
  opt_state = opt_init(params)

  network_apply = jax.jit(network.apply)

  @jax.jit
  def loss_fn(params, graph, targets, mask):
    decoded_nodes = network_apply(params, graph) * mask
    return (jnp.sum((decoded_nodes - targets)**2 * mask) /
            jnp.sum(mask))

  @jax.jit
  def update(params, opt_state, graph, targets, mask):
    loss, grads = jax.value_and_grad(loss_fn)(params, graph, targets, mask)
    updates, opt_state = opt_update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state, loss

  train_stats = []
  i = 0
  logging.info('Start training')
  for epoch in range(n_epochs):
    logging.info('Start epoch %r', epoch)
    random.shuffle(training_data)
    for graph, targets, mask in training_data:
      graph = apply_random_rotation(graph)
      params, opt_state, loss = update(params, opt_state, graph, targets, mask)
      train_stats.append(loss)

      if (i+1) % measurement_store_interval == 0:
        logging.info('Start evaluation run')
        test_stats = []
        for test_graph, test_targets, test_mask in test_data:
          predictions = network_apply(params, test_graph)
          test_stats.append(np.corrcoef(
              predictions[test_mask == 1], test_targets[test_mask == 1])[0, 1])
        logging.info('Train loss %r', np.mean(train_stats))
        logging.info('Test correlation %r', np.mean(test_stats))
        train_stats = []
      i += 1
