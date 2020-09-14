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
"""Tools to compute the connectivity of the graph."""

import numpy as np
from sklearn import neighbors
import tensorflow.compat.v1 as tf


def _compute_connectivity(positions, radius):
  """Get the indices of connected edges with radius connectivity.

  Args:
    positions: Positions of nodes in the graph. Shape:
      [num_nodes_in_graph, num_dims].
    radius: Radius of connectivity.

  Returns:
    senders indices [num_edges_in_graph]
    receiver indices [num_edges_in_graph]

  """
  tree = neighbors.KDTree(positions)
  receivers_list = tree.query_radius(positions, r=radius)
  num_nodes = len(positions)
  senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
  receivers = np.concatenate(receivers_list, axis=0)
  return senders, receivers


def _compute_connectivity_for_batch(positions, n_node, radius):
  """`compute_connectivity` for a batch of graphs.

  Args:
    positions: Positions of nodes in the batch of graphs. Shape:
      [num_nodes_in_batch, num_dims].
    n_node: Number of nodes for each graph in the batch. Shape:
      [num_graphs in batch].
    radius: Radius of connectivity.

  Returns:
    senders indices [num_edges_in_batch]
    receiver indices [num_edges_in_batch]
    number of edges per graph [num_graphs_in_batch]

  """

  # TODO(alvarosg): Consider if we want to support batches here or not.
  # Separate the positions corresponding to particles in different graphs.
  positions_per_graph_list = np.split(positions, np.cumsum(n_node[:-1]), axis=0)
  receivers_list = []
  senders_list = []
  n_edge_list = []
  num_nodes_in_previous_graphs = 0

  # Compute connectivity for each graph in the batch.
  for positions_graph_i in positions_per_graph_list:
    senders_graph_i, receivers_graph_i = _compute_connectivity(
        positions_graph_i, radius)

    num_edges_graph_i = len(senders_graph_i)
    n_edge_list.append(num_edges_graph_i)

    # Because the inputs will be concatenated, we need to add offsets to the
    # sender and receiver indices according to the number of nodes in previous
    # graphs in the same batch.
    receivers_list.append(receivers_graph_i + num_nodes_in_previous_graphs)
    senders_list.append(senders_graph_i + num_nodes_in_previous_graphs)

    num_nodes_graph_i = len(positions_graph_i)
    num_nodes_in_previous_graphs += num_nodes_graph_i

  # Concatenate all of the results.
  senders = np.concatenate(senders_list, axis=0).astype(np.int32)
  receivers = np.concatenate(receivers_list, axis=0).astype(np.int32)
  n_edge = np.stack(n_edge_list).astype(np.int32)

  return senders, receivers, n_edge


def compute_connectivity_for_batch_pyfunc(positions, n_node, radius):
  """`_compute_connectivity_for_batch` wrapped in a pyfunc."""
  senders, receivers, n_edge = tf.py_function(
      _compute_connectivity_for_batch,
      [positions, n_node, radius], [tf.int32, tf.int32, tf.int32])
  senders.set_shape([None])
  receivers.set_shape([None])
  n_edge.set_shape(n_node.get_shape())
  return senders, receivers, n_edge

