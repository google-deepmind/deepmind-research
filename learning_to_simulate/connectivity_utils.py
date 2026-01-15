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
"""Tools to compute the connectivity of the graph.

============================================================================
BEGINNER'S GUIDE TO THIS FILE
============================================================================

This file handles GRAPH CONSTRUCTION - finding which particles should be
connected by edges in our graph neural network.

KEY CONCEPT: RADIUS-BASED CONNECTIVITY
- Two particles are connected if distance < connectivity_radius
- This is how we model "local interactions" in physics
- Example: Water molecules only interact with nearby molecules

WHY KD-TREES?
- Naive approach: Check all pairs → O(n²) complexity
- KD-tree: Spatial data structure for efficient neighbor queries → O(n log n)
- Essential for scaling to thousands of particles!

WHAT THIS FILE DOES:
1. Build a KD-tree from particle positions
2. Query neighbors within radius for each particle
3. Return sender/receiver pairs (edge list format)

============================================================================
"""

import functools

import numpy as np
from sklearn import neighbors
import tensorflow.compat.v1 as tf


def _compute_connectivity(positions, radius, add_self_edges):
  """Get the indices of connected edges with radius connectivity.
  
  BEGINNER NOTE:
  This function answers: "Which particles are close enough to interact?"
  
  Algorithm:
  1. Build KD-tree from positions (efficient spatial index)
  2. For each particle, query all neighbors within radius
  3. Return as edge list: (sender[i], receiver[i]) are connected

  Args:
    positions: Positions of nodes in the graph. Shape:
      [num_nodes_in_graph, num_dims].
    radius: Radius of connectivity.
      # BEGINNER: Particles closer than this distance get connected
    add_self_edges: Whether to include self edges or not.
      # BEGINNER: Should a particle be connected to itself?

  Returns:
    senders indices [num_edges_in_graph]
    receiver indices [num_edges_in_graph]

  """
  # Build KD-tree for efficient spatial queries
  # BEGINNER: KD-tree is like a "spatial index" - makes neighbor lookups fast
  tree = neighbors.KDTree(positions)
  
  # Query all neighbors within radius for each particle
  receivers_list = tree.query_radius(positions, r=radius)
  
  # Convert to edge list format
  num_nodes = len(positions)
  senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
  receivers = np.concatenate(receivers_list, axis=0)

  # Optionally remove self-edges (particle connected to itself)
  if not add_self_edges:
    mask = senders != receivers
    senders = senders[mask]
    receivers = receivers[mask]

  return senders, receivers


def _compute_connectivity_for_batch(
    positions, n_node, radius, add_self_edges):
  """`compute_connectivity` for a batch of graphs.
  
  BEGINNER NOTE:
  When training, we process multiple simulations at once (batching).
  This function handles the bookkeeping:
  - Split positions by which simulation they belong to
  - Compute connectivity for each simulation separately
  - Add index offsets so edges point to correct nodes in combined array

  Args:
    positions: Positions of nodes in the batch of graphs. Shape:
      [num_nodes_in_batch, num_dims].
      # BEGINNER: All particles from all simulations, concatenated
    n_node: Number of nodes for each graph in the batch. Shape:
      [num_graphs in batch].
      # BEGINNER: e.g., [100, 150, 120] means 3 simulations with those particle counts
    radius: Radius of connectivity.
    add_self_edges: Whether to include self edges or not.

  Returns:
    senders indices [num_edges_in_batch]
    receiver indices [num_edges_in_batch]
    number of edges per graph [num_graphs_in_batch]

  """

  # Split positions by simulation (graph)
  # BEGINNER: cumsum gives split points: [100, 250, 370] for n_node=[100,150,120]
  positions_per_graph_list = np.split(positions, np.cumsum(n_node[:-1]), axis=0)
  receivers_list = []
  senders_list = []
  n_edge_list = []
  num_nodes_in_previous_graphs = 0

  # Compute connectivity for each simulation separately
  for positions_graph_i in positions_per_graph_list:
    senders_graph_i, receivers_graph_i = _compute_connectivity(
        positions_graph_i, radius, add_self_edges)

    num_edges_graph_i = len(senders_graph_i)
    n_edge_list.append(num_edges_graph_i)

    # Add offset to indices so they point to correct nodes in combined array
    # BEGINNER: If sim 1 has 100 nodes, sim 2's node 0 is actually index 100
    receivers_list.append(receivers_graph_i + num_nodes_in_previous_graphs)
    senders_list.append(senders_graph_i + num_nodes_in_previous_graphs)

    num_nodes_graph_i = len(positions_graph_i)
    num_nodes_in_previous_graphs += num_nodes_graph_i

  # Concatenate results from all simulations
  senders = np.concatenate(senders_list, axis=0).astype(np.int32)
  receivers = np.concatenate(receivers_list, axis=0).astype(np.int32)
  n_edge = np.stack(n_edge_list).astype(np.int32)

  return senders, receivers, n_edge


def compute_connectivity_for_batch_pyfunc(
    positions, n_node, radius, add_self_edges=True):
  """`_compute_connectivity_for_batch` wrapped in a TensorFlow pyfunc.
  
  BEGINNER NOTE:
  TensorFlow graphs can't directly call NumPy/sklearn code.
  tf.py_function wraps Python code so it can be called from TF graphs.
  
  The trade-off:
  - Flexibility: Use any Python library (sklearn KD-tree here)
  - Performance: Slightly slower than pure TensorFlow ops
  """
  partial_fn = functools.partial(
      _compute_connectivity_for_batch, add_self_edges=add_self_edges)
  senders, receivers, n_edge = tf.py_function(
      partial_fn,
      [positions, n_node, radius],
      [tf.int32, tf.int32, tf.int32])
  senders.set_shape([None])
  receivers.set_shape([None])
  n_edge.set_shape(n_node.get_shape())
  return senders, receivers, n_edge

