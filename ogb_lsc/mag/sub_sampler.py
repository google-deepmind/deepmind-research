# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for subsampling the MAG dataset."""

import collections

import jraph
import numpy as np


def get_or_sample_row(node_id: int,
                      nb_neighbours: int,
                      csr_matrix, remove_duplicates: bool):
  """Either obtain entire row or a subsampled set of neighbours."""
  if node_id + 1 >= csr_matrix.indptr.shape[0]:
    lo = 0
    hi = 0
  else:
    lo = csr_matrix.indptr[node_id]
    hi = csr_matrix.indptr[node_id + 1]
  if lo == hi:  # Skip empty neighbourhoods
    neighbours = None
  elif hi - lo <= nb_neighbours:
    neighbours = csr_matrix.indices[lo:hi]
  elif hi - lo < 5 * nb_neighbours:  # For small surroundings, sample directly
    nb_neighbours = min(nb_neighbours, hi - lo)
    inds = lo + np.random.choice(hi - lo, size=(nb_neighbours,), replace=False)
    neighbours = csr_matrix.indices[inds]
  else:  # Otherwise, do not slice -- sample indices instead
    # To extend GraphSAGE ("uniform w/ replacement"), modify this call
    inds = np.random.randint(lo, hi, size=(nb_neighbours,))
    if remove_duplicates:
      inds = np.unique(inds)
    neighbours = csr_matrix.indices[inds]
  return neighbours


def get_neighbours(node_id: int,
                   node_type: int,
                   neighbour_type: int,
                   nb_neighbours: int,
                   remove_duplicates: bool,
                   author_institution_csr, institution_author_csr,
                   author_paper_csr, paper_author_csr,
                   paper_paper_csr, paper_paper_transpose_csr):
  """Fetch the edge indices from one node to corresponding neighbour type."""
  if node_type == 0 and neighbour_type == 0:
    csr = paper_paper_transpose_csr  # Citing
  elif node_type == 0 and neighbour_type == 1:
    csr = paper_author_csr
  elif node_type == 0 and neighbour_type == 3:
    csr = paper_paper_csr  # Cited
  elif node_type == 1 and neighbour_type == 0:
    csr = author_paper_csr
  elif node_type == 1 and neighbour_type == 2:
    csr = author_institution_csr
  elif node_type == 2 and neighbour_type == 1:
    csr = institution_author_csr
  else:
    raise ValueError('Non-existent edge type requested')
  return get_or_sample_row(node_id, nb_neighbours, csr, remove_duplicates)


def get_senders(neighbour_type: int,
                sender_index,
                paper_features):
  """Get the sender features from given neighbours."""
  if neighbour_type == 0 or neighbour_type == 3:
    sender_features = paper_features[sender_index]
  elif neighbour_type == 1 or neighbour_type == 2:
    sender_features = np.zeros((sender_index.shape[0],
                                paper_features.shape[1]))  # Consider averages
  else:
    raise ValueError('Non-existent node type requested')
  return sender_features


def make_edge_type_feature(node_type: int, neighbour_type: int):
  edge_feats = np.zeros(7)
  edge_feats[node_type] = 1.0
  edge_feats[neighbour_type + 3] = 1.0
  return edge_feats


def subsample_graph(paper_id: int,
                    author_institution_csr,
                    institution_author_csr,
                    author_paper_csr,
                    paper_author_csr,
                    paper_paper_csr,
                    paper_paper_transpose_csr,
                    max_nb_neighbours_per_type,
                    max_nodes=None,
                    max_edges=None,
                    paper_years=None,
                    remove_future_nodes=False,
                    deduplicate_nodes=False) -> jraph.GraphsTuple:
  """Subsample a graph around given paper ID."""
  if paper_years is not None:
    root_paper_year = paper_years[paper_id]
  else:
    root_paper_year = None
  # Add the center node as "node-zero"
  sub_nodes = [paper_id]
  num_nodes_in_subgraph = 1
  num_edges_in_subgraph = 0
  reached_node_budget = False
  reached_edge_budget = False
  node_and_type_to_index_in_subgraph = dict()
  node_and_type_to_index_in_subgraph[(paper_id, 0)] = 0
  # Store all (integer) depths as an additional feature
  depths = [0]
  types = [0]
  sub_edges = []
  sub_senders = []
  sub_receivers = []

  # Store all unprocessed neighbours
  # Each neighbour is stored as a 4-tuple (node_index in original graph,
  # node_index in subsampled graph, type, number of hops away from source).
  # TYPES: 0: paper, 1: author, 2: institution, 3: paper (for bidirectional)
  neighbour_deque = collections.deque([(paper_id, 0, 0, 0)])

  max_depth = len(max_nb_neighbours_per_type)

  while neighbour_deque and not reached_edge_budget:
    left_entry = neighbour_deque.popleft()
    node_index, node_index_in_sampled_graph, node_type, node_depth = left_entry

    # Expand from this node, to a node of related type
    for neighbour_type in range(4):
      if reached_edge_budget:
        break  # Budget may have been reached in previous type; break here.
      nb_neighbours = max_nb_neighbours_per_type[node_depth][node_type][neighbour_type]  # pylint:disable=line-too-long
      # Only extend if we want to sample further in this edge type
      if nb_neighbours > 0:
        sampled_neighbors = get_neighbours(
            node_index,
            node_type,
            neighbour_type,
            nb_neighbours,
            deduplicate_nodes,
            author_institution_csr,
            institution_author_csr,
            author_paper_csr,
            paper_author_csr,
            paper_paper_csr,
            paper_paper_transpose_csr,
        )

        if sampled_neighbors is not None:
          if remove_future_nodes and root_paper_year is not None:
            if neighbour_type in [0, 3]:
              sampled_neighbors = [
                  x for x in sampled_neighbors
                  if paper_years[x] <= root_paper_year
              ]
              if not sampled_neighbors:
                continue

          nb_neighbours = len(sampled_neighbors)
          edge_feature = make_edge_type_feature(node_type, neighbour_type)

          for neighbor_original_idx in sampled_neighbors:
            # Key into dict of existing nodes using both node id and type.
            neighbor_key = (neighbor_original_idx, neighbour_type % 3)
            # Get existing idx in subgraph if it exists.
            neighbor_subgraph_idx = node_and_type_to_index_in_subgraph.get(
                neighbor_key, None)
            if (not reached_node_budget and
                (not deduplicate_nodes or neighbor_subgraph_idx is None)):
              # If it does not exist already, or we are not deduplicating,
              # just create a new node and update the dict.
              neighbor_subgraph_idx = num_nodes_in_subgraph
              node_and_type_to_index_in_subgraph[neighbor_key] = (
                  neighbor_subgraph_idx)
              num_nodes_in_subgraph += 1
              sub_nodes.append(neighbor_original_idx)
              types.append(neighbour_type % 3)
              depths.append(node_depth + 1)
              if max_nodes is not None and num_nodes_in_subgraph >= max_nodes:
                reached_node_budget = True
                continue  # Move to next neighbor which might already exist.
              if node_depth < max_depth - 1:
                # If the neighbours are to be further expanded, enqueue them.
                # Expand only if the nodes did not already exist.
                neighbour_deque.append(
                    (neighbor_original_idx, neighbor_subgraph_idx,
                     neighbour_type % 3, node_depth + 1))
            # The neighbor id within graph is now fixed; just add edges.
            if neighbor_subgraph_idx is not None:
              # Either node existed before or was successfully added.
              sub_senders.append(neighbor_subgraph_idx)
              sub_receivers.append(node_index_in_sampled_graph)
              sub_edges.append(edge_feature)
              num_edges_in_subgraph += 1
            if max_edges is not None and num_edges_in_subgraph >= max_edges:
              reached_edge_budget = True
              break  # Break out of adding edges for this neighbor type

  # Stitch the graph together
  sub_nodes = np.array(sub_nodes, dtype=np.int32)

  if sub_senders:
    sub_senders = np.array(sub_senders, dtype=np.int32)
    sub_receivers = np.array(sub_receivers, dtype=np.int32)
    sub_edges = np.stack(sub_edges, axis=0)
  else:
    # Use empty arrays.
    sub_senders = np.zeros([0], dtype=np.int32)
    sub_receivers = np.zeros([0], dtype=np.int32)
    sub_edges = np.zeros([0, 7])

  # Finally, derive the sizes
  sub_n_node = np.array([sub_nodes.shape[0]])
  sub_n_edge = np.array([sub_senders.shape[0]])
  assert sub_nodes.shape[0] == num_nodes_in_subgraph
  assert sub_edges.shape[0] == num_edges_in_subgraph
  if max_nodes is not None:
    assert num_nodes_in_subgraph <= max_nodes
  if max_edges is not None:
    assert num_edges_in_subgraph <= max_edges

  types = np.array(types)
  depths = np.array(depths)
  sub_nodes = {
      'index': sub_nodes.astype(np.int32),
      'type': types.astype(np.int16),
      'depth': depths.astype(np.int16),
  }

  return jraph.GraphsTuple(nodes=sub_nodes,
                           edges=sub_edges.astype(np.float16),
                           senders=sub_senders.astype(np.int32),
                           receivers=sub_receivers.astype(np.int32),
                           globals=np.array([0], dtype=np.int16),
                           n_node=sub_n_node.astype(dtype=np.int32),
                           n_edge=sub_n_edge.astype(dtype=np.int32))
