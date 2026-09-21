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

"""Dynamic batching utilities."""

from typing import Generator, Iterable, Iterator, Sequence, Tuple

import jax.tree_util as tree
import jraph
import numpy as np

_NUMBER_FIELDS = ("n_node", "n_edge", "n_graph")


def dynamically_batch(graphs_tuple_iterator: Iterator[jraph.GraphsTuple],
                      n_node: int, n_edge: int,
                      n_graph: int) -> Generator[jraph.GraphsTuple, None, None]:
  """Dynamically batches trees with `jraph.GraphsTuples` to `graph_batch_size`.

  Elements of the `graphs_tuple_iterator` will be incrementally added to a batch
  until the limits defined by `n_node`, `n_edge` and `n_graph` are reached. This
  means each element yielded by this generator

  For situations where you have variable sized data, it"s useful to be able to
  have variable sized batches. This is especially the case if you have a loss
  defined on the variable shaped element (for example, nodes in a graph).

  Args:
    graphs_tuple_iterator: An iterator of `jraph.GraphsTuples`.
    n_node: The maximum number of nodes in a batch.
    n_edge: The maximum number of edges in a batch.
    n_graph: The maximum number of graphs in a batch.

  Yields:
    A `jraph.GraphsTuple` batch of graphs.

  Raises:
    ValueError: if the number of graphs is < 2.
    RuntimeError: if the `graphs_tuple_iterator` contains elements which are not
      `jraph.GraphsTuple`s.
    RuntimeError: if a graph is found which is larger than the batch size.
  """
  if n_graph < 2:
    raise ValueError("The number of graphs in a batch size must be greater or "
                     f"equal to `2` for padding with graphs, got {n_graph}.")
  valid_batch_size = (n_node - 1, n_edge, n_graph - 1)
  accumulated_graphs = []
  num_accumulated_nodes = 0
  num_accumulated_edges = 0
  num_accumulated_graphs = 0
  for element in graphs_tuple_iterator:
    element_nodes, element_edges, element_graphs = _get_graph_size(element)
    if _is_over_batch_size(element, valid_batch_size):
      graph_size = element_nodes, element_edges, element_graphs
      graph_size = {k: v for k, v in zip(_NUMBER_FIELDS, graph_size)}
      batch_size = {k: v for k, v in zip(_NUMBER_FIELDS, valid_batch_size)}
      raise RuntimeError("Found graph bigger than batch size. Valid Batch "
                         f"Size: {batch_size}, Graph Size: {graph_size}")

    if not accumulated_graphs:
      # If this is the first element of the batch, set it and continue.
      accumulated_graphs = [element]
      num_accumulated_nodes = element_nodes
      num_accumulated_edges = element_edges
      num_accumulated_graphs = element_graphs
      continue
    else:
      # Otherwise check if there is space for the graph in the batch:
      if ((num_accumulated_graphs + element_graphs > n_graph - 1) or
          (num_accumulated_nodes + element_nodes > n_node - 1) or
          (num_accumulated_edges + element_edges > n_edge)):
        # If there is, add it to the batch
        batched_graph = _batch_np(accumulated_graphs)
        yield jraph.pad_with_graphs(batched_graph, n_node, n_edge, n_graph)
        accumulated_graphs = [element]
        num_accumulated_nodes = element_nodes
        num_accumulated_edges = element_edges
        num_accumulated_graphs = element_graphs
      else:
        # Otherwise, return the old batch and start a new batch.
        accumulated_graphs.append(element)
        num_accumulated_nodes += element_nodes
        num_accumulated_edges += element_edges
        num_accumulated_graphs += element_graphs

  # We may still have data in batched graph.
  if accumulated_graphs:
    batched_graph = _batch_np(accumulated_graphs)
    yield jraph.pad_with_graphs(batched_graph, n_node, n_edge, n_graph)


def _batch_np(graphs: Sequence[jraph.GraphsTuple]) -> jraph.GraphsTuple:
  # Calculates offsets for sender and receiver arrays, caused by concatenating
  # the nodes arrays.
  offsets = np.cumsum(np.array([0] + [np.sum(g.n_node) for g in graphs[:-1]]))

  def _map_concat(nests):
    concat = lambda *args: np.concatenate(args)
    return tree.tree_map(concat, *nests)

  return jraph.GraphsTuple(
      n_node=np.concatenate([g.n_node for g in graphs]),
      n_edge=np.concatenate([g.n_edge for g in graphs]),
      nodes=_map_concat([g.nodes for g in graphs]),
      edges=_map_concat([g.edges for g in graphs]),
      globals=_map_concat([g.globals for g in graphs]),
      senders=np.concatenate([g.senders + o for g, o in zip(graphs, offsets)]),
      receivers=np.concatenate(
          [g.receivers + o for g, o in zip(graphs, offsets)]))


def _get_graph_size(graph: jraph.GraphsTuple) -> Tuple[int, int, int]:
  n_node = np.sum(graph.n_node)
  n_edge = len(graph.senders)
  n_graph = len(graph.n_node)
  return n_node, n_edge, n_graph


def _is_over_batch_size(
    graph: jraph.GraphsTuple,
    graph_batch_size: Iterable[int],
) -> bool:
  graph_size = _get_graph_size(graph)
  return any([x > y for x, y in zip(graph_size, graph_batch_size)])



