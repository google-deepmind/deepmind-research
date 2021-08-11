# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
#
# WikiGraphs is licensed under the terms of the Creative Commons
# Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license.
#
# WikiText-103 data (unchanged) is licensed by Salesforce.com, Inc. under the
# terms of the Creative Commons Attribution-ShareAlike 4.0 International
# (CC BY-SA 4.0) license. You can find details about CC BY-SA 4.0 at:
#
#     https://creativecommons.org/licenses/by-sa/4.0/legalcode
#
# Freebase data is licensed by Google LLC under the terms of the Creative
# Commons CC BY 4.0 license. You may obtain a copy of the License at:
#
#     https://creativecommons.org/licenses/by/4.0/legalcode
#
# ==============================================================================
"""Graph net utils."""

from typing import Union, List, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np


ArrayType = Union[np.ndarray, jnp.ndarray]


def pad_size(in_size: int):
  out_size = 1
  while out_size < in_size:
    out_size *= 2
  return out_size


def pad_graphs(
    graphs: jraph.GraphsTuple,
    pad_n_nodes: Optional[int] = None,
    pad_n_edges: Optional[int] = None) -> jraph.GraphsTuple:
  """Pad graphs to have a canonical number of nodes and edges.

  Here we pad the number of nodes and number of edges to powers of 2 by adding a
  placeholder graph to the end of the batch.  So that the batch gets at most 2x
  as large as before, and number of graphs increase by 1.

  Note this method always adds at least one new node to the placeholder graph to
  make sure any edges if added are valid.

  Args:
    graphs: a batch of graphs.
    pad_n_nodes: (optional) number of nodes to pad to.
    pad_n_edges: (optional) number of edges to pad to.

  Returns:
    padded: the input batch padded to canonical sizes.
  """
  n_nodes, node_dim = graphs.nodes.shape
  n_edges, edge_dim = graphs.edges.shape
  # Add at least one extra node to the placeholder graph.
  if pad_n_nodes is None:
    pad_n_nodes = pad_size(n_nodes + 1)
  if pad_n_edges is None:
    pad_n_edges = pad_size(n_edges)

  nodes = np.concatenate([
      graphs.nodes,
      np.zeros((pad_n_nodes - n_nodes, node_dim), dtype=graphs.nodes.dtype)
  ], axis=0)
  edges = np.concatenate([
      graphs.edges,
      np.zeros((pad_n_edges - n_edges, edge_dim), dtype=graphs.edges.dtype)
  ], axis=0)
  # Add padding edges
  senders = np.concatenate([
      graphs.senders,
      np.full(pad_n_edges - n_edges, n_nodes, dtype=graphs.senders.dtype)
  ], axis=0)
  receivers = np.concatenate([
      graphs.receivers,
      np.full(pad_n_edges - n_edges, n_nodes, dtype=graphs.receivers.dtype)
  ], axis=0)
  n_node = np.concatenate([
      graphs.n_node, np.full(1, pad_n_nodes - n_nodes)], axis=0)
  n_edge = np.concatenate([
      graphs.n_edge, np.full(1, pad_n_edges - n_edges)], axis=0)
  return jraph.GraphsTuple(
      nodes=nodes, edges=edges, senders=senders, receivers=receivers,
      n_node=n_node, n_edge=n_edge, globals=None)


def batch_graphs_by_device(
    graphs: List[jraph.GraphsTuple],
    num_devices: int
    ) -> List[jraph.GraphsTuple]:
  """Batch a list of graphs into num_devices batched graphs.

  The input graphs are grouped into num_devices groups. Within each group the
  graphs are merged. This is needed for parallelizing the graphs using pmap.

  Args:
    graphs: a list of graphs to be merged.
    num_devices: the number of local devices.

  Returns:
    graph: a size num_devices list of merged graphs.
  """
  bs = len(graphs)
  assert bs % num_devices == 0, (
      'Batch size {} is not divisible by {} devices.'.format(bs, num_devices))
  bs_per_device = bs // num_devices
  graphs_on_devices = []
  for i in range(num_devices):
    graphs_on_device_i = graphs[i*bs_per_device:(i+1)*bs_per_device]
    graphs_on_device_i = jraph.batch(graphs_on_device_i)
    graphs_on_devices.append(graphs_on_device_i)
  return graphs_on_devices


def pad_graphs_by_device(graphs: List[jraph.GraphsTuple]) -> jraph.GraphsTuple:
  """Pad and concatenate the list of graphs.

  Each graph in the list is padded according to the maximum n_nodes and n_edges
  in the list, such that all graphs have the same length. Then they are
  concatenated. This is need for pmap.

  Args:
    graphs: a list of graphs.

  Returns:
    graph: a single padded and merged graph.
  """
  # Add at least one extra node to the placeholder graph.
  pad_n_nodes = pad_size(max([g.nodes.shape[0] for g in graphs]) + 1)
  pad_n_edges = pad_size(max([g.edges.shape[0] for g in graphs]))
  padded_graphs = [pad_graphs(g, pad_n_nodes, pad_n_edges) for g in graphs]
  nodes = []
  edges = []
  senders = []
  receivers = []
  n_node = []
  n_edge = []
  for g in padded_graphs:
    assert g.nodes.shape[0] == pad_n_nodes
    assert g.edges.shape[0] == pad_n_edges
    assert g.senders.size == pad_n_edges
    assert g.receivers.size == pad_n_edges
    assert g.n_node.size == padded_graphs[0].n_node.size
    assert g.n_edge.size == padded_graphs[0].n_edge.size
    nodes.append(g.nodes)
    edges.append(g.edges)
    senders.append(g.senders)
    receivers.append(g.receivers)
    n_node.append(g.n_node)
    n_edge.append(g.n_edge)

  return jraph.GraphsTuple(
      nodes=np.concatenate(nodes, axis=0),
      edges=np.concatenate(edges, axis=0),
      senders=np.concatenate(senders, axis=0),
      receivers=np.concatenate(receivers, axis=0),
      n_node=np.concatenate(n_node, axis=0),
      n_edge=np.concatenate(n_edge, axis=0),
      globals=None)


class MLPMessagePassingLayer(hk.Module):
  """Message passing layer implemented as MLPs."""

  def __init__(self,
               node_hidden_sizes: List[int],
               msg_hidden_sizes: List[int],
               residual: bool = True,
               layer_norm: bool = False,
               name: Optional[str] = None):
    """Constructor.

    Args:
      node_hidden_sizes: hidden sizes for the node update model.
      msg_hidden_sizes: hidden sizes for the edge message model.
      residual: set to True to use residual connections, this will also mean the
        input dimension is appended to `node_hidden_sizes` as the output size.
      layer_norm: whether to apply layer norm on the node representations.
      name: name for this module.
    """
    super().__init__(name=name)
    self._node_hidden_sizes = node_hidden_sizes
    self._msg_hidden_sizes = msg_hidden_sizes
    self._residual = residual
    self._layer_norm = layer_norm

  def _compute_messages(self, graph: jraph.GraphsTuple) -> ArrayType:
    """Compute the messages on each edge."""
    x = jnp.concatenate([graph.nodes[graph.senders],
                         graph.nodes[graph.receivers],
                         graph.edges], axis=-1)
    return hk.nets.MLP(self._msg_hidden_sizes, activate_final=True)(x)

  def _update_nodes(self, graph: jraph.GraphsTuple,
                    messages: ArrayType) -> ArrayType:
    """Compute updated node representations."""
    x = jax.ops.segment_sum(messages, graph.receivers,
                            num_segments=graph.nodes.shape[0])
    x = jnp.concatenate([graph.nodes, x], axis=-1)

    layer_sizes = self._node_hidden_sizes[:]
    if self._residual:
      layer_sizes += [graph.nodes.shape[-1]]

    x = hk.nets.MLP(layer_sizes, activate_final=False)(x)
    if self._layer_norm:
      x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)

    if self._residual:
      return graph.nodes + x
    else:
      return x

  def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Apply this layer on the input graph."""
    messages = self._compute_messages(graph)
    updated_nodes = self._update_nodes(graph, messages)
    return graph._replace(nodes=updated_nodes)


class SimpleGraphNet(hk.Module):
  """A simple graph net module, a stack of message passing layers."""

  def __init__(self,
               num_layers: int,
               msg_hidden_size_factor: int = 2,
               layer_norm: bool = False,
               name: Optional[str] = None):
    """Constructor.

    Args:
      num_layers: number of message passing layers in the network.
      msg_hidden_size_factor: size of message module hidden sizes as a factor of
        the input node feature dimensionality.
      layer_norm: whether to apply layer norm on node updates.
      name: name of this module.
    """
    super().__init__(name=name)
    self._num_layers = num_layers
    self._msg_hidden_size_factor = msg_hidden_size_factor
    self._layer_norm = layer_norm

  def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Run the simple graph net on the input data.

    Args:
      graph: input graph.

    Returns:
      graph: output graph.
    """
    input_node_dim = graph.nodes.shape[-1]
    msg_hidden_size = input_node_dim * self._msg_hidden_size_factor

    for _ in range(self._num_layers):
      graph = MLPMessagePassingLayer(
          node_hidden_sizes=[],
          msg_hidden_sizes=[msg_hidden_size],
          layer_norm=self._layer_norm)(graph)
    return graph


def add_reverse_edges(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
  """Add edges in the reverse direction, copy edge features."""
  senders = np.concatenate([graph.senders, graph.receivers], axis=0)
  receivers = np.concatenate([graph.receivers, graph.senders], axis=0)
  edges = np.concatenate([graph.edges, graph.edges], axis=0)
  return graph._replace(senders=senders, receivers=receivers, edges=edges)
