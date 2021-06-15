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

"""MAG240M-LSC models."""

from typing import Callable, NamedTuple, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import jraph


_REDUCER_NAMES = {
    'sum':
        jax.ops.segment_sum,
    'mean':
        jraph.segment_mean,
    'softmax':
        jraph.segment_softmax,
}


class ModelOutput(NamedTuple):
  node_embeddings: jnp.ndarray
  node_embedding_projections: jnp.ndarray
  node_projection_predictions: jnp.ndarray
  node_logits: jnp.ndarray


def build_update_fn(
    name: str,
    output_sizes: Sequence[int],
    activation: Callable[[jnp.ndarray], jnp.ndarray],
    normalization_type: str,
    is_training: bool,
):
  """Builds update function."""

  def single_mlp(inner_name: str):
    """Creates a single MLP performing the update."""
    mlp = hk.nets.MLP(
        output_sizes=output_sizes,
        name=inner_name,
        activation=activation)
    mlp = jraph.concatenated_args(mlp)
    if normalization_type == 'layer_norm':
      norm = hk.LayerNorm(
          axis=-1,
          create_scale=True,
          create_offset=True,
          name=name + '_layer_norm')
    elif normalization_type == 'batch_norm':
      batch_norm = hk.BatchNorm(
          create_scale=True,
          create_offset=True,
          decay_rate=0.9,
          name=f'{inner_name}_batch_norm',
          cross_replica_axis=None if hk.running_init() else 'i',
      )
      norm = lambda x: batch_norm(x, is_training)
    elif normalization_type == 'none':
      return mlp
    else:
      raise ValueError(f'Unknown normalization type {normalization_type}')
    return jraph.concatenated_args(hk.Sequential([mlp, norm]))

  return single_mlp(f'{name}_homogeneous')


def build_gn(
    output_sizes: Sequence[int],
    activation: Callable[[jnp.ndarray], jnp.ndarray],
    suffix: str,
    use_sent_edges: bool,
    is_training: bool,
    dropedge_rate: float,
    normalization_type: str,
    aggregation_function: str,
):
  """Builds an InteractionNetwork with MLP update functions."""
  node_update_fn = build_update_fn(
      f'node_processor_{suffix}',
      output_sizes,
      activation=activation,
      normalization_type=normalization_type,
      is_training=is_training,
  )
  edge_update_fn = build_update_fn(
      f'edge_processor_{suffix}',
      output_sizes,
      activation=activation,
      normalization_type=normalization_type,
      is_training=is_training,
  )

  def maybe_dropedge(x):
    """Dropout on edge messages."""
    if not is_training:
      return x
    return x * hk.dropout(
        hk.next_rng_key(),
        dropedge_rate,
        jnp.ones([x.shape[0], 1]),
    )

  dropped_edge_update_fn = lambda *args: maybe_dropedge(edge_update_fn(*args))
  return jraph.InteractionNetwork(
      update_edge_fn=dropped_edge_update_fn,
      update_node_fn=node_update_fn,
      aggregate_edges_for_nodes_fn=_REDUCER_NAMES[aggregation_function],
      include_sent_messages_in_node_update=use_sent_edges,
  )


def _get_activation_fn(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
  if name == 'identity':
    return lambda x: x
  if hasattr(jax.nn, name):
    return getattr(jax.nn, name)
  raise ValueError('Unknown activation function %s specified. '
                   'See https://jax.readthedocs.io/en/latest/jax.nn.html'
                   'for the list of supported function names.')


class NodePropertyEncodeProcessDecode(hk.Module):
  """Node Property Prediction Encode Process Decode Model."""

  def __init__(
      self,
      mlp_hidden_sizes: Sequence[int],
      latent_size: int,
      num_classes: int,
      num_message_passing_steps: int = 2,
      activation: str = 'relu',
      dropout_rate: float = 0.0,
      dropedge_rate: float = 0.0,
      use_sent_edges: bool = False,
      disable_edge_updates: bool = False,
      normalization_type: str = 'layer_norm',
      aggregation_function: str = 'sum',
      name='NodePropertyEncodeProcessDecode',
  ):
    super().__init__(name=name)
    self._num_classes = num_classes
    self._latent_size = latent_size
    self._output_sizes = list(mlp_hidden_sizes) + [latent_size]
    self._num_message_passing_steps = num_message_passing_steps
    self._activation = _get_activation_fn(activation)
    self._dropout_rate = dropout_rate
    self._dropedge_rate = dropedge_rate
    self._use_sent_edges = use_sent_edges
    self._disable_edge_updates = disable_edge_updates
    self._normalization_type = normalization_type
    self._aggregation_function = aggregation_function

  def _dropout_graph(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    node_key, edge_key = hk.next_rng_keys(2)
    nodes = hk.dropout(node_key, self._dropout_rate, graph.nodes)
    edges = graph.edges
    if not self._disable_edge_updates:
      edges = hk.dropout(edge_key, self._dropout_rate, edges)
    return graph._replace(nodes=nodes, edges=edges)

  def _encode(
      self,
      graph: jraph.GraphsTuple,
      is_training: bool,
  ) -> jraph.GraphsTuple:
    node_embed_fn = build_update_fn(
        'node_encoder',
        self._output_sizes,
        activation=self._activation,
        normalization_type=self._normalization_type,
        is_training=is_training,
    )
    edge_embed_fn = build_update_fn(
        'edge_encoder',
        self._output_sizes,
        activation=self._activation,
        normalization_type=self._normalization_type,
        is_training=is_training,
    )
    gn = jraph.GraphMapFeatures(edge_embed_fn, node_embed_fn)
    graph = gn(graph)
    if is_training:
      graph = self._dropout_graph(graph)
    return graph

  def _process(
      self,
      graph: jraph.GraphsTuple,
      is_training: bool,
  ) -> jraph.GraphsTuple:
    for idx in range(self._num_message_passing_steps):
      net = build_gn(
          output_sizes=self._output_sizes,
          activation=self._activation,
          suffix=str(idx),
          use_sent_edges=self._use_sent_edges,
          is_training=is_training,
          dropedge_rate=self._dropedge_rate,
          normalization_type=self._normalization_type,
          aggregation_function=self._aggregation_function)
      residual_graph = net(graph)
      graph = graph._replace(nodes=graph.nodes + residual_graph.nodes)
      if not self._disable_edge_updates:
        graph = graph._replace(edges=graph.edges + residual_graph.edges)
      if is_training:
        graph = self._dropout_graph(graph)
    return graph

  def _node_mlp(
      self,
      graph: jraph.GraphsTuple,
      is_training: bool,
      output_size: int,
      name: str,
  ) -> jnp.ndarray:
    decoder_sizes = list(self._output_sizes[:-1]) + [output_size]
    net = build_update_fn(
        name,
        decoder_sizes,
        self._activation,
        normalization_type=self._normalization_type,
        is_training=is_training,
    )
    return net(graph.nodes)

  def __call__(
      self,
      graph: jraph.GraphsTuple,
      is_training: bool,
      stop_gradient_embedding_to_logits: bool = False,
  ) -> ModelOutput:
    # Note that these update configs may need to change if
    # we switch back to GraphNetwork rather than InteractionNetwork.

    graph = self._encode(graph, is_training)
    graph = self._process(graph, is_training)
    node_embeddings = graph.nodes
    node_projections = self._node_mlp(graph, is_training, self._latent_size,
                                      'projector')
    node_predictions = self._node_mlp(
        graph._replace(nodes=node_projections),
        is_training,
        self._latent_size,
        'predictor',
    )
    if stop_gradient_embedding_to_logits:
      graph = jax.tree_map(jax.lax.stop_gradient, graph)
    node_logits = self._node_mlp(graph, is_training, self._num_classes,
                                 'logits_decoder')
    return ModelOutput(
        node_embeddings=node_embeddings,
        node_logits=node_logits,
        node_embedding_projections=node_projections,
        node_projection_predictions=node_predictions,
    )
