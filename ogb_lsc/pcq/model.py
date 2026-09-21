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

"""PCQM4M-LSC models."""

import copy
import functools
from typing import Any, Dict, Mapping, Sequence, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
from ml_collections import config_dict


_REDUCER_NAMES = {
    "sum": jax.ops.segment_sum,
    "mean": jraph.segment_mean,
}

_NUM_EDGE_FEATURES = 13
_NUM_NODE_FEATURES = 173


@chex.dataclass
class RegressionLossConfig:
  """Regression Loss Config."""
  # For normalization and denormalization.
  std: float
  mean: float
  kwargs: Mapping[str, Any]
  out_size: int = 1


def _sigmoid_cross_entropy(
    logits: jax.Array,
    labels: jax.Array,
) -> jax.Array:
  log_p = jax.nn.log_sigmoid(logits)
  log_not_p = jax.nn.log_sigmoid(-logits)
  return -labels * log_p - (1. - labels) * log_not_p


def _softmax_cross_entropy(
    logits: jax.Array,
    targets: jax.Array,
) -> jax.Array:
  logits = jax.nn.log_softmax(logits)
  return -jnp.sum(targets * logits, axis=-1)


def _regression_loss(
    pred: jnp.ndarray,
    targets: jnp.ndarray,
    exponent: int,
) -> jnp.ndarray:
  """Regression loss."""
  error = pred - targets
  if exponent == 2:
    return error ** 2
  elif exponent == 1:
    return jnp.abs(error)
  else:
    raise ValueError(f"Unsupported exponent value {exponent}.")


def _build_mlp(
    name: str,
    output_sizes: Sequence[int],
    use_layer_norm=False,
    activation=jax.nn.relu,
):
  """Builds an MLP, optionally with layernorm."""
  net = hk.nets.MLP(
      output_sizes=output_sizes, name=name + "_mlp", activation=activation)
  if use_layer_norm:
    layer_norm = hk.LayerNorm(
        axis=-1,
        create_scale=True,
        create_offset=True,
        name=name + "_layer_norm")
    net = hk.Sequential([net, layer_norm])
  return jraph.concatenated_args(net)


def _compute_relative_displacement_and_distance(
    graph: jraph.GraphsTuple,
    normalization_factor: float,
    use_target: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Computes relative displacements and distances."""

  if use_target:
    node_positions = graph.nodes["positions_targets"]
  else:
    node_positions = graph.nodes["positions"]
  relative_displacement = node_positions[
      graph.receivers] - node_positions[graph.senders]

  # Note due to the random rotations in space, mean across all nodes across
  # all batches is guaranteed to be zero, and the standard deviation is
  # guaranteed to be the same for all 3 coordinates, so we only need to scale
  # by a single value.
  relative_displacement /= normalization_factor
  relative_distance = jnp.linalg.norm(
      relative_displacement, axis=-1, keepdims=True)
  return relative_displacement, relative_distance


def _broadcast_global_to_nodes(
    global_feature: jnp.ndarray,
    graph: jraph.GraphsTuple,
) -> jnp.ndarray:
  graph_idx = jnp.arange(graph.n_node.shape[0])
  sum_n_node = jax.tree_leaves(graph.nodes)[0].shape[0]
  node_graph_idx = jnp.repeat(
      graph_idx, graph.n_node, axis=0, total_repeat_length=sum_n_node)
  return global_feature[node_graph_idx]


def _broadcast_global_to_edges(
    global_feature: jnp.ndarray,
    graph: jraph.GraphsTuple,
) -> jnp.ndarray:
  graph_idx = jnp.arange(graph.n_edge.shape[0])
  sum_n_edge = graph.senders.shape[0]
  edge_graph_idx = jnp.repeat(
      graph_idx, graph.n_edge, axis=0, total_repeat_length=sum_n_edge)
  return global_feature[edge_graph_idx]


class GraphPropertyEncodeProcessDecode(hk.Module):
  """Encode-process-decode model for graph property prediction."""

  def __init__(
      self,
      loss_config: config_dict.ConfigDict,
      mlp_hidden_size: int,
      mlp_layers: int,
      latent_size: int,
      use_layer_norm: bool,
      num_message_passing_steps: int,
      shared_message_passing_weights: bool,
      mask_padding_graph_at_every_step: bool,
      loss_config_name: str,
      loss_kwargs: config_dict.ConfigDict,
      processor_mode: str,
      global_reducer: str,
      node_reducer: str,
      dropedge_rate: float,
      dropnode_rate: float,
      aux_multiplier: float,
      ignore_globals: bool,
      ignore_globals_from_final_layer_for_predictions: bool,
      add_relative_distance: bool = False,
      add_relative_displacement: bool = False,
      add_absolute_positions: bool = False,
      position_normalization: float = 1.,
      relative_displacement_normalization: float = 1.,
      add_misc_node_features: bool = None,
      name="GraphPropertyEncodeProcessDecode",
  ):
    super(GraphPropertyEncodeProcessDecode, self).__init__()
    self._loss_config = loss_config

    self._config = config_dict.ConfigDict(dict(
        loss_config=loss_config,
        mlp_hidden_size=mlp_hidden_size,
        mlp_layers=mlp_layers,
        latent_size=latent_size,
        use_layer_norm=use_layer_norm,
        num_message_passing_steps=num_message_passing_steps,
        shared_message_passing_weights=shared_message_passing_weights,
        mask_padding_graph_at_every_step=mask_padding_graph_at_every_step,
        loss_config_name=loss_config_name,
        loss_kwargs=loss_kwargs,
        processor_mode=processor_mode,
        global_reducer=global_reducer,
        node_reducer=node_reducer,
        dropedge_rate=dropedge_rate,
        dropnode_rate=dropnode_rate,
        aux_multiplier=aux_multiplier,
        ignore_globals=ignore_globals,
        ignore_globals_from_final_layer_for_predictions=ignore_globals_from_final_layer_for_predictions,
        add_relative_distance=add_relative_distance,
        add_relative_displacement=add_relative_displacement,
        add_absolute_positions=add_absolute_positions,
        position_normalization=position_normalization,
        relative_displacement_normalization=relative_displacement_normalization,
        add_misc_node_features=add_misc_node_features,
    ))

  def __call__(self, graph: jraph.GraphsTuple) -> chex.ArrayTree:
    """Model inference step."""
    out = self._forward(graph, is_training=False)
    if isinstance(self._loss_config, RegressionLossConfig):
      out["globals"] = out[
          "globals"]*self._loss_config.std + self._loss_config.mean
    return out

  @hk.experimental.name_like("__call__")
  def get_loss(
      self,
      graph: jraph.GraphsTuple,
      is_training: bool = True,
  ) -> Tuple[jnp.ndarray, chex.ArrayTree]:
    """Model loss."""
    scalars = get_utilization_scalars(graph)
    targets = copy.deepcopy(graph.globals["target"])
    if len(targets.shape) == 1:
      targets = targets[:, None]
    del graph.globals["target"]
    target_mask = None
    if "target_mask" in graph.globals:
      target_mask = copy.deepcopy(graph.globals["target_mask"])
      del graph.globals["target_mask"]

    out = self._forward(graph, is_training)

    if isinstance(self._loss_config, RegressionLossConfig):
      normalized_targets = (
          (targets - self._loss_config.mean) / self._loss_config.std)
      per_graph_and_head_loss = _regression_loss(
          out["globals"], normalized_targets, **self._loss_config.kwargs)
    else:
      raise TypeError(type(self._loss_config))

    # Mask out nans
    if target_mask is None:
      per_graph_and_head_loss = jnp.mean(per_graph_and_head_loss, axis=1)
    else:
      per_graph_and_head_loss = jnp.sum(
          per_graph_and_head_loss * target_mask, axis=1)
      per_graph_and_head_loss /= jnp.sum(target_mask + 1e-8, axis=1)

    g_mask = jraph.get_graph_padding_mask(graph)
    loss = _mean_with_mask(per_graph_and_head_loss, g_mask)
    scalars.update({"loss": loss})

    if self._config.aux_multiplier > 0:
      atom_loss = self._get_node_auxiliary_loss(
          graph, out["atom_one_hots"], graph.nodes["atom_one_hots_targets"],
          is_regression=False)
      bond_loss = self._get_edge_auxiliary_loss(
          graph, out["bond_one_hots"], graph.edges["bond_one_hots_targets"],
          is_regression=False)
      loss += (atom_loss + bond_loss)*self._config.aux_multiplier
      scalars.update({"atom_loss": atom_loss, "bond_loss": bond_loss})

    scaled_loss = loss / jax.device_count()
    scalars.update({"total_loss": loss})
    return scaled_loss, scalars

  @hk.transparent
  def _prepare_features(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Prepares features keys into flat node, edge and global features."""

    # Collect edge features.
    edge_features_list = [graph.edges["bond_one_hots"]]
    if (self._config.add_relative_displacement or
        self._config.add_relative_distance):
      (relative_displacement, relative_distance
       ) = _compute_relative_displacement_and_distance(
           graph, self._config.relative_displacement_normalization,
           use_target=False)

      if self._config.add_relative_displacement:
        edge_features_list.append(relative_displacement)
      if self._config.add_relative_distance:
        edge_features_list.append(relative_distance)
      mask_at_edges = _broadcast_global_to_edges(
          graph.globals["positions_nan_mask"], graph)
      edge_features_list.append(mask_at_edges[:, None].astype(jnp.float32))

    edge_features = jnp.concatenate(edge_features_list, axis=-1)

    # Collect node features
    node_features_list = [graph.nodes["atom_one_hots"]]

    if self._config.add_absolute_positions:
      node_features_list.append(
          graph.nodes["positions"] / self._config.position_normalization)
      mask_at_nodes = _broadcast_global_to_nodes(
          graph.globals["positions_nan_mask"], graph)
      node_features_list.append(mask_at_nodes[:, None].astype(jnp.float32))

    node_features = jnp.concatenate(node_features_list, axis=-1)

    global_features = jnp.zeros((len(graph.n_node), self._config.latent_size))
    chex.assert_tree_shape_prefix(global_features, (len(graph.n_node),))
    return graph._replace(
        nodes=node_features, edges=edge_features, globals=global_features)

  @hk.transparent
  def _encoder(
      self,
      graph: jraph.GraphsTuple,
      is_training: bool,
  ) -> jraph.GraphsTuple:
    """Builds the encoder."""
    del is_training  # unused
    graph = self._prepare_features(graph)

    # Run encoders in all of the node, edge and global features.
    output_sizes = [self._config.mlp_hidden_size] * self._config.mlp_layers
    output_sizes += [self._config.latent_size]
    build_mlp = functools.partial(
        _build_mlp,
        output_sizes=output_sizes,
        use_layer_norm=self._config.use_layer_norm,
    )

    gmf = jraph.GraphMapFeatures(
        embed_edge_fn=build_mlp("edge_encoder"),
        embed_node_fn=build_mlp("node_encoder"),
        embed_global_fn=None
        if self._config.ignore_globals else build_mlp("global_encoder"),
    )
    return gmf(graph)

  @hk.transparent
  def _processor(
      self,
      graph: jraph.GraphsTuple,
      is_training: bool,
  ) -> jraph.GraphsTuple:
    """Builds the processor."""
    output_sizes = [self._config.mlp_hidden_size] * self._config.mlp_layers
    output_sizes += [self._config.latent_size]
    build_mlp = functools.partial(
        _build_mlp,
        output_sizes=output_sizes,
        use_layer_norm=self._config.use_layer_norm,
    )

    shared_weights = self._config.shared_message_passing_weights
    node_reducer = _REDUCER_NAMES[self._config.node_reducer]
    global_reducer = _REDUCER_NAMES[self._config.global_reducer]

    def dropout_if_training(fn, dropout_rate: float):
      def wrapped(*args):
        out = fn(*args)
        if is_training:
          mask = hk.dropout(hk.next_rng_key(), dropout_rate,
                            jnp.ones([out.shape[0], 1]))
          out = out * mask
        return out
      return wrapped

    num_mps = self._config.num_message_passing_steps
    for step in range(num_mps):
      if step == 0 or not shared_weights:
        suffix = "shared" if shared_weights else step

        update_edge_fn = dropout_if_training(
            build_mlp(f"edge_processor_{suffix}"),
            dropout_rate=self._config.dropedge_rate)

        update_node_fn = dropout_if_training(
            build_mlp(f"node_processor_{suffix}"),
            dropout_rate=self._config.dropnode_rate)

        if self._config.ignore_globals:
          gnn = jraph.InteractionNetwork(
              update_edge_fn=update_edge_fn,
              update_node_fn=update_node_fn,
              aggregate_edges_for_nodes_fn=node_reducer)
        else:
          gnn = jraph.GraphNetwork(
              update_edge_fn=update_edge_fn,
              update_node_fn=update_node_fn,
              update_global_fn=build_mlp(f"global_processor_{suffix}"),
              aggregate_edges_for_nodes_fn=node_reducer,
              aggregate_nodes_for_globals_fn=global_reducer,
              aggregate_edges_for_globals_fn=global_reducer,
          )

      mode = self._config.processor_mode

      if mode == "mlp":
        graph = gnn(graph)

      elif mode == "resnet":
        new_graph = gnn(graph)
        graph = graph._replace(
            nodes=graph.nodes + new_graph.nodes,
            edges=graph.edges + new_graph.edges,
            globals=graph.globals + new_graph.globals,
        )
      else:
        raise ValueError(f"Unknown processor_mode `{mode}`")

      if self._config.mask_padding_graph_at_every_step:
        graph = _mask_out_padding_graph(graph)

    return graph

  @hk.transparent
  def _decoder(
      self,
      graph: jraph.GraphsTuple,
      input_graph: jraph.GraphsTuple,
      is_training: bool,
  ) -> chex.ArrayTree:
    """Builds the decoder."""
    del is_training  # unused.

    output_sizes = [self._config.mlp_hidden_size] * self._config.mlp_layers
    output_sizes += [self._loss_config.out_size]
    net = _build_mlp("regress_out", output_sizes, use_layer_norm=False)
    summed_nodes = _aggregate_nodes_to_globals(graph, graph.nodes)
    inputs_to_global_decoder = [summed_nodes]
    if not self._config.ignore_globals_from_final_layer_for_predictions:
      inputs_to_global_decoder.append(graph.globals)

    out = net(jnp.concatenate(inputs_to_global_decoder, axis=-1))
    out_dict = {}
    out_dict["globals"] = out

    # Note "linear" names are for compatibility with pre-trained model names.
    out_dict["bond_one_hots"] = hk.Linear(
        _NUM_EDGE_FEATURES, name="linear")(graph.edges)
    out_dict["atom_one_hots"] = hk.Linear(
        _NUM_NODE_FEATURES, name="linear_1")(graph.nodes)
    return out_dict

  @hk.transparent
  def _forward(self, graph: jraph.GraphsTuple, is_training: bool):
    input_graph = jraph.GraphsTuple(*graph)
    with hk.experimental.name_scope("encoder_scope"):
      graph = self._encoder(graph, is_training)
    with hk.experimental.name_scope("processor_scope"):
      graph = self._processor(graph, is_training)
    with hk.experimental.name_scope("decoder_scope"):
      out = self._decoder(graph, input_graph, is_training)
    return out

  def _get_node_auxiliary_loss(
      self, graph, pred, targets, is_regression, additional_mask=None):
    loss = self._get_loss(pred, targets, is_regression)
    target_mask = jraph.get_node_padding_mask(graph)

    if additional_mask is not None:
      loss *= additional_mask
      target_mask = jnp.logical_and(target_mask, additional_mask)

    return _mean_with_mask(loss, target_mask)

  def _get_edge_auxiliary_loss(
      self, graph, pred, targets, is_regression, additional_mask=None):
    loss = self._get_loss(pred, targets, is_regression)
    target_mask = jraph.get_edge_padding_mask(graph)

    if additional_mask is not None:
      loss *= additional_mask
      target_mask = jnp.logical_and(target_mask, additional_mask)

    return _mean_with_mask(loss, target_mask)

  def _get_loss(self, pred, targets, is_regression):
    if is_regression:
      loss = ((pred - targets)**2).mean(axis=-1)
    else:
      targets /= jnp.maximum(1., jnp.sum(targets, axis=-1, keepdims=True))
      loss = _softmax_cross_entropy(pred, targets)
    return loss


def get_utilization_scalars(
    padded_graph: jraph.GraphsTuple) -> Dict[str, float]:
  padding_nodes = jraph.get_number_of_padding_with_graphs_nodes(padded_graph)
  all_nodes = len(jax.tree_leaves(padded_graph.nodes)[0])
  padding_edges = jraph.get_number_of_padding_with_graphs_edges(padded_graph)
  all_edges = len(jax.tree_leaves(padded_graph.edges)[0])
  padding_graphs = jraph.get_number_of_padding_with_graphs_graphs(padded_graph)
  all_graphs = len(padded_graph.n_node)
  return {"node_utilization": 1 - (padding_nodes / all_nodes),
          "edge_utilization": 1 - (padding_edges / all_edges),
          "graph_utilization": 1 - (padding_graphs / all_graphs)}


def sum_with_mask(array: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
  return (mask * array).sum(0)


def _mean_with_mask(array: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
  num_valid_rows = mask.sum(0)
  return sum_with_mask(array, mask) / num_valid_rows


def _mask_out_padding_graph(
    padded_graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
  return padded_graph._replace(
      nodes=jnp.where(
          jraph.get_node_padding_mask(
              padded_graph)[:, None], padded_graph.nodes, 0.),
      edges=jnp.where(
          jraph.get_edge_padding_mask(
              padded_graph)[:, None], padded_graph.edges, 0.),
      globals=jnp.where(
          jraph.get_graph_padding_mask(
              padded_graph)[:, None], padded_graph.globals, 0.),
      )


def _aggregate_nodes_to_globals(graph, node_features):
  n_graph = graph.n_node.shape[0]
  sum_n_node = jax.tree_leaves(graph.nodes)[0].shape[0]
  graph_idx = jnp.arange(n_graph)
  node_gr_idx = jnp.repeat(
      graph_idx, graph.n_node, axis=0, total_repeat_length=sum_n_node)
  return jax.ops.segment_sum(node_features, node_gr_idx, num_segments=n_graph)
