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

"""Graph network implementation accompanying ICML 2020 submission.

   "Learning to Simulate Complex Physics with Graph Networks"

   Alvaro Sanchez-Gonzalez*, Jonathan Godwin*, Tobias Pfaff*, Rex Ying,
   Jure Leskovec, Peter W. Battaglia

   https://arxiv.org/abs/2002.09405

The Sonnet `EncodeProcessDecode` module provided here implements the learnable
parts of the model.
It assumes an encoder preprocessor has already built a graph with
connectivity and features as described in the paper, with features normalized
to zero-mean unit-variance.

Dependencies include Tensorflow 1.x, Sonnet 1.x and the Graph Nets 1.1 library.
"""

import graph_nets as gn
import sonnet as snt
import tensorflow as tf


def build_mlp(
    hidden_size: int, num_hidden_layers: int, output_size: int) -> snt.Module:
  """Builds an MLP."""
  return snt.nets.MLP(
      output_sizes=[hidden_size] * num_hidden_layers + [output_size])


class EncodeProcessDecode(snt.AbstractModule):
  """Encode-Process-Decode function approximator for learnable simulator."""

  def __init__(
      self,
      latent_size: int,
      mlp_hidden_size: int,
      mlp_num_hidden_layers: int,
      num_message_passing_steps: int,
      output_size: int,
      name: str = "EncodeProcessDecode"):
    """Inits the model.

    Args:
      latent_size: Size of the node and edge latent representations.
      mlp_hidden_size: Hidden layer size for all MLPs.
      mlp_num_hidden_layers: Number of hidden layers in all MLPs.
      num_message_passing_steps: Number of message passing steps.
      output_size: Output size of the decode node representations as required
        by the downstream update function.
      name: Name of the model.
    """

    super().__init__(name=name)

    self._latent_size = latent_size
    self._mlp_hidden_size = mlp_hidden_size
    self._mlp_num_hidden_layers = mlp_num_hidden_layers
    self._num_message_passing_steps = num_message_passing_steps
    self._output_size = output_size

    with self._enter_variable_scope():
      self._networks_builder()

  def _build(self, input_graph: gn.graphs.GraphsTuple) -> tf.Tensor:
    """Forward pass of the learnable dynamics model."""

    # Encode the input_graph.
    latent_graph_0 = self._encode(input_graph)

    # Do `m` message passing steps in the latent graphs.
    latent_graph_m = self._process(latent_graph_0)

    # Decode from the last latent graph.
    return self._decode(latent_graph_m)

  def _networks_builder(self):
    """Builds the networks."""

    def build_mlp_with_layer_norm():
      mlp = build_mlp(
          hidden_size=self._mlp_hidden_size,
          num_hidden_layers=self._mlp_num_hidden_layers,
          output_size=self._latent_size)
      return snt.Sequential([mlp, snt.LayerNorm()])

    # The encoder graph network independently encodes edge and node features.
    encoder_kwargs = dict(
        edge_model_fn=build_mlp_with_layer_norm,
        node_model_fn=build_mlp_with_layer_norm)
    self._encoder_network = gn.modules.GraphIndependent(**encoder_kwargs)

    # Create `num_message_passing_steps` graph networks with unshared parameters
    # that update the node and edge latent features.
    # Note that we can use `modules.InteractionNetwork` because
    # it also outputs the messages as updated edge latent features.
    self._processor_networks = []
    for _ in range(self._num_message_passing_steps):
      self._processor_networks.append(
          gn.modules.InteractionNetwork(
              edge_model_fn=build_mlp_with_layer_norm,
              node_model_fn=build_mlp_with_layer_norm))

    # The decoder MLP decodes node latent features into the output size.
    self._decoder_network = build_mlp(
        hidden_size=self._mlp_hidden_size,
        num_hidden_layers=self._mlp_num_hidden_layers,
        output_size=self._output_size)

  def _encode(
      self, input_graph: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:
    """Encodes the input graph features into a latent graph."""

    # Copy the globals to all of the nodes, if applicable.
    if input_graph.globals is not None:
      broadcasted_globals = gn.blocks.broadcast_globals_to_nodes(input_graph)
      input_graph = input_graph.replace(
          nodes=tf.concat([input_graph.nodes, broadcasted_globals], axis=-1),
          globals=None)

    # Encode the node and edge features.
    latent_graph_0 = self._encoder_network(input_graph)
    return latent_graph_0

  def _process(
      self, latent_graph_0: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:
    """Processes the latent graph with several steps of message passing."""

    # Do `m` message passing steps in the latent graphs.
    # (In the shared parameters case, just reuse the same `processor_network`)
    latent_graph_prev_k = latent_graph_0
    for processor_network_k in self._processor_networks:
      latent_graph_k = self._process_step(
          processor_network_k, latent_graph_prev_k)
      latent_graph_prev_k = latent_graph_k

    latent_graph_m = latent_graph_k
    return latent_graph_m

  def _process_step(
      self, processor_network_k: snt.Module,
      latent_graph_prev_k: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:
    """Single step of message passing with node/edge residual connections."""

    # One step of message passing.
    latent_graph_k = processor_network_k(latent_graph_prev_k)

    # Add residuals.
    latent_graph_k = latent_graph_k.replace(
        nodes=latent_graph_k.nodes+latent_graph_prev_k.nodes,
        edges=latent_graph_k.edges+latent_graph_prev_k.edges)
    return latent_graph_k

  def _decode(self, latent_graph: gn.graphs.GraphsTuple) -> tf.Tensor:
    """Decodes from the latent graph."""
    return self._decoder_network(latent_graph.nodes)
