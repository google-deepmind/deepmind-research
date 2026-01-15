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

============================================================================
BEGINNER'S GUIDE TO THIS FILE
============================================================================

This file implements the "Encode-Process-Decode" architecture for graph
neural networks, which is the core learning component of the simulator.

THE THREE STAGES:

1. ENCODE: Raw features → Latent representations
   - Node features (velocity, position) → Node embeddings
   - Edge features (displacement, distance) → Edge embeddings
   
2. PROCESS: Message passing (the "magic")
   - Nodes send messages to neighbors along edges
   - Each node aggregates incoming messages
   - This repeats for M steps (typically 10-15)
   - RESIDUAL CONNECTIONS: new = old + update (helps training)
   
3. DECODE: Latent representations → Output (acceleration)
   - Node embeddings → Predicted acceleration per particle

WHY THIS WORKS:
- Message passing lets particles "communicate" about their state
- Multiple steps allow information to propagate across the graph
- The network learns WHAT information to send and HOW to use it

============================================================================

The Sonnet `EncodeProcessDecode` module provided here implements the learnable
parts of the model.
It assumes an encoder preprocessor has already built a graph with
connectivity and features as described in the paper, with features normalized
to zero-mean unit-variance.

Dependencies include Tensorflow 1.x, Sonnet 1.x and the Graph Nets 1.1 library.
"""
from typing import Callable

import graph_nets as gn
import sonnet as snt
import tensorflow as tf

Reducer = Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]


def build_mlp(
    hidden_size: int, num_hidden_layers: int, output_size: int) -> snt.Module:
  """Builds a Multi-Layer Perceptron (MLP).
  
  BEGINNER NOTE:
  An MLP is just stacked linear layers with activations between them.
  Example with hidden_size=128, num_hidden_layers=2, output_size=64:
    Input → [Linear(128), ReLU] → [Linear(128), ReLU] → [Linear(64)] → Output
  """
  return snt.nets.MLP(
      output_sizes=[hidden_size] * num_hidden_layers + [output_size])  # pytype: disable=bad-return-type  # gen-stub-imports


class EncodeProcessDecode(snt.AbstractModule):
  """Encode-Process-Decode function approximator for learnable simulator.
  
  BEGINNER NOTE:
  This is the "brain" of the physics simulator. It processes graphs through:
  
  1. ENCODER: Convert raw features into learned representations
  2. PROCESSOR: Run message passing M times to propagate information  
  3. DECODER: Convert node representations to output (acceleration)
  
  The architecture uses:
  - MLPs (neural networks) to transform features
  - LayerNorm for training stability
  - Residual connections (skip connections) to help gradient flow
  """

  def __init__(
      self,
      latent_size: int,
      mlp_hidden_size: int,
      mlp_num_hidden_layers: int,
      num_message_passing_steps: int,
      output_size: int,
      reducer: Reducer = tf.math.unsorted_segment_sum,
      name: str = "EncodeProcessDecode"):
    """Inits the model.

    Args:
      latent_size: Size of the node and edge latent representations.
        # BEGINNER: Dimension of learned embeddings (typically 128)
      mlp_hidden_size: Hidden layer size for all MLPs.
        # BEGINNER: Width of neural network layers
      mlp_num_hidden_layers: Number of hidden layers in all MLPs.
        # BEGINNER: Depth of neural networks (typically 2)
      num_message_passing_steps: Number of message passing steps.
        # BEGINNER: How many times particles "talk" to neighbors (10-15)
      output_size: Output size of the decode node representations as required
        by the downstream update function.
        # BEGINNER: Usually = num_dimensions (2D or 3D acceleration)
      reducer: Reduction to be used when aggregating the edges in the nodes in
        the interaction network. This should be a callable whose signature
        matches tf.math.unsorted_segment_sum.
        # BEGINNER: How to combine messages from multiple neighbors (sum)
      name: Name of the model.
    """

    super().__init__(name=name)

    self._latent_size = latent_size
    self._mlp_hidden_size = mlp_hidden_size
    self._mlp_num_hidden_layers = mlp_num_hidden_layers
    self._num_message_passing_steps = num_message_passing_steps
    self._output_size = output_size
    self._reducer = reducer

    with self._enter_variable_scope():
      self._networks_builder()

  def _build(self, input_graph: gn.graphs.GraphsTuple) -> tf.Tensor:
    """Forward pass of the learnable dynamics model.
    
    BEGINNER: This is where Encode → Process → Decode happens!
    """

    # Step 1: ENCODE - Convert raw features to latent space
    latent_graph_0 = self._encode(input_graph)

    # Step 2: PROCESS - Run M message passing steps
    latent_graph_m = self._process(latent_graph_0)

    # Step 3: DECODE - Convert latent nodes to output acceleration
    return self._decode(latent_graph_m)

  def _networks_builder(self):
    """Builds all the neural network components.
    
    BEGINNER NOTE:
    This creates three types of networks:
    1. ENCODER: GraphIndependent - processes nodes and edges separately
    2. PROCESSORS: InteractionNetwork - does message passing (one per step)
    3. DECODER: MLP - converts final node states to output
    """

    def build_mlp_with_layer_norm():
      """MLP followed by LayerNorm for stable training."""
      mlp = build_mlp(
          hidden_size=self._mlp_hidden_size,
          num_hidden_layers=self._mlp_num_hidden_layers,
          output_size=self._latent_size)
      return snt.Sequential([mlp, snt.LayerNorm()])

    # ENCODER: Process nodes and edges independently (no message passing yet)
    encoder_kwargs = dict(
        edge_model_fn=build_mlp_with_layer_norm,
        node_model_fn=build_mlp_with_layer_norm)
    self._encoder_network = gn.modules.GraphIndependent(**encoder_kwargs)

    # PROCESSORS: Create M InteractionNetworks for message passing
    # Each has its OWN parameters (not shared) for expressivity
    self._processor_networks = []
    for _ in range(self._num_message_passing_steps):
      self._processor_networks.append(
          gn.modules.InteractionNetwork(
              edge_model_fn=build_mlp_with_layer_norm,
              node_model_fn=build_mlp_with_layer_norm,
              reducer=self._reducer))

    # DECODER: Convert final node latents to output acceleration
    self._decoder_network = build_mlp(
        hidden_size=self._mlp_hidden_size,
        num_hidden_layers=self._mlp_num_hidden_layers,
        output_size=self._output_size)

  def _encode(
      self, input_graph: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:
    """Encodes the input graph features into a latent graph.
    
    BEGINNER NOTE:
    This transforms raw features into learned representations:
    - Raw node features (velocity, etc.) → Node embeddings
    - Raw edge features (displacement) → Edge embeddings
    
    If global features exist, they're broadcast (copied) to each node.
    """

    # Broadcast global features to all nodes if present
    if input_graph.globals is not None:
      broadcasted_globals = gn.blocks.broadcast_globals_to_nodes(input_graph)
      input_graph = input_graph.replace(
          nodes=tf.concat([input_graph.nodes, broadcasted_globals], axis=-1),
          globals=None)

    # Apply encoder MLPs to nodes and edges
    latent_graph_0 = self._encoder_network(input_graph)
    return latent_graph_0

  def _process(
      self, latent_graph_0: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:
    """Processes the latent graph with several steps of message passing.
    
    BEGINNER NOTE - MESSAGE PASSING:
    This is where the "learning" really happens! Each step:
    1. Edges compute messages based on connected nodes
    2. Nodes aggregate messages from all incoming edges
    3. Nodes update their state based on aggregated messages
    
    After M steps, each node has information from M-hop neighbors!
    """

    # Run M message passing steps sequentially
    latent_graph_prev_k = latent_graph_0
    latent_graph_k = latent_graph_0
    for processor_network_k in self._processor_networks:
      latent_graph_k = self._process_step(
          processor_network_k, latent_graph_prev_k)
      latent_graph_prev_k = latent_graph_k

    latent_graph_m = latent_graph_k
    return latent_graph_m

  def _process_step(
      self, processor_network_k: snt.Module,
      latent_graph_prev_k: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:
    """Single step of message passing with residual connections.
    
    BEGINNER NOTE - RESIDUAL CONNECTIONS:
    Instead of: output = network(input)
    We do:      output = input + network(input)
    
    This helps gradients flow during training (prevents vanishing gradients)
    and lets the network learn "adjustments" rather than full transformations.
    """

    # One step of message passing
    latent_graph_k = processor_network_k(latent_graph_prev_k)

    # Add residual connections for nodes and edges
    # new_state = old_state + learned_update
    latent_graph_k = latent_graph_k.replace(
        nodes=latent_graph_k.nodes+latent_graph_prev_k.nodes,
        edges=latent_graph_k.edges+latent_graph_prev_k.edges)
    return latent_graph_k

  def _decode(self, latent_graph: gn.graphs.GraphsTuple) -> tf.Tensor:
    """Decodes node latent features to output (acceleration).
    
    BEGINNER NOTE:
    Final step - convert learned node representations to actual predictions.
    Only nodes are decoded (edges were just for message passing).
    """
    return self._decoder_network(latent_graph.nodes)
