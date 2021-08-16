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
"""Transformer embedding modules."""

from typing import List, Optional

import haiku as hk
from haiku import initializers as init
import jax
import jax.numpy as jnp
import jraph

from wikigraphs.model import graph_net as gn


def get_pos_start(timesteps: int, batch_size: int) -> jnp.ndarray:
  """Find the right slice of positional embeddings for incremental sampling."""
  pos_start = hk.get_state(
      'cache_progress_idx', [batch_size], dtype=jnp.int32, init=jnp.zeros)
  hk.set_state('cache_progress_idx', pos_start + timesteps)
  return pos_start


class SinusoidalPositionEmbedding(hk.Module):
  """Position encoding, using mixture of sinusoidal signals."""

  def __init__(self,
               dim: int,
               cache_steps: int = 0,
               reverse_order: bool = False,
               clamp_len: Optional[int] = None,
               name: Optional[str] = None):
    """Initialize a SinusoidalPositionEmbedding.

    Args:
      dim: Embedding dimension.
      cache_steps: The length of the memory.
      reverse_order: If set to True, position index is reversed.
      clamp_len: position beyond clamp_len will be reset to clamp_len, default
        to not clamping.
      name: Optional name for this Haiku module.
    """
    super(SinusoidalPositionEmbedding, self).__init__(name=name)
    self._dim = dim
    self._cache_steps = cache_steps
    self._reverse_order = reverse_order
    self._clamp_len = clamp_len
    self._inv_freq = 1.0 / (
        10000 ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim))

  def __call__(self, timesteps: int, batch_size: int) -> jnp.ndarray:
    """Computes the sinusoidal position embedding.

    Args:
      timesteps: The length of the sequence.
      batch_size: The size of the batch.

    Returns:
      Sinusoidal position embedding.
    """
    full_length = timesteps + self._cache_steps

    if self._reverse_order:
      positions = jnp.arange(full_length - 1, -1, -1)
      positions = jnp.repeat(positions[None, :], batch_size, axis=0)
    else:
      if self._cache_steps > 0:
        positions = (get_pos_start(timesteps, batch_size)[:, None]
                     + jnp.arange(timesteps)[None, :])
      else:
        positions = jnp.arange(0, full_length)
        positions = jnp.repeat(positions[None, :], batch_size, axis=0)

    if self._clamp_len is not None:
      positions = jnp.minimum(positions, self._clamp_len)

    scaled_time = positions[:, :, None] * self._inv_freq[None, None, :]
    return jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=2)


def relative_shift(x: jnp.ndarray) -> jnp.ndarray:
  """Shift the relative logits."""
  x_shape = list(x.shape)
  x = jnp.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
  x = jnp.reshape(
      x, [x_shape[0], x_shape[1], x_shape[3] + 1, x_shape[2]])[:, :, 1:, :]
  x = jnp.reshape(x, x_shape)
  return x


class RelativePositionEmbedding(hk.Module):
  """Position encoding, using relative positions than absolute positions."""

  def __init__(self,
               dim: int,
               dropout_rate: float,
               r_w_bias: jnp.ndarray,
               r_r_bias: jnp.ndarray,
               init_scale: float = 0.02,
               clamp_len: Optional[int] = None,
               name: Optional[str] = None):
    """Initialize a RelativePositionEmbedding.

    Args:
      dim: Embedding dimension.
      dropout_rate: dropout rate.
      r_w_bias: global content bias.
      r_r_bias: global positional bias.
      init_scale: the initialization scale of the RandomNormal used for the
        linear layer.
      clamp_len: position beyond clamp_len will be reset to clamp_len, default
        to not clamping.
      name: Optional name for this Haiku module.
    """
    super(RelativePositionEmbedding, self).__init__(name=name)
    self._dim = dim
    self._dropout_rate = dropout_rate
    self._r_w_bias = r_w_bias
    self._r_r_bias = r_r_bias
    self._init_scale = init_scale
    self._sinusoidal_pos_emb = SinusoidalPositionEmbedding(
        dim=dim,
        reverse_order=True,
        clamp_len=clamp_len,
        name=name)

  def __call__(self, q: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
    """Computes the relative position embedding.

    Args:
      q: The query.
      k: The key.

    Returns:
      Relative position embedding.
    """
    # Use key instead of query to obtain the length.
    batch_size, key_length, num_heads, head_dim = list(k.shape)
    # Content based addressing and global content bias
    content_score = jnp.einsum('bthd,bThd->bhtT', q + self._r_w_bias, k)

    # Relative position encoding
    positional_encodings = self._sinusoidal_pos_emb(key_length, batch_size)
    positional_encodings = hk.dropout(hk.next_rng_key(), self._dropout_rate,
                                      positional_encodings)
    rel_pos_emb = hk.Conv1D(
        output_channels=self._dim, kernel_shape=1, with_bias=False,
        w_init=init.RandomNormal(stddev=self._init_scale))(positional_encodings)
    rel_pos_emb = jnp.reshape(rel_pos_emb, [
        batch_size, key_length, num_heads, head_dim])

    # Content dependent positional bias and global positional bias
    rel_pos_score = jnp.einsum('bthd,bThd->bhtT', q + self._r_r_bias,
                               rel_pos_emb)
    rel_pos_score = relative_shift(rel_pos_score)
    assert content_score.shape == rel_pos_score.shape
    return content_score + rel_pos_score


def hierarchical_logprobs(
    logits: jnp.ndarray,
    class_logits: jnp.ndarray,
    cutoffs: List[int]) -> jnp.ndarray:
  """Hierarchical log-probs for adaptive softmax."""
  sizes = [y - x for x, y in zip(cutoffs[:-1], cutoffs[1:])]
  num_tails = len(sizes) - 1
  split_logits = jnp.split(logits, cutoffs[1:-1], axis=-1)
  all_head_logits = jnp.concatenate([split_logits[0], class_logits], -1)
  # Mask out item 0, the NULL token
  all_head_logits += jnp.concatenate(
      [jnp.ones([1], dtype=logits.dtype) * -10,
       jnp.zeros([sizes[0] + num_tails - 1], dtype=logits.dtype)], 0)
  all_head_logprobs = jax.nn.log_softmax(all_head_logits)
  head_logprobs, class_logprobs = jnp.split(all_head_logprobs,
                                            [sizes[0]], axis=-1)
  tail_logprobs = []
  for i, tail_size in enumerate(sizes[1:]):  # pylint: disable=unused-variable
    tail_logprobs += [jax.nn.log_softmax(split_logits[i + 1])
                      + class_logprobs[..., [i]]]
  return jnp.concatenate([head_logprobs] + tail_logprobs, -1)


class AdaptiveSoftmaxEmbedding(hk.Module):
  """Adaptive inputs and softmax (https://arxiv.org/abs/1809.10853)."""

  def __init__(self,
               dim: int,
               vocab_size: int,
               cutoffs: List[int],
               tail_shrink_factor: int = 4,
               hierarchical: bool = True,
               init_std: float = 0.02,
               init_proj_std: float = 0.01,
               dtype: jnp.dtype = jnp.float32,
               name: Optional[str] = None):
    """Initialize a AdaptiveSoftmaxEmbedding.

    Args:
      dim: dimensionality of the hidden space.
      vocab_size: the size of the vocabulary.
      cutoffs: the cutoff indices of the vocabulary used for the adaptive
        softmax embedding.
      tail_shrink_factor: how many times to shrink the hidden dimensionality
        for low-frequency vocabulary after each cutoff.
      hierarchical: whether to use hierarchical softmax.
      init_std: standard deviation of the Normal distribution used to initialize
        the embedding weights.
      init_proj_std: standard deviation of the Normal distribution used to
        initialize the projection weights.
      dtype: Optional data type default to jnp.float32.
      name: Optional name for this Haiku module.
    """
    super(AdaptiveSoftmaxEmbedding, self).__init__(name=name)
    self._hidden_size = dim
    self._vocab_size = vocab_size
    self._cutoffs = [0] + list(cutoffs) + [self._vocab_size]
    self._tail_shrink_factor = tail_shrink_factor
    self._hierarchical = hierarchical
    self._dtype = dtype
    self._embeddings = []
    self._projections = []

    self._bias = hk.get_parameter(
        'bias', [self._vocab_size], dtype=self._dtype, init=jnp.zeros)

    l_cutoffs = self._cutoffs[:-1]
    r_cutoffs = self._cutoffs[1:]
    for i, (l_cutoff, r_cutoff) in enumerate(zip(l_cutoffs, r_cutoffs)):
      hidden_size = self._hidden_size // (self._tail_shrink_factor ** i)
      embedding = hk.get_parameter(
          f'embeddings_{l_cutoff}_{r_cutoff}',
          [r_cutoff - l_cutoff, hidden_size],
          dtype=self._dtype,
          init=hk.initializers.RandomNormal(stddev=init_std))
      self._embeddings += [embedding]
      if self._tail_shrink_factor != 1:
        projection = hk.get_parameter(
            f'projection_{l_cutoff}_{r_cutoff}',
            [hidden_size, self._hidden_size],
            dtype=self._dtype,
            init=hk.initializers.RandomNormal(stddev=init_proj_std))
        self._projections += [projection]

    if self._tail_shrink_factor != 1:
      self._output_projection = hk.get_parameter(
          'output_head_projection',
          [self._hidden_size, self._hidden_size],
          dtype=self._dtype,
          init=hk.initializers.RandomNormal(stddev=init_proj_std))

    if self._hierarchical:
      self._class_weights = hk.get_parameter(
          'tail_class_weights',
          [self._hidden_size, len(cutoffs)],
          init=hk.initializers.RandomNormal(stddev=init_std))
      self._class_bias = hk.get_parameter(
          'tail_class_bias',
          [len(cutoffs)],
          dtype=self._dtype,
          init=jnp.zeros)

  @hk.transparent
  def build_embeddings(self):
    """Builds input embeddings."""
    if self._projections:
      embedding_mat = [
          jnp.dot(emb, proj) for emb, proj in zip(self._embeddings,
                                                  self._projections)]
    else:
      embedding_mat = self._embeddings
    input_embeddings = jnp.concatenate(embedding_mat, 0)
    return input_embeddings

  @hk.transparent
  def build_output_embeddings(self):
    """Builds separate output embeddings."""
    if self._projections:
      projections = [self._output_projection] + self._projections[1:]
      embedding_mat = [jnp.dot(emb, proj)
                       for emb, proj in zip(self._embeddings, projections)]
    else:
      embedding_mat = self._embeddings
    output_embeddings = jnp.concatenate(embedding_mat, 0)
    return jnp.transpose(output_embeddings)

  def embed_input(self, input_tokens: jnp.ndarray) -> jnp.ndarray:
    """Embeds the input."""
    assert jnp.issubdtype(input_tokens.dtype, jnp.integer)
    input_embeddings = self.build_embeddings()
    embedded_inputs = input_embeddings[input_tokens]
    return embedded_inputs * self._hidden_size ** 0.5

  def embed_output(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Outputs logits."""
    output_embs = self.build_output_embeddings()
    logits = jnp.einsum('btd,dv->btv', inputs, output_embs) + self._bias
    if self._hierarchical:
      class_logits = jnp.dot(inputs, self._class_weights) + self._class_bias
      logprobs = hierarchical_logprobs(logits, class_logits, self._cutoffs)
      return logprobs
    else:
      return logits


class GraphEmbeddingModel(hk.Module):
  """A single graph network for embedding graph data."""

  def __init__(self,
               embed_dim: int,
               num_layers: int,
               msg_hidden_size_factor: int = 2,
               use_layer_norm: bool = False,
               name: Optional[str] = None):
    """Constructor.

    Args:
      embed_dim: node embedding size.
      num_layers: number of message passing layers to use.
      msg_hidden_size_factor: size of the message network hiddens as a factor
        of embed_dim.
      use_layer_norm: whether to apply layer norm on node updates.
      name: optional name for this module.
    """
    super().__init__(name=name)
    self._embed_dim = embed_dim
    self._num_layers = num_layers
    self._msg_hidden_size_factor = msg_hidden_size_factor
    self._use_layer_norm = use_layer_norm

  def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Compute embeddings for each node in the graphs.

    Args:
      graphs: a set of graphs batched into a single graph.  The nodes and edges
        are represented as feature tensors.

    Returns:
      graphs: new graph with node embeddings updated (shape [n_nodes,
        embed_dim]).
    """
    nodes = hk.Linear(self._embed_dim)(graphs.nodes)
    edges = hk.Linear(self._embed_dim)(graphs.edges)

    nodes = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
        jax.nn.gelu(nodes))
    edges = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
        jax.nn.gelu(edges))

    graphs = graphs._replace(nodes=nodes, edges=edges)
    graphs = gn.SimpleGraphNet(
        num_layers=self._num_layers,
        msg_hidden_size_factor=self._msg_hidden_size_factor,
        layer_norm=self._use_layer_norm)(graphs)
    return graphs
