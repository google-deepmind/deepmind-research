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
"""Jax implementation of the Transformer-XL model."""

from typing import Dict, List, Optional, Tuple

import haiku as hk
from haiku import initializers as init
import jax
import jax.numpy as jnp
import jraph
import numpy as np

from wikigraphs.model import transformer_block
from wikigraphs.model.embedding import AdaptiveSoftmaxEmbedding
from wikigraphs.model.embedding import GraphEmbeddingModel


# For WikiText-103
DEFAULT_CUTOFFS = (20000 + 1, 40000 + 1, 200000 + 1)


def sequence_prediction_metrics(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None
    ) -> Dict[str, float]:
  """Compute the metrics for sequence prediction.

  Args:
    logits: [B, T, V] array of logits.
    labels: [B, T] array of labels.
    mask: [B, T] array of binary masks, if provided.

  Returns:
    metrics: a dictionary of metrics.
  """
  vocab_size = logits.shape[-1]
  logps = jax.nn.log_softmax(logits)
  labels_one_hot = hk.one_hot(labels, vocab_size)
  class_logps = jnp.sum(logps * labels_one_hot, axis=-1)
  prediction_correct = jnp.argmax(logits, axis=-1) == labels
  if mask is not None:
    masked_logps = mask * class_logps
    total_count = jnp.sum(mask)
    tokens_correct = jnp.sum(prediction_correct * mask)
    seq_correct = jnp.all(
        jnp.logical_or(prediction_correct, jnp.logical_not(mask)), axis=-1)
  else:
    masked_logps = class_logps
    total_count = np.prod(class_logps.shape)
    tokens_correct = jnp.sum(prediction_correct)
    seq_correct = jnp.all(prediction_correct, axis=-1)

  token_accuracy = tokens_correct.astype(jnp.float32) / total_count
  seq_accuracy = jnp.mean(seq_correct)
  log_probs = jnp.mean(jnp.sum(masked_logps, axis=-1))
  total_loss = -jnp.sum(masked_logps)
  loss = total_loss / total_count
  return dict(
      loss=loss,
      total_loss=total_loss,
      total_count=total_count,
      token_accuracy=token_accuracy,
      seq_accuracy=seq_accuracy,
      log_probs=log_probs,
  )


class TransformerXL(hk.Module):
  """TransformerXL language model with memory using GPT2 blocks.

  TransformerXL: https://arxiv.org/abs/1901.02860
  GPT-2: http://www.persagen.com/files/misc/radford2019language.pdf
  """

  def __init__(self,
               vocab_size: int = 256,
               emb_dim: int = 256,
               num_layers: int = 10,
               num_heads: int = 8,
               dropout_prob: float = 0.1,
               dropout_attn_prob: float = 0.0,
               self_att_init_scale: float = 0.02,
               dense_init_scale: float = 0.02,
               dense_dim: int = 2100,
               cutoffs: List[int] = DEFAULT_CUTOFFS,
               tail_shrink_factor: int = 1,
               relative_pos_clamp_len: Optional[int] = None,
               name: Optional[str] = None):
    """Initialize a TransformerXL.

    Args:
      vocab_size: the size of the vocabulary.
      emb_dim: the dimensionality of the embeddings.
      num_layers: number of transformer blocks.
      num_heads: number of attention heads.
      dropout_prob: dropout probability.
      dropout_attn_prob: dropout probability of the attention module.
      self_att_init_scale: the initialization scale of the VarianceScaling
        used for the linear layer in the attention module.
      dense_init_scale: the initialization scale of the VarianceScaling
        used for the linear layer in the feedforward module.
      dense_dim: feature size of the feedforward block.
      cutoffs: the cutoff indices of the vocabulary used for the adaptive
        softmax embedding.
      tail_shrink_factor: how many times to shrink the hidden dimensionality
        for low-frequency vocabulary after each cutoff in the adaptive softmax
        embedding.
      relative_pos_clamp_len: clamp length of the relative position embeddings.
      name: Optional name for this Haiku module.
    """
    super().__init__(name=name)
    self._vocab_size = vocab_size
    self._emb_dim = emb_dim
    self._num_layers = num_layers
    self._num_heads = num_heads
    self._dropout_prob = dropout_prob
    self._dropout_attn_prob = dropout_attn_prob
    self._self_att_init_scale = self_att_init_scale
    self._dense_init_scale = dense_init_scale
    self._dense_dim = dense_dim
    self._relative_pos_clamp_len = relative_pos_clamp_len
    self._io_emb = AdaptiveSoftmaxEmbedding(
        emb_dim, vocab_size, cutoffs=cutoffs,
        tail_shrink_factor=tail_shrink_factor)

  def __call__(self,
               x: jnp.ndarray,
               mask: Optional[jnp.ndarray] = None,
               is_training: bool = True,
               should_reset: Optional[jnp.ndarray] = None,
               cache_steps: int = 0,
               extra: Optional[jnp.ndarray] = None,
               extra_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Computes the outputs of the TransformerXL.

    Args:
      x: [batch, timesteps]. Inputs at time step t.
      mask: [batch, timesteps]. It indicates what tokens to be predicted. In
        other words it corresponds to non-pad tokens in x_{t+1}.
      is_training: whether the current stage is training or not.
      should_reset: reset marker [batch, timesteps].
      cache_steps: number of timesteps in the cache.
      extra: if provided should be extra key-value input
        [batch, extra_timesteps, in_dim].
      extra_mask: if provided should be the mask for extra key-value input,
        [batch, extra_timesteps].

    Returns:
      output: transformer output [batch, timesteps].
    """
    if cache_steps == 0:
      cache_steps = x.shape[1]
    if should_reset is None:
      should_reset = jnp.where(x == 1, 1, 0)
    h = self._io_emb.embed_input(x)

    if mask is not None:
      attention_mask = mask[:, None, None, :]
    else:
      attention_mask = None

    head_dim = self._emb_dim // self._num_heads
    assert self._emb_dim % self._num_heads == 0, 'Head dim should be an int.'

    # Biases for relative position embedding shared across all layers
    r_w_bias = hk.get_parameter(
        'r_w_bias', [1, 1, self._num_heads, head_dim],
        init=init.RandomNormal(stddev=self._self_att_init_scale))
    r_r_bias = hk.get_parameter(
        'r_r_bias', [1, 1, self._num_heads, head_dim],
        init=init.RandomNormal(stddev=self._self_att_init_scale))

    for i in range(self._num_layers):
      if mask is not None:
        h *= mask[:, :, None]
      h = transformer_block.GPT2Block(
          r_w_bias=r_w_bias,
          r_r_bias=r_r_bias,
          causal=True,
          dense_dim=self._dense_dim,
          dropout_prob=self._dropout_prob,
          dropout_attn_prob=self._dropout_attn_prob,
          num_heads=self._num_heads,
          self_att_init_scale=self._self_att_init_scale,
          dense_init_scale=self._dense_init_scale,
          relative_pos_clamp_len=self._relative_pos_clamp_len,
          name='transformer_block_{}'.format(i),
          )(
              h, mask=attention_mask, is_training=is_training,
              should_reset=should_reset, cache_steps=cache_steps,
              extra=extra, extra_mask=extra_mask)

    if mask is not None:
      h *= mask[:, :, None]
    return self._io_emb.embed_output(h)

  def loss(self,
           inputs: jnp.ndarray,
           labels: jnp.ndarray,
           mask: Optional[jnp.ndarray] = None,
           is_training: bool = True,
           should_reset: Optional[jnp.ndarray] = None,
           cache_steps: int = 0,
           extra: Optional[jnp.ndarray] = None,
           extra_mask: Optional[jnp.ndarray] = None
           ) -> Tuple[float, Dict[str, float]]:
    """Computes the loss of the TransformerXL.

    Args:
      inputs: [batch, timesteps].
      labels: [batch, timesteps].
      mask: [batch, timesteps]. It indicates what tokens to be predicted. In
        other words it corresponds to non-pad tokens in the `labels`.
      is_training: whether the current stage is training or not.
      should_reset: reset marker [batch, timesteps].
      cache_steps: number of timesteps in the cache.
      extra: if provided should be extra key-value input
        [batch, extra_timesteps, in_dim].
      extra_mask: if provided should be the mask for extra key-value input,
        [batch, extra_timesteps].

    Returns:
      output: loss and a dict containing metrics.
    """
    # [B, T, V]
    logits = self(inputs, mask=mask, is_training=is_training,
                  should_reset=should_reset, cache_steps=cache_steps,
                  extra=extra, extra_mask=extra_mask)

    metrics = sequence_prediction_metrics(logits, labels, mask)
    return metrics['loss'], metrics


def repeat_rows(a: jnp.ndarray, repeats: int, out_length: int) -> jnp.ndarray:
  """Repeat rows of input tensor a.

  Output is
    [a[0],
     a[0],
     ...
     a[0],  # A total of repeats[0] copies of a[0].
     a[1],
     a[1],
     ...,
     a[1],  # A total of repeats[1] copies of a[1].
     ...
     a[n-1]],  # A total of repeats[n-1] copies of a[n-1].

  Args:
    a: [n_rows, ...] input tensor.
    repeats: [n_rows] int tensor, the number of repeats for each row.
    out_length: number of rows in the output, it should be the same as
      sum(repeats), provided to be static for jit.

  Returns:
    out: [out_length, ...] output tensor.
  """
  a = jnp.asarray(a)
  n = a.shape[0]
  assert n == repeats.size
  chunk_start = jnp.cumsum(repeats)
  idx = jnp.sum(jnp.arange(out_length)[:, None] >= chunk_start[None, :],
                axis=-1)
  return a[idx]


def unpack_and_pad(
    packed: jnp.ndarray,
    split_sizes: jnp.ndarray,
    pad_size: int,
    pad_value: int = 0) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Unpack and pad tensors to a standard size.

  Args:
    packed: a [total_size, ...] tensor, which contains n individual tensors
      concatenated along the 0-th axis.
    split_sizes: size [n] int tensor, size of each individual tensor.
    pad_size: size for each split to pad to.
    pad_value: the value to use for padding.

  Returns:
    tensors: [n, pad_size, ...] tensor, tensors[i] is the i-th individual tensor
      padded to pad_size length.
    mask: [n, pad_size] mask tensor indicating which value is padded.
  """
  in_shape = list(packed.shape)
  total_size = in_shape[0]
  n_splits = split_sizes.shape[0]
  idx = jnp.arange(pad_size)
  masks = split_sizes[:, None] > idx[None, :]

  out_shape = in_shape[:]
  out_shape[0] = n_splits * pad_size
  out = jnp.full(out_shape, pad_value, dtype=packed.dtype)
  # Index for the rows of `packed`:
  # Define split_start[k] = sum_{i=0}^{k-1} split_sizes[i], which is the
  # starting index of split k.  So if split_start[k] <= i < split_start[k+1]
  # then index belongs to split k.  We therefore have:
  # idx[i] = k * pad_size + i - split_start[k]
  cumsum = jnp.concatenate([jnp.array([0], dtype=split_sizes.dtype),
                            jnp.cumsum(split_sizes)[:-1]])
  idx = jnp.arange(total_size)
  idx += repeat_rows(jnp.arange(n_splits), split_sizes, total_size) * pad_size
  idx -= repeat_rows(cumsum, split_sizes, total_size)
  out = out.at[idx].set(packed)
  out = out.reshape([n_splits, pad_size] + out_shape[1:])
  return out, masks


class Graph2TextTransformer(hk.Module):
  """A graph2text TransformerXL model.

  It embeds the graph with a simple graph neural network model, and passes the
  graph embeddings to the TransformerXL model, which are presented as the extra
  inputs to attend to in addition to the text embeddings inputs.
  """

  def __init__(self,
               *transformer_args,
               gnn_embed_dim: int = 128,
               gnn_num_layers: int = 5,
               gnn_layer_norm: bool = False,
               name: Optional[str] = None,
               **transformer_kwargs):
    """Constructor.

    Args:
      *transformer_args: args for the transformer module.
      gnn_embed_dim: node embedding size.
      gnn_num_layers: number of message passing layers to use.
      gnn_layer_norm: whether to use layer norm in the GNN.
      name: optional name for this module.
      **transformer_kwargs: kwargs for the transformer module.
    """
    super().__init__(name=name)
    self._transformer = TransformerXL(*transformer_args, **transformer_kwargs)
    self._gnn = GraphEmbeddingModel(
        embed_dim=gnn_embed_dim,
        num_layers=gnn_num_layers,
        use_layer_norm=gnn_layer_norm)

  def _encode_graphs(self,
                     graphs: jraph.GraphsTuple,
                     pad_n_nodes: Optional[int] = None,
                     padded: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Encode graphs so that it can be used in the transformer.

    Args:
      graphs: a graph structured using jraph.GraphsTuple.
      pad_n_nodes: size for each node to pad to.
      padded: Whether to pad each graph to the same number of nodes.

    Returns:
      tensors: unpacked and padded graph nodes.
      mask: mask tensor indicating which value is padded.
    """
    graphs = self._gnn(graphs)
    if pad_n_nodes is None:
      pad_n_nodes = graphs.n_node.max()
    out, mask = unpack_and_pad(graphs.nodes, graphs.n_node, pad_n_nodes)
    if padded:
      # Remove the padding graph from the batch
      return out[:-1], mask[:-1]
    else:
      return out, mask

  def __call__(self,
               graphs: jraph.GraphsTuple,
               pad_n_nodes: int,
               batch_padded: bool,
               *args, **kwargs):
    """Computes the outputs of the graph2text TransformerXL.

    Args:
      graphs: a graph structured using graph_net.Graph.
      pad_n_nodes: size for each node to pad to.
      batch_padded: whether the graph batch is padded or not.
      *args: args to the TransformerXL model.
      **kwargs: kwargs to the TransformerXL model.

    Returns:
      output: transformer output [batch, timesteps].
    """
    extra, extra_mask = self._encode_graphs(graphs, pad_n_nodes, batch_padded)
    return self._transformer(
        *args, extra=extra, extra_mask=extra_mask, **kwargs)

  def loss(self,
           graphs: jraph.GraphsTuple,
           pad_n_nodes: int,
           batch_padded: bool,
           inputs: jnp.ndarray,
           labels: jnp.ndarray,
           mask: jnp.ndarray,
           **kwargs):
    """Computes the loss of the graph2text TransformerXL.

    Args:
      graphs: a graph structured using graph_net.Graph.
      pad_n_nodes: size for each node to pad to.
      batch_padded: whether the graph batch is padded or not.
      inputs: [batch, timesteps].
      labels: [batch, timesteps].
      mask: [batch, timesteps].
      **kwargs: kwargs to the TransformerXL model.

    Returns:
      output: loss and a dict containing metrics.
    """
    extra, extra_mask = self._encode_graphs(graphs, pad_n_nodes, batch_padded)
    return self._transformer.loss(
        inputs, labels, mask, extra=extra, extra_mask=extra_mask, **kwargs)


class Bow2TextTransformer(hk.Module):
  """A bag-of-words to text TransformerXL model.

  This model embeds bag-of-words into vectors and the text transformer can then
  condition on these vectors to generate text.

  More specifically, the bow embedded vectors will be treated as extra tokens
  that the transformer can attend to, in addition to the text data it is already
  modelling.

  To make the model more expressive, we allow each bag-of-words to be embedded
  into potentially more than 1 vectors, and the transformer will treat them as
  more than 1 extra tokens correspondingly.
  """

  def __init__(self,
               *transformer_args,
               bow_embedding_dim: int = 256,
               bow_n_tokens: int = 1,
               name: Optional[str] = None,
               **transformer_kwargs):
    """Constructor.

    Args:
      *transformer_args: the TransformerXL constructor arguments.
      bow_embedding_dim: dimensionality for the bag-of-words embeddings.
      bow_n_tokens: number of extra tokens to create for the bag-of-words
        representations.
      name: optional name for this module.
      **transformer_kwargs: kwargs for the transformer module.
    """
    super().__init__(name=name)
    self._transformer = TransformerXL(*transformer_args, **transformer_kwargs)
    self._bow_embedding_dim = bow_embedding_dim
    self._bow_n_tokens = bow_n_tokens

  def _encode_bow(self, bow: jnp.ndarray) -> jnp.ndarray:
    """Encode the bag-of-words into tensors that can be used by the transormer.

    Args:
      bow: a [batch_size, bow_vocab_size] tensor, each row is a bow vector.

    Returns:
      embeddings: [batch_size, bow_n_tokens, bow_embedding_dim] tensor.
    """
    batch_size = bow.shape[0]
    bow = bow.astype(jnp.float32)

    # [B, D * n]
    embeddings = hk.Linear(self._bow_embedding_dim * self._bow_n_tokens)(bow)
    embeddings = transformer_block.layer_norm(jax.nn.gelu(embeddings))
    return jnp.reshape(
        embeddings, [batch_size, self._bow_n_tokens, self._bow_embedding_dim])

  def __call__(self, bow: jnp.ndarray, *args, **kwargs):
    """Compute the output of this bag-of-words-to-text transformer model.

    Args:
      bow: a [batch_size, bow_vocab_size] tensor, each row is a bow vector.
      *args: args to the TransformerXL model.
      **kwargs: kwargs to the TransformerXL model.

    Returns:
      output: transformer output [batch, timesteps].
    """
    return self._transformer(*args, extra=self._encode_bow(bow), **kwargs)

  def loss(self, bow: jnp.ndarray, *args, **kwargs):
    """Computes the loss of the graph2text TransformerXL.

    Args:
      bow: a [batch_size, bow_vocab_size] tensor, each row is a bow vector.
      *args: args to the TransformerXL model.
      **kwargs: kwargs to the TransformerXL model.

    Returns:
      output: loss and a dict containing metrics.
    """
    return self._transformer.loss(*args, extra=self._encode_bow(bow), **kwargs)
