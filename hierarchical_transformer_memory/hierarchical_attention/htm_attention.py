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

"""Haiku module implementing hierarchical attention over memory."""

from typing import Optional, NamedTuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


_EPSILON = 1e-3


class HierarchicalMemory(NamedTuple):
  """Structure of the hierarchical memory.

  Where 'B' is batch size, 'M' is number of memories, 'C' is chunk size, and 'D'
  is memory dimension.
  """
  keys: jnp.ndarray  # [B, M, D]
  contents: jnp.ndarray  # [B, M, C, D]
  steps_since_last_write: jnp.ndarray  # [B], steps since last memory write
  accumulator: jnp.ndarray  # [B, C, D], accumulates experiences before write


def sinusoid_position_encoding(
    sequence_length: int,
    hidden_size: int,
    min_timescale: float = 2.,
    max_timescale: float = 1e4,
) -> jnp.ndarray:
  """Creates sinusoidal encodings.

  Args:
    sequence_length: length [L] of sequence to be position encoded.
    hidden_size: dimension [D] of the positional encoding vectors.
    min_timescale: minimum timescale for the frequency.
    max_timescale: maximum timescale for the frequency.

  Returns:
    An array of shape [L, D]
  """
  freqs = np.arange(0, hidden_size, min_timescale)
  inv_freq = max_timescale**(-freqs / hidden_size)
  pos_seq = np.arange(sequence_length - 1, -1, -1.0)
  sinusoid_inp = np.einsum("i,j->ij", pos_seq, inv_freq)
  pos_emb = np.concatenate(
      [np.sin(sinusoid_inp), np.cos(sinusoid_inp)], axis=-1)
  return pos_emb


class HierarchicalMemoryAttention(hk.Module):
  """Multi-head attention over hierarchical memory."""

  def __init__(self,
               feature_size: int,
               k: int,
               num_heads: int = 1,
               memory_position_encoding: bool = True,
               init_scale: float = 2.,
               name: Optional[str] = None) -> None:
    """Constructor.

    Args:
      feature_size: size of feature dimension of attention-over-memories
        embedding.
      k: number of memories to sample.
      num_heads: number of attention heads.
      memory_position_encoding: whether to add positional encodings to memories
        during within memory attention.
      init_scale: scale factor for Variance weight initializers.
      name: module name.
    """
    super().__init__(name=name)
    self._size = feature_size
    self._k = k
    self._num_heads = num_heads
    self._weights = None
    self._memory_position_encoding = memory_position_encoding
    self._init_scale = init_scale

  @property
  def num_heads(self):
    return self._num_heads

  @hk.transparent
  def _singlehead_linear(self,
                         inputs: jnp.ndarray,
                         hidden_size: int,
                         name: str):
    linear = hk.Linear(
        hidden_size,
        with_bias=False,
        w_init=hk.initializers.VarianceScaling(scale=self._init_scale),
        name=name)
    out = linear(inputs)
    return out

  def __call__(
      self,
      queries: jnp.ndarray,
      hm_memory: HierarchicalMemory,
      hm_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Do hierarchical attention over the stored memories.

    Args:
       queries: Tensor [B, Q, E] Query(ies) in, for batch size B, query length
         Q, and embedding dimension E.
       hm_memory: Hierarchical Memory.
       hm_mask: Optional boolean mask tensor of shape [B, Q, M]. Where false,
         the corresponding query timepoints cannot attend to the corresponding
         memory chunks. This can be used for enforcing causal attention on the
         learner, not attending to memories from prior episodes, etc.

    Returns:
      Value updates for each query slot: [B, Q, D]
    """
    # some shape checks
    batch_size, query_length, _ = queries.shape
    (memory_batch_size, num_memories,
     memory_chunk_size, mem_embbedding_size) = hm_memory.contents.shape
    assert batch_size == memory_batch_size
    chex.assert_shape(hm_memory.keys,
                      (batch_size, num_memories, mem_embbedding_size))
    chex.assert_shape(hm_memory.accumulator,
                      (memory_batch_size, memory_chunk_size,
                       mem_embbedding_size))
    chex.assert_shape(hm_memory.steps_since_last_write,
                      (memory_batch_size,))
    if hm_mask is not None:
      chex.assert_type(hm_mask, bool)
      chex.assert_shape(hm_mask,
                        (batch_size, query_length, num_memories))
    query_head = self._singlehead_linear(queries, self._size, "query")
    key_head = self._singlehead_linear(
        jax.lax.stop_gradient(hm_memory.keys), self._size, "key")

    # What times in the input [t] attend to what times in the memories [T].
    logits = jnp.einsum("btd,bTd->btT", query_head, key_head)

    scaled_logits = logits / np.sqrt(self._size)

    # Mask last dimension, replacing invalid logits with large negative values.
    # This allows e.g. enforcing causal attention on learner, or blocking
    # attention across episodes
    if hm_mask is not None:
      masked_logits = jnp.where(hm_mask, scaled_logits, -1e6)
    else:
      masked_logits = scaled_logits

    # identify the top-k memories and their relevance weights
    top_k_logits, top_k_indices = jax.lax.top_k(masked_logits, self._k)
    weights = jax.nn.softmax(top_k_logits)

    # set up the within-memory attention
    assert self._size % self._num_heads == 0
    mha_key_size = self._size // self._num_heads
    attention_layer = hk.MultiHeadAttention(
        key_size=mha_key_size,
        model_size=self._size,
        num_heads=self._num_heads,
        w_init_scale=self._init_scale,
        name="within_mem_attn")

    # position encodings
    augmented_contents = hm_memory.contents
    if self._memory_position_encoding:
      position_embs = sinusoid_position_encoding(
          memory_chunk_size, mem_embbedding_size)
      augmented_contents += position_embs[None, None, :, :]

    def _within_memory_attention(sub_inputs, sub_memory_contents, sub_weights,
                                 sub_top_k_indices):
      top_k_contents = sub_memory_contents[sub_top_k_indices, :, :]

      # Now we go deeper, with another vmap over **tokens**, because each token
      # can each attend to different memories.
      def do_attention(sub_sub_inputs, sub_sub_top_k_contents):
        tiled_inputs = jnp.tile(sub_sub_inputs[None, None, :],
                                reps=(self._k, 1, 1))
        sub_attention_results = attention_layer(
            query=tiled_inputs,
            key=sub_sub_top_k_contents,
            value=sub_sub_top_k_contents)
        return sub_attention_results
      do_attention = jax.vmap(do_attention, in_axes=0)
      attention_results = do_attention(sub_inputs, top_k_contents)
      attention_results = jnp.squeeze(attention_results, axis=2)
      # Now collapse results across k memories
      attention_results = sub_weights[:, :, None] * attention_results
      attention_results = jnp.sum(attention_results, axis=1)
      return attention_results

    # vmap across batch
    batch_within_memory_attention = jax.vmap(_within_memory_attention,
                                             in_axes=0)
    outputs = batch_within_memory_attention(
        queries,
        jax.lax.stop_gradient(augmented_contents),
        weights,
        top_k_indices)

    return outputs
