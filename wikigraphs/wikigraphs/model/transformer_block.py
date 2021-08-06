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
"""Transformer blocks."""

import math
from typing import Callable, Optional

import haiku as hk
from haiku import initializers as init
import jax
import jax.numpy as jnp

from wikigraphs.model.embedding import RelativePositionEmbedding


def conv1d(x, num_units, init_scale=0.02, with_bias=True):
  return hk.Conv1D(
      output_channels=num_units, kernel_shape=1, with_bias=with_bias,
      w_init=init.RandomNormal(stddev=init_scale))(x)


def layer_norm(x):
  return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)


class FeedForwardBlock(hk.Module):
  """Feed forward block."""

  def __init__(self,
               dense_dim: int = 2100,
               dropout_prob: float = 0.1,
               init_scale: float = 1.,
               name: Optional[str] = None):
    """Initializes a FeedForwardBlock.

    Args:
      dense_dim: feature size of the feedforward block.
      dropout_prob: dropout probability.
      init_scale: the initialization scale of the VarianceScaling used for the
        feedforward layer.
      name: Optional name for this Haiku module.
    """
    super(FeedForwardBlock, self).__init__(name=name)
    self._dense_dim = dense_dim
    self._dropout_prob = dropout_prob
    self._init_scale = init_scale

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    hiddens = x.shape[-1]
    x = conv1d(x, num_units=self._dense_dim, init_scale=self._init_scale)
    x = jax.nn.relu(x)
    x = hk.dropout(hk.next_rng_key(), self._dropout_prob, x)
    x = conv1d(x, num_units=hiddens, init_scale=self._init_scale)
    return hk.dropout(hk.next_rng_key(), self._dropout_prob, x)


def get_reset_attention_mask(should_reset: jnp.ndarray) -> jnp.ndarray:
  """Maps a reset token vector into an attention mask that consists of blocks.

  A sequence of should reset tokens such as:
    [0, 1, 0, 1, 0, 0]
  transforms into an attention mask such as:
    [[1, 0, 0, 0, 0, 0],
     [0, 1, 1, 0, 0, 0],
     [0, 1, 1, 0, 0, 0],
     [0, 0, 0, 1, 1, 1],
     [0, 0, 0, 1, 1, 1],
     [0, 0, 0, 1, 1, 1]]
  Args:
    should_reset: Reset tokens with shape [batch, timesteps].
  Returns:
    attention_mask: Attention mask with shape [batch, timesteps, timesteps].
  """
  should_reset = jnp.cumsum(should_reset, axis=-1)
  attention_mask = should_reset[:, :, None] == should_reset[:, None, :]
  return attention_mask.astype(jnp.float32)


def attend(q: jnp.ndarray,
           k: jnp.ndarray,
           v: jnp.ndarray,
           mask: Optional[jnp.ndarray] = None,
           attend_fn:
           Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
           dropout_prob: float = 0.0,
           extra_k: Optional[jnp.ndarray] = None,
           extra_v: Optional[jnp.ndarray] = None,
           extra_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Computes multi-head attention using the given query, key and value.

  Args:
    q: Query with shape [batch, q_timesteps, num_heads, head_dim].
    k: Key with shape [batch, timesteps, num_heads, head_dim].
    v: Value with shape [batch, timesteps, num_heads, head_dim].
    mask: Attention mask to apply [batch, 1, q_timesteps, timesteps].
    attend_fn: An optionally defined attend function. The default attend_fn is
      is jnp.einsum('bthd,bThd->bhtT', q, k).
    dropout_prob: dropout probability on the attention weights.
    extra_k: Extra keys to attend to, if provided.  Note the extra keys and
      values do not apply the specified attention_fn, but instead use the
      default dot-product attention. [batch, timesteps_extra, num_heads,
      head_dim].
    extra_v: Extra values to attend to, if provided. [batch, timesteps_extra,
      num_heads, head_dim].
    extra_mask: Extra attention mask to apply on the extra inputs [batch, 1,
      q_timesteps, timesteps_extra].

  Returns:
    Output of the attention with shape [batch, timesteps, hiddens]
  """
  infinity_proxy = 1e9
  batch, q_time, num_heads, head_dim = q.shape
  hiddens = num_heads * head_dim

  _, kv_time, _, _ = k.shape
  expected_kv_shape = (batch, kv_time, num_heads, head_dim)

  if k.shape != expected_kv_shape:
    raise ValueError(
        f'Expected key shape {expected_kv_shape} but got shape {k.shape}')
  if v.shape != expected_kv_shape:
    raise ValueError(
        f'Expected value shape {expected_kv_shape} but got shape {v.shape}')

  if attend_fn is not None:
    attention = attend_fn(q, k)
  else:
    attention = jnp.einsum('bthd,bThd->bhtT', q, k)

  if mask is not None:
    attention = attention * mask - infinity_proxy * (1 - mask)

  if extra_k is not None and extra_v is not None:
    extra_time = extra_k.shape[1]
    expected_extra_shape = (batch, extra_time, num_heads, head_dim)
    if extra_k.shape != expected_extra_shape:
      raise ValueError(
          f'Expected extra key shape {expected_extra_shape} but got'
          f' {extra_k.shape}')
    if extra_v.shape != expected_extra_shape:
      raise ValueError(
          f'Expected extra value shape {expected_extra_shape} but got'
          f' {extra_v.shape}')

    # [B, H, t, T']
    extra_attention = jnp.einsum('bthd,bThd->bhtT', q, extra_k)
    if extra_mask is not None:
      extra_attention = extra_attention * extra_mask - infinity_proxy * (
          1 - extra_mask)

    # [B, H, t, T+T']
    attention = jnp.concatenate([attention, extra_attention], axis=-1)
    # [B, T+T', H, D]
    v = jnp.concatenate([v, extra_v], axis=1)

  scale = 1. / math.sqrt(head_dim)
  attention *= scale
  normalized = jax.nn.softmax(attention)
  if dropout_prob > 0:
    normalized = hk.dropout(hk.next_rng_key(), dropout_prob, normalized)
  summed = jnp.einsum('bhtT,bThd->bthd', normalized, v)
  return jnp.reshape(summed, [batch, q_time, hiddens])


class Attention(hk.Module):
  """Attention with memory (https://arxiv.org/abs/1901.02860).

  This implementation leverages the `state` in Haiku, in which the inputs are
  stored as `states`. At each step, these states in memory are updated with a
  rolling window.
  """

  def __init__(self,
               r_w_bias: Optional[jnp.ndarray] = None,
               r_r_bias: Optional[jnp.ndarray] = None,
               num_heads: int = 8,
               init_scale: float = 1.0,
               with_final_bias: bool = False,
               final_init_scale_multiplier: float = 1.,
               relative_pos_clamp_len: Optional[int] = None,
               dropout_prob: float = 0.0,
               name: Optional[str] = None):
    """Initializes a Attention module.

    Args:
      r_w_bias: global content bias.
      r_r_bias: global positional bias.
      num_heads: number of attention heads.
      init_scale: the initialization scale of the VarianceScaling used for the
        linear layer.
      with_final_bias: whether to let final layer have biases.
      final_init_scale_multiplier: how much to scale the initialization scale of
        the output layer.
      relative_pos_clamp_len: clamp length of the relative position embeddings.
      dropout_prob: dropout probability.
      name: Optional name for this Haiku module.
    """
    super(Attention, self).__init__(name=name)
    self._r_w_bias = r_w_bias
    self._r_r_bias = r_r_bias
    self._num_heads = num_heads
    self._init_scale = init_scale
    self._with_final_bias = with_final_bias
    self._final_init_scale = final_init_scale_multiplier * init_scale
    self._relative_pos_clamp_len = relative_pos_clamp_len
    self._dropout_prob = dropout_prob

  def _update_cache(self,
                    key: jnp.ndarray,
                    value: jnp.ndarray,
                    cache_steps: Optional[int] = None,
                    axis: int = 1) -> jnp.ndarray:
    """Update the cache stored in hk.state."""
    cache_shape = list(value.shape)
    value_steps = cache_shape[axis]
    if cache_steps is not None:
      cache_shape[axis] += cache_steps
    cache = hk.get_state(
        key, shape=cache_shape, dtype=value.dtype, init=jnp.zeros)

    # Overwrite at index 0, then rotate timesteps left so what was just
    # inserted is first.
    value = jax.lax.dynamic_update_slice(
        cache, value, jnp.zeros(len(cache_shape), dtype=jnp.int32))
    value = jnp.roll(value, -value_steps, axis)
    hk.set_state(key, value)
    return value

  def _update_memory(self,
                     mem: jnp.ndarray,
                     mask: jnp.ndarray,
                     input_length: int,
                     cache_steps: int,
                     should_reset: jnp.ndarray) -> jnp.ndarray:
    """Logic for using and updating cached activations."""
    batch_size = mem.shape[0]
    if cache_steps > 0:
      # Tells us how much of the cache should be used.
      cache_progress_idx = hk.get_state(
          'cache_progress_idx', [batch_size], dtype=jnp.int32, init=jnp.zeros)
      hk.set_state('cache_progress_idx', cache_progress_idx + input_length)
      mem = self._update_cache('mem', mem, cache_steps=cache_steps)
      if mask is None:
        mask = jnp.ones((batch_size, 1, input_length, input_length))
      cache_mask = (jnp.arange(cache_steps - 1, -1, -1)[None, None, None, :]
                    < cache_progress_idx[:, None, None, None])
      cache_mask = jnp.broadcast_to(
          cache_mask, (batch_size, 1, input_length, cache_steps))
      mask = jnp.concatenate([cache_mask, mask], axis=-1)
    if should_reset is not None:
      if cache_steps > 0:
        should_reset = self._update_cache('should_reset', should_reset,
                                          cache_steps=cache_steps)
      reset_mask = get_reset_attention_mask(should_reset)[:, None, :, :]
      mask *= reset_mask[:, :, cache_steps:, :]
    return mem, mask

  def __call__(self,
               x: jnp.ndarray,
               mask: Optional[jnp.ndarray] = None,
               should_reset: Optional[jnp.ndarray] = None,
               cache_steps: int = 0,
               extra: Optional[jnp.ndarray] = None,
               extra_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Compute the multi-head attention.

    Args:
      x: input [batch, x_timesteps, in_dim].
      mask: attention mask [batch, 1, x_timesteps, y_timesteps].
      should_reset: reset marker [batch, timesteps].
      cache_steps: number of timesteps in the cache.
      extra: if provided should be extra key-value input
        [batch, extra_timesteps, in_dim'].
      extra_mask: if provided should be the mask for extra key-value input,
        [batch, extra_timesteps].

    Returns:
      output: attention output [batch, x_timesteps, in_dim].
    """
    hiddens_in = x.shape[-1]
    steps = x.shape[1]
    qkv_hiddens = hiddens_in

    y, mask = self._update_memory(x, mask, steps, cache_steps, should_reset)

    q = conv1d(x, qkv_hiddens, init_scale=self._init_scale, with_bias=False)
    k = conv1d(y, qkv_hiddens, init_scale=self._init_scale, with_bias=False)
    v = conv1d(y, qkv_hiddens, init_scale=self._init_scale, with_bias=False)

    batch, q_time, _ = q.shape
    _, kv_time, _ = k.shape
    head_dim = qkv_hiddens // self._num_heads
    assert qkv_hiddens % self._num_heads == 0, 'Head dim should be an integer.'
    q = jnp.reshape(q, [batch, q_time, self._num_heads, head_dim])
    k = jnp.reshape(k, [batch, kv_time, self._num_heads, head_dim])
    v = jnp.reshape(v, [batch, kv_time, self._num_heads, head_dim])

    attend_fn = RelativePositionEmbedding(
        dim=qkv_hiddens, dropout_rate=self._dropout_prob,
        r_w_bias=self._r_w_bias, r_r_bias=self._r_r_bias,
        init_scale=self._init_scale, clamp_len=self._relative_pos_clamp_len)

    if extra is not None:
      extra_k = conv1d(extra, qkv_hiddens, init_scale=self._init_scale,
                       with_bias=False)
      extra_v = conv1d(extra, qkv_hiddens, init_scale=self._init_scale,
                       with_bias=False)
      extra_time = extra.shape[1]
      extra_k = jnp.reshape(
          extra_k, [batch, extra_time, self._num_heads, head_dim])
      extra_v = jnp.reshape(
          extra_v, [batch, extra_time, self._num_heads, head_dim])
      if extra_mask is not None:
        extra_mask = extra_mask[:, None, None, :]
      attn_vec = attend(q, k, v, mask=mask, attend_fn=attend_fn,
                        dropout_prob=self._dropout_prob,
                        extra_k=extra_k, extra_v=extra_v, extra_mask=extra_mask)
    else:
      attn_vec = attend(q, k, v, mask=mask, attend_fn=attend_fn,
                        dropout_prob=self._dropout_prob)
    attn_out = conv1d(attn_vec, hiddens_in, with_bias=self._with_final_bias,
                      init_scale=self._final_init_scale)
    return hk.dropout(hk.next_rng_key(), self._dropout_prob, attn_out)


class SelfAttentionBlock(hk.Module):
  """Self attention block."""

  def __init__(self,
               r_w_bias: Optional[jnp.ndarray] = None,
               r_r_bias: Optional[jnp.ndarray] = None,
               causal: bool = False,
               num_heads: int = 8,
               dropout_prob: float = 0.1,
               dropout_attn_prob: float = 0.0,
               init_scale: float = 1.0,
               relative_pos_clamp_len: Optional[int] = None,
               name: Optional[str] = None):
    """Initializes a SelfAttentionBlock.

    Args:
      r_w_bias: global content bias.
      r_r_bias: global positional bias.
      causal: whether to apply a causal mask to the input.
      num_heads: number of attention heads.
      dropout_prob: dropout probability.
      dropout_attn_prob: dropout probability of the attention module.
      init_scale: the initialization scale of the VarianceScaling used for the
        linear layer.
      relative_pos_clamp_len: clamp length of the relative position embeddings.
      name: Optional name for this Haiku module.
    """
    super(SelfAttentionBlock, self).__init__(name=name)
    self._r_w_bias = r_w_bias
    self._r_r_bias = r_r_bias
    self._causal = causal
    self._num_heads = num_heads
    self._dropout_prob = dropout_prob
    self._dropout_attn_prob = dropout_attn_prob
    self._init_scale = init_scale

    self._relative_pos_clamp_len = relative_pos_clamp_len

  def __call__(self,
               x: jnp.ndarray,
               mask: Optional[jnp.ndarray] = None,
               should_reset: Optional[jnp.ndarray] = None,
               cache_steps: int = 0,
               extra: Optional[jnp.ndarray] = None,
               extra_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Computes the outputs of the self attention block.

    Args:
      x: query input [batch, x_timesteps, in_dim].
      mask: attention mask [batch, 1, 1, x_timesteps].
      should_reset: reset marker [batch, timesteps].
      cache_steps: number of timesteps in the cache.
      extra: if provided should be extra key-value input
        [batch, extra_timesteps, in_dim'].
      extra_mask: if provided should be the mask for extra key-value input,
        [batch, extra_timesteps].

    Returns:
      output: block output [batch, x_timesteps, in_dim].
    """
    if self._causal:
      timesteps = x.shape[1]
      batch_size = x.shape[0]
      t = jnp.arange(timesteps, dtype=jnp.int32)
      causal_mask = (t[:, None] >= t[None, :])[None, None, :, :]
      causal_mask = causal_mask.astype(x.dtype)
      if mask is None:
        mask = jnp.broadcast_to(
            causal_mask, (batch_size, 1, timesteps, timesteps))
      else:
        mask *= causal_mask
      x = Attention(
          self._r_w_bias,
          self._r_r_bias,
          num_heads=self._num_heads,
          init_scale=self._init_scale,
          relative_pos_clamp_len=self._relative_pos_clamp_len,
          dropout_prob=self._dropout_attn_prob)(
              x, mask=mask, should_reset=should_reset,
              cache_steps=cache_steps, extra=extra, extra_mask=extra_mask)
    else:
      x = Attention(
          self._r_w_bias,
          self._r_r_bias,
          num_heads=self._num_heads,
          init_scale=self._init_scale,
          dropout_prob=self._dropout_attn_prob)(
              x, mask=mask, extra=extra, extra_mask=extra_mask)
    return hk.dropout(hk.next_rng_key(), self._dropout_prob, x)


class GPT2Block(hk.Module):
  """GPT-2 style transformer block with memory."""

  def __init__(self,
               r_w_bias: Optional[jnp.ndarray] = None,
               r_r_bias: Optional[jnp.ndarray] = None,
               causal: bool = True,
               dense_dim: int = 2100,
               dropout_prob: float = 0.1,
               dropout_attn_prob: float = 0.0,
               num_heads: int = 8,
               self_att_init_scale: float = 0.02,
               dense_init_scale: float = 0.02,
               relative_pos_clamp_len: Optional[int] = None,
               name: Optional[str] = None):
    """Initializes a GPT2Block.

    Args:
      r_w_bias: global content bias.
      r_r_bias: global positional bias.
      causal: whether to apply a causal mask to the input.
      dense_dim: feature size of the feedforward block.
      dropout_prob: dropout probability.
      dropout_attn_prob: dropout probability of the attention module.
      num_heads: number of attention heads.
      self_att_init_scale: the initialization scale of the VarianceScaling
        used for the linear layer in the attention module.
      dense_init_scale: the initialization scale of the VarianceScaling
        used for the linear layer in the feedforward module.
      relative_pos_clamp_len: clamp length of the relative position embeddings.
      name: Optional name for this Haiku module.
    """
    super(GPT2Block, self).__init__(name=name)
    self._r_w_bias = r_w_bias
    self._r_r_bias = r_r_bias
    self._causal = causal
    self._dense_dim = dense_dim
    self._dropout_prob = dropout_prob
    self._dropout_attn_prob = dropout_attn_prob
    self._num_heads = num_heads
    self._self_att_init_scale = self_att_init_scale
    self._dense_init_scale = dense_init_scale
    self._relative_pos_clamp_len = relative_pos_clamp_len

  def __call__(self,
               x: jnp.ndarray,
               mask: Optional[jnp.ndarray] = None,
               is_training: bool = True,
               should_reset: Optional[jnp.ndarray] = None,
               cache_steps: int = 0,
               extra: Optional[jnp.ndarray] = None,
               extra_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Computes the outputs of the GPT-2 block.

    Args:
      x: query input [batch, x_timesteps, in_dim].
      mask: attention mask [batch, 1, 1, x_timesteps].
      is_training: whether the current stage is training or not.
      should_reset: reset marker [batch, timesteps].
      cache_steps: number of timesteps in the cache.
      extra: if provided should be extra key-value input
        [batch, extra_timesteps, in_dim'].
      extra_mask: if provided should be the mask for extra key-value input,
        [batch, extra_timesteps].

    Returns:
      output: block output [batch, x_timesteps, in_dim].
    """
    dropout_prob = self._dropout_prob if is_training else 0.0
    dropout_attn_prob = self._dropout_attn_prob if is_training else 0.0
    x = layer_norm(x + SelfAttentionBlock(
        self._r_w_bias,
        self._r_r_bias,
        causal=self._causal,
        num_heads=self._num_heads,
        dropout_prob=dropout_prob,
        dropout_attn_prob=dropout_attn_prob,
        init_scale=self._self_att_init_scale,
        relative_pos_clamp_len=self._relative_pos_clamp_len)(
            x, mask=mask, should_reset=should_reset,
            cache_steps=cache_steps, extra=extra, extra_mask=extra_mask))
    x = layer_norm(x + FeedForwardBlock(
        dense_dim=self._dense_dim,
        dropout_prob=dropout_prob,
        init_scale=self._dense_init_scale)(x))
    return x
