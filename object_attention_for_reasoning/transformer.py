# Fork of Sonnet transformer model with small modifications
#
# Copyright 2017 The Sonnet Authors. All Rights Reserved.
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
"""Implementation of Transformer networks.

Size glossary:
  * Batch size (B).
  * Sequence length (N).
  * Memory size (M). The size of the optional memory, passed in via `state`.
  * Number of heads (H): the number of attention heads.
  * Value size (V): the size of each value embedding per head.
  * Key size (K): the size of each key embedding per head. Equally, the size
      of each query embedding per head. Typically K <= V.
  * Embedding size (HV). The size of the activation or embedding relating to
      each input between layers. Equal to value_size * num_heads.
  * All attention size (F). The size of all attention activations over every
      head.
  * QKV size (F / H): The size of the query, key and value per head. Equal to
      2K + V or equivalently F / H.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
from sonnet.python.modules import base
from sonnet.python.modules import basic
from sonnet.python.modules import layer_norm as snt_ln
from sonnet.python.modules import util
from sonnet.python.modules.nets import mlp as snt_mlp
import tensorflow.compat.v1 as tf

AttentionState = collections.namedtuple('AttentionState',
                                        ('queries', 'keys', 'values', 'logits',
                                         'weights', 'embeddings', 'read_words'))

CompressedMemoryState = collections.namedtuple(
    'CompressedMemoryState', ('episodic_memory', 'compressed_memory', 'index'))


def rel_shift(position_logits):
  """Shifting of logits for relative attention.

  Args:
    position_logits: A tensor of shape [B, H, N, N + M].

  Returns:
    The shifted logits. Example, for input (H=1, B=1):
      [5, 4, 3, 2, 1]
      [5, 4, 3, 2, 1]
      [5, 4, 3, 2, 1]
      [5, 4, 3, 2, 1]
      [5, 4, 3, 2, 1]

    the function outputs:
      [1, 0, 5, 4, 3]
      [2, 1, 0, 5, 4]
      [3, 2, 1, 0, 5]
      [4, 3, 2, 1, 0]
      [5, 4, 3, 2, 1]

  Raises:
    ValueError if position_logits is not 4D.

  Note: this is not an exact shift as the upper triangle is non-zero. This
  works as intended in the causally-masked case. If this is used with un-masked
  attention, we'd want these to also be zero.
  """
  if position_logits.get_shape().ndims != 4:
    raise ValueError('Expected 4D position logits.')

  input_shape = position_logits.shape
  batch_size = input_shape[0]
  num_heads = input_shape[1]
  t1 = input_shape[2]
  t2 = input_shape[3]
  # We prepend zeros on the final timescale dimension.
  to_pad = tf.zeros([batch_size, num_heads, t1, 1])
  position_logits = tf.concat([to_pad, position_logits], -1)
  # Reshape trick to shift input.
  position_logits = tf.reshape(position_logits,
                               [batch_size, num_heads, t2 + 1, t1])
  # Remove extra time dimension and re-shape.
  position_logits = position_logits[:, :, 1:]
  position_logits = tf.reshape(position_logits, input_shape)
  return position_logits


def _layer_norm(inputs):
  if inputs.get_shape().ndims > 2:
    return basic.BatchApply(snt_ln.LayerNorm())(inputs)
  else:
    return snt_ln.LayerNorm()(inputs)


def _concat_and_slice(prev_memory, new_memory):
  original_memory_size = prev_memory.get_shape().as_list()[1]
  concat_memory = tf.concat([prev_memory, new_memory], 1)
  memory = concat_memory[:, -original_memory_size:]
  return memory, concat_memory


def simple_attention(queries, keys, values):
  logits = tf.matmul(queries, keys, transpose_b=True)
  weights = tf.nn.softmax(logits)
  return tf.matmul(weights, values)


class ResidualDropoutWrapper(base.AbstractModule):
  """Wrapper class that applies residual connections, dropout and layer norm.

  By default applies a relu to the module output before the other operations.
  """

  def __init__(self,
               layer,
               dropout_rate,
               layer_norm='input',
               name='residual_dropout_wrapper'):
    self._module = layer
    self._dropout_rate = dropout_rate
    self._layer_norm = layer_norm
    super(ResidualDropoutWrapper, self).__init__(name=name)

  def _build(self, inputs, *args, **kwargs):
    if self._layer_norm in ('both', 'input'):
      normed_inputs = _layer_norm(inputs)
    else:
      normed_inputs = inputs
    module_output = self._module(normed_inputs, *args, **kwargs)
    module_state = None
    # If module outputs multiple items, assumes (output, state) tuple.
    if isinstance(module_output, tuple):
      module_output, module_state = module_output
    if kwargs['is_training']:  # kwargs must contain is_training.
      module_output = tf.nn.dropout(module_output, rate=self._dropout_rate)
    output = inputs + module_output
    if self._layer_norm in ('both', 'output'):
      output = _layer_norm(output)
    if module_state is None:
      return output
    else:
      return output, module_state


def future_mask(chunk_size, dtype):
  """Creates attention mask to ensure an element i cannot attend to j > i."""
  square = tf.ones([chunk_size, chunk_size], dtype=dtype)
  # Create upper diagonal matrix and remove diagonal entries (allow self-attn).
  mask = tf.matrix_band_part(square, 0, -1) - tf.matrix_band_part(square, 0, 0)
  # Multiply by -1e6 and expand to broadcast with [B, H, N, N] logits.
  mask = -1e6 * tf.reshape(mask, [1, 1, chunk_size, chunk_size])
  return mask


def _memory_size(state):
  if isinstance(state, CompressedMemoryState):
    return (state.episodic_memory.get_shape().as_list()[1] +
            state.compressed_memory.get_shape().as_list()[1])
  else:
    return state.get_shape().as_list()[1]


def create_mask(inputs, state, equal_window):
  """Creates mask for future sequence positions.

  Args:
    inputs: inputs tensor of shape [B, N, D]
    state: optional tensor of shape [B, M, D], CompressedMemoryState or a list
      where the ith entry corresponds to the ith layer's state.
    equal_window: if True, then each activation has an equally-sized attention
      window of length 'M'. This only makes sense if a state is given.

  Returns:
    Float tensor of shape [1, 1, N, N + M], to be summed with logits.
  """
  chunk_size = inputs.get_shape().as_list()[1]
  dtype = inputs.dtype
  mask = future_mask(chunk_size, dtype)
  if state is not None:
    if isinstance(state, (tuple, list)):
      largest_memory_layer = np.argmax([_memory_size(s) for s in state])
      state = state[largest_memory_layer]
    mem_size = _memory_size(state)
    mask = tf.concat(
        [tf.zeros([1, 1, chunk_size, mem_size], dtype=dtype), mask], 3)

  if equal_window:
    attn_mask = tf.ones([chunk_size, chunk_size], dtype=dtype)
    mask_dia = tf.cast(tf.matrix_band_part(attn_mask, 0, 0), dtype=dtype)
    mask_l = tf.cast(tf.matrix_band_part(attn_mask, -1, 0), dtype=dtype)
    start_mask = tf.reshape(mask_l - mask_dia,
                            [1, 1, chunk_size, chunk_size]) * -1e6
    mask = tf.concat(
        [mask[:, :, :, :chunk_size] + start_mask, mask[:, :, :, chunk_size:]],
        3)
  return mask


def default_mlp(hidden_sizes, activate_final=False, init_std=2., **kwargs):
  """Standard batch-applied MLP for transformer modules."""
  init = {'w': tf.variance_scaling_initializer(init_std, distribution='normal')}
  mlp = snt_mlp.MLP(
      hidden_sizes,
      activate_final=activate_final,
      use_dropout=True,
      initializers=init,
      **kwargs)
  return basic.BatchApply(mlp)


def get_position_encodings(sequence_length,
                           hidden_size,
                           clamp_value,
                           max_timescale=10000.,
                           min_timescale=2.0):
  """Creates sinusoidal encodings of shape [1, N + M, D]."""
  # NOTE: when not using relative position encodings, min_timescale must be 2.0
  # and hidden_size must be an even number. Otherwise, the dimensions do not
  # match.
  pos_seq = tf.range(sequence_length - 1, -1, -1.0)
  if clamp_value > 0:
    pos_seq = tf.minimum(pos_seq, clamp_value)
  freqs = tf.range(0, hidden_size, min_timescale)
  inv_freq = 1 / (max_timescale**(freqs / hidden_size))
  sinusoid_inp = tf.einsum('i,j->ij', pos_seq, inv_freq)
  pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
  pos_emb = tf.expand_dims(pos_emb, 0)

  output_dim = pos_emb.get_shape().as_list()[-1]
  if output_dim != hidden_size:
    raise ValueError(
        'position embedding dimension ({}) does not match that of the input ({}).'
        .format(output_dim, hidden_size))
  return pos_emb


class MultiheadAttention(base.AbstractModule):
  """Implements multi-head attention with optional state context."""

  def __init__(self,
               value_size,
               key_size,
               num_heads,
               mask=None,
               scaling=True,
               positional_encodings=None,
               use_relative_positions=False,
               init_std=2.,
               name='multihead_attention'):
    """Creates a MultiheadAttention module.

    Args:
      value_size: V parameter. See size glossary in class docstring.
      key_size: K parameter. See size glossary in class docstring.
      num_heads: The number of independent queries per timestep.
      mask: Optional mask to attention logits. This can prevent attending to
        future positions or unused memory slots.
      scaling: Whether to scale the attention logits.
      positional_encodings: Either None (none given), or an iterable of
        `(key_positional_encodings, query_positional_encodings)` tuples, where
        the first encodings in the list indicate the oldest entries in memory
        and the final encodings indicate the newest entries in memory and the
        sequence.
      use_relative_positions: If True then relative positions are incorporated,
        vs absolute, into the attention logits. This is done exactly as
        described in the TransformerXL, Dai et al. 2019.
      init_std: scaling of standard deviation for weight matrices init.
      name: Name of module.
    """

    super(MultiheadAttention, self).__init__(name=name)
    self._value_size = value_size
    self._key_size = key_size
    self._sizes = {
        'value': self._value_size,
        'key': self._key_size,
        'query': self._key_size,
        'relative_keys': self._key_size,
        'relative_keys_0': self._key_size,
    }
    self._num_heads = num_heads
    self._mask = mask
    self._scaling = scaling
    self._positional_encodings = positional_encodings
    self._use_relative_positions = use_relative_positions
    self._init = {'w': tf.variance_scaling_initializer(init_std)}

  @util.reuse_variables
  def multihead_linear(self, inputs, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      hidden_size = self._sizes[name]
      input_size = inputs.shape[-1].value
      w = tf.get_variable(
          'linear/w',
          shape=[input_size, self._num_heads * hidden_size],
          initializer=self._init['w'])
      w = tf.reshape(w, [input_size, self._num_heads, hidden_size])
      out = tf.einsum('bij,jhk->bhik', inputs, w)
      return out

  def _build(self,
             inputs,
             query_inputs=None,
             state=None,
             is_training=False,
             dropout_keep_prob=0.5,
             key_value_inputs=None):
    """Calculates multi-layer self attention.

    Args:
      inputs: Tensor of shape [batch_size, num_steps, output_dim_size]. Inputs
        used as the query, key, and value to the attention layer.
      query_inputs: optional Tensor of shape [batch_size, num_steps,
        output_dim_size]. Query inputs to the attention layer. Set when
        query_inputs is different from the inputs argument.
      state: optional CompressedMemoryState or a Tensor of shape [batch_size,
        memory_size, dim_size] concatenated to the inputs. Set when attend to
        the memory from previous steps.
      is_training: if currently training.
      dropout_keep_prob: dropout rate applied to attention weights.
      key_value_inputs: optional Tensor of shape [batch_size, num_steps,
        output_dim_size]. It is used as the key and value of the multihead
        attention. Set when the key and value are different from the inputs
        argument.

    Returns:
      output: the result Tensor of shape
        [batch_size, num_steps, output_dim_size].
      attention_state: named tuple of AttentionState.
    """
    if key_value_inputs is not None and state is not None:
      raise ValueError('Only one of the key_value_input and state is needed.')
    embedding_size = self._value_size * self._num_heads

    q_inputs = inputs if query_inputs is None else query_inputs
    # Denoted by L. If query_inputs is None, L = N.
    _, query_size = q_inputs.get_shape().as_list()[:2]

    if key_value_inputs is not None:
      k_inputs = key_value_inputs
      v_inputs = k_inputs
    elif state is not None:
      if isinstance(state, CompressedMemoryState):
        state_memory_list = [state.compressed_memory, state.episodic_memory]
      else:
        state_memory_list = [state]

      k_inputs = tf.concat(state_memory_list + [inputs], 1)
      v_inputs = k_inputs
    else:
      k_inputs = inputs
      v_inputs = inputs

    # Batch size denoted by B
    batch_size = tf.shape(inputs)[0]
    # Chunk_size denoted by N
    chunk_size = inputs.get_shape().as_list()[1]
    # Denoted by N + M
    att_size = k_inputs.get_shape().as_list()[1]

    if self._positional_encodings and not self._use_relative_positions:
      if len(self._positional_encodings) != 1:
        raise ValueError(
            'Absolute positional encodings only supported for 1 memory. '
            'Found %i.' % len(self._positional_encodings))
      key_positions, query_positions = self._positional_encodings[0]
      k_inputs += key_positions
      q_inputs += query_positions

    # [B, H, L, K]
    q = self.multihead_linear(q_inputs, 'query')
    # [B, H, N + M, K]
    k = self.multihead_linear(k_inputs, 'key')
    # [B, H, N + M, V]
    v = self.multihead_linear(v_inputs, 'value')

    # Scaling the dot-product
    if self._scaling:
      q *= self._key_size**-0.5

    # [B, H, L, N + M]
    if self._use_relative_positions:
      r_w_bias = tf.get_variable(
          'r_w_bias', [1, self._num_heads, 1, self._key_size],
          dtype=inputs.dtype)
      content_logits = tf.matmul(q + r_w_bias, k, transpose_b=True)
      all_relative_logits = []
      # Loop over multiple positional encodings, for the case of multiple
      # memory types.
      for i, positional_encodings in enumerate(self._positional_encodings):
        key_positions, query_positions = positional_encodings
        if key_positions.get_shape().as_list()[-1] != att_size:
          key_positions = key_positions[:, -att_size:]  # Crop to layer mem size
        is_final = i == len(self._positional_encodings) - 1
        suffix = '' if is_final else '_%d' % i
        relative_keys = self.multihead_linear(
            key_positions, name='relative_keys' + suffix)
        # [B, H, N, D]
        r_r_bias = tf.get_variable(
            'r_r_bias' + suffix, [1, self._num_heads, 1, self._key_size],
            dtype=inputs.dtype)
        relative_keys = tf.tile(relative_keys, [batch_size, 1, 1, 1])
        relative_logits = tf.matmul(
            q + r_r_bias, relative_keys, transpose_b=True)
        relative_logits = rel_shift(relative_logits)
        if not is_final:  # Include relative positions for input sequence.
          relative_logits = relative_logits[:, :, :, :-chunk_size]
        all_relative_logits.append(relative_logits)
      all_relative_logits = tf.concat(all_relative_logits, 3)
      logits = content_logits + all_relative_logits
    else:
      # [B, H, N, N + M]
      logits = tf.matmul(q, k, transpose_b=True)
      content_logits = logits

    if self._mask is not None:
      if self._mask.get_shape().as_list()[-1] != att_size:
        mask = self._mask[:, :, :, -att_size:]
      else:
        mask = self._mask
      logits += mask

    weights = tf.nn.softmax(logits)
    if is_training:
      weights = tf.nn.dropout(weights, dropout_keep_prob)
    # [B, L, H, V], where V is value_size
    output_transpose = tf.einsum('bhij,bhjk->bihk', weights, v)

    # [B, L, H, V] -> [B, L, HV]
    attended_inputs = basic.BatchReshape([query_size, embedding_size])(
        output_transpose)
    # Apply final mlp to mix information between heads.
    output = basic.BatchApply(basic.Linear(embedding_size))(attended_inputs)

    attention_state = AttentionState(
        queries=q,
        keys=k,
        values=v,
        weights=weights,
        logits=content_logits,
        embeddings=inputs,
        read_words=output)
    return output, attention_state


class TransformerTower(base.AbstractModule):
  """Transformer tower.

  Deep residual network using blocks of attention and MLPs, specified in
  Vaswani et al. 2017.
  """

  def __init__(self,
               value_size,
               num_heads,
               num_layers,
               causal=True,
               key_size=None,
               shared_attention=False,
               output_size=None,
               mlp_hidden_sizes=tuple([1024]),
               dropout_rate=0.1,
               use_relative_positions=True,
               clamp_time_range=0,
               same_attention_length=False,
               layer_norm='input',
               name='transformer_tower'):
    """Initializes TransformerTower.

    Args:
      value_size: dimensionality of values per-head.
      num_heads: number of attention heads.
      num_layers: number of transformer blocks, where each block contains a
        multi-head attention layer and an MLP.
      causal: if True, applies a causal mask.
      key_size: optional dimensionality of key size. If unspecified then it is
        set to `value_size`.
      shared_attention: if True, attention params are shared across all layers.
      output_size: if set, the desired output dimensionality. By default the
        output size is `value_size` x `num_heads`.
      mlp_hidden_sizes: tuple containing dimensionality of mlp layer(s). If
        multiple values are specified, the mlp contains multiple layers for each
        transformer block.
      dropout_rate: dropout rate applied to hidden activations, attention, and
        positional encodings.
      use_relative_positions: if False, applies absolute positional encodings.
        If true, uses relative positional encodings from Dai et al. 2019.
      clamp_time_range: clamps max temporal positional encoding if specified.
      same_attention_length: if True, attention is masked to ensure each
        position in the sequence contains the same length of attention.
      layer_norm: Where to apply layer-norm in Transformer block. Can be one of
        'input' (Vaswani et al. 2017), 'output', or 'both'.
      name: name of variable scope.
    """
    super(TransformerTower, self).__init__(name=name)
    self._causal = causal
    self._mask = None

    if key_size is None:
      key_size = value_size
    self._key_size = key_size
    self._value_size = value_size
    self._shared_attention = shared_attention
    self._num_heads = num_heads
    self._num_layers = num_layers
    self._output_size = output_size
    self._embedding_size = self._value_size * self._num_heads
    self._mlp_hidden_sizes = list(mlp_hidden_sizes) + [self._embedding_size]
    self._multihead_attention = None
    self._object_embeddings = None
    self._dropout_rate = dropout_rate
    self._positional_encodings = None
    self._use_relative_positions = use_relative_positions
    self._clamp_time_range = clamp_time_range
    self._same_attention_length = same_attention_length
    self._layer_norm = layer_norm
    self._attention_modules = []
    self._object_mlps = []

  def get_sublayers(self, is_training):
    if self._multihead_attention is None or not self._shared_attention:
      attention_module = MultiheadAttention(
          value_size=self._value_size,
          key_size=self._key_size,
          num_heads=self._num_heads,
          mask=self._mask,
          positional_encodings=self._positional_encodings,
          use_relative_positions=self._use_relative_positions,
          init_std=2. / np.sqrt(self._num_layers),
      )
      self._multihead_attention = ResidualDropoutWrapper(
          attention_module, self._dropout_rate, layer_norm=self._layer_norm)
    mlp = default_mlp(
        self._mlp_hidden_sizes, init_std=2. / np.sqrt(self._num_layers))
    object_mlp = ResidualDropoutWrapper(
        mlp, self._dropout_rate, layer_norm=self._layer_norm)

    self._attention_modules.append(attention_module)
    self._object_mlps.append(mlp)
    return self._multihead_attention, object_mlp

  def _build(self,
             inputs,
             state=None,
             condition=None,
             is_training=True,
             final_layer_key_value_inputs=None):
    """Calculates multi-layer self attention and mlp transformation.

    Args:
      inputs: Tensor of shape [batch_size, num_steps, dim_size].
      state: optional list of length num_layers of tensors of shape
        [batch_size, memory_size, dim_size].
      condition: optional tensor to condition on. The shape is shape
        [batch_size, dim_size].
      is_training: If true, dropout is applied.
      final_layer_key_value_inputs: optional Tensor to be used as the key and
        value for the final multi-head attention layer of shape
        [batch_size, num_steps, dim_size]. Useful when the tower is a Seq2Seq
        decoder and it can attend to encoder outputs.

    Returns:
      output: tensor of shape [batch_size, num_steps, output_dim_size].
      state: list of length `num_layers` containing AttentionState tuples.
    """
    # inputs: [B, N, F]
    if final_layer_key_value_inputs is not None and state is not None and len(
        state) == (self._num_layers - 1):
      raise ValueError('When the final_layer_key_value_input is set, exclude'
                       'the state of the last layer.')

    if condition is not None:
      condition_tile = tf.tile(
          tf.expand_dims(condition, 1), [1, tf.shape(inputs)[1], 1])
      inputs = tf.concat([inputs, condition_tile], -1)

    # Map inputs to be of `embedding_size` dimension.
    if inputs.get_shape().as_list()[-1] != self._embedding_size:
      inputs = default_mlp([self._embedding_size], activate_final=True)(
          inputs,
          is_training=is_training,
          dropout_keep_prob=1 - self._dropout_rate)

    if state is None:
      memory_sizes = [0]
    elif isinstance(state[0], CompressedMemoryState):
      cm_mem_size = max(_memory_size(s.compressed_memory) for s in state)
      em_mem_size = max(_memory_size(s.episodic_memory) for s in state)
      memory_sizes = [cm_mem_size, em_mem_size]
    else:
      memory_sizes = [max([_memory_size(s) for s in state])]
    chunk_size = inputs.get_shape().as_list()[1]
    self._positional_encodings = []
    # Creates positional encodings for different memory types.
    for i, memory_size in enumerate(memory_sizes):
      seq_len = chunk_size + memory_size
      key_positions = get_position_encodings(
          sequence_length=seq_len,
          hidden_size=inputs.get_shape().as_list()[2],
          clamp_value=self._clamp_time_range,
      )
      if is_training:
        key_positions = tf.nn.dropout(key_positions, rate=self._dropout_rate)
      key_positions = tf.cast(key_positions, dtype=inputs.dtype)
      query_positions = key_positions[:, -chunk_size:, :]
      self._positional_encodings.append((key_positions, query_positions))

    if self._causal:
      self._mask = create_mask(inputs, state, self._same_attention_length)

    layer_i_inputs = inputs
    attention_states = []
    key_value_inputs = None

    for i in range(self._num_layers):
      with tf.variable_scope('layer_%d' % i, reuse=tf.AUTO_REUSE):
        multihead_attention, object_mlp = self.get_sublayers(is_training)
        # Multihead attention with residuals.
        state_i = None if state is None else state[i]
        if i == (self._num_layers -
                 1) and final_layer_key_value_inputs is not None:
          # When the final_layer_key_value_inputs is set, the finaly layer
          # of attention will use it as the key & value, thus no need for state.
          key_value_inputs = final_layer_key_value_inputs
          state_i = None

        attention_outputs, attention_state = multihead_attention(
            layer_i_inputs,
            state=state_i,
            is_training=is_training,
            dropout_keep_prob=1. - self._dropout_rate,
            key_value_inputs=key_value_inputs)
        attention_states.append(attention_state)
        # Feed-forward with residuals.
        output = object_mlp(
            attention_outputs,
            is_training=is_training,
            dropout_keep_prob=1 - self._dropout_rate)
        layer_i_inputs = output

    if self._output_size is not None:
      output = basic.BatchApply(
          basic.Linear(self._output_size, use_bias=False))(
              output)

    return output, attention_states

  def attention_module(self, i):
    """Returns the i-th layer attention module."""
    return self._attention_modules[i]
